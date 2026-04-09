# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def apply_rotary_emb_wan(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensors.
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class MovaRotaryPosEmbed(nn.Module):
    """
    Rotary position embeddings for 3D video data (temporal + spatial dimensions).
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Split dimensions for temporal, height, width
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = self._get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    @staticmethod
    def _get_1d_rotary_pos_embed(
        dim: int,
        max_seq_len: int,
        theta: float,
        freqs_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
        t = torch.arange(max_seq_len, dtype=freqs_dtype)
        freqs = torch.outer(t, freqs)
        freqs_cos = freqs.cos().float().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().float().repeat_interleave(2, dim=-1)
        return freqs_cos, freqs_sin

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class MovaPatchEmbedding(nn.Module):

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Conv3dLayer(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MovaSelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
        )

        self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-6)

        self.to_out = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=True,
            input_is_parallel=True,
        )

        self.attn_op = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x, rotary_emb):
        B, L, _ = x.shape
        qkv, _ = self.to_qkv(x)
        
        q_size = self.to_qkv.num_heads * self.head_dim
        k_size = self.to_qkv.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, k_size, k_size], dim=-1)

        # Reshape for norm
        q = q.view(B, L, -1, self.head_dim)
        k = k.view(B, L, -1, self.head_dim)
        v = v.view(B, L, -1, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply rotary embeddings
        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            q = apply_rotary_emb_wan(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb_wan(k, freqs_cos, freqs_sin)

        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
        attn_metadata = AttentionMetadata()
        x = self.attn_op(q, k, v, attn_metadata=attn_metadata)

        x = x.reshape(B, L, -1)
        x, _ = self.to_out(x)
        return x


class MovaCrossAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.to_k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.to_v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)

        self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-6)

        self.to_out = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)

        self.attn_op = Attention(
            num_heads=num_heads, # Simplified for cross-attention
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x, context, mask=None):
        B, L, _ = x.shape
        _, L_ctx, _ = context.shape

        q, _ = self.to_q(x)
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)

        q = q.view(B, L, -1, self.head_dim)
        k = k.view(B, L_ctx, -1, self.head_dim)
        v = v.view(B, L_ctx, -1, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
        attn_metadata = AttentionMetadata(attn_mask=mask)
        out = self.attn_op(q, k, v, attn_metadata=attn_metadata)

        out = out.reshape(B, L, -1)
        out, _ = self.to_out(out)
        return out


class MovaTransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = MovaSelfAttention(dim=dim, num_heads=num_heads)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.norm3 = RMSNorm(dim, eps=1e-6)
        self.cross_attn = MovaCrossAttention(dim=dim, num_heads=num_heads)

        self.ffn = nn.ModuleDict({
            "proj_1": ColumnParallelLinear(dim, ffn_dim, bias=True, gather_output=False),
            "proj_2": RowParallelLinear(ffn_dim, dim, bias=True, input_is_parallel=True)
        })

    def forward(self, x, context, t_mod, rotary_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod.chunk(6, dim=1)

        norm_x = self.norm1(x)
        msa_input = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(msa_input, rotary_emb)

        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context)

        norm_x2 = self.norm2(x)
        mlp_input = norm_x2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        mlp_out, _ = self.ffn["proj_1"](mlp_input)
        mlp_out = F.gelu(mlp_out, approximate='tanh')
        mlp_out, _ = self.ffn["proj_2"](mlp_out)
        
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class Mova720PTransformer2DModel(nn.Module):
    _sp_plan = {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.od_config = od_config
        self.patch_size = patch_size
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels

        self.patch_embed = MovaPatchEmbedding(patch_size, in_channels, self.inner_dim)
        self.rope = MovaRotaryPosEmbed(attention_head_dim, patch_size, 1024) # Typical max frames*spatial

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.inner_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim * 6)
        )

        self.blocks = nn.ModuleList([
            MovaTransformerBlock(self.inner_dim, num_attention_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.inner_dim, eps=eps, elementwise_affine=False)
        self.head_modulation = nn.Parameter(torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5)

        self.proj_out = RowParallelLinear(
            input_size=self.inner_dim,
            output_size=out_channels * math.prod(patch_size),
            bias=True,
            input_is_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        b, c, t, h, w = hidden_states.shape
        
        # Compute RoPE before flattening
        rotary_emb = self.rope(hidden_states)
        
        hidden_states = self.patch_embed(hidden_states)

        # Handle timestep feature
        t_feat = self.time_embedding(kwargs.get('t_emb'))
        t_mod = self.time_projection(t_feat) # [B, inner_dim * 6]
        
        context = self.text_embedding(encoder_hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, context, t_mod, rotary_emb)

        mod_params = self.head_modulation + t_feat.unsqueeze(1)
        shift, scale = mod_params.chunk(2, dim=1)
        
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        hidden_states, _ = self.proj_out(hidden_states)

        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            b, t // p_t, h // p_h, w // p_w, p_t, p_h, p_w, self.out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.reshape(b, self.out_channels, t, h, w)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from a pretrained model, handling the mapping from
        separate Q/K/V projections to fused QKV projections for self-attention.
        """
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Stacked params mapping for self-attention QKV fusion
        stacked_params_mapping = [
            (".attn.to_qkv", ".attn.to_q", "q"),
            (".attn.to_qkv", ".attn.to_k", "k"),
            (".attn.to_qkv", ".attn.to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loader = AutoWeightsLoader(self)

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            # Handle QKV fusion for self-attention
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                if lookup_name in params_dict:
                    param = params_dict[lookup_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(original_name)
                    break
            else:
                # Handle weight name remapping for cross-attention and FFN
                # diffusers: ffn.0 -> our: ffn.proj_1, diffusers: ffn.2 -> our: ffn.proj_2
                if ".ffn.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.0.", ".ffn.proj_1.")
                elif ".ffn.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.2.", ".ffn.proj_2.")
                
                if lookup_name not in params_dict:
                    logger.debug(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                # Handle RMSNorm and other sharded weights that need manual TP slicing
                # if not using a weight_loader that handles it automatically
                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [".norm_q.", ".norm_k.", ".attn.norm_q.", ".attn.norm_k."]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(original_name)

        return loaded_params
