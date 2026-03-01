# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
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
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    # 增加维度以适配 [B, L, H, D] 的广播要求
    cos = freqs_cos.view(1, freqs_cos.shape[0], 1, -1)
    sin = freqs_sin.view(1, freqs_sin.shape[0], 1, -1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class MovaRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int = 1048576,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        # 在实际实现中，这里应包含 RoPE 的预计算逻辑
        self.register_buffer("freqs_cos", torch.zeros(max_seq_len, attention_head_dim), persistent=False)
        self.register_buffer("freqs_sin", torch.zeros(max_seq_len, attention_head_dim), persistent=False)

    def forward(self, seq_len: int):
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]


class MovaPatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
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
    def __init__(self, dim: int, num_heads: int, parallel_config: Any):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            pc=parallel_config,
        )

        self.norm_q = RMSNorm(dim, eps=1e-6)
        self.norm_k = RMSNorm(dim, eps=1e-6)

        self.to_out = ReplicatedLinear(
            input_size=dim,
            output_size=dim,
            pc=parallel_config,
        )

        self.attn_op = Attention(
            num_heads=num_heads // (parallel_config.tp_size if parallel_config else 1),
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
        )

    def forward(self, x, freqs_cos, freqs_sin):
        B, L, _ = x.shape
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.view(B, L, -1, self.head_dim)
        k = k.view(B, L, -1, self.head_dim)
        v = v.view(B, L, -1, self.head_dim)

        q = apply_rotary_emb_wan(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb_wan(k, freqs_cos, freqs_sin)

        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
        attn_metadata = AttentionMetadata()
        x = self.attn_op(q, k, v, attn_metadata=attn_metadata)

        x = x.reshape(B, L, -1)
        x, _ = self.to_out(x)
        return x


class MovaCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, parallel_config: Any):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = ReplicatedLinear(dim, dim, pc=parallel_config)
        self.to_k = ReplicatedLinear(dim, dim, pc=parallel_config)
        self.to_v = ReplicatedLinear(dim, dim, pc=parallel_config)

        self.norm_q = RMSNorm(dim, eps=1e-6)
        self.norm_k = RMSNorm(dim, eps=1e-6)

        self.to_out = ReplicatedLinear(dim, dim, pc=parallel_config)

        self.attn_op = Attention(
            num_heads=num_heads // (parallel_config.tp_size if parallel_config else 1),
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
        )

    def forward(self, x, context, mask=None):
        B, L, _ = x.shape
        _, L_ctx, _ = context.shape

        q, _ = self.to_q(x)
        q = self.norm_q(q)
        k, _ = self.to_k(context)
        k = self.norm_k(k)
        v, _ = self.to_v(context)

        q = q.view(B, L, -1, self.head_dim)
        k = k.view(B, L_ctx, -1, self.head_dim)
        v = v.view(B, L_ctx, -1, self.head_dim)

        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
        attn_metadata = AttentionMetadata(attn_mask=mask)
        out = self.attn_op(q, k, v, attn_metadata=attn_metadata)

        out = out.reshape(B, L, -1)
        out, _ = self.to_out(out)
        return out


class MovaTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, parallel_config: Any):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = MovaSelfAttention(dim=dim, num_heads=num_heads, parallel_config=parallel_config)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.norm3 = RMSNorm(dim, eps=1e-6)
        self.cross_attn = MovaCrossAttention(dim=dim, num_heads=num_heads, parallel_config=parallel_config)

        self.ffn = nn.Sequential(
            ReplicatedLinear(dim, ffn_dim, pc=parallel_config),
            nn.GELU(approximate='tanh'),
            ReplicatedLinear(ffn_dim, dim, pc=parallel_config),
        )

    def forward(self, x, context, t_mod, freqs_cos, freqs_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod.chunk(6, dim=1)

        norm_x = self.norm1(x)
        msa_input = norm_x * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(msa_input, freqs_cos, freqs_sin)

        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context)

        norm_x2 = self.norm2(x)
        mlp_input = norm_x2 * (1 + scale_mlp) + shift_mlp
        mlp_out, _ = self.ffn[0](mlp_input)
        mlp_out = self.ffn[1](mlp_out)
        mlp_out, _ = self.ffn[2](mlp_out)
        x = x + gate_mlp * mlp_out

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
        self.parallel_config = od_config.parallel_config if od_config else None
        self.patch_size = patch_size
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels

        self.patch_embed = MovaPatchEmbedding(patch_size, in_channels, self.inner_dim)
        self.rope = MovaRotaryPosEmbed(attention_head_dim, patch_size, 1048576)

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
            MovaTransformerBlock(self.inner_dim, num_attention_heads, ffn_dim, self.parallel_config)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.inner_dim, eps=eps, elementwise_affine=False)
        self.head_modulation = nn.Parameter(torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5)

        self.proj_out = ReplicatedLinear(
            input_size=self.inner_dim,
            output_size=out_channels * math.prod(patch_size),
            pc=self.parallel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        b, c, t, h, w = hidden_states.shape
        hidden_states = self.patch_embed(hidden_states)

        # 这里的 t_emb 需要根据实际的 timestep 预处理得到，此处假设已有
        t_feat = self.time_embedding(kwargs.get('t_emb'))
        t_mod = self.time_projection(t_feat).view(b, 6, -1)
        context = self.text_embedding(encoder_hidden_states)

        freqs_cos, freqs_sin = self.rope(hidden_states.shape[1])

        for block in self.blocks:
            hidden_states = block(hidden_states, context, t_mod, freqs_cos, freqs_sin)

        mod_params = self.head_modulation + t_feat.unsqueeze(1)
        shift, scale = mod_params.chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        hidden_states, _ = self.proj_out(hidden_states)

        p_t, p_h, p_w = self.patch_size
        # 严格遵循 MOVA 的维度还原顺序：b, c, (f p_t), (h p_h), (w p_w)
        hidden_states = hidden_states.reshape(
            b, t // p_t, h // p_h, w // p_w, p_t, p_h, p_w, self.out_channels
        )
        # permute(batch, channel, f, p_t, h, p_h, w, p_w)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.reshape(b, self.out_channels, t, h, w)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = default_weight_loader
        for name, loaded_weight in weights:
            # 这里的核心是处理 QKV 合并后的加载
            if "attn.to_qkv" in name:
                # vLLM 的 QKVParallelLinear 内部存储方式通常为 [Q, K, V] 依次排列
                # 我们需要根据加载的 sub-key (q, k, 或 v) 来确定偏移量
                param = self.get_parameter(name)
                # 假设 loaded_weight 已经是拼接好的，或者通过 loader 的分片逻辑处理
                loader(param, loaded_weight)
            elif "attn.to_q" in name or "attn.to_k" in name or "attn.to_v" in name:
                # 针对 MOVA 原始分开存储权重的兼容逻辑
                param_name = name.replace("to_q", "to_qkv").replace("to_k", "to_qkv").replace("to_v", "to_qkv")
                param = self.get_parameter(param_name)
                
                # 计算偏移量：q=0, k=1, v=2
                qkv_dim = param.shape[0] // 3
                if "to_q" in name:
                    offset = 0
                elif "to_k" in name:
                    offset = qkv_dim
                else:
                    offset = 2 * qkv_dim
                
                # 使用 slice 填入对应的权重位置
                param.data[offset:offset+qkv_dim].copy_(loaded_weight)
            else:
                try:
                    loader(self.get_parameter(name), loaded_weight)
                except AttributeError:
                    logger.debug(f"Skipping weight {name}")
