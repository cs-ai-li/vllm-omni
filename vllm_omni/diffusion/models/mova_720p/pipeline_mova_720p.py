import os
import re
import html
import math
import json
from typing import Iterable, Optional, List, Union, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5TokenizerFast, UMT5EncoderModel
from tqdm import tqdm

# 尝试导入 ftfy，如果不存在则回退到基础清洗
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

from .mova_720p_transformer import Mova720PTransformer2DModel


def basic_clean(text):
    if HAS_FTFY:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def sinusoidal_embedding_1d(dim, t):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) /
        half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Mova720PPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Load components from checkpoint
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only)
        
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only).to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only)
        self.vae = AutoencoderKL.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only).to(self.device)

        # Initialize transformer with vLLM-Omni config
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config, Mova720PTransformer2DModel)
        self.transformer = Mova720PTransformer2DModel(
            od_config=od_config, **transformer_kwargs)

        # Weight loading configuration
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                prefix="transformer.",
            )
        ]

        # VAE and latent settings
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = 64

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
        
        prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # MOVA 720p 默认 VAE 缩放
        vae_spatial_factor = 8
        vae_temporal_factor = 4
        
        num_latent_frames = (num_frames - 1) // vae_temporal_factor + 1
        latent_height = height // vae_spatial_factor
        latent_width = width // vae_spatial_factor

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(shape, device=device, dtype=dtype)
        
        return latents

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        # 1. 提取请求参数
        prompts = req.prompts
        if prompts is not None:
            prompt = [
                p if isinstance(p, str) else (p.get("prompt") or "")
                for p in prompts
            ]
        else:
            prompt = [""]

        sampling_params = req.sampling_params
        num_inference_steps = sampling_params.num_inference_steps or 50
        guidance_scale = sampling_params.guidance_scale or 5.0
        # MOVA 720p 默认尺寸: 720x1280
        height = sampling_params.height or 720
        width = sampling_params.width or 1280
        num_frames = getattr(sampling_params, "num_frames", 1)

        # 2. 准备 Latents
        batch_size = len(prompt)
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.bfloat16,
            self.device,
        )

        # 3. 准备 Text Embeddings (支持 CFG)
        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        if guidance_scale > 1.0:
            negative_prompt_embeds = self._get_t5_prompt_embeds([""] * batch_size)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. 扩散循环
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(tqdm(timesteps)):
            # 扩展 latent 以适应 CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # 计算时间步嵌入
            t_emb = sinusoidal_embedding_1d(self.transformer.freq_dim, t.expand(latent_model_input.shape[0]))
            t_emb = t_emb.to(device=self.device, dtype=torch.bfloat16)

            # Transformer 推理
            noise_pred = self.transformer(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                t_emb=t_emb,
            )

            # CFG 引导
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler 步进
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 5. VAE 解码
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def get_mova_720p_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    创建 MOVA 720p 模型的后处理函数。
    """
    from diffusers.image_processor import VaeImageProcessor

    # 加载 VAE 配置以获取 scale factor
    model_path = od_config.model
    if not os.path.exists(model_path):
        from vllm_omni.diffusion.model_loader.utils import download_weights_from_hf_specific
        model_path = download_weights_from_hf_specific(model_path, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)

    # 创建图像处理器
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        # 如果是视频 Tensor [B, C, F, H, W]，此处可能需要特殊处理
        # 目前先处理单图或简单的序列
        if images.ndim == 5:
            # 取第一帧或压平 batch 处理
            images = images[:, :, 0, :, :]
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func
