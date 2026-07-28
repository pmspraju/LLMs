"""Model loading and (optional) LoRA setup.

Returns the four SD1.5 sub-models plus the training noise scheduler:
  * tokenizer / text_encoder : frozen, turns captions into conditioning
  * vae                       : frozen, maps images <-> latents
  * unet                      : the thing we actually train
  * noise_scheduler (DDPM)    : defines the forward noising process for training

For `finetune_mode == "full"` the whole UNet is trainable and kept in fp32
(mixed precision is applied via autocast in the training loop). For
`finetune_mode == "lora"` the UNet weights are frozen and small LoRA adapters
are injected into the attention projections -- this is what makes the local
4080 debug run cheap.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .config import TrainConfig


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def load_models(
    cfg: TrainConfig,
) -> Tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler]:
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model, subfolder="tokenizer", revision=cfg.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model, subfolder="text_encoder", revision=cfg.revision
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model, subfolder="vae", revision=cfg.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model, subfolder="unet", revision=cfg.revision
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model, subfolder="scheduler", revision=cfg.revision
    )

    # The VAE and text encoder are always frozen -- we never train them here.
    _freeze(vae)
    _freeze(text_encoder)

    if cfg.finetune_mode == "lora":
        _freeze(unet)
        _add_lora(cfg, unet)
    elif cfg.finetune_mode == "full":
        unet.requires_grad_(True)
    else:
        raise ValueError(f"Unknown finetune_mode: {cfg.finetune_mode!r}")

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    return tokenizer, text_encoder, vae, unet, noise_scheduler


def _add_lora(cfg: TrainConfig, unet: UNet2DConditionModel) -> None:
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)


def trainable_parameters(unet: UNet2DConditionModel) -> List[nn.Parameter]:
    return [p for p in unet.parameters() if p.requires_grad]


def count_parameters(unet: UNet2DConditionModel) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    return trainable, total
