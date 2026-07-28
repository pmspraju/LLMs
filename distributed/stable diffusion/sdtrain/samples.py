"""Fixed-prompt sample generation.

We run the classifier-free-guidance text->image loop *by hand* (rather than
building a full `StableDiffusionPipeline`) for two reasons:

  1. It is transparent -- you can see exactly the same denoising loop you would
     write in inference, which is useful while learning.
  2. It sidesteps dtype friction: during training the UNet lives in fp32 while
     the VAE/text-encoder may live in fp16/bf16. Wrapping this loop in autocast
     lets each op cast its inputs as needed instead of fighting the pipeline's
     internal `.to(dtype)` assumptions.

Sampling always uses a fixed seed and a fixed prompt list so that images at
step 250, 500, 750, ... are directly comparable (and later comparable across
the baseline / DDP / FSDP runs).
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import List

import torch
from PIL import Image

from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid

from .config import TrainConfig


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def generate_samples(
    cfg: TrainConfig,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Image.Image:
    """Return a single PIL grid image (one column per prompt)."""
    was_training = unet.training
    grad_ckpt_was_on = getattr(unet, "is_gradient_checkpointing", False)
    unet.eval()
    if grad_ckpt_was_on:
        unet.disable_gradient_checkpointing()

    prompts: List[str] = cfg.sample_prompts
    n = len(prompts)

    scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(cfg.sample_num_inference_steps, device=device)

    # --- text conditioning (unconditional + conditional for CFG) ----------
    def encode(text_list):
        toks = tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(toks)[0]

    with _autocast(device, weight_dtype):
        cond = encode(prompts)
        uncond = encode([""] * n)
        context = torch.cat([uncond, cond], dim=0)

        # --- initial latents (fixed seed for reproducibility) -------------
        generator = torch.Generator(device=device).manual_seed(cfg.sample_seed)
        latent_h = cfg.resolution // 8
        latents = torch.randn(
            (n, unet.config.in_channels, latent_h, latent_h),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        latents = latents * scheduler.init_noise_sigma

        # --- denoising loop with classifier-free guidance -----------------
        for t in scheduler.timesteps:
            latent_in = torch.cat([latents] * 2)
            latent_in = scheduler.scale_model_input(latent_in, t)
            noise_pred = unet(latent_in, t, encoder_hidden_states=context).sample
            uncond_pred, cond_pred = noise_pred.chunk(2)
            noise_pred = uncond_pred + cfg.sample_guidance_scale * (cond_pred - uncond_pred)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- decode latents -> pixels -------------------------------------
        images = vae.decode(latents / vae.config.scaling_factor).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in images]
    grid = make_image_grid(pil_images, rows=1, cols=n)

    # restore training state
    if grad_ckpt_was_on:
        unet.enable_gradient_checkpointing()
    if was_training:
        unet.train()
    return grid


def save_grid(grid: Image.Image, output_dir: str, step: int) -> str:
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, f"samples_step{step:06d}.png")
    grid.save(path)
    return path
