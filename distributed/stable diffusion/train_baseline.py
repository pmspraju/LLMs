"""Single-GPU baseline fine-tuning of Stable Diffusion 1.5.

Stage 2 of the learning path: the reference run that DDP (stage 3) and FSDP
(stage 4) will be compared against.

Two modes (see sdtrain/config.py):
    python train_baseline.py --mode local_debug   # LoRA, 256px, ~100 steps (RTX 4080)
    python train_baseline.py --mode cloud_full     # full UNet, 512px, 3000 steps (A6000)

Logged every optimizer step (CSV, and W&B if --use_wandb):
    training loss, learning rate, gradient norm, step wall-time,
    images/sec, peak GPU memory (MB).
Fixed-prompt samples and checkpoints are written at configured intervals.

Only the main process (rank 0) logs / samples / checkpoints. For this single-GPU
baseline that is always true, but the guards are already here so the DDP/FSDP
conversions are minimal.
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from diffusers.optimization import get_scheduler

from sdtrain.config import TrainConfig, get_config
from sdtrain.data import build_dataloader
from sdtrain.dist_utils import get_rank, get_world_size, is_main_process
from sdtrain.logging_utils import (
    CSVLogger,
    WandbLogger,
    peak_memory_mb,
    reset_peak_memory,
)
from sdtrain.models import count_parameters, load_models, trainable_parameters
from sdtrain.samples import generate_samples, save_grid


CSV_FIELDS = [
    "step",
    "epoch",
    "loss",
    "lr",
    "grad_norm",
    "step_time_s",
    "images_per_sec",
    "peak_mem_mb",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD1.5 single-GPU baseline")
    p.add_argument("--mode", choices=["local_debug", "cloud_full"], default="local_debug")
    # Common overrides (leave as None to keep the mode preset's value).
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pretrained_model", default=None)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    for name in [
        "output_dir",
        "pretrained_model",
        "max_train_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "resolution",
        "learning_rate",
        "mixed_precision",
        "seed",
        "run_name",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.use_wandb:
        cfg.use_wandb = True
    return cfg


def resolve_dtype(mixed_precision: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[mixed_precision]


def infinite_loader(dataloader, sampler):
    """Yield (epoch, batch) forever, re-shuffling the DistributedSampler each epoch."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield epoch, batch
        epoch += 1


def compute_loss(cfg, batch, unet, vae, text_encoder, noise_scheduler, device, weight_dtype):
    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)

    # Image -> latent space (frozen VAE). Scaling factor keeps latents ~unit variance.
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Forward diffusion: add noise at a random timestep per sample.
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Text conditioning (frozen text encoder).
    encoder_hidden_states = text_encoder(input_ids)[0]

    # The UNet predicts either the noise (epsilon) or the velocity (v-prediction),
    # depending on how the checkpoint's scheduler was configured. SD1.5 is epsilon.
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    # Loss in fp32 for numerical stability regardless of autocast dtype.
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def save_checkpoint(cfg, unet, step):
    ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if cfg.finetune_mode == "lora":
        from peft.utils import get_peft_model_state_dict

        lora_state = get_peft_model_state_dict(unet)
        torch.save(lora_state, os.path.join(ckpt_dir, "unet_lora.pt"))
    else:
        unet.save_pretrained(os.path.join(ckpt_dir, "unet"))
    return ckpt_dir


def main():
    args = parse_args()
    cfg = apply_overrides(get_config(args.mode), args)

    world_size = get_world_size()
    rank = get_rank()
    main_process = is_main_process()

    torch.manual_seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_dtype(cfg.mixed_precision)

    # ---- output dir + provenance ----------------------------------------
    if main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.to_json(os.path.join(cfg.output_dir, "config.json"))

    # ---- models ----------------------------------------------------------
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(cfg)

    # Frozen models run in the low-precision weight dtype to save memory.
    # The *trainable* UNet stays in fp32; autocast handles the fast math.
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=torch.float32)

    if main_process:
        trainable, total = count_parameters(unet)
        print(f"[rank {rank}] mode={cfg.mode} finetune={cfg.finetune_mode} "
              f"device={device} dtype={weight_dtype}")
        print(f"[rank {rank}] UNet trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")

    # ---- data ------------------------------------------------------------
    dataloader, sampler = build_dataloader(cfg, tokenizer, world_size, rank)

    # ---- optimizer + LR schedule ----------------------------------------
    params = trainable_parameters(unet)
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )

    use_scaler = cfg.mixed_precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---- loggers (rank 0 only) ------------------------------------------
    csv_logger = wandb_logger = None
    if main_process:
        csv_logger = CSVLogger(os.path.join(cfg.output_dir, "metrics.csv"), CSV_FIELDS)
        if cfg.use_wandb:
            from dataclasses import asdict
            wandb_logger = WandbLogger(cfg.wandb_project, cfg.run_name, asdict(cfg))

    # ---- training loop ---------------------------------------------------
    unet.train()
    data_iter = infinite_loader(dataloader, sampler)
    accum = cfg.gradient_accumulation_steps
    images_per_step = cfg.train_batch_size * accum * world_size

    reset_peak_memory(device)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    step_start = time.perf_counter()
    accum_loss = 0.0
    current_epoch = 0

    while global_step < cfg.max_train_steps:
        for micro in range(accum):
            current_epoch, batch = next(data_iter)
            autocast_ctx = (
                torch.autocast("cuda", dtype=weight_dtype)
                if device.type == "cuda" and weight_dtype != torch.float32
                else nullcontext()
            )
            with autocast_ctx:
                loss = compute_loss(
                    cfg, batch, unet, vae, text_encoder, noise_scheduler, device, weight_dtype
                )
            accum_loss += loss.item()
            scaler.scale(loss / accum).backward()

        # ---- optimizer step (once per `accum` micro-batches) -------------
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        global_step += 1

        # ---- logging (rank 0) --------------------------------------------
        if main_process and global_step % cfg.log_every == 0:
            row = {
                "step": global_step,
                "epoch": current_epoch,
                "loss": accum_loss / accum,
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "step_time_s": step_time,
                "images_per_sec": images_per_step / step_time,
                "peak_mem_mb": peak_memory_mb(device),
            }
            csv_logger.log(row)
            if wandb_logger is not None:
                wandb_logger.log(row, step=global_step)
            print(
                f"step {global_step:>5}/{cfg.max_train_steps} | "
                f"loss {row['loss']:.4f} | lr {row['lr']:.2e} | "
                f"grad {row['grad_norm']:.3f} | {row['images_per_sec']:.2f} img/s | "
                f"{row['peak_mem_mb']:.0f} MB"
            )

        # ---- fixed-prompt samples (rank 0) -------------------------------
        if main_process and cfg.sample_every > 0 and global_step % cfg.sample_every == 0:
            grid = generate_samples(
                cfg, unet, vae, text_encoder, tokenizer, noise_scheduler, device, weight_dtype
            )
            path = save_grid(grid, cfg.output_dir, global_step)
            print(f"  saved samples -> {path}")

        # ---- checkpoint (rank 0) -----------------------------------------
        if main_process and cfg.checkpoint_every > 0 and global_step % cfg.checkpoint_every == 0:
            ckpt = save_checkpoint(cfg, unet, global_step)
            print(f"  saved checkpoint -> {ckpt}")

        accum_loss = 0.0
        step_start = time.perf_counter()

    # ---- final artifacts -------------------------------------------------
    if main_process:
        ckpt = save_checkpoint(cfg, unet, global_step)
        grid = generate_samples(
            cfg, unet, vae, text_encoder, tokenizer, noise_scheduler, device, weight_dtype
        )
        save_grid(grid, cfg.output_dir, global_step)
        print(f"Training complete. Final checkpoint: {ckpt}")
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
