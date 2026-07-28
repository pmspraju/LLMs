"""Configuration presets for the two run modes.

`local_debug`  -> LoRA, low resolution, tiny batch, few steps. Fits an RTX 4080.
                  Purpose: prove the pipeline runs end-to-end before paying for cloud.
`cloud_full`   -> full UNet fine-tune at 512px, real batch size, many steps.
                  Purpose: the true single-GPU baseline on a Lambda A6000 (48GB).

Everything is a plain dataclass so it can be serialized to JSON (for run
provenance) and overridden from the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class TrainConfig:
    # --- run identity -----------------------------------------------------
    mode: str = "local_debug"          # "local_debug" | "cloud_full"
    finetune_mode: str = "lora"        # "lora" | "full"

    # --- model ------------------------------------------------------------
    # NOTE: the original "runwayml/stable-diffusion-v1-5" repo was removed from
    # the Hub. This community re-upload is the current canonical SD1.5 repo.
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: Optional[str] = None

    # --- data -------------------------------------------------------------
    #dataset_name: str = "lambdalabs/pokemon-blip-captions"
    dataset_name: str = "diffusers/pokemon-gpt4-captions"
    dataset_split: str = "train"
    resolution: int = 256
    center_crop: bool = True
    random_flip: bool = True
    num_workers: int = 0

    # --- optimization -----------------------------------------------------
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 100
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant_with_warmup"  # see diffusers.optimization
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # --- precision / memory ----------------------------------------------
    mixed_precision: str = "bf16"      # "no" | "fp16" | "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42

    # --- LoRA (only used when finetune_mode == "lora") -------------------
    lora_rank: int = 4
    lora_alpha: int = 4

    # --- logging / checkpoints -------------------------------------------
    output_dir: str = "outputs/local_debug"
    log_every: int = 1                 # write a CSV row every N optimizer steps
    sample_every: int = 50             # generate fixed-prompt samples every N steps
    checkpoint_every: int = 100        # save weights every N steps
    use_wandb: bool = False
    wandb_project: str = "sd15-distributed-learning"
    run_name: Optional[str] = None

    # --- fixed-prompt sampling -------------------------------------------
    sample_prompts: List[str] = field(default_factory=lambda: [
        "a cute yellow electric pokemon, cartoon",
        "a blue water-type dragon pokemon",
        "a fiery red lizard pokemon with a flame tail",
        "a small green grass pokemon with a leaf",
    ])
    sample_guidance_scale: float = 7.5
    sample_num_inference_steps: int = 25
    sample_seed: int = 1234

    # --------------------------------------------------------------------
    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def local_debug_config() -> TrainConfig:
    """Cheap correctness check on the RTX 4080 (LoRA, 256px, ~100 steps)."""
    return TrainConfig(
        mode="local_debug",
        finetune_mode="lora",
        resolution=256,
        num_workers=0,                 # 0 avoids Windows multiprocessing headaches
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_steps=100,
        learning_rate=1e-4,
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=10,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        lora_rank=4,
        lora_alpha=4,
        output_dir="outputs/local_debug",
        log_every=1,
        sample_every=50,
        checkpoint_every=100,
    )


def cloud_full_config() -> TrainConfig:
    """True single-GPU baseline on a Lambda A6000 (full UNet fine-tune, 512px)."""
    return TrainConfig(
        mode="cloud_full",
        finetune_mode="full",
        resolution=512,
        num_workers=4,
        train_batch_size=4,
        gradient_accumulation_steps=1,
        max_train_steps=3000,
        learning_rate=1e-5,            # full fine-tunes want a smaller LR than LoRA
        lr_scheduler="cosine",
        lr_warmup_steps=300,
        mixed_precision="bf16",        # A6000 (Ampere) has fast bf16, avoids GradScaler
        gradient_checkpointing=True,
        output_dir="outputs/cloud_full_baseline",
        log_every=1,
        sample_every=250,
        checkpoint_every=500,
    )


def get_config(mode: str) -> TrainConfig:
    if mode == "local_debug":
        return local_debug_config()
    if mode == "cloud_full":
        return cloud_full_config()
    raise ValueError(f"Unknown mode: {mode!r}. Use 'local_debug' or 'cloud_full'.")
