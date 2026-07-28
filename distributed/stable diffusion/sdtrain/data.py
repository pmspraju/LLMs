"""Data pipeline for lambdalabs/pokemon-blip-captions.

The dataset has two columns: `image` (PIL.Image) and `text` (a BLIP caption).
Images are resized/cropped to `resolution` and normalized to [-1, 1] (the range
the VAE expects). Captions are tokenized with the SD1.5 CLIP tokenizer.

`build_dataloader` returns a `(dataloader, sampler)` pair. In the single-GPU
baseline `sampler` is None; the DDP/FSDP stages will swap in a
`DistributedSampler` here without touching the rest of the training loop.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from transformers import CLIPTokenizer

from .config import TrainConfig


def _build_transforms(cfg: TrainConfig) -> transforms.Compose:
    ops = [
        transforms.Resize(
            cfg.resolution, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(cfg.resolution)
        if cfg.center_crop
        else transforms.RandomCrop(cfg.resolution),
    ]
    if cfg.random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),                     # -> [0, 1], CxHxW
        transforms.Normalize([0.5], [0.5]),        # -> [-1, 1]
    ]
    return transforms.Compose(ops)


def build_dataloader(
    cfg: TrainConfig,
    tokenizer: CLIPTokenizer,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[DataLoader, Optional[torch.utils.data.Sampler]]:
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    image_transforms = _build_transforms(cfg)

    def preprocess(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        pixel_values = [image_transforms(img) for img in images]
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": tokens.input_ids}

    dataset.set_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([e["input_ids"] for e in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Single-GPU baseline: no distributed sampler yet.
    sampler: Optional[torch.utils.data.Sampler] = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, sampler
