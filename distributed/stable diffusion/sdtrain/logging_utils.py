"""Lightweight metric logging.

Design choice -- CSV + (optional) Weights & Biases:

  * CSV is the source of truth. It has zero external dependencies, works fully
    offline, survives a killed cloud instance, and -- crucially for this project
    -- lets us load `baseline.csv`, `ddp.csv`, `fsdp.csv` side by side in step 5
    and overlay them with matplotlib. It is also trivial to reason about when the
    lesson is specifically "did rank 0 write exactly one row per step?".

  * W&B (enabled with --use_wandb) is nice for *live* curves while a long cloud
    run is in flight, but it needs an account/network and hides less of the
    plumbing than we want while learning. So it is strictly additive: every
    metric still goes to CSV; W&B just mirrors it.

All writes are guarded by `is_main_process()` at the call site so that in the
DDP/FSDP stages only rank 0 touches disk.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import torch


class CSVLogger:
    """Append-only CSV writer with a fixed, ordered set of columns."""

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Fresh file with header. (Resuming runs is a later concern.)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: Dict[str, float]) -> None:
        # Flush every row so a crashed cloud instance still leaves us the data.
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


class WandbLogger:
    """Thin optional wrapper. Only imported if the user opts in."""

    def __init__(self, project: str, run_name: Optional[str], config: dict):
        import wandb  # local import so wandb is never a hard dependency
        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config)

    def log(self, row: Dict[str, float], step: int) -> None:
        self._wandb.log(row, step=step)

    def finish(self) -> None:
        self._wandb.finish()


# --- GPU memory helpers ---------------------------------------------------

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0
