"""Distributed helpers.

For the single-GPU baseline these all report a world of size 1 / rank 0, so the
"only rank 0 logs" pattern is *already* in place. When we move to DDP/FSDP with
`torchrun`, `torch.distributed` gets initialized and these same functions start
returning real values -- the training scripts do not need to change how they ask
"am I the main process?".
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def get_local_rank() -> int:
    """Rank within the local node (used to pick the CUDA device)."""
    import os
    if is_dist_initialized():
        return int(os.environ.get("LOCAL_RANK", get_rank()))
    return 0


def is_main_process() -> bool:
    """True only for the process that is allowed to log / save / sample."""
    return get_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()
