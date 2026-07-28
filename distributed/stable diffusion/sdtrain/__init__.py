"""Reusable building blocks for the SD1.5 distributed-training learning project.

The package is intentionally split so the *same* modules are reused across the
single-GPU baseline, the DDP stage, and the FSDP stage. Only the training entry
script changes between stages; data/logging/model/sampling logic stays put.
"""
