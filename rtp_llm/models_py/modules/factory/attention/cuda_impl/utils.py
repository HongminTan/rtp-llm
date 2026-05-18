"""Shared utilities for CUDA attention implementations."""

import functools

import torch


@functools.cache
def _sm_major() -> int:
    return torch.cuda.get_device_capability()[0]


def is_sm_90() -> bool:
    """SM90: Hopper (H100/H200/H800/H20)."""
    return _sm_major() == 9


def is_sm_100() -> bool:
    """SM100: Blackwell B200/GB200."""
    return _sm_major() == 10


def is_sm_120() -> bool:
    """SM120: Blackwell RTX 5090/DGX Spark."""
    return _sm_major() == 12
