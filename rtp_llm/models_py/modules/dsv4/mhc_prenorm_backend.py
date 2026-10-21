"""Shared mHC backend selection and split sizing for execution and warmup."""

import functools
import logging

from rtp_llm.utils.module_util import has_module

_logger = logging.getLogger(__name__)

DEEPGEMM = "deepgemm"
TILELANG = "tilelang"

_TILELANG_HIDDEN_BLOCK = 256


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def largest_divisor_le(n: int, want: int) -> int:
    # Each split must cover a whole number of hidden blocks.
    d = min(max(int(want), 1), n)
    while d > 1 and n % d:
        d -= 1
    return max(d, 1)


def num_split_for(
    *, m: int, k: int, num_sms: int, block_m: int = 64, block_k: int = 64
) -> int:
    grid_size = _ceil_div(max(int(m), 1), block_m)
    split_k = max(int(num_sms), 1) // max(grid_size, 1)
    num_block_k = _ceil_div(int(k), block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def resolve_n_splits(
    *, mhc_hidden_size: int, num_tokens: int, num_sms: int, backend: str
) -> int:
    if backend not in (DEEPGEMM, TILELANG):
        raise ValueError(f"Unknown mHC prenorm backend: {backend}")
    n_splits = num_split_for(m=num_tokens, k=mhc_hidden_size, num_sms=num_sms)
    # Choose a divisor of the K-block count so every split processes
    # the same number of complete blocks.
    block_k = _TILELANG_HIDDEN_BLOCK if backend == TILELANG else 64
    n_splits = largest_divisor_le(int(mhc_hidden_size) // block_k, n_splits)
    validate_n_splits(mhc_hidden_size, n_splits, backend)
    return n_splits


@functools.lru_cache(maxsize=None, typed=True)
def validate_n_splits(mhc_hidden_size: int, n_splits: int, backend: str) -> None:
    if backend not in (DEEPGEMM, TILELANG):
        raise ValueError(f"Unknown mHC prenorm backend: {backend}")
    block_k = _TILELANG_HIDDEN_BLOCK if backend == TILELANG else 64
    if type(n_splits) is not int or n_splits <= 0:
        raise ValueError(f"n_splits must be a positive integer, got {n_splits!r}")
    if mhc_hidden_size <= 0 or mhc_hidden_size % (block_k * n_splits):
        raise ValueError(
            f"{backend} requires equal partitions of {block_k}-element K blocks: "
            f"K={mhc_hidden_size}, n_splits={n_splits}"
        )


def device_num_sms(device=None) -> int:
    try:
        import deep_gemm

        num_sms = int(deep_gemm.get_num_sms())
        if num_sms > 0:
            return num_sms
    except Exception:
        pass

    import torch

    if device is None:
        device = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


@functools.cache
def device_supports_pdl(device) -> bool:
    import torch

    return torch.cuda.get_device_capability(device)[0] >= 9


@functools.cache
def resolve_backend() -> str:
    if has_module("deep_gemm"):
        try:
            import deep_gemm
        except (ImportError, OSError, RuntimeError):
            pass
        else:
            if callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
                return DEEPGEMM
    if not has_module("tilelang"):
        raise RuntimeError(
            "mHC prenorm requires DeepGEMM tf32_hc_prenorm_gemm "
            "or TileLang; neither is available"
        )
    _logger.warning(
        "DeepGEMM tf32_hc_prenorm_gemm is unavailable; "
        "falling back to TileLang mHC split-K"
    )
    return TILELANG
