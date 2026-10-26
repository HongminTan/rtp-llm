"""Shared access to FlashInfer's SM90 FP8 block-scale GEMM."""

import torch

from rtp_llm.models_py.utils.math import align, ceil_div

# FlashInfer fp8_blockscale_gemm_sm90 uses swapAB for M < 32 and the
# normal kernel otherwise. Recheck this threshold when updating FlashInfer.
FLASHINFER_M_THRESHOLD = 32


def get_flashinfer_sm90_fp8_gemm():
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90
    except Exception:
        return None


def use_flashinfer_fp8_gemm(m: int) -> bool:
    return 0 < m < FLASHINFER_M_THRESHOLD


def is_flashinfer_activation_scale(scale: torch.Tensor, m: int, k: int) -> bool:
    """Whether ``scale`` is the FP32 ``[m, k/128]`` TMA-aligned MN-major view
    FlashInfer reads; its binding passes only the data pointer."""
    groups = ceil_div(k, 128)
    if scale.dtype != torch.float32 or tuple(scale.shape) != (m, groups):
        return False
    return (m == 1 or scale.stride(0) == 1) and (
        groups == 1 or scale.stride(1) == align(m, 4)
    )
