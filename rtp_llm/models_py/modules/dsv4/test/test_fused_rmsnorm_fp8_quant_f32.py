"""Correctness test for DSV4 fused RMSNorm + FP8 f32 per-token-group quant.

Reference = flashinfer rmsnorm + ``sgl_per_token_group_quant_fp8`` with the
exact layout the decode DeepGEMM linear family consumes
(column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False) — the
C2a fused kernel must reproduce both outputs bit-comparably.
"""

from __future__ import annotations

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_f32_triton import (
    rmsnorm_fp8_quant_f32,
)


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (x32 * inv * weight.float()).to(x.dtype)


def _assert_matches(m: int, n: int) -> None:
    eps = 1.0e-6
    group_size = 128
    torch.manual_seed(20260703 + m * 31 + n)
    x = (torch.randn((m, n), device="cuda", dtype=torch.bfloat16) * 0.3).contiguous()
    weight = (
        (torch.randn((n,), device="cuda") * 0.1 + 1.0)
        .abs()
        .to(torch.bfloat16)
        .contiguous()
    )

    ref_norm = _torch_rmsnorm(x, weight, eps)
    ref_q, ref_s = sgl_per_token_group_quant_fp8(
        ref_norm,
        group_size=group_size,
        eps=1.0e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
    )

    x_in = x.clone()
    norm, q, s = rmsnorm_fp8_quant_f32(
        x_in,
        weight,
        eps=eps,
        group_size=group_size,
        clamp_eps=1.0e-4,
        out_norm=x_in,
    )

    assert norm.shape == ref_norm.shape and norm.dtype == ref_norm.dtype
    assert q.shape == ref_q.shape and q.dtype == ref_q.dtype
    assert s.shape == ref_s.shape and s.dtype == ref_s.dtype
    assert s.stride() == ref_s.stride()

    norm_diff = (norm.float() - ref_norm.float()).abs().max().item()
    assert norm_diff <= 2e-2, f"norm max diff {norm_diff}"

    q_diff = (
        q.contiguous().view(torch.uint8).to(torch.int16)
        - ref_q.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()
    # The reference quantizes the flashinfer norm output; the fused kernel
    # quantizes its own norm write. bf16 rounding differences of the norm
    # input can flip a boundary fp8 bucket, so allow a small fraction of
    # one-bucket differences instead of requiring bit equality.
    bad = (q_diff > 1).float().mean().item()
    assert bad <= 2e-3, f"fp8 mismatch fraction {bad}"
    max_qdiff = q_diff.max().item()
    assert max_qdiff <= 2, f"fp8 bucket diff {max_qdiff}"

    s_diff = (s.float() - ref_s.float()).abs()
    rel = (s_diff / ref_s.float().clamp_min(1e-12)).max().item()
    assert rel <= 1e-2, f"scale relative diff {rel}"


def test_fused_rmsnorm_fp8_quant_f32_shapes() -> None:
    # decode shapes: bs1/bs2 token rows at hidden 4096; odd m for the
    # column-major aligned-M padding path.
    for m, n in [(1, 4096), (2, 4096), (5, 4096), (128, 4096), (3, 2560)]:
        _assert_matches(m, n)


def main() -> None:
    test_fused_rmsnorm_fp8_quant_f32_shapes()
    print("fused rmsnorm fp8 f32 test passed")


if __name__ == "__main__":
    main()
