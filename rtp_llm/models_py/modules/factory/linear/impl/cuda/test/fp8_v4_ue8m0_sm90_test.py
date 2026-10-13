"""V4-style UE8M0 dense FP8 linears on SM90 (Hopper).

DeepSeek-V4 checkpoints store dense FP8 weights with UE8M0 block scales.
On SM100 those repack to DeepGEMM's packed-int32 TMA layout; on SM90 the
linear must consume plain FP32 scales (UE8M0 -> FP32 is exact). This test
validates the SM90 conversion numerically against a dequantized reference
through the real ``create_fp8_linear`` factory path.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.quantized_linear import (
    create_fp8_linear,
)


def _quant_weight_ue8m0(weight: torch.Tensor):
    """[N, K] fp32 -> fp8 e4m3 + UE8M0 [N/128, K/128] block scales."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
        ceil_to_ue8m0,
    )

    n, k = weight.shape
    padded = torch.zeros((n // 128 * 128, k // 128 * 128), device=weight.device)
    padded[:n, :k] = weight
    blocks = padded.view(-1, 128, padded.size(1) // 128, 128)
    sf = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    sf = ceil_to_ue8m0(sf)
    q = (blocks / sf).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view_as(padded)[:n, :k].contiguous(), sf.view(
        n // 128, k // 128
    ).contiguous().to(torch.float8_e8m0fnu)


class Fp8V4Ue8m0Sm90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this SM90 target")

    def test_v4_ue8m0_linear_matches_dequant_reference(self):
        if is_deep_gemm_e8m0_used():
            self.skipTest("SM90-only path: e8m0 packing is used on this arch")

        torch.manual_seed(20261013)
        N, K = 512, 1024  # deliberately non-square
        M = 64
        w_ref = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
        w_fp8, s_ue8m0 = _quant_weight_ue8m0(w_ref)
        self.assertEqual(s_ue8m0.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(s_ue8m0.shape), (N // 128, K // 128))

        linear = create_fp8_linear(w_fp8, s_ue8m0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            y = linear(x)

        self.assertEqual(y.shape, (M, N))
        # Reference uses the SAME activation quantization as the linear
        # (per-token-group FP8) so the comparison isolates weight/scale
        # layout correctness from FP8 rounding.
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
        x_deq = (
            x_fp8.float().view(M, K // 128, 128) * x_scale.float().unsqueeze(-1)
        ).view(M, K)
        s_f32 = s_ue8m0.float()
        w_deq = w_fp8.float() * s_f32.repeat_interleave(128, 0).repeat_interleave(
            128, 1
        )[:N, :K]
        ref = x_deq @ w_deq.t()
        rel = (y.float() - ref).abs() / ref.abs().clamp(min=1e-2)
        self.assertLess(
            rel.mean().item(), 5e-3, f"mean rel {rel.mean().item():.4f} too large"
        )
        self.assertLess(
            rel.max().item(), 0.2, f"max rel {rel.max().item():.4f} too large"
        )


if __name__ == "__main__":
    unittest.main()
