"""Equivalence test for ``_silu_mul_fp8_quant_triton.silu_mul_fp8_quant_packed``.

Compares the fused Triton kernel against the legacy 5-step torch chain it
replaces (clamp + silu + mul + bf16-cast + sgl_per_token_group_quant_fp8).
Tolerance: bit-equivalent on FP8 quantized output (UE8M0 quantization is
deterministic given the same fp32→bf16-rounded input), tight on the packed
UE8M0 scale.

Runs on any CUDA device with Triton — no DeepGEMM dependency. Verified
locally before the SM100 smoke validation.
"""

from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulFp8QuantPackedTest(unittest.TestCase):
    """Compare the fused kernel vs. the legacy reference implementation."""

    @classmethod
    def setUpClass(cls):
        # Direct file-path imports — bypass ``rtp_llm.models_py.modules``
        # package init (which pulls in fastapi/aiohttp/rtp_kernel/etc.).
        # The kernel under test depends only on torch + triton.
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_packed,
        )

        cls._fused = staticmethod(silu_mul_fp8_quant_packed)

    @staticmethod
    def _ref(
        gate_up: torch.Tensor,
        clamp_limit: float,
    ):
        """Pure-torch reference for fused silu+clamp+mul + per-token-group
        FP8 quant + UE8M0 packed scale.

        Mirrors the legacy 5-step chain (lines 514-534 of
        ``moe/strategies/grouped_fp4.py``) that we're replacing in Phase 2
        optimization 3, AND the layout that
        ``sgl_per_token_group_quant_fp8(..., column_major_scales=True,
        scale_tma_aligned=True, scale_ue8m0=True)`` produces (so the fused
        kernel's output layout matches DeepGEMM's contiguous-GEMM expectation).

        UE8M0 scale = ``2^ceil(log2(absmax/fp8_max))`` per (row, group),
        packed 4-per-int32 along K, then transposed to column-major
        ``[M, num_packed_groups]`` with M padded to multiple of 4.
        """
        import torch.nn.functional as F

        N = gate_up.size(-1)
        inter = N // 2
        group_size = 128
        # Step 1-3: legacy silu+clamp+mul → bf16 round
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        if clamp_limit > 0:
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
            gate = torch.clamp(gate, max=clamp_limit)
        hidden_f32 = F.silu(gate) * up
        hidden_bf16 = hidden_f32.to(torch.bfloat16)
        # Re-cast to fp32 for quantization (legacy passes bf16 to the
        # quantization kernel which then upcasts internally — same effect
        # as a single bf16 round + fp32 quant arithmetic).
        y = hidden_bf16.to(torch.float32)

        M = y.size(0)
        num_groups = inter // group_size

        # Step 4: per-(row, group) absmax → UE8M0 exponent
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_max = float(finfo.max)
        fp8_min = float(finfo.min)
        y_grouped = y.view(M, num_groups, group_size)
        absmax = y_grouped.abs().amax(dim=-1).clamp_min(1e-10)
        scale_raw = absmax / fp8_max
        exponent = torch.ceil(torch.log2(scale_raw))
        scale = torch.exp2(exponent)
        # Step 5: quantize
        y_q = torch.clamp(y_grouped / scale.unsqueeze(-1), fp8_min, fp8_max)
        out_q = y_q.to(torch.float8_e4m3fn).view(M, inter)

        # Step 6: pack 4 UE8M0 exponents per int32, column-major TMA-aligned M
        exp_biased = (exponent + 127.0).clamp(0, 255).to(torch.int32)  # [M, num_groups]
        num_packed_groups = (num_groups + 3) // 4
        # Pad num_groups up to multiple of 4 with zero-byte placeholders
        if num_groups % 4 != 0:
            pad = 4 - (num_groups % 4)
            exp_biased = torch.cat(
                [
                    exp_biased,
                    torch.zeros((M, pad), dtype=torch.int32, device=exp_biased.device),
                ],
                dim=1,
            )
        # Reshape into [M, num_packed_groups, 4] and pack via shifts
        exp_packed = exp_biased.view(M, num_packed_groups, 4)
        out_scale_rowmajor = (
            exp_packed[:, :, 0]
            | (exp_packed[:, :, 1] << 8)
            | (exp_packed[:, :, 2] << 16)
            | (exp_packed[:, :, 3] << 24)
        )  # [M, num_packed_groups] int32
        # Match the production layout: column-major TMA-aligned M
        tma_aligned_M = ((M + 3) // 4) * 4
        out_scale_colmajor = torch.zeros(
            (num_packed_groups, tma_aligned_M),
            dtype=torch.int32,
            device=out_scale_rowmajor.device,
        ).T[:M, :]
        out_scale_colmajor.copy_(out_scale_rowmajor)

        return out_q, out_scale_colmajor

    def _check(self, M: int, inter: int, clamp_limit: float, seed: int = 0):
        torch.manual_seed(seed)
        device = "cuda:0"
        gate_up = torch.randn(M, 2 * inter, dtype=torch.bfloat16, device=device) * 4.0

        ref_q, ref_s = self._ref(gate_up, clamp_limit)
        fused_q, fused_s = self._fused(gate_up, clamp_limit=clamp_limit, group_size=128)

        # Output shape contract
        self.assertEqual(fused_q.shape, (M, inter))
        self.assertEqual(fused_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(fused_s.dtype, torch.int32)

        # FP8 values: convert to fp32 for comparison (FP8 equality after
        # cast is the bit-identical check we want; small UE8M0 quantization
        # noise is OK but should be near-zero given the bf16 round-through).
        ref_q_f32 = ref_q.to(torch.float32)
        fused_q_f32 = fused_q.to(torch.float32)
        # Allow a few representable-FP8-step differences from independent
        # rounding paths (UE8M0 is deterministic; ULP drift here would point
        # to a kernel bug).
        max_abs = (ref_q_f32 - fused_q_f32).abs().max().item()
        self.assertLess(
            max_abs,
            1e-2,
            f"FP8 output diff too large (M={M}, inter={inter}, clamp={clamp_limit}): "
            f"max_abs={max_abs}",
        )

        # Scale: should be exactly equal (UE8M0 packing is deterministic).
        # Compare element-wise on int32 — the layouts must match
        # (column-major TMA-aligned).
        self.assertEqual(fused_s.shape, ref_s.shape, "scale shape mismatch")
        scale_diff = (fused_s.cpu() != ref_s.cpu()).sum().item()
        self.assertEqual(
            scale_diff,
            0,
            f"UE8M0 scale mismatch (M={M}, inter={inter}, clamp={clamp_limit}): "
            f"{scale_diff} elements differ out of {fused_s.numel()}",
        )

    # --- Realistic V4-Flash shapes -----------------------------------------

    def test_v4_flash_no_clamp_small(self):
        # 1k tokens × inter=2048 (V4 small batch decode-ish)
        self._check(M=1024, inter=2048, clamp_limit=0.0)

    def test_v4_flash_with_clamp_small(self):
        self._check(M=1024, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_medium(self):
        # 4k tokens × inter=2048 (typical prefill)
        self._check(M=4096, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_padded(self):
        # M not a multiple of 4 — exercises tma_aligned_M padding
        self._check(M=4099, inter=2048, clamp_limit=10.0)

    def test_v4_flash_with_clamp_large(self):
        # 16k tokens (max prefill)
        self._check(M=16384, inter=2048, clamp_limit=10.0)

    def test_smaller_inter(self):
        # inter=512: 4 groups of 128
        self._check(M=512, inter=512, clamp_limit=5.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulFp8QuantF32ScaleMaskedTest(unittest.TestCase):
    """Correctness of silu_mul_fp8_quant_f32_scale with/without masked_m.

    The masked mode serves the DeepEP low-latency grouped-capacity layout
    [E, M, 2*inter]: rows past each expert's count stay unwritten (the
    masked grouped GEMM never reads them) and valid rows must match the
    2D unmasked result bit-for-bit-equivalently.
    """

    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_f32_scale,
        )

        cls._fused = staticmethod(silu_mul_fp8_quant_f32_scale)

    @staticmethod
    def _check_valid_rows(
        got_q: torch.Tensor,
        got_s: torch.Tensor,
        gate_up_2d: torch.Tensor,
        clamp_limit: float,
        group_size: int = 128,
    ) -> None:
        """Reference check on the VALID rows of one expert slice."""
        import torch.nn.functional as F

        inter = gate_up_2d.size(-1) // 2
        gate = gate_up_2d[:, :inter].float()
        up = gate_up_2d[:, inter:].float()
        if clamp_limit > 0:
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
            gate = torch.clamp(gate, max=clamp_limit)
        y = (F.silu(gate) * up).to(torch.bfloat16).to(torch.float32)
        num_groups = inter // group_size
        yg = y.view(y.size(0), num_groups, group_size)
        absmax = yg.abs().amax(dim=-1).clamp_min(1e-10)
        scale = (absmax / 448.0).clamp_min(1e-10)
        ref_q = (
            (yg / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        ).view(y.size(0), inter)

        gq = got_q.to(torch.float32)
        rq = ref_q.to(torch.float32)
        max_abs = (gq - rq).abs().max().item()
        assert max_abs < 1e-2, f"FP8 output diff too large: {max_abs}"
        assert torch.allclose(got_s, scale, rtol=1e-3, atol=1e-9), (
            f"FP32 scale mismatch: max rel "
            f"{((got_s - scale).abs() / scale.clamp_min(1e-10)).max().item()}"
        )

    def test_masked_partial_counts(self):
        torch.manual_seed(1234)
        E, M, inter = 8, 256, 2048
        gate_up = torch.randn(E, M, 2 * inter, device="cuda", dtype=torch.bfloat16)
        counts = torch.randint(0, M + 1, (E,), device="cuda", dtype=torch.int32)
        counts[0] = 0  # empty expert
        counts[1] = M  # full expert
        q, s = self._fused(gate_up, clamp_limit=10.0, group_size=128, masked_m=counts)
        self.assertEqual(q.shape, (E, M, inter))
        self.assertEqual(s.shape, (E, M, inter // 128))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.float32)
        for e in range(E):
            c = int(counts[e])
            if c == 0:
                continue
            self._check_valid_rows(q[e, :c], s[e, :c], gate_up[e, :c], 10.0)

    def test_masked_full_equals_unmasked(self):
        torch.manual_seed(4321)
        E, M, inter = 4, 512, 1024
        gate_up = torch.randn(E, M, 2 * inter, device="cuda", dtype=torch.bfloat16)
        counts = torch.full((E,), M, device="cuda", dtype=torch.int32)
        q_m, s_m = self._fused(gate_up, clamp_limit=10.0, group_size=128, masked_m=counts)
        q_p, s_p = self._fused(
            gate_up.view(E * M, 2 * inter), clamp_limit=10.0, group_size=128
        )
        # All rows valid → masked output equals the plain 2D output.
        self.assertTrue(
            torch.equal(q_m.view(E * M, inter).to(torch.float32), q_p.to(torch.float32))
        )
        self.assertTrue(torch.equal(s_m.view(E * M, inter // 128), s_p))

    def test_unmasked_2d_regression(self):
        torch.manual_seed(999)
        M, inter = 1024, 2048
        gate_up = torch.randn(M, 2 * inter, device="cuda", dtype=torch.bfloat16)
        q, s = self._fused(gate_up, clamp_limit=10.0, group_size=128)
        self.assertEqual(q.shape, (M, inter))
        self._check_valid_rows(q, s, gate_up, 10.0)


if __name__ == "__main__":
    unittest.main()
