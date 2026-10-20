"""Numerics and write boundaries for the FP32-scale SwiGLU kernel."""

import unittest

import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulFp8QuantF32ScaleTest(unittest.TestCase):
    @staticmethod
    def _reference(inputs, clamp):
        gate, up = inputs.float().chunk(2, dim=-1)
        if clamp:
            gate = gate.clamp(max=clamp)
            up = up.clamp(-clamp, clamp)
        hidden = (F.silu(gate) * up).bfloat16().float()
        groups = hidden.reshape(*hidden.shape[:-1], hidden.shape[-1] // 128, 128)
        scales = (groups.abs().amax(-1) / 448.0).clamp_min(1e-10)
        q = (groups / scales[..., None]).to(torch.float8_e4m3fn)
        return q.reshape_as(hidden).float(), scales

    def _assert_quantization(self, q, scales, ref_q, ref_scales):
        torch.testing.assert_close(scales, ref_scales, rtol=0.01, atol=1e-12)
        # Permit an FP8 rounding step at exp thresholds, but bound total error.
        torch.testing.assert_close(q.float(), ref_q, rtol=0.13, atol=2**-9)
        relative_error = (q.float() - ref_q).norm() / ref_q.norm()
        self.assertLess(relative_error.item(), 0.005)

    def test_contiguous_shared_expert_quantization(self):
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_f32_scale,
        )

        torch.manual_seed(43)
        inputs = torch.randn((11, 512), device="cuda", dtype=torch.bfloat16) * 4
        inputs[0] = 0
        q = torch.empty((11, 256), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((11, 2), device="cuda", dtype=torch.float32)
        silu_mul_fp8_quant_f32_scale(
            inputs, clamp_limit=1.5, output_q=q, output_scale=scales
        )
        self._assert_quantization(q, scales, *self._reference(inputs, 1.5))

    def test_masked_partial_tiles_and_preallocated_outputs(self):
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_f32_scale,
        )

        torch.manual_seed(42)
        experts, capacity, inter = 4, 11, 256
        counts = torch.tensor([0, 1, 9, 11], dtype=torch.int32, device="cuda")
        inputs = (
            torch.randn(
                (experts, capacity, 2 * inter), device="cuda", dtype=torch.bfloat16
            )
            * 4
        )
        # Exercise the zero-absmax scale floor in a valid group.
        inputs[3, 0, :128] = 0
        valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        for clamp in (0.0, 1.5):
            with self.subTest(clamp=clamp):
                q_storage = torch.full(
                    (experts * capacity + 1, inter),
                    7.0,
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                )
                scale_storage = torch.full(
                    (experts * capacity + 1, inter // 128),
                    -123.0,
                    dtype=torch.float32,
                    device="cuda",
                )
                q = q_storage[:-1].view(experts, capacity, inter)
                scales = scale_storage[:-1].view(experts, capacity, inter // 128)
                actual_q, actual_scales = silu_mul_fp8_quant_f32_scale(
                    inputs,
                    clamp_limit=clamp,
                    masked_m=counts,
                    output_q=q,
                    output_scale=scales,
                )
                self.assertIs(actual_q, q)
                self.assertIs(actual_scales, scales)

                ref_q, ref_scales = self._reference(inputs, clamp)
                self._assert_quantization(
                    q.float()[valid], scales[valid], ref_q[valid], ref_scales[valid]
                )
                self.assertTrue(torch.all(q.float()[~valid] == 7.0).item())
                self.assertTrue(torch.all(scales[~valid] == -123.0).item())
                self.assertTrue(torch.all(q_storage[-1].float() == 7.0).item())
                self.assertTrue(torch.all(scale_storage[-1] == -123.0).item())


if __name__ == "__main__":
    unittest.main()
