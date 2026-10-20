"""Check checkpoint conversion independently of grouped GEMM and DeepEP."""

import unittest
from types import SimpleNamespace

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GroupedFp4Sm90WeightTest(unittest.TestCase):
    @staticmethod
    def _checkpoint(experts, rows, cols, seed):
        generator = torch.Generator().manual_seed(seed)
        codes = torch.randint(0, 16, (experts, rows, cols), generator=generator)
        codes[0, :128, :128] = 0
        exponents = torch.randint(
            -5, 3, (experts, rows, cols // 32), generator=generator
        )
        # Decode E2M1 from its sign, exponent and mantissa, without the
        # production lookup table or dequantization helper.
        magnitude = codes & 7
        exponent = magnitude >> 1
        mantissa = magnitude & 1
        values = torch.where(
            exponent == 0,
            mantissa.float() * 0.5,
            (1.0 + mantissa.float() * 0.5) * torch.exp2(exponent.float() - 1),
        )
        values = torch.where(codes < 8, values, -values)
        dense = values * torch.exp2(exponents.float()).repeat_interleave(32, -1)
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
        scales = (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        return packed.view(torch.int8).cuda(), scales.cuda(), dense.cuda()

    def _assert_conversion(self, quantized, scales, dense):
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        experts, rows, cols = dense.shape
        self.assertEqual(scales.shape, (experts, rows // 128, cols // 128))
        for row in range(0, rows, 128):
            for col in range(0, cols, 128):
                block = dense[:, row : row + 128, col : col + 128]
                expected_scale = (block.abs().flatten(1).amax(1) / 448.0).clamp_min(
                    1e-12
                )
                expected_q = (block / expected_scale[:, None, None]).to(
                    torch.float8_e4m3fn
                )
                torch.testing.assert_close(
                    scales[:, row // 128, col // 128], expected_scale, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    quantized[:, row : row + 128, col : col + 128].float(),
                    expected_q.float(),
                    rtol=0,
                    atol=0,
                )

    def test_setup_converts_both_layouts_and_partial_expert_slice(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4_sm90 import (
            GroupedFp4Sm90Executor,
        )
        from rtp_llm.utils.model_weight import W

        experts, dim, inter = 3, 384, 128
        w13, s13, dense13 = self._checkpoint(experts, 2 * inter, dim, seed=1)
        w2, s2, dense2 = self._checkpoint(experts, dim, inter, seed=2)
        for layout in ("gate_up", "up_gate"):
            with self.subTest(layout=layout):
                if layout == "up_gate":
                    checkpoint_w13 = w13.roll(inter, dims=1)
                    checkpoint_s13 = (
                        s13.view(torch.uint8).roll(inter, dims=1).view(s13.dtype)
                    )
                else:
                    checkpoint_w13, checkpoint_s13 = w13, s13
                executor = SimpleNamespace(
                    cfg=SimpleNamespace(
                        dim=dim,
                        moe_inter_dim=inter,
                        n_routed_experts=6,
                        ep_size=2,
                        moe_w1_layout=layout,
                    ),
                    # Three experts leave a one-expert final slice.
                    _DEQUANT_EXPERT_SLICE=2,
                )
                GroupedFp4Sm90Executor.setup_weights(
                    executor,
                    {
                        W.moe_w1: checkpoint_w13,
                        W.moe_s1: checkpoint_s13,
                        W.moe_w2: w2,
                        W.moe_s2: s2,
                    },
                )
                self._assert_conversion(executor._w13, executor._s13, dense13)
                self._assert_conversion(executor._w2, executor._s2, dense2)


if __name__ == "__main__":
    unittest.main()
