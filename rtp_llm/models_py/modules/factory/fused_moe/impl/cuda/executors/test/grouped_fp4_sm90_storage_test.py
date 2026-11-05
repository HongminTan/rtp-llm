"""Resident FP4 conversion, shared scratch, and grouped execution parity."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4_sm90 import (
    GroupedFp4Sm90Executor,
    _dequant_fp4_to_fp8_blockwise,
)
from rtp_llm.models_py.triton_kernels.moe.fp4_to_fp8 import (
    compute_fp4_to_fp8_scales,
    convert_fp4_to_fp8,
)
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class StorageTest(unittest.TestCase):
    def checkpoint(self, e, n, k):
        w = torch.randint(-128, 128, (e, n, k // 2), dtype=torch.int8, device="cuda")
        s = torch.randint(100, 140, (e, n, k // 32), dtype=torch.uint8, device="cuda")
        w[0, :128, :64] = 0
        s[0, -1, :] = 0  # E8M0 subnormal scale with nonzero weights.
        return w, s.view(torch.float8_e8m0fnu)

    def test_conversion_and_changing_graph_counts(self):
        for e, n, k in ((3, 256, 384), (2, 4096, 2048)):
            with self.subTest(shape=(e, n, k)):
                w, s4 = self.checkpoint(e, n, k)
                ref, s8 = _dequant_fp4_to_fp8_blockwise(w, s4, n, k, 2)
                got = compute_fp4_to_fp8_scales(w, s4)
                torch.testing.assert_close(got, s8, rtol=0, atol=0)
                out = torch.empty_like(ref)
                convert_fp4_to_fp8(w, s4, got, out)
                torch.testing.assert_close(
                    out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                )
                counts = torch.ones(e, device="cuda", dtype=torch.int32)
                convert_fp4_to_fp8(w, s4, got, out, counts)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    convert_fp4_to_fp8(w, s4, got, out, counts)
                expert_ids = torch.arange(e, device="cuda")
                for active in (
                    expert_ids < 0,
                    expert_ids >= 0,
                    expert_ids % 2 == 0,
                    expert_ids % 2 == 1,
                ):
                    counts.copy_(active.int())
                    out.fill_(7)
                    graph.replay()
                    expected = torch.where(
                        active[:, None, None],
                        ref.view(torch.uint8),
                        torch.full_like(ref, 7).view(torch.uint8),
                    )
                    torch.testing.assert_close(
                        out.view(torch.uint8), expected, rtol=0, atol=0
                    )

    def make_executor(self, mode, weights):
        executor = GroupedFp4Sm90Executor.__new__(GroupedFp4Sm90Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(
            dim=1024,
            moe_inter_dim=512,
            n_routed_experts=6,
            ep_size=2,
            moe_w1_layout="gate_up",
            dsv4_moe_sm90_storage=mode,
            swiglu_limit=10.0,
        )
        executor.setup_weights(dict(weights))
        return executor

    def test_executor_paths_and_workspace_reuse(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("SM90 grouped GEMM required")
        w13, s13 = self.checkpoint(3, 1024, 1024)
        w2, s2 = self.checkpoint(3, 1024, 512)
        weights = {W.moe_w1: w13, W.moe_s1: s13, W.moe_w2: w2, W.moe_s2: s2}
        a, b = self.make_executor("fp8", weights), self.make_executor("fp4", weights)
        self.assertIsNone(a._weight_workspace)
        self.assertEqual(a.weight_workspace_numel(), 0)
        self.assertEqual(b._w13.data_ptr(), w13.data_ptr())
        scratch = torch.empty(
            b.weight_workspace_numel(), dtype=torch.float8_e4m3fn, device="cuda"
        )
        b.bind_weight_workspace(scratch)
        x = torch.randn(12, 1024, device="cuda", dtype=torch.bfloat16)
        q, s = sgl_per_token_group_quant_fp8(
            x,
            128,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        routing = torch.ones(12, 1, device="cuda")
        for experts in ([0, 1, 2], [0, 2], [1]):
            ids = torch.tensor(experts, dtype=torch.int32, device="cuda")
            ids = ids.repeat(12 // len(experts)).view(-1, 1)
            scratch.fill_(float("nan"))
            torch.testing.assert_close(
                a._forward_local(q, s, routing, ids),
                b._forward_local(q, s, routing, ids),
                rtol=0,
                atol=0,
            )
        x_masked = torch.randn(3 * 128, 1024, device="cuda", dtype=torch.bfloat16)
        q_masked, s_masked = sgl_per_token_group_quant_fp8(
            x_masked,
            128,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        counts = torch.tensor([4, 0, 2], dtype=torch.int32, device="cuda")
        payload = SimpleNamespace(
            expert_x=q_masked.view(3, 128, 1024),
            expert_x_scale=s_masked.view(3, 128, 8),
            expert_tokens_meta=SimpleNamespace(expert_num_tokens=counts, expected_m=2),
        )
        b._forward_masked(payload)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = b._forward_masked(payload)
        for values in ([4, 0, 2], [0, 3, 4], [1, 2, 0]):
            counts.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            expected = a._forward_masked(payload)
            graph.replay()
            for e, count in enumerate(values):
                torch.testing.assert_close(
                    output[e, :count], expected[e, :count], rtol=0, atol=0
                )
        self.assertEqual(b._weight_workspace.data_ptr(), scratch.data_ptr())
        with self.assertRaisesRegex(ValueError, "dsv4_moe_sm90_storage"):
            self.make_executor("typo", weights)


if __name__ == "__main__":
    unittest.main()
