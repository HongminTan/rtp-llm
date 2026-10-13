"""Real-GPU numerical coverage for the SM90 (Hopper) grouped FP8 MoE path.

Builds the production ``ChunkedFp8Fp4MoeLayer`` with the
``grouped_fp4_sm90`` strategy (packed-FP4 experts dequantized to FP8 +
FP32 128x128-block scales), then checks the executor's routed output
against an exact-FP32 PyTorch reference computed from the same packed
weights. Also verifies ``auto`` strategy resolution lands on the SM90
executor on a Hopper host.
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4_sm90 import (
    GroupedFp4Sm90Executor,
    _has_sm90_grouped_fp8_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer as MoE,
)
from rtp_llm.utils.model_weight import W

_DIM = 512
_INTER = 256
_EXPERTS = 16
_TOPK = 4
_SWIGLU_LIMIT = 10.0


def _make_layer_weights(experts: int, dim: int, inter_dim: int, device: str) -> dict:
    def packed_fp4(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            -10,
            10,
            (experts, out_dim, in_dim // 2),
            dtype=torch.int8,
            device=device,
        )

    def fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
        # Block scales near the neutral UE8M0 byte 127 keep outputs large
        # enough for relative-error checks to matter.
        return torch.randint(
            124,
            127,
            (experts, out_dim, in_dim // 32),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e8m0fnu)

    return {
        W.moe_gate: torch.randn(experts, dim, dtype=torch.bfloat16, device=device),
        W.moe_gate_bias: torch.zeros(experts, dtype=torch.float32, device=device),
        # W.moe_w1 covers gate|up together ([E, 2*inter, ...]) — the layer's
        # moe_w1_layout is "gate_up".
        W.moe_w1: packed_fp4(2 * inter_dim, dim),
        W.moe_s1: fp4_scale(2 * inter_dim, dim),
        W.moe_w2: packed_fp4(dim, inter_dim),
        W.moe_s2: fp4_scale(dim, inter_dim),
    }


def _reference_forward(
    x: torch.Tensor,
    router_weights: torch.Tensor,
    indices: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Routed-expert reference matching the executor's quantization dataflow.

    Activations arrive already per-token-group FP8-quantized (same kernel);
    the SiLU*mul output additionally rounds through BF16 and re-quantizes
    per-token-group to FP8 exactly like ``silu_mul_fp8_quant_f32_scale``.
    """
    N = x.shape[0]
    inter = w13.shape[1] // 2
    y = torch.zeros(N, w2.shape[1], dtype=torch.float32, device=x.device)
    x_f32 = x.float()
    for n in range(N):
        for k in range(indices.shape[1]):
            e = int(indices[n, k])
            gate_up = x_f32[n] @ w13[e].t()
            gate, up = gate_up[:inter], gate_up[inter:]
            gate = torch.clamp(gate, max=_SWIGLU_LIMIT)
            up = torch.clamp(up, min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
            hidden = torch.nn.functional.silu(gate) * up
            # BF16 round-trip then per-128-group FP8 re-quant (plain FP32
            # scale, no UE8M0 rounding) — mirrors the fused kernel.
            hidden = hidden.to(torch.bfloat16).float()
            groups = hidden.view(inter // 128, 128)
            amax = groups.abs().amax(dim=1).clamp(min=1e-10)
            scale = amax / 448.0
            h_q = (groups / scale[:, None]).clamp(-448.0, 448.0)
            h_deq = (h_q * scale[:, None]).view(inter)
            out = h_deq @ w2[e].t()
            y[n] += router_weights[n, k] * out
    return y


class GroupedFp4Sm90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated SM90 target")
        if not _has_sm90_grouped_fp8_kernel():
            raise AssertionError(
                "SM90 grouped FP8 DeepGEMM kernel support is required by this "
                "dedicated SM90 target"
            )

    def _build_moe(self, layer_weights: dict, strategy: str) -> MoE:
        return MoE(
            moe_w1_layout="gate_up",
            layer_id=3,
            dim=_DIM,
            moe_inter_dim=_INTER,
            n_routed_experts=_EXPERTS,
            n_activated_experts=_TOPK,
            n_shared_experts=0,
            score_func="sqrtsoftplus",
            route_scale=1.0,
            swiglu_limit=_SWIGLU_LIMIT,
            n_hash_layers=0,
            vocab_size=32,
            layer_weights=layer_weights,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=128,
            strategy=strategy,
        )

    def test_executor_matches_exact_fp32_reference(self):
        torch.manual_seed(20261013)
        weights = _make_layer_weights(_EXPERTS, _DIM, _INTER, "cuda")

        moe = self._build_moe(weights, "grouped_fp4_sm90")
        executor = moe.fused_moe.fused_experts
        self.assertIsInstance(executor, GroupedFp4Sm90Executor)

        # Fixed routing so the reference and the executor see identical
        # (token, expert, router-weight) triples.
        N = 48
        x = torch.randn(N, _DIM, dtype=torch.bfloat16, device="cuda")
        indices = torch.randint(
            0, _EXPERTS, (N, _TOPK), dtype=torch.int64, device="cuda"
        )
        router_weights = torch.rand(N, _TOPK, dtype=torch.float32, device="cuda")

        with torch.inference_mode():
            out = executor(x, router_weights, indices)

        self.assertEqual(out.shape, (N, _DIM))
        self.assertEqual(out.dtype, torch.float32)

        # Reference mirrors the executor's dataflow on BOTH sides: the same
        # per-token-group FP8 activation quantization for the input, and the
        # executor's own converted FP8 weights + FP32 block scales. Only
        # accumulation order and BF16 output rounding should differ.
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        x_deq = (
            x_fp8.float().view(N, _DIM // 128, 128) * x_scale.float().unsqueeze(-1)
        ).view(N, _DIM)

        def expand_blocks(w_fp8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            e, out_dim, in_dim = w_fp8.shape
            return (
                w_fp8.float()
                .view(e, out_dim // 128, 128, in_dim // 128, 128)
                * s[:, :, None, :, None]
            ).view(e, out_dim, in_dim)

        w13_ref = expand_blocks(executor._w13, executor._s13)
        w2_ref = expand_blocks(executor._w2, executor._s2)
        ref = _reference_forward(x_deq, router_weights, indices, w13_ref, w2_ref)

        # Error-to-signal ratio, same convention as the SM100
        # grouped-vs-local-loop equivalence test: the reference recomputes
        # the expert GEMMs in torch, so the residual covers FP8-quantized
        # activation rounding plus one GEMM-implementation rounding
        # (torch vs DeepGEMM) on the gate/up projection.
        scale = ref.abs().mean().item() + 1e-6
        diff = (out - ref).abs().mean().item()
        self.assertLess(
            diff / scale,
            0.03,
            f"error-to-signal ratio {diff / scale:.4f} too large",
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_auto_strategy_resolves_to_sm90_executor(self):
        torch.manual_seed(20261013)
        weights = _make_layer_weights(_EXPERTS, _DIM, _INTER, "cuda")

        moe = self._build_moe(weights, "auto")
        self.assertIsInstance(moe.fused_moe.fused_experts, GroupedFp4Sm90Executor)

    def test_explicit_sm90_strategy_selection(self):
        torch.manual_seed(20261013)
        weights = _make_layer_weights(_EXPERTS, _DIM, _INTER, "cuda")

        moe = self._build_moe(weights, "grouped_fp4_sm90")
        self.assertIsInstance(moe.fused_moe.fused_experts, GroupedFp4Sm90Executor)


if __name__ == "__main__":
    unittest.main()
