"""Numerical comparisons for DSV4 FP8 attention projections."""

from types import SimpleNamespace

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import _stack_fp8_rows
from rtp_llm.models_py.modules.dsv4.test.test_fused_inv_rope_fp8_quant import (
    _eager_ref,
    _unpack_scale,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear


def _make_wo_a_weight(G: int, R: int, K: int, scale_ue8m0: bool, seed: int = 0):
    """Synthesize wo_a weights with the scale format required by the device."""
    torch.manual_seed(seed)
    w_fp32 = torch.randn(G * R, K, dtype=torch.float32, device="cuda") * 0.3
    weight_fp8 = w_fp32.to(torch.float8_e4m3fn).view(G, R, K).contiguous()
    scale_bytes = torch.randint(
        115, 125, (G * R // 128, K // 128), dtype=torch.uint8, device="cuda"
    )
    scale_raw = scale_bytes.view(torch.float8_e8m0fnu).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    if not scale_ue8m0:
        return weight_fp8, scale_fp32
    idx = torch.arange(R, device="cuda") // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return weight_fp8, s_stk


def test_end_to_end_via_einsum():
    """Compare the selected wo_a projection against eager quantization and einsum."""
    torch.manual_seed(7)
    for n_groups, cases in (
        (8, [(1, 1), (2, 1), (3, 1), (4, 1), (16, 1), (64, 1), (2, 3)]),
        (2, [(3, 1)]),  # Six flattened rows require padding to eight scale rows.
    ):
        n_heads, head_dim, rope_dim = 64, 512, 64
        nope_dim = head_dim - rope_dim
        heads_per_group = n_heads // n_groups
        R = 1024
        K = heads_per_group * head_dim

        scale_ue8m0 = is_deep_gemm_e8m0_used()
        w_stk, s_stk = _make_wo_a_weight(n_groups, R, K, scale_ue8m0, seed=1)
        recipe = (1, 1 if scale_ue8m0 else 128, 128)
        attn = None
        if torch.cuda.get_device_capability()[0] == 9:
            from rtp_llm.models_py.kernels.cuda.flashinfer_gemm import (
                FLASHINFER_M_THRESHOLD,
                get_flashinfer_sm90_fp8_gemm,
            )
            from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

            attn = SimpleNamespace(
                n_groups=n_groups,
                o_lora_rank=R,
                _wo_a_stk_w=w_stk,
                _wo_a_stk_s=s_stk,
                _wo_a_swapab_gemm=get_flashinfer_sm90_fp8_gemm(),
                _wo_a_swapab_max_rows=FLASHINFER_M_THRESHOLD,
            )
            assert (
                attn._wo_a_swapab_gemm is not None
            ), "SM90 wo_a coverage requires FlashInfer FP8 block-scale GEMM"

        for B, S in cases:
            o = (
                torch.randn(
                    B, S, n_heads, head_dim, dtype=torch.bfloat16, device="cuda"
                )
                * 0.3
            )
            ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
            freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)
            M = B * S

            # Ref path
            fp8_ref, scale_ref, _, _ = _eager_ref(o, freqs, n_groups, heads_per_group)
            if not scale_ue8m0:
                scale_ref = _unpack_scale(scale_ref)
            out_ref = torch.empty(M, n_groups, R, dtype=torch.bfloat16, device="cuda")
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (fp8_ref, scale_ref),
                (w_stk, s_stk),
                out_ref,
                recipe=recipe,
            )

            use_swapab = attn is not None and AttentionFP8._wo_a_swapab_supported(
                attn, M
            )
            fp8_cand, scale_cand = fused_inv_rope_fp8_quant(
                o,
                freqs,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
                scale_ue8m0=scale_ue8m0,
                flatten_groups=use_swapab,
            )
            if use_swapab:
                out_cand = AttentionFP8._wo_a_project_swapab(
                    attn, fp8_cand, scale_cand, B, S
                ).view(M, n_groups, R)
            else:
                out_cand = torch.empty(
                    M, n_groups, R, dtype=torch.bfloat16, device="cuda"
                )
                deep_gemm.fp8_einsum(
                    "bhr,hdr->bhd",
                    (fp8_cand, scale_cand),
                    (w_stk, s_stk),
                    out_cand,
                    recipe=recipe,
                )

            d = (out_ref.float() - out_cand.float()).abs()
            rel = d / (out_ref.float().abs() + 1e-6)
            print(
                f"  [G={n_groups} B={B} S={S} M={M} swapab={use_swapab}]  max={d.max():.4e}  mean={d.mean():.4e}  "
                f"rel-max={rel.max():.4e}"
            )
            assert d.max() <= 5e-2, f"end-to-end einsum diff @ B={B} S={S}: {d.max()}"


def test_stacked_qkv_matches_separate():
    torch.manual_seed(1)
    q_rows, kv_rows, k = 1024, 512, 4096
    weights = {}
    for name, rows in (("q", q_rows), ("kv", kv_rows)):
        weights[name + "_w"] = (torch.randn(rows, k, device="cuda") * 0.05).to(
            torch.float8_e4m3fn
        )
        exps = torch.randint(110, 125, (rows // 128, k // 128), device="cuda")
        scale = exps.to(torch.uint8).view(torch.float8_e8m0fnu)
        weights[name + "_s"] = scale
    reference_q = _v4_fp8_linear(weights["q_w"], weights["q_s"])
    reference_kv = _v4_fp8_linear(weights["kv_w"], weights["kv_s"])
    weight, scale = _stack_fp8_rows(weights, ("q_w", "q_s"), ("kv_w", "kv_s"))
    stacked = _v4_fp8_linear(weight, scale)
    fallback_q = _v4_fp8_linear(weights["q_w"], weights["q_s"])
    fallback_kv = _v4_fp8_linear(weights["kv_w"], weights["kv_s"])
    for m in (1, 3, 31, 64, 512):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        quant = reference_q.quantize_input(x)
        q_ref = reference_q.forward_quantized(*quant)
        kv_ref = reference_kv.forward_quantized(*quant)
        y = stacked.forward_quantized(*quant)
        torch.testing.assert_close(y[:, :q_rows], q_ref, rtol=1.6e-2, atol=1e-2)
        torch.testing.assert_close(y[:, q_rows:], kv_ref, rtol=1.6e-2, atol=1e-2)
        torch.testing.assert_close(
            fallback_q.forward_quantized(*quant), q_ref, rtol=1.6e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fallback_kv.forward_quantized(*quant), kv_ref, rtol=1.6e-2, atol=1e-2
        )


if __name__ == "__main__":
    test_end_to_end_via_einsum()
    test_stacked_qkv_matches_separate()
