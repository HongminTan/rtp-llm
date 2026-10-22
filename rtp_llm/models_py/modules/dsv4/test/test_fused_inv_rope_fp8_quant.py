"""Validate inverse RoPE quantization and wo_a projection against eager references."""

from __future__ import annotations

from types import SimpleNamespace

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import fp8_max, fp8_min
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched
from rtp_llm.ops.compute_ops import per_token_group_quant_fp8


def _eager_ref(
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    G: int,
    R: int,
):
    """Eager inverse RoPE and per-group quantization with packed UE8M0 scales."""
    B, S, H, D = o.shape
    M = B * S
    tma_M = ((M + 3) // 4) * 4
    rd = freqs_cis.shape[-1] * 2

    # 1) Inverse RoPE on the last rd columns (torch path).
    o2 = o.clone()
    apply_rotary_emb_batched(o2[..., -rd:], freqs_cis, inverse=True)

    # 2) reshape [B, S, H, D] → [B, S, G, K]  (K = hpg * D)
    K = H * D // G
    o_bsgk = o2.reshape(B, S, G, K)
    # → [G, M, K] contiguous for per-group quant
    x_gmk = o_bsgk.reshape(M, G, K).permute(1, 0, 2).contiguous()

    a_fp8_3d = torch.empty(G, M, K, dtype=torch.float8_e4m3fn, device=o.device)
    scale_buf = torch.empty(G * (K // 512) * tma_M, dtype=torch.int32, device=o.device)
    a_scale_3d = scale_buf.as_strided((G, M, K // 512), (K // 512 * tma_M, 1, tma_M))
    for g in range(G):
        per_token_group_quant_fp8(
            x_gmk[g],
            a_fp8_3d[g],
            a_scale_3d[g],
            128,
            1e-10,
            fp8_min,
            fp8_max,
            True,
        )
    # transpose to match fused kernel's output: [M, G, …]
    return a_fp8_3d.permute(1, 0, 2).contiguous(), a_scale_3d.transpose(0, 1), M, tma_M


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


def _unpack_scale(scale):
    exponents = scale.contiguous().view(torch.uint8).to(torch.int32)
    return torch.exp2((exponents - 127).float())


def test_correctness_fp8_and_scale():
    """Check exact agreement between output formats and tolerance against eager quantization."""
    torch.manual_seed(42)
    # Production dsv4 V4-Flash dims.
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups

    for B, S in [(1, 1), (3, 1), (8, 1), (16, 1), (64, 1)]:
        o = (
            torch.randn(B, S, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)

        fp8_ref, scale_ref, M, tma_M = _eager_ref(o, freqs, n_groups, heads_per_group)
        fp8_cand, scale_cand = fused_inv_rope_fp8_quant(
            o,
            freqs,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_head_dim=rope_dim,
        )

        for flatten_groups in (False, True):
            fp8_f32, scale_f32 = fused_inv_rope_fp8_quant(
                o,
                freqs,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
                scale_ue8m0=False,
                flatten_groups=flatten_groups,
            )
            expected_fp8 = fp8_cand
            expected_scale = _unpack_scale(scale_cand)
            if flatten_groups:
                expected_fp8 = expected_fp8.transpose(0, 1).reshape(n_groups * M, -1)
                expected_scale = expected_scale.transpose(0, 1).reshape(
                    n_groups * M, -1
                )
            assert torch.equal(
                fp8_f32.view(torch.uint8), expected_fp8.view(torch.uint8)
            )
            torch.testing.assert_close(scale_f32, expected_scale, rtol=0, atol=0)

        # Compare FP8 byte agreement; this is not a floating-point ULP metric.
        diff_fp8 = (
            fp8_ref.contiguous().view(torch.uint8).to(torch.int16)
            - fp8_cand.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        exact_rate = (diff_fp8 == 0).float().mean().item()
        max_byte_diff = diff_fp8.max().item()

        # Compare the individual packed exponent bytes.
        sr_c = scale_ref.contiguous().view(torch.int32)
        sc_c = scale_cand.contiguous().view(torch.int32)
        diff_s = (
            sr_c.view(torch.uint8).to(torch.int16)
            - sc_c.view(torch.uint8).to(torch.int16)
        ).abs()
        s_exact = (diff_s == 0).float().mean().item()
        s_max = diff_s.max().item()

        print(
            f"  [B={B} S={S} M={M}]  "
            f"fp8 exact={exact_rate*100:.2f}% max_byte_diff={max_byte_diff}   "
            f"scale exact={s_exact*100:.2f}% max_byte_diff={s_max}"
        )
        # Eager and fused RoPE arithmetic can cross different scale-rounding
        # boundaries. Require byte agreement rates and limit exponent differences;
        # the projection test below separately bounds absolute output error.
        assert exact_rate >= 0.95, f"fp8 byte exactness {exact_rate:.3f} < 0.95"
        assert s_exact >= 0.99, f"scale byte exactness {s_exact:.4f} < 0.99"
        assert s_max <= 1, f"scale byte diff {s_max} > 1 exp (boundary ok)"


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


def _bench(fn, *args, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def bench_decode():
    """Split (apply_rotary_emb_batched + G per-group quants) vs fused."""
    n_heads, head_dim, rope_dim = 64, 512, 64
    nope_dim = head_dim - rope_dim
    n_groups = 8
    heads_per_group = n_heads // n_groups

    print(f"  [n_heads={n_heads} head_dim={head_dim} G={n_groups}  decode S=1]")
    print(
        "    {:>5}  {:>12}  {:>10}  {:>10}".format(
            "B", "eager+quant", "fused", "speedup"
        )
    )

    for B in [1, 8, 16, 64, 256]:
        o = (
            torch.randn(B, 1, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        ang = torch.rand(B, rope_dim // 2, device="cuda") * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)

        def run_eager(o):
            return _eager_ref(o, freqs, n_groups, heads_per_group)

        def run_fused(o):
            return fused_inv_rope_fp8_quant(
                o,
                freqs,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_head_dim=rope_dim,
            )

        t_e = _bench(run_eager, o)
        t_f = _bench(run_fused, o)
        print(
            "    {:5d}  {:10.2f}us  {:8.2f}us  {:8.2f}x".format(
                B, t_e * 1e3, t_f * 1e3, t_e / t_f
            )
        )


if __name__ == "__main__":
    print("== Correctness (direct fp8 + scale) ==")
    test_correctness_fp8_and_scale()
    print("\n== Correctness (end-to-end via fp8_einsum) ==")
    test_end_to_end_via_einsum()
    print("\n== Benchmark (decode B sweep) ==")
    bench_decode()
    print("\nOK")
