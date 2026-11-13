"""Validate native DeepGEMM FP8 indexer scoring and TopK. """

from __future__ import annotations

import contextlib

import torch

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as _indexer_score_mod
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_gather_triton import (
    gather_indexer_k_for_prefill,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    quantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score,
    supports_mqa_logits,
)
from rtp_llm.models_py.triton_kernels import (
    legacy_fp8_indexer_score as _legacy_fallback_mod,
)


class _CountingDeepGemm:
    """Delegate to DeepGEMM while counting the non-paged entry point."""

    def __init__(self, inner):
        self._inner = inner
        self.mqa = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def fp8_fp4_mqa_logits(self, *args, **kwargs):
        self.mqa += 1
        return self._inner.fp8_fp4_mqa_logits(*args, **kwargs)


@contextlib.contextmanager
def _count_mqa_dispatch():
    """Count DeepGEMM vs Triton-fallback calls around the wrapper invocations.

    ``fp8_mqa_indexer_score`` resolves ``_deep_gemm`` and the fallback symbol at
    call time, so swapping both handles proves which path the shared predicate
    selected and that the case really executed on a kernel.
    """
    inner = _indexer_score_mod._deep_gemm
    counted = _CountingDeepGemm(inner)
    fallback = {"calls": 0}
    original_fallback = _legacy_fallback_mod.legacy_fp8_mqa_logits

    def _counted_fallback(*args, **kwargs):
        fallback["calls"] += 1
        return original_fallback(*args, **kwargs)

    _indexer_score_mod._deep_gemm = counted
    _legacy_fallback_mod.legacy_fp8_mqa_logits = _counted_fallback
    try:
        yield counted, fallback
    finally:
        _indexer_score_mod._deep_gemm = inner
        _legacy_fallback_mod.legacy_fp8_mqa_logits = original_fallback


def _assert_mqa_path(counted, fallback, expected_calls: int) -> None:
    """DeepGEMM-path assertion: the dispatch must match the shared predicate.

    Predicate true → the numeric assertions above ran on DeepGEMM and the
    entry point was called ``expected_calls`` times; predicate false → they ran
    on the Triton fallback instead.
    """
    expects_deepgemm = supports_mqa_logits(torch.device("cuda"))
    expected_deepgemm = expected_calls if expects_deepgemm else 0
    expected_fallback = 0 if expects_deepgemm else expected_calls
    assert counted.mqa == expected_deepgemm, (
        f"expected {expected_deepgemm} deep_gemm.fp8_fp4_mqa_logits call(s) "
        f"(predicate says DeepGEMM={expects_deepgemm}), got {counted.mqa} — "
        "the capability predicate and the dispatch drifted"
    )
    assert fallback["calls"] == expected_fallback, (
        f"expected {expected_fallback} Triton fallback call(s) "
        f"(predicate says DeepGEMM={expects_deepgemm}), got {fallback['calls']} "
        "— the case did not run on either kernel path"
    )


def _make_packed_cache(k_bf16: torch.Tensor, block_size: int):
    """Quantize ``k_bf16 [T, 128]`` into a paged FP8 pool. Returns
    ``pool_uint8 [(num_blocks+1)*block_size, 132] uint8``. Physical block
    id 0 is invalid, so slot i lives at absolute offset ``block_size + i``."""
    T = k_bf16.shape[0]
    num_blocks = (T + block_size - 1) // block_size
    pool_uint8 = torch.zeros(
        (num_blocks + 1) * block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=k_bf16.device,
    )
    pool_3d = pool_uint8.view(num_blocks + 1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T, dtype=torch.int64, device=k_bf16.device) + block_size
    quantize_indexer_k(k_bf16, slots, pool_3d)
    return pool_uint8


def _bf16_reference(q_bf16, k_bf16_dequant, weights, q_pos, ratio):
    """[B, S, T] fp32 — bf16 reference path."""
    return v4_indexer_score(
        q_bf16.contiguous(),
        k_bf16_dequant.contiguous(),
        weights.contiguous(),
        q_pos=q_pos,
        compress_ratio=ratio,
    )


def _fp8_path(
    q_bf16, weights, pool_uint8, T_live, q_pos_kernel, ratio, block_size: int = 64
):
    """[B, S, T_live] fp32 — DeepGEMM FP8 non-paged path.

    ``pool_uint8`` is the flat ``[num_blocks*block_size, 132]`` view; the
    gather kernel needs the original ``block_size`` to decode the
    per-block K|scale grouped layout, so we reshape back to 3D here."""
    B, S, H, D = q_bf16.shape
    assert B == 1, "prefill bsz==1"

    total_slots = pool_uint8.shape[0]
    assert total_slots % block_size == 0
    num_blocks = total_slots // block_size
    pool_3d = pool_uint8.view(num_blocks, block_size, INDEXER_ENTRY_BYTES)
    slot_mapping = (
        torch.arange(T_live, dtype=torch.int64, device=q_bf16.device) + block_size
    )
    k_quant, k_scale = gather_indexer_k_for_prefill(
        pool_3d,
        slot_mapping,
        head_dim=D,
    )

    q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
    M = B * S
    if q_pos_kernel is not None:
        ke = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(T_live)
        cu_ke = ke.reshape(M).contiguous()
    else:
        cu_ke = torch.full((M,), T_live, dtype=torch.int32, device=q_bf16.device)
    cu_ks = torch.zeros(M, dtype=torch.int32, device=q_bf16.device)

    logits = fp8_mqa_indexer_score(
        q_fp8.view(M, H, D),
        w_fold.view(M, H),
        k_quant,
        k_scale,
        cu_ks,
        cu_ke,
        clean_logits=True,  # mask invalid columns before comparing
    )
    return logits.view(B, S, T_live)


def _compare(label, ref, cand, k_topk, q_pos_kernel=None, ratio=1):
    """mean_abs < 10% of score_max + top-K overlap >= K-2 per row."""
    # Mask out causally-invalid positions in both tensors before comparing.
    # ref already has -inf past threshold (when q_pos was passed); FP8 path
    # leaves garbage past cu_seqlen_ke. Mask cand to match.
    if q_pos_kernel is not None:
        thr = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(cand.shape[-1])
        T = cand.shape[-1]
        col = torch.arange(T, device=cand.device).view(1, 1, T)
        valid = col < thr.view(*thr.shape, 1)
        cand = torch.where(valid, cand, torch.full_like(cand, float("-inf")))

    # ref may have -inf in masked positions; replace with 0 for diff stat.
    ref_finite = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
    cand_finite = torch.where(torch.isfinite(cand), cand, torch.zeros_like(cand))
    diff = (ref_finite - cand_finite).abs()
    score_max = ref_finite.abs().amax().item()
    print(
        f"  [{label}] mean_abs={diff.mean().item():.4f} "
        f"max_abs={diff.amax().item():.4f} score_max={score_max:.4f}"
    )
    assert (
        score_max == 0 or diff.mean().item() < 0.10 * score_max
    ), f"{label}: FP8 mean error >10% of score range"

    # Top-K overlap per row.
    B, S, T = ref.shape
    bad = 0
    K_eff = min(k_topk, T)
    counted = 0
    for s in range(S):
        ref_row = ref[0, s]
        cand_row = cand[0, s]
        # Skip rows whose entire range is masked out (early rows of fresh prefill).
        n_valid = (torch.isfinite(ref_row) & (ref_row > float("-inf"))).sum().item()
        K_row = min(K_eff, max(int(n_valid), 0))
        # Tiny-K rows (n_valid < 16) are dominated by FP8 noise on near-tied
        # scores — skip from the bad-row count (they're not useful indexer
        # selections in practice).
        if K_row < 16:
            continue
        counted += 1
        top_ref = set(ref_row.topk(K_row)[1].tolist())
        top_cand = set(cand_row.topk(K_row)[1].tolist())
        overlap = len(top_ref & top_cand)
        # Allow up to 5% per-row mismatches (FP8 noise budget) — what matters
        # for the indexer is that the *bulk* of high-magnitude positions agree.
        if overlap < int(K_row * 0.95):
            bad += 1
    print(
        f"  [{label} top-{K_eff} bad rows] {bad}/{counted} (skipped {S-counted} tiny-K rows)"
    )
    assert bad <= max(
        1, counted // 10
    ), f"{label}: {bad}/{counted} rows failed top-K overlap"


def test_prefill_fresh():
    torch.manual_seed(0)
    B, S, H, D = 1, 128, 64, INDEXER_HEAD_DIM
    T_live = 128
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    qpos = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)
    ref = _bf16_reference(q_bf16, k_recon, weights, qpos, ratio)
    with _count_mqa_dispatch() as (counted, fallback):
        cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos, ratio)
    _compare("fresh S=T=128", ref, cand, k_topk=16, q_pos_kernel=qpos, ratio=ratio)
    _assert_mqa_path(counted, fallback, expected_calls=1)


def test_prefill_continuation():
    torch.manual_seed(1)
    B, S, H, D = 1, 8, 64, INDEXER_HEAD_DIM
    T_live = 200
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    # No causal mask in continuation.
    ref = _bf16_reference(q_bf16, k_recon, weights, None, ratio)
    with _count_mqa_dispatch() as (counted, fallback):
        cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, None, ratio)
    _compare("continuation S=8 T=200", ref, cand, k_topk=32)
    _assert_mqa_path(counted, fallback, expected_calls=1)

    # Multiple request offsets, an empty row and an unaligned native stride.
    if supports_mqa_logits(torch.device("cuda")):
        q, wf = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
        k, sf = gather_indexer_k_for_prefill(pool_3d, slots, head_dim=D)
        starts = torch.tensor(
            [0, 3, 17, 31, 0, 80, 1, 199], device="cuda", dtype=torch.int32
        )
        ends = torch.tensor(
            [200, 120, 17, 130, 64, 200, 199, 200], device="cuda", dtype=torch.int32
        )
        with _count_mqa_dispatch() as (counted, fallback):
            relative = fp8_mqa_indexer_score(
                q.view(S, H, D),
                wf.view(S, H),
                k,
                sf.reshape(-1),
                starts,
                ends,
                max_seqlen_k=200,
            )
            cleaned_scores = fp8_mqa_indexer_score(
                q.view(S, H, D),
                wf.view(S, H),
                k,
                sf.reshape(-1),
                starts,
                ends,
                clean_logits=True,
            )
        assert relative.stride(0) == 256
        for row in range(S):
            lo, hi = int(starts[row]), int(ends[row])
            if hi > lo:
                # Independently score this request's K slice from column zero.
                # This checks coordinates without conflating native FP8 GEMM
                # accumulation with an unrelated FP32/FP64 matmul reference.
                expected = fp8_mqa_indexer_score(
                    q.view(S, H, D)[row : row + 1],
                    wf.view(S, H)[row : row + 1],
                    k[lo:hi].clone(),
                    sf[lo:hi].clone(),
                    torch.zeros((1,), dtype=torch.int32, device="cuda"),
                    torch.full((1,), hi - lo, dtype=torch.int32, device="cuda"),
                )
                torch.testing.assert_close(
                    relative[row, : hi - lo], expected[0], rtol=2e-3, atol=2e-3
                )
            torch.testing.assert_close(
                cleaned_scores[row, : hi - lo], relative[row, : hi - lo]
            )
            assert torch.isneginf(cleaned_scores[row, hi - lo :]).all()
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_prefill_topk

        selected = torch.empty((S, 16), dtype=torch.int32, device="cuda")
        _run_prefill_topk(
            relative, torch.zeros_like(starts), ends - starts, selected, 16, ratio
        )
        for row in range(S):
            length = int(ends[row] - starts[row])
            count = min(16, length)
            assert (selected[row, count:] == -1).all()
            if count:
                ids = selected[row, :count].long()
                assert ((ids >= 0) & (ids < length)).all()
                actual = relative[row, ids].sort(descending=True).values
                expected = relative[row, :length].topk(count).values
                torch.testing.assert_close(actual, expected)
        _assert_mqa_path(counted, fallback, expected_calls=2)


def test_prefill_long_S():
    torch.manual_seed(2)
    B, S, H, D = 1, 4096, 64, INDEXER_HEAD_DIM
    T_live = 4096
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    qpos = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)
    ref = _bf16_reference(q_bf16, k_recon, weights, qpos, ratio)
    with _count_mqa_dispatch() as (counted, fallback):
        cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos, ratio)
    _compare("long S=T=4096", ref, cand, k_topk=64, q_pos_kernel=qpos, ratio=ratio)
    _assert_mqa_path(counted, fallback, expected_calls=1)


def test_prefill_chunked():
    """Split S into 2 chunks; each runs its own fp8_mqa call. Concat rows,
    compare vs single-shot fp8 (and bf16 reference)."""
    torch.manual_seed(3)
    B, S, H, D = 1, 256, 64, INDEXER_HEAD_DIM
    T_live = 256
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )
    qpos_full = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)

    ref = _bf16_reference(q_bf16, k_recon, weights, qpos_full, ratio)
    with _count_mqa_dispatch() as (counted, fallback):
        cand_single = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos_full, ratio)

        # Chunked: split S → [0, mid), [mid, S). Each chunk's own qpos / output.
        mid = S // 2
        parts = []
        for s_start, s_end in ((0, mid), (mid, S)):
            q_chunk = q_bf16[:, s_start:s_end]
            w_chunk = weights[:, s_start:s_end]
            qpos_chunk = qpos_full[:, s_start:s_end]
            parts.append(
                _fp8_path(q_chunk, w_chunk, pool_uint8, T_live, qpos_chunk, ratio)
            )
        cand_chunked = torch.cat(parts, dim=1)

    # Chunked must match single-shot on finite (masked) positions. Both
    # were called with clean_logits=True so masked positions are -inf —
    # diff would be NaN, so finite-mask first.
    finite = torch.isfinite(cand_single) & torch.isfinite(cand_chunked)
    cs = torch.where(finite, cand_single, torch.zeros_like(cand_single))
    cc = torch.where(finite, cand_chunked, torch.zeros_like(cand_chunked))
    diff_cs = (cs - cc).abs().amax().item()
    print(f"  [chunked vs single-shot fp8] max_abs={diff_cs:.6f}")
    assert diff_cs < 1e-3, f"chunked != single-shot fp8 (max_abs={diff_cs})"

    _compare(
        "chunked vs bf16",
        ref,
        cand_chunked,
        k_topk=32,
        q_pos_kernel=qpos_full,
        ratio=ratio,
    )
    _assert_mqa_path(counted, fallback, expected_calls=3)  # 1 single + 2 chunks


if __name__ == "__main__":
    print("== FP8 non-paged (prefill) indexer score: equivalence ==")
    test_prefill_fresh()
    test_prefill_continuation()
    test_prefill_long_S()
    test_prefill_chunked()
    print("\nOK")
