"""DSv4 Indexer FP8-paged score path via DeepGEMM.

Wraps ``deep_gemm.fp8_fp4_paged_mqa_logits`` so it can drop into the indexer
decode loop in place of the bf16 ``v4_indexer_score`` Triton kernel
when the cache is FP8 packed (132B/slot).

End-to-end shape contract:

  q_fp8       [B, next_n, H, D]            float8_e4m3fn  (per-(t,h) quant)
  w_fold      [B*next_n, H]                fp32           (per-token Q
                                                            scale folded in)
  kv_cache    [num_blocks, block_size, 1, D+4]  uint8     (132B per slot:
                                                            128 FP8 K + 4B fp32 scale)
  context_lens[B, next_n]                  int32          (live K length per row)
  block_table [B, max_blocks]              int32          (logical→physical block id)

Returns ``[B*next_n, max_ctx_len] fp32`` logits — same semantics as
``v4_indexer_score``: each row is the per-K-token score after fused
einsum + ReLU + per-head weighted sum.

Caller is responsible for FP8 quantizing Q via
:func:`indexer_q_fp8_quant_fold`, building the block_table, and the
2D context_lens shape DeepGEMM requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.utils.arch import is_sm90

try:
    import deep_gemm as _deep_gemm
except ImportError:
    _deep_gemm = None


def deepgemm_supports_block_kv(device, block_kv: int) -> bool:
    """DeepGEMM's paged-MQA metadata contract per device generation.

    ``get_paged_mqa_logits_metadata`` accepts ``block_kv == 64`` on SM90/SM100
    and ``block_kv == 32`` on SM100 only (a pre-SM90 device such as L20/SM89
    rejects either); sizes outside that list are unconstrained here.
    """
    major = torch.cuda.get_device_capability(device)[0]
    if block_kv == 32:
        return major == 10
    if block_kv == 64:
        return major in (9, 10)
    return True


def supports_paged_mqa_logits(device, block_kv: Optional[int] = None) -> bool:
    """Whether this group-128 call selects native DeepGEMM, not overall availability."""
    return (
        (
            _deep_gemm is not None
            and callable(getattr(_deep_gemm, "fp8_fp4_paged_mqa_logits", None))
        )
        and is_sm90(device)
        and (block_kv is None or block_kv == 64)
    )


def supports_mqa_logits(device) -> bool:
    """Whether this group-128 call selects native DeepGEMM, not overall availability."""
    return (
        _deep_gemm is not None
        and callable(getattr(_deep_gemm, "fp8_fp4_mqa_logits", None))
    ) and is_sm90(device)


def has_fp8_paged_mqa_logits() -> bool:
    """The group-128 scorer is available through DeepGEMM or Triton."""
    return True


def has_fp8_mqa_logits() -> bool:
    """The group-128 scorer is available through DeepGEMM or Triton."""
    return True


@dataclass
class PagedIndexerMetadata:
    block_table: torch.Tensor
    context_lens: torch.Tensor
    schedule: Optional[torch.Tensor]


def prepare_paged_indexer_metadata(
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> PagedIndexerMetadata:
    """Prepare one row per query, sharing physical KV pages across queries."""
    batch_size, next_n = context_lens.shape
    table = block_table[:batch_size].to(torch.int32).contiguous()
    if next_n != 1:
        table = table.repeat_interleave(next_n, dim=0)
    lengths = context_lens.reshape(-1, 1).contiguous()
    schedule = None
    if supports_paged_mqa_logits(lengths.device, block_size):
        schedule = _deep_gemm.get_paged_mqa_logits_metadata(
            lengths, block_size, _deep_gemm.get_num_sms()
        )
    return PagedIndexerMetadata(table, lengths, schedule)


def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [total_slots, 132] uint8 — flat pool view
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
    *,
    metadata: Optional[PagedIndexerMetadata] = None,
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes; downstream TopK must use the context lengths.
    """
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
    # DeepGEMM kv_cache shape: [num_blocks, block_size, 1, D+4] uint8.
    # Our pool is a flat [total_slots, 132] view; reshape into the 4D
    # layout (no copy — just a metadata change).
    total_slots = kv_pool_uint8.shape[0]
    assert (
        total_slots % block_size == 0
    ), f"total_slots={total_slots} not divisible by block_size={block_size}"
    num_blocks = total_slots // block_size
    kv_4d = kv_pool_uint8.view(num_blocks, block_size, 1, INDEXER_ENTRY_BYTES)

    if metadata is None:
        metadata = prepare_paged_indexer_metadata(block_table, context_lens, block_size)
    q_rows = q_fp8.reshape(-1, 1, q_fp8.shape[-2], q_fp8.shape[-1])
    if metadata.schedule is None:
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_paged_mqa_logits,
        )

        return legacy_fp8_paged_mqa_logits(
            q_rows,
            kv_4d,
            w_fold,
            metadata.context_lens,
            metadata.block_table,
            max_ctx_len,
        )
    return _deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_rows.contiguous(), None),
        kv_4d,
        w_fold.contiguous(),
        metadata.context_lens,
        metadata.block_table,
        metadata.schedule,
        max_context_len=max_ctx_len,
        indices=None,
    )


# ---------------------------------------------------------------------------
# Prefill (non-paged) wrapper around ``deep_gemm.fp8_fp4_mqa_logits``.
#
# Shape contract:
#   q_fp8        [M, H, D]    float8_e4m3fn  (M = total query tokens)
#   w_fold       [M, H]       fp32           (per-(token, head) Q scale folded in)
#   k_quant      [N, D]       float8_e4m3fn  (N = total key tokens — gathered
#                                              contiguous from the FP8 cache)
#   k_scale      [N]          float32
#   cu_seqlen_ks [M]          int32          (K start, inclusive)
#   cu_seqlen_ke [M]          int32          (K end,   exclusive)
#
# Native columns are request-relative; Triton retains global K columns.
# ---------------------------------------------------------------------------


def fp8_mqa_indexer_score(
    q_fp8: torch.Tensor,  # [M, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [M, H]    fp32
    k_quant: torch.Tensor,  # [N, D]    float8_e4m3fn
    k_scale: torch.Tensor,  # [N]       float32
    cu_seqlen_ks: torch.Tensor,  # [M]       int32
    cu_seqlen_ke: torch.Tensor,  # [M]       int32
    *,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
) -> torch.Tensor:
    """Return FP32 logits using the backend selected for the input format.

    Native DeepGEMM columns are request-relative; Triton columns are global.

    Valid columns are [0, ke-ks) for DeepGEMM and [ks, ke) for Triton.
    clean_logits fills invalid columns with negative infinity.
    The native output row stride can include padding.
    """
    assert q_fp8.dtype == torch.float8_e4m3fn and q_fp8.dim() == 3
    assert q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert w_fold.shape[0] == q_fp8.shape[0]
    assert k_quant.dtype == torch.float8_e4m3fn and k_quant.dim() == 2
    assert k_quant.shape[-1] == INDEXER_HEAD_DIM
    assert k_scale.dtype == torch.float32 and k_scale.dim() == 1
    assert k_scale.shape[0] == k_quant.shape[0]
    assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ks.dim() == 1
    assert cu_seqlen_ke.dtype == torch.int32 and cu_seqlen_ke.dim() == 1
    assert cu_seqlen_ks.shape[0] == q_fp8.shape[0]
    assert cu_seqlen_ke.shape[0] == q_fp8.shape[0]

    if not supports_mqa_logits(q_fp8.device):
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_mqa_logits,
        )

        return legacy_fp8_mqa_logits(
            q_fp8,
            (k_quant, k_scale),
            w_fold,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
            max_seqlen_k=max_seqlen_k,
        )
    width = max_seqlen_k if max_seqlen_k > 0 else k_quant.shape[0]
    if q_fp8.shape[0] == 0 or width == 0:
        return torch.empty(
            (q_fp8.shape[0], width), dtype=torch.float32, device=q_fp8.device
        )
    scores = _deep_gemm.fp8_fp4_mqa_logits(
        (q_fp8.contiguous(), None),
        (k_quant.contiguous(), k_scale.contiguous()),
        w_fold.contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        max_seqlen_k=width,
    )
    if clean_logits:
        columns = torch.arange(width, device=scores.device)
        scores.masked_fill_(
            columns[None, :] >= (cu_seqlen_ke - cu_seqlen_ks)[:, None],
            -torch.inf,
        )
    return scores
