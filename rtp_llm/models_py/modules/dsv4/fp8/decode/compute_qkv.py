"""DSv4 decode Q/KV projection — extracted from ``AttentionFP8._forward_decode_body``.

Mirrors ``AttentionFP8._prefill_compute_qkv`` for the decode path
(per-request batched, including target-verify ``q_len > 1``).

The Attention module is passed in as ``attn`` and treated as a bag of
weights + tiny helpers (``_lin`` / ``_rmsnorm_weighted``) — this matches
the existing free-function pattern in :mod:`.decode_attn_metadata`, and
keeps ``attention.py`` thin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8


class DecodeQKV(NamedTuple):
    """Q/KV intermediate produced by :func:`decode_compute_qkv`.

    ``qr``  — ``[B, S, q_lora_rank]`` bf16 — fed to the CSA indexer.
    ``q``   — ``[B, S, H, D]`` bf16 — dense Q for sparse attn.
    ``kv``  — ``[B, S, D]`` bf16 — single MQA head, written to SWA pool.
    ``freqs_cis`` — ``[T, freqs_dim]`` — per-token RoPE table
        lookup, reused by the output-proj inverse-RoPE.
    """

    qr: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis: torch.Tensor


def decode_compute_qkv(
    attn: "AttentionFP8",
    x: torch.Tensor,  # [B, S, dim] bf16
    position_ids: torch.Tensor,  # [T] int32 absolute position per token
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    ``position_ids`` is flat over the token-major ``[B, S]`` layout. For
    normal decode ``S == 1``; target verify passes the full verify span.
    """
    rd = attn.rope_head_dim
    position_ids = position_ids.reshape(-1).to(
        device=attn.freqs_cis.device, dtype=torch.long
    )
    freqs_cis = attn.freqs_cis.index_select(0, position_ids).contiguous()

    # Quantize the shared bf16 input once and feed both wq_a/wkv GEMMs
    # (same pattern as the prefill qkv path): the two per-layer input
    # quants were bit-identical work duplicated on every decode step.
    if attn._can_reuse_qkv_input_quant():
        shared_quant = attn.wq_a.quantize_input(x.reshape(-1, x.shape[-1]))
        q_a_out = attn._lin_from_shared_quant(attn.wq_a, shared_quant, x.shape)
        kv_in = attn._lin_from_shared_quant(attn.wkv, shared_quant, x.shape)
    else:
        q_a_out = attn._lin(attn.wq_a, x)
        kv_in = attn._lin(attn.wkv, x)

    # Q path
    qr = attn._rmsnorm_weighted(
        q_a_out, attn.q_norm
    )  # [B, 1, q_lora_rank]
    q = attn._lin(attn.wq_b, qr).unflatten(
        -1, (attn.n_heads, attn.head_dim)
    )  # [B, S, H, D]
    q = fused_rmsnorm_rope(q, None, freqs_cis, rd, eps=attn.eps)

    # KV path (single MQA head) — per-token RoPE using the same table lookup.
    kv = fused_rmsnorm_rope(
        kv_in,
        attn.kv_norm,
        freqs_cis,
        rd,
        eps=attn.eps,
    )

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis=freqs_cis)
