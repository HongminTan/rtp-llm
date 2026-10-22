"""DSV4 decode Q/KV projection, normalization and RoPE for [B, S] tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

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
    position_ids: torch.Tensor,  # [B*S] absolute token positions
    *,
    freqs_cis: Optional[torch.Tensor] = None,
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    Optional ``freqs_cis`` skips lookup in ``attn.freqs_cis``. It must be a
    contiguous complex64 tensor [B*S, rope_head_dim//2] on x.device, matching
    position_ids and the token order of x.reshape(-1, x.shape[-1]).
    """
    rd = attn.rope_head_dim
    if freqs_cis is not None:
        expected_shape = (x.shape[0] * x.shape[1], rd // 2)
        if (
            freqs_cis.shape != expected_shape
            or freqs_cis.dtype != torch.complex64
            or freqs_cis.device != x.device
            or not freqs_cis.is_contiguous()
        ):
            raise ValueError(
                f"freqs_cis must be contiguous complex64 {expected_shape} on {x.device}"
            )
    if freqs_cis is None:
        positions = position_ids.reshape(-1).to(
            device=attn.freqs_cis.device, dtype=torch.long
        )
        freqs_cis = attn.freqs_cis.index_select(0, positions)

    if attn._can_reuse_qkv_input_quant():
        shared_quant = attn.wq_a.quantize_input(x.reshape(-1, x.shape[-1]))
        if attn.wqkv_a is not None:
            q_a_out, kv_in = attn._qkv_a_from_shared_quant(shared_quant, x.shape)
        else:
            q_a_out = attn._lin_from_shared_quant(attn.wq_a, shared_quant, x.shape)
            kv_in = attn._lin_from_shared_quant(attn.wkv, shared_quant, x.shape)
    else:
        q_a_out = attn._lin(attn.wq_a, x)
        kv_in = attn._lin(attn.wkv, x)

    # Q path
    qr = attn._rmsnorm_weighted(q_a_out, attn.q_norm)  # [B, S, q_lora_rank]
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
