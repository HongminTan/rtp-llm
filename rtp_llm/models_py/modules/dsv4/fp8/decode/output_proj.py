"""DSv4 decode output projection: inverse RoPE, FP8 quantization, wo_a,
wo_b, and optional TP reduction. Quantization emits the layout required by
the selected wo_a backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import dequantize_fp8_weight
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8


def decode_output_proj(
    attn: "AttentionFP8",
    o: torch.Tensor,  # [B, q_len, H, D]
    freqs_cis: torch.Tensor,  # [B, freqs_dim] or flat [B*q_len, freqs_dim]
    bsz: int,
    q_len: int,
) -> torch.Tensor:
    """Project attention output using the selected FP8 backend on CUDA.

    CPU and empty inputs use inverse RoPE and a BF16 grouped projection.
    """
    rd = attn.rope_head_dim

    if o.is_cuda and o.numel() > 0:
        use_swapab = attn._wo_a_swapab_supported(bsz * q_len)
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            freqs_cis,
            n_groups=attn.n_groups,
            heads_per_group=attn.n_heads // attn.n_groups,
            nope_dim=attn.head_dim - attn.rope_head_dim,
            rope_head_dim=attn.rope_head_dim,
            scale_ue8m0=attn._wo_a_scale_ue8m0,
            flatten_groups=use_swapab,
        )
        if use_swapab:
            o = attn._wo_a_project_swapab(o_fp8, o_scale, bsz, q_len)
        else:
            o = attn._wo_a_project_from_fp8(o_fp8, o_scale, bsz, q_len)
    else:
        if freqs_cis.dim() == 2 and int(freqs_cis.shape[0]) == bsz:
            apply_rotary_emb_batched(o[..., -rd:], freqs_cis, inverse=True)
        else:
            apply_rotary_emb(
                o[..., -rd:],
                freqs_cis.reshape(-1, freqs_cis.shape[-1]).contiguous(),
                inverse=True,
            )
        o = o.reshape(bsz, q_len, attn.n_groups, -1)
        wo_a_bf16 = dequantize_fp8_weight(attn.wo_a_w, attn.wo_a_s).to(o.dtype)
        wo_a = wo_a_bf16.view(attn.n_groups, attn.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    out = attn._lin(attn.wo_b, o.flatten(2))
    if attn.tp_size > 1:
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        reduced = all_reduce(out, Group.TP, inplace=True)
        if reduced is not out:
            out.copy_(reduced)
    return out
