"""In-graph multi-step decode loop (channel B minimal arm, R216 design).

The captured CUDA graph body chains ``n_steps`` decode steps:

* Step 0 reads the captured ``inputs.input_ids`` (filled per replay by the
  runner's ``prepare_cuda_graph`` / fusedCopy flush).
* Steps ``i > 0`` consume the previous step's greedy argmax token — the
  token stays on device, so the whole chain runs inside one graph replay.
* Between steps, :func:`update_decode_metadata_in_place` recomputes every
  metadata buffer from ``start_pos + 1``. That function is pure device
  torch ops writing into persistent (``forbid_realloc``) buffers, so the
  per-step shift is itself part of the captured graph; the per-replay
  prepare re-initializes the chain before each replay.

Head: the python ``V4Transformer`` always carries ``norm`` (applied at the
end of ``forward_layers``) and ``head_weight`` (transformer.py:308/:310),
so the in-graph sampler is plain ``F.linear + argmax`` — no C++ bridge.

Pre-registered constraints (perf-log R209.2 / R205.2 / R216):

* greedy sampling only (temperature 0 / top_k 1 request semantics); non
  greedy requests must fall back to the single-step graph.
* KV-boundary valve (executor-side eligibility, mirroring the M1
  ``g_m1_gate_skip_kv_boundary`` gate) must guarantee that no request
  crosses a pool block boundary inside the N-step window — paged write
  slots advance from block tables and ``start_pos + 1``, which is only
  correct within one block.
* EOS truncation is post-replay host work: at most ``n_steps - 1`` extra
  tokens are computed past an EOS (delivery truncates to the EOS
  position), matching the M1 one-step-lag semantics scaled by N.

This module is imported only from the multi-step capture path (runner
wiring lands in a later increment); nothing here runs in production until
the capture entry point calls it.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    update_decode_metadata_in_place,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import as_attention_inputs_by_tag


def greedy_next_token(hidden: torch.Tensor, head_weight: torch.Tensor) -> torch.Tensor:
    """Greedy head for one decode step: post-norm hidden -> argmax token.

    ``hidden`` is [B, dim] (already final-normed — ``forward_layers``
    applies ``v4.norm`` at the end); ``head_weight`` is [vocab, dim].
    Returns [B] int64 token ids on the hidden's device.
    """
    logits = F.linear(hidden.to(head_weight.dtype), head_weight).float()
    return torch.argmax(logits, dim=-1)


def _captured_paged_tables(
    inputs: Any, fmha_impl: Any
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, int]]]:
    """Mirrors DSv4DecodeFmhaImpl.prepare's tagged block-table extraction.

    Returns (tables, entries_per_block) from the *captured* inputs and the
    impl's snapshotted config; (None, None) on the legacy non-paged path.
    Inside a capture these are device tensors whose storage the runner
    re-fills per replay, so referencing them from the graph is safe.
    """
    entries = getattr(fmha_impl, "_paged_entries_per_block", None)
    if not entries:
        return None, None
    by_tag = as_attention_inputs_by_tag(inputs.attention_inputs)
    group_tags = getattr(getattr(fmha_impl, "config", None), "group_tags", None)
    if group_tags and len(by_tag) == 1 and len(group_tags) == 1:
        by_tag = {group_tags[0]: next(iter(by_tag.values()))}
    tables: Dict[str, torch.Tensor] = {}
    for tag, tagged in by_tag.items():
        if tag not in entries:
            continue
        block_table = getattr(tagged, "kv_cache_kernel_block_id_device", None)
        if block_table is None or block_table.numel() == 0:
            continue
        tables[tag] = block_table
    return (tables or None), entries


def advance_decode_metadata(
    meta: Any,
    paged_block_tables: Optional[Dict[str, torch.Tensor]],
    paged_pool_entries_per_block: Optional[Dict[str, int]],
) -> None:
    """One in-graph step shift: recompute every metadata buffer for
    ``start_pos + 1`` in place. Pure device ops — safe inside a capture
    and self-chaining (each call reads the previous call's write)."""
    update_decode_metadata_in_place(
        meta,
        meta.start_pos + 1,
        forbid_realloc=True,
        paged_block_tables=paged_block_tables,
        paged_pool_entries_per_block=paged_pool_entries_per_block,
    )


def forward_decode_multistep(
    v4: Any,
    kv_cache: Optional[Any],
    v4_args: Any,
    inputs: Any,  # PyModelInputs (capture instance)
    fmha_impl: Any = None,
    n_steps: int = 4,
    prepare_hidden_fn: Optional[Any] = None,
) -> torch.Tensor:
    """Run ``n_steps`` chained decode steps (one capture body).

    Returns [n_steps, B] int64 greedy token ids. The caller (graph runner
    wiring, later increment) stages this tensor into a capture buffer the
    executor reads after the replay.
    """
    if n_steps <= 1:
        raise ValueError(f"n_steps must be >= 2 for a multi-step body, got {n_steps}")
    if fmha_impl is None or not hasattr(fmha_impl, "metadata"):
        raise RuntimeError(
            "forward_decode_multistep requires a graph-path fmha_impl with "
            "persistent metadata (DSv4DecodeFmhaImpl family)"
        )
    meta = fmha_impl.metadata
    if int(meta.q_len_per_req) != 1:
        raise RuntimeError(
            f"multi-step capture requires pure decode (q_len == 1), got {meta.q_len_per_req}"
        )

    paged_tables, paged_entries = _captured_paged_tables(inputs, fmha_impl)

    # Lazy import: forward.py drags the CUDA-closure (decode ops / fmha impls)
    # that dsv4_decode_cpu deliberately excludes; only the capture entry
    # point (runner wiring) ever executes this body.
    from rtp_llm.models_py.modules.dsv4.decode.forward import forward_layers

    param_dev = v4.embed.weight.device
    tok = inputs.input_ids
    if tok.dim() == 0:
        tok = tok.unsqueeze(0)
    if tok.device != param_dev:
        tok = tok.to(param_dev)
    tok = tok.reshape(-1)

    tokens_out = []
    for i in range(n_steps):
        if i > 0:
            advance_decode_metadata(meta, paged_tables, paged_entries)
            tok = next_tok
        h = forward_layers(
            v4,
            kv_cache,
            tok,
            meta,
            prepare_hidden_fn=prepare_hidden_fn,
        )  # [B, 1, dim], post final norm
        hidden = h.reshape(-1, h.shape[-1])
        next_tok = greedy_next_token(hidden, v4.head_weight)
        tokens_out.append(next_tok)

    return torch.stack(tokens_out, dim=0)  # [n_steps, B]
