"""Fused sender-side packing for the capacity-padded static-EP decode path.

Replaces the aten chain in ``grouped_fp4_sm90.py`` ``_forward_ep_static``
(validity/where -> stable argsort -> scatter_add counts -> cumsum starts ->
arange/index_select/repeat_interleave -> four index_put writes, ~25-30
launches per layer) with three Triton kernels:

  * ``_rank_scan`` (single CTA): within[i] = number of j < i routed to the
    same destination rank -- the stable counting-sort rank. This is the aten
    chain's ``pos - starts[dest_sorted]`` re-indexed by assignment, proven
    equivalent in ``logs/mr1-algorithm-proof-01.py`` (R483).
  * ``_send_fill`` (one launch): zero payload + the ``-1`` eid sentinel
    column, replacing ``torch.zeros`` + the column write.
  * ``_send_pack``: per assignment, the bf16 token row (``x[i // K]``), the
    eid column (invalid ids read as 0 -- the aten chain zeroed them before
    the dest split, so those rows still ship to rank 0 with eid 0 and are
    zero-weighted at the combine), and the assign-index column, stored
    directly at ``(dest[i], within[i])``.

Numerical parity: the slot layout is bit-identical to the aten chain (same
stable order, same bf16 casts, same sentinel); the GPU parity gate in
``modules/dsv4/test/test_ep_static_pack_parity.py`` asserts it bitwise.

CUDA-graph safe: no host sync anywhere (the counting-sort replaces the
count/cumsum chain the aten form already kept device-side).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rank_scan(
    ids_ptr,  # int64 [M] raw router ids (may contain invalid sentinels)
    within_ptr,  # int32 [M] out: stable per-dest rank of assignment i
    M,
    E_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single CTA, WORLD sequential passes over the assignment stream. Decode
    # M = N*K is at most a few hundred; the chunk-eager variant is not wired
    # to this kernel (executor-side scope note).
    for d in tl.static_range(WORLD):
        running = tl.zeros((), dtype=tl.int32)
        for tile in range(0, tl.cdiv(M, BLOCK)):
            offs = tile * BLOCK + tl.arange(0, BLOCK)
            mask = offs < M
            ids = tl.load(ids_ptr + offs, mask=mask, other=0)
            valid = (ids >= 0) & (ids < WORLD * E_LOCAL)
            ids_v = tl.where(valid, ids, 0)
            dest = ids_v // E_LOCAL
            flags = (dest == d) & mask
            csum = tl.cumsum(flags.to(tl.int32), axis=0)
            rank = running + csum - 1
            tl.store(within_ptr + offs, rank, mask=flags)
            running += tl.sum(flags.to(tl.int32), axis=0)


@triton.jit
def _send_fill(
    send_ptr,  # bf16 [world * cap, D + 2]
    D_TOTAL: tl.constexpr,  # D + 2
    SENTINEL_COL: tl.constexpr,  # D
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    base = send_ptr + row.to(tl.int64) * D_TOTAL
    for col in range(0, tl.cdiv(D_TOTAL, BLOCK_D)):
        offs = col * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D_TOTAL
        val = tl.where(offs == SENTINEL_COL, -1.0, 0.0)
        tl.store(base + offs, val.to(tl.bfloat16), mask=mask)


@triton.jit
def _send_pack(
    ids_ptr,  # int64 [M] raw router ids
    x_ptr,  # bf16 [N, D]
    within_ptr,  # int32 [M] from _rank_scan
    send_ptr,  # bf16 [world * cap, D + 2]
    M,
    cap,
    K: tl.constexpr,
    D: tl.constexpr,
    D_TOTAL: tl.constexpr,  # D + 2
    E_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    ids = tl.load(ids_ptr + offs, mask=mask, other=0)
    valid = (ids >= 0) & (ids < WORLD * E_LOCAL)
    ids_v = tl.where(valid, ids, 0)
    dest = ids_v // E_LOCAL
    within = tl.load(within_ptr + offs, mask=mask, other=0)

    # slot row: send[dest * cap + within, :] (flattened world * cap rows)
    slot = dest.to(tl.int64) * cap + within.to(tl.int64)
    rows = (offs // K).to(tl.int64)

    # payload columns :D (bf16 row copy)
    for col in range(0, tl.cdiv(D, BLOCK_D)):
        cols = col * BLOCK_D + tl.arange(0, BLOCK_D)
        cmask = cols < D
        vals = tl.load(
            x_ptr + rows[:, None] * D + cols[None, :],
            mask=mask[:, None] & cmask[None, :],
            other=0.0,
        )
        tl.store(
            send_ptr + slot[:, None] * D_TOTAL + cols[None, :],
            vals,
            mask=mask[:, None] & cmask[None, :],
        )
    # eid column D (bf16; invalid ids read 0, matching the aten zeroing)
    tl.store(send_ptr + slot * D_TOTAL + D, ids_v.to(tl.bfloat16), mask=mask)
    # assign-index column D + 1 (bf16; exact for cap-1 < 256 on the capture
    # path -- the same R62.1 domain note as the aten form)
    tl.store(send_ptr + slot * D_TOTAL + D + 1, offs.to(tl.bfloat16), mask=mask)


def ep_static_pack_send(
    ids: torch.Tensor,
    x: torch.Tensor,
    world: int,
    e_local: int,
    cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused replacement for the aten sender-side packing chain.

    Args:
      ids: int64 [N, K] raw router ids (invalid sentinels allowed).
      x: bf16 [N, D] token activations.
      world / e_local: EP geometry.
      cap: per-destination slot count (N * K on the capture path).

    Returns:
      send_buf: bf16 [world, cap, D + 2] -- payload, eid column (D), and
        assign-index column (D + 1); padding slots carry eid -1.
      within: int32 [N * K] -- stable per-dest rank of each assignment, for
        the combine-side slot lookup ``dest * cap + within``.
    """
    assert ids.dtype == torch.int64
    assert x.dtype == torch.bfloat16
    N, D = x.shape
    M = N * ids.size(1)
    K = ids.size(1)

    flat_ids = ids.reshape(-1)
    within = torch.empty(M, dtype=torch.int32, device=x.device)
    send_buf = torch.empty(world * cap * (D + 2), dtype=torch.bfloat16, device=x.device)

    _rank_scan[(1,)](
        flat_ids,
        within,
        M,
        E_LOCAL=e_local,
        WORLD=world,
        BLOCK=128,
    )
    _send_fill[(world * cap,)](
        send_buf,
        D_TOTAL=D + 2,
        SENTINEL_COL=D,
        BLOCK_D=1024,
    )
    grid = (triton.cdiv(M, 128),)
    _send_pack[grid](
        flat_ids,
        x,
        within,
        send_buf,
        M,
        cap,
        K=K,
        D=D,
        D_TOTAL=D + 2,
        E_LOCAL=e_local,
        WORLD=world,
        BLOCK=128,
        BLOCK_D=64,
    )
    return send_buf.view(world, cap, D + 2), within
