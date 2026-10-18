"""Fused mHC post + next-layer pre boundary (small-batch decode path).

Port of the SGLang ``mhc_fused_post_pre`` wrapper (snapshot 9a1b1d2d5e,
``kernels/ops/layernorm/mhc.py:1508``) onto this tree's vendored kernel
stack: the FMA kernel in ``....mhc.fused_post_pre_fma_kernel`` computes
the post residual plus the pre-norm GEMM / square-sum partials in one
launch, then the unchanged ``_mhc_pre_big_fuse`` stage consumes the
partials. The boundary between one sublayer's HC post and the next HC
pre collapses from three launches (post, GEMM, big_fuse) to two (fused
FMA, big_fuse) at decode token counts, and the residual never round-trips
through a separate post kernel.

Only the small-batch path is ported (num_tokens <= threshold): larger
batches keep the unfused high-throughput decomposition, matching the
SGLang source's own split.
"""

from __future__ import annotations

import torch

from ....mhc.fused_post_pre_fma_kernel import mhc_fused_post_pre_fma
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse

# The scalar-FMA kernel wins only where launch overhead dominates; the
# threshold matches the SGLang source (32 tokens).
_FMA_TOKEN_THRESHOLD = 32


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse one HC post with the next HC pre's GEMM partials.

    Args mirror the SGLang wrapper: ``x`` is the sublayer output
    (the hidden_in of the post step); ``residual``/``post_layer_mix``/
    ``comb_res_mix`` are the HC state produced by the *previous* ``pre``;
    ``fn``/``mhc_scale``/``mhc_base`` belong to the *next* HC unit.

    Returns:
        residual_cur: post-mapped residual, (T, hc, H) bf16.
        post_mix_cur: (T, hc, 1) float32 — the next unit's post mix.
        comb_mix_cur: (T, hc, hc) float32 — the next unit's comb mix.
        layer_input_cur: (T, H) bf16 — the next sublayer's input.
    """
    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        # Some DP/EP ranks receive no tokens; keep the same contracts as
        # the unfused pre/post pair so later layers see typed empties.
        return (
            torch.empty_like(residual),
            torch.empty(
                (*residual.shape[:-2], mhc_mult, 1),
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                (*residual.shape[:-2], mhc_mult, mhc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                (*residual.shape[:-2], hidden_size),
                dtype=torch.bfloat16,
                device=residual.device,
            ),
        )

    assert num_tokens <= _FMA_TOKEN_THRESHOLD, (
        f"mhc_fused_post_pre is the small-batch decode path "
        f"(num_tokens={num_tokens} > {_FMA_TOKEN_THRESHOLD}); "
        "use the unfused post/pre pair instead"
    )
    x_flat = x.view(num_tokens, hidden_size)

    # Tile selection mirrors the SGLang wrapper: narrower tiles for very
    # small batches, more splits only when the hidden dim is small enough
    # that splits stay above the warp-reduction floor.
    tile_n = 2 if num_tokens < 8 else 3
    if num_tokens < 8 and hidden_size <= 4096:
        n_splits = 8

    post_mix_flat = post_layer_mix.view(num_tokens, mhc_mult)
    if post_mix_flat.shape[-1] != mhc_mult:
        post_mix_flat = post_layer_mix.reshape(num_tokens, mhc_mult)

    residual_cur, gemm_out_mul, gemm_out_sqrsum = mhc_fused_post_pre_fma(
        comb_res_mix.view(num_tokens, mhc_mult, mhc_mult),
        residual_flat,
        post_mix_flat,
        x_flat,
        fn.view(mhc_mult3, mhc_mult, hidden_size),
        tile_mix_outputs=tile_n,
        split_k=n_splits,
    )

    post_mix_cur = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix_cur = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input_cur = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    )

    outer = residual.shape[:-2]
    return (
        residual_cur.view(*outer, mhc_mult, hidden_size),
        post_mix_cur.view(*outer, mhc_mult, 1),
        comb_mix_cur.view(*outer, mhc_mult, mhc_mult),
        layer_input_cur.view(*outer, hidden_size),
    )
