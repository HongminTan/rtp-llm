from __future__ import annotations

import torch

from ....mhc.fused_post_pre_fma_kernel import mhc_fused_post_pre_fma
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse
from .pre_big_fuse import _policy


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
    output_norm_weight: torch.Tensor | None = None,
    output_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    assert num_tokens > 0
    x_flat = x.view(num_tokens, hidden_size)

    tile_n = 2 if num_tokens < 8 else 3
    target_splits = 8 if num_tokens < 8 and hidden_size <= 4096 else 4
    n_splits = _policy().largest_divisor_le(max(1, hidden_size // 256), target_splits)
    use_pdl = _policy().device_supports_pdl(residual.device)
    post_mix_flat = post_layer_mix.view(num_tokens, mhc_mult)

    residual_cur, gemm_out_mul, gemm_out_sqrsum = mhc_fused_post_pre_fma(
        comb_res_mix.view(num_tokens, mhc_mult, mhc_mult),
        residual_flat,
        post_mix_flat,
        x_flat,
        fn.view(mhc_mult3, mhc_mult, hidden_size),
        tile_mix_outputs=tile_n,
        split_k=n_splits,
        use_pdl=use_pdl,
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

    if output_norm_weight is not None:
        assert output_norm_weight.dtype == torch.bfloat16
        assert (
            output_norm_weight.shape == (hidden_size,)
            and output_norm_weight.is_contiguous()
        )
        assert output_norm_weight.device == residual.device
    kernel = _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        use_pdl=use_pdl,
        output_norm_eps=output_norm_eps if output_norm_weight is not None else None,
    )
    kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        *(() if output_norm_weight is None else (output_norm_weight,)),
    )

    outer = residual.shape[:-2]
    return (
        residual_cur.view(*outer, mhc_mult, hidden_size),
        post_mix_cur.view(*outer, mhc_mult, 1),
        comb_mix_cur.view(*outer, mhc_mult, mhc_mult),
        layer_input_cur.view(*outer, hidden_size),
    )
