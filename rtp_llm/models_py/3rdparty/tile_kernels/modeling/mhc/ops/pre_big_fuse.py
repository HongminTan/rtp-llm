import functools

import torch

from ....mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse


@functools.cache
def _policy():
    from rtp_llm.models_py.modules.dsv4 import mhc_prenorm_backend

    return mhc_prenorm_backend


@functools.cache
def device_num_sms(device: torch.device) -> int:
    return _policy().device_num_sms(device)


def _run_tilelang_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_mult3: int,
    mhc_hidden_size: int,
    n_splits: int,
) -> None:
    fn = round_to_tf32(fn)
    kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size, n_splits)
    kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul.view(n_splits, -1, 1, mhc_mult3),
        gemm_out_sqrsum.view(n_splits, -1, 1),
    )


def _run_deepgemm_splitk_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    policy = _policy()
    backend = policy.resolve_backend()
    if n_splits is None:
        n_splits = policy.resolve_n_splits(
            mhc_hidden_size=mhc_hidden_size,
            num_tokens=num_tokens,
            num_sms=device_num_sms(residual.device),
            backend=backend,
        )
    else:
        policy.validate_n_splits(mhc_hidden_size, n_splits, backend)

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    if backend == "deepgemm":
        _run_deepgemm_splitk_gemm(
            residual_flat.view(num_tokens, mhc_hidden_size),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    elif backend == "tilelang":
        _run_tilelang_gemm(
            residual_flat,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_mult3,
            mhc_hidden_size,
            n_splits,
        )
    else:
        raise ValueError(
            "Unsupported mHC prenorm backend: "
            f"{backend!r}; expected deepgemm or tilelang."
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
        use_pdl=policy.device_supports_pdl(residual.device),
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
