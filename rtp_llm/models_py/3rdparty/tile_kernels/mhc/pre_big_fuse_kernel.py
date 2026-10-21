import math

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
    use_pdl: bool = False,
    output_norm_eps: float | None = None,
):
    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.macro
    def compute(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        output_norm_weight,
    ):
        with T.Kernel(num_tokens, threads=96) as pid:
            if use_pdl:
                T.pdl_sync()
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            mixes_shared[j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                if output_norm_eps is not None:
                    norm_input = T.alloc_shared(hidden_size, T.bfloat16)
                    sqrsum = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(sqrsum)
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    if output_norm_eps is not None:
                        # RMSNorm consumes the BF16 output of HC mixing.
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] = T.float32(T.bfloat16(ol[i1_h]))
                            sqrsum[i1_h] += ol[i1_h] * ol[i1_h]
                        T.copy(ol, norm_input[i0_h * hidden_block], disable_tma=True)
                    else:
                        T.copy(
                            ol, layer_input[pid, i0_h * hidden_block], disable_tma=True
                        )

                if output_norm_eps is not None:
                    total = T.alloc_fragment(1, T.float32)
                    T.reduce_sum(sqrsum, total, dim=0)
                    total[0] = T.rsqrt(total[0] / hidden_size + output_norm_eps)
                    for i0_h in T.serial(hidden_size // hidden_block):
                        normalized = T.alloc_fragment(hidden_block, T.float32)
                        for i1_h in T.Parallel(hidden_block):
                            h = i0_h * hidden_block + i1_h
                            normalized[i1_h] = (
                                T.float32(norm_input[h]) * total[0]
                            ) * T.float32(output_norm_weight[h])
                        T.copy(
                            normalized,
                            layer_input[pid, i0_h * hidden_block],
                            disable_tma=True,
                        )

            if use_pdl:
                T.pdl_trigger()

    if output_norm_eps is None:

        @T.prim_func
        def mhc_pre_big_fuse(
            gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
            gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
            mhc_scale: T.Tensor[(3,), T.float32],
            mhc_base: T.Tensor[(mhc_mult3,), T.float32],
            residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
            post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
            comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
            layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        ) -> None:
            compute(
                gemm_out_mul,
                gemm_out_sqrsum,
                mhc_scale,
                mhc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                None,
            )

        return mhc_pre_big_fuse

    @T.prim_func
    def mhc_pre_big_fuse_with_norm(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        output_norm_weight: T.Tensor[(hidden_size,), T.bfloat16],
    ) -> None:
        compute(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            output_norm_weight,
        )

    return mhc_pre_big_fuse_with_norm
