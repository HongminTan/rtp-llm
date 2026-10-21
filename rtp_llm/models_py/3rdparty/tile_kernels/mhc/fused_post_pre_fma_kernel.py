"""Fused mHC post and split-K prenorm partials."""

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
def _mhc_fused_post_pre_fma(
    mhc: int,
    hidden: int,
    num_mix_outputs: int,
    n_thr: int = 256,
    tile_mix_outputs: int = 1,
    split_k: int = 1,
    use_pdl: bool = False,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")

    hidden_per_split = (hidden + split_k - 1) // split_k
    num_mix_output_tiles = (num_mix_outputs + tile_mix_outputs - 1) // tile_mix_outputs
    num_warps = n_thr // 32

    @T.prim_func
    def _mhc_fused_post_pre_fma_kernel(
        prev_comb_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32],
        prev_residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16],
        prev_post_mix: T.Tensor[(num_tokens, mhc), T.float32],
        hidden_in: T.Tensor[(num_tokens, hidden), T.bfloat16],
        pre_fn: T.Tensor[(num_mix_outputs, mhc, hidden), T.float32],
        mixes_partial_out: T.Tensor[(split_k, num_tokens, num_mix_outputs), T.float32],
        sqrsum_partial_out: T.Tensor[(split_k, num_tokens), T.float32],
        cur_residual_out: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, num_mix_output_tiles, split_k, threads=n_thr) as (
            token_idx,
            mix_output_tile_idx,
            hidden_split_idx,
        ):
            thread_idx = T.get_thread_binding()
            warp_idx = T.get_warp_idx()
            lane_idx = T.get_lane_idx()

            warp_partials = T.alloc_shared((num_warps, tile_mix_outputs + 1), T.float32)
            post_mix_smem = T.alloc_shared((mhc,), T.float32)
            comb_mix_smem = T.alloc_shared((mhc, mhc), T.float32)

            post_mix_for_token = T.alloc_local((mhc,), T.float32)
            comb_mix_for_token = T.alloc_local((mhc, mhc), T.float32)

            mix_acc = T.alloc_local((tile_mix_outputs,), T.float32)
            sqrsum_acc = T.alloc_local((1,), T.float32)
            cur_residual_values = T.alloc_local((mhc,), T.float32)

            T.clear(mix_acc)
            T.clear(sqrsum_acc)

            if use_pdl:
                T.pdl_sync()
            T.copy(prev_post_mix[token_idx, 0], post_mix_smem)
            T.copy(prev_comb_mix[token_idx, 0, 0], comb_mix_smem)

            for route_idx in T.unroll(mhc):
                post_mix_for_token[route_idx] = post_mix_smem[route_idx]

            for old_route_idx in T.unroll(mhc):
                for new_route_idx in T.unroll(mhc):
                    comb_mix_for_token[old_route_idx, new_route_idx] = comb_mix_smem[
                        old_route_idx, new_route_idx
                    ]

            hidden_iters_per_thread = (hidden_per_split + n_thr - 1) // n_thr
            for hidden_iter in T.serial(hidden_iters_per_thread):
                hidden_idx = (
                    hidden_split_idx * hidden_per_split
                    + hidden_iter * n_thr
                    + thread_idx
                )

                if (
                    hidden_idx < hidden
                    and hidden_idx < (hidden_split_idx + 1) * hidden_per_split
                ):
                    for new_route_idx in T.unroll(mhc):
                        cur_residual_values[new_route_idx] = (
                            post_mix_for_token[new_route_idx]
                            * hidden_in[token_idx, hidden_idx]
                        )
                        for old_route_idx in T.unroll(mhc):
                            cur_residual_values[new_route_idx] += (
                                comb_mix_for_token[old_route_idx, new_route_idx]
                                * prev_residual[token_idx, old_route_idx, hidden_idx]
                            )

                    # Prenorm consumes the BF16-rounded post residual.
                    for route_idx in T.unroll(mhc):
                        cur_residual_values[route_idx] = T.bfloat16(
                            cur_residual_values[route_idx]
                        )

                    if mix_output_tile_idx == 0:
                        for route_idx in T.unroll(mhc):
                            cur_residual_out[token_idx, route_idx, hidden_idx] = (
                                cur_residual_values[route_idx]
                            )
                            sqrsum_acc[0] += (
                                cur_residual_values[route_idx]
                                * cur_residual_values[route_idx]
                            )

                    for tile_col_idx in T.unroll(tile_mix_outputs):
                        mix_output_idx = (
                            mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                        )
                        if mix_output_idx < num_mix_outputs:
                            for route_idx in T.unroll(mhc):
                                mix_acc[tile_col_idx] += (
                                    pre_fn[mix_output_idx, route_idx, hidden_idx]
                                    * cur_residual_values[route_idx]
                                )

            for tile_col_idx in T.unroll(tile_mix_outputs):
                mix_acc[tile_col_idx] = T.warp_reduce_sum(mix_acc[tile_col_idx])
            if mix_output_tile_idx == 0:
                sqrsum_acc[0] = T.warp_reduce_sum(sqrsum_acc[0])

            if lane_idx == 0:
                for tile_col_idx in T.unroll(tile_mix_outputs):
                    warp_partials[warp_idx, tile_col_idx] = mix_acc[tile_col_idx]
                if mix_output_tile_idx == 0:
                    warp_partials[warp_idx, tile_mix_outputs] = sqrsum_acc[0]

            T.sync_threads()

            if warp_idx == 0:
                for tile_col_idx in T.unroll(tile_mix_outputs):
                    mix_output_idx = (
                        mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                    )
                    if mix_output_idx < num_mix_outputs and lane_idx == tile_col_idx:
                        mix_output_partial = T.alloc_var(T.float32, init=0.0)
                        for reduce_warp_idx in T.unroll(num_warps):
                            mix_output_partial += warp_partials[
                                reduce_warp_idx, tile_col_idx
                            ]
                        mixes_partial_out[
                            hidden_split_idx, token_idx, mix_output_idx
                        ] = mix_output_partial

                if mix_output_tile_idx == 0 and lane_idx == 0:
                    sqrsum_partial = T.alloc_var(T.float32, init=0.0)
                    for reduce_warp_idx in T.unroll(num_warps):
                        sqrsum_partial += warp_partials[
                            reduce_warp_idx, tile_mix_outputs
                        ]
                    sqrsum_partial_out[hidden_split_idx, token_idx] = sqrsum_partial

            if use_pdl:
                T.pdl_trigger()

    return _mhc_fused_post_pre_fma_kernel


def mhc_fused_post_pre_fma(
    comb_res_mix: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    x: torch.Tensor,
    fn: torch.Tensor,
    tile_mix_outputs: int = 1,
    split_k: int = 1,
    n_thr: int = 256,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc, hidden = residual.shape
    num_mix_outputs = fn.shape[0]

    gemm_out_mul = torch.empty(
        split_k,
        num_tokens,
        num_mix_outputs,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        split_k, num_tokens, dtype=torch.float32, device=residual.device
    )
    residual_cur = torch.empty_like(residual)

    _mhc_fused_post_pre_fma(
        hc,
        hidden,
        num_mix_outputs,
        n_thr=n_thr,
        tile_mix_outputs=tile_mix_outputs,
        split_k=split_k,
        use_pdl=use_pdl,
    )(
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_cur,
    )
    return residual_cur, gemm_out_mul, gemm_out_sqrsum
