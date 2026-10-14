"""Single-rank SM90 (Hopper) FP8-activation x FP4-weight MoE executor.

Hopper has no FP4 tensor cores: DeepGEMM's native ``fp8_fp4`` family and the
MegaMoE kernels are SM100-only. This executor dequantizes the packed-FP4
routed-expert weights into FP8 e4m3 with FP32 128x128-block scaling factors
once at weight-bind time, then runs the grouped FP8 contiguous entry point
(``m_grouped_fp8_gemm_nt_contiguous``; DeepGEMM >= 2.8 also exposes it as
``m_grouped_fp8_fp4_gemm_nt_contiguous``), which dispatches on SM90 to
``sm90_m_grouped_fp8_gemm_contiguous_1d2d`` (FP8 operands, FP32 scales).
The FP8 name is used directly so the cu129-pinned DeepGEMM 2.1.1 build —
which carries the same SM90 kernel but not the unified FP8/FP4 dispatcher —
runs this path too.

Numerics: unpacking FP4 e2m1 x UE8M0 (power-of-two) is exact in FP32; the
per-128x128-block requantization to e4m3 adds the same class of rounding a
native FP8 checkpoint carries, so the Hopper path is a bounded perturbation
of the SM100 FP4 math rather than a semantic change.

Forward mirrors :class:`GroupedFp4Executor`'s 4-opt path; the differences are
all scale-format related:
  - activation quant and ``ep_scatter`` run with ``scale_ue8m0=False`` and
    row-major FP32 scales (the layout the SM90 contiguous kernels take);
  - the fused SiLU+clamp+mul+quant emits FP32 scales
    (``silu_mul_fp8_quant_f32_scale``);
  - GEMMs pass FP8 weight tensors plus ``[E, ceil(N/128), ceil(K/128)]``
    FP32 block scales and no explicit recipe (SM90 default is 128x128).

EP > 1 dispatch is MegaMoE's job on SM100; on SM90 this executor requires
``ep_size == 1`` like ``grouped_fp4``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    _FP4_LUT,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_scatter_v2,

    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_f32_scale,
    per_token_group_quant_f32_scale_masked,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on both SM90 and SM100).
_GROUPED_ALIGNMENT = 128


def _has_sm90_grouped_fp8_kernel() -> bool:
    """Whether the dequant-then-grouped-FP8 path is runnable on this host.

    Requires the grouped FP8 contiguous entry point — available as
    ``m_grouped_fp8_gemm_nt_contiguous`` since DeepGEMM 2.1.1, and aliased by
    >= 2.8's unified ``m_grouped_fp8_fp4_gemm_nt_contiguous`` dispatcher —
    plus an SM90 device.
    """
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "get_mk_alignment_for_contiguous_layout"):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 9


def _dequant_fp4_to_fp8_blockwise(
    weight_int8: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    out_dim: int,
    in_dim: int,
    expert_slice: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize stacked packed-FP4 experts into FP8 + FP32 128-block scales.

    Args:
      weight_int8: ``[E, out_dim, in_dim // 2]`` int8 (packed e2m1).
      scale_ue8m0: ``[E, out_dim, in_dim // FP4_BLOCK]`` float8_e8m0fnu.
      out_dim / in_dim: logical expert linear dims.
      expert_slice: how many experts to convert per chunk (memory-bound
        transient FP32 blow-up control; 0 converts everything at once).

    Returns:
      w_fp8: ``[E, out_dim, in_dim]`` torch.float8_e4m3fn.
      s_fp32: ``[E, ceil(out_dim / 128), ceil(in_dim / 128)]`` torch.float32.
    """
    E = weight_int8.shape[0]
    assert weight_int8.dim() == 3 and scale_ue8m0.dim() == 3
    assert weight_int8.shape[2] * 2 == in_dim
    assert scale_ue8m0.shape[2] * FP4_BLOCK == in_dim

    device = weight_int8.device
    n_blocks_n = ceil_div(out_dim, FP8_BLOCK)
    n_blocks_k = ceil_div(in_dim, FP8_BLOCK)
    w_fp8 = torch.empty((E, out_dim, in_dim), dtype=torch.float8_e4m3fn, device=device)
    s_fp32 = torch.empty((E, n_blocks_n, n_blocks_k), dtype=torch.float32, device=device)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    lut = _FP4_LUT.to(device)

    step = E if expert_slice <= 0 else min(expert_slice, E)
    for start in range(0, E, step):
        end = min(start + step, E)
        # [e, out, in] exact FP32 dequant of one expert chunk.
        packed = weight_int8[start:end].to(torch.int32) & 0xFF
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        nib = torch.empty(
            (end - start, out_dim, in_dim), dtype=torch.int64, device=device
        )
        nib[..., 0::2] = low.long()
        nib[..., 1::2] = high.long()
        w_f32 = lut[nib] * scale_ue8m0[start:end].to(torch.float32).repeat_interleave(
            FP4_BLOCK, dim=-1
        )
        del packed, low, high, nib

        # Per-128x128-block requantization: scale = block_amax / fp8_max.
        blocks = w_f32.view(end - start, n_blocks_n, FP8_BLOCK, n_blocks_k, FP8_BLOCK)
        amax = blocks.abs().amax(dim=(2, 4))
        scale = (amax / fp8_max).clamp_(min=1e-12)
        q = (blocks / scale[:, :, None, :, None]).clamp_(-fp8_max, fp8_max)
        w_fp8[start:end] = q.view(end - start, out_dim, in_dim).to(
            torch.float8_e4m3fn
        )
        s_fp32[start:end] = scale
        del w_f32, blocks, amax, scale, q
    return w_fp8, s_fp32


class GroupedFp4Sm90Executor(Fp8Fp4ExecutorBase):
    """Grouped FP8-GEMM MoE executor for Hopper (SM90) hosts.

    Consumes the same checkpoint tensors as :class:`GroupedFp4Executor`
    (packed FP4 + UE8M0 group-32 scales) but executes them through the
    dequantized FP8 + FP32-block-scale grouped contiguous kernels.
    """

    # Chunk experts through the FP32 dequant transient: ~2 GB at a time for
    # the full-size w13 stack instead of the whole ~17 GB.
    _DEQUANT_EXPERT_SLICE = 16

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.dim % FP8_BLOCK == 0)
        checker.check(config.moe_inter_dim % FP8_BLOCK == 0)
        checker.check(config.world_size == config.ep_size)
        checker.check(_has_sm90_grouped_fp8_kernel())

    def setup_weights(self, layer_weights: Dict) -> None:
        """Dequantize the EP-sliced routed-expert stacks into FP8 + FP32 scales.

        Pops canonical ``W.moe_w{1,2}`` and ``W.moe_s{1,2}`` keys from
        ``layer_weights`` (each shaped ``[E_local, ...]``); keeps only the
        dequantized FP8/FP32 forms alive.
        """
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        D, inter = cfg.dim, cfg.moe_inter_dim
        expected_local = max(int(cfg.n_routed_experts) // int(cfg.ep_size), 1)

        w13, s13_raw = normalize_moe_w13_gate_up(
            layer_weights.pop(W.moe_w1),
            layer_weights.pop(W.moe_s1),
            inter,
            cfg.moe_w1_layout,
        )
        w2, s2_raw = layer_weights.pop(W.moe_w2), layer_weights.pop(W.moe_s2)
        if w13.shape[0] != expected_local or w2.shape[0] != expected_local:
            raise ValueError(
                f"EP-sliced expert stacks {[w13.shape[0], w2.shape[0]]} do not "
                f"match expected local expert count {expected_local}"
            )

        self._e_local = expected_local
        self._w13, self._s13 = _dequant_fp4_to_fp8_blockwise(
            w13, s13_raw, 2 * inter, D, self._DEQUANT_EXPERT_SLICE
        )
        del w13, s13_raw
        self._w2, self._s2 = _dequant_fp4_to_fp8_blockwise(
            w2, s2_raw, D, inter, self._DEQUANT_EXPERT_SLICE
        )
        del w2, s2_raw

        # Return the freed FP4 blocks to CUDA so the KV-cache planner sees
        # the real residual HBM (same reason as GroupedFp4Executor).
        torch.cuda.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """4-opt prefill path, FP32-scale variant; returns ``[N, D] fp32``."""
        cfg = self.cfg
        N, D = x.shape
        E = self._e_local
        inter = cfg.moe_inter_dim
        device = x.device

        if cfg.ep_size > 1:
            if self._mega_moe_enabled():
                # DeepGEMM SM90 MegaMoE (nv_dev lineage, PR #383): one fused
                # dispatch + 2xGEMM + SwiGLU + combine kernel over
                # torch.distributed._symmetric_memory. Fastest SM90 decode
                # backend (3-10x vs DeepEP LL at DSv4 shapes, RUN
                # build_logs/moe-benchmark-report.md). Opt-in via
                # DSV4_EP_MEGA_MOE=1; token counts must stay within the
                # buffer capacity.
                return self._forward_ep_mega(x, weights, indices)
            if self._deepep_ll_enabled():
                # DeepEP low-latency mode (the production decode+graph EP
                # pattern): fixed-capacity dispatch/combine buffers, all
                # ranks enter symmetrically, weight application happens in
                # the combine kernel. Opt-in via DSV4_EP_DEEPEP_LL=1 — the
                # decode-role process sets it; token counts must stay within
                # the buffer capacity (prefill belongs to the prefill role).
                return self._forward_ep_deepep(x, weights, indices)
            if torch.cuda.is_current_stream_capturing():
                # Graph capture: batch shapes are uniform across ranks, so
                # the capacity-padded equal-split path applies (no host
                # syncs, constant shapes).
                return self._forward_ep_static(x, weights, indices)
            # Eager: dynamic splits (DP ranks may hold different token
            # counts, including empty batches).
            return self._forward_ep(x, weights, indices)
        if N == 0:
            return torch.zeros(N, D, dtype=torch.float32, device=device)
        if torch.cuda.is_current_stream_capturing():
            return self._forward_capture_topk(x, weights, indices)
        return self._forward_local(x, weights, indices)

    def _forward_local(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Eager single-rank compute core over LOCAL expert indices."""
        cfg = self.cfg
        N, D = x.shape
        E = self._e_local
        inter = cfg.moe_inter_dim
        device = x.device

        # (1) Quant input ONCE — row-major FP32 scales: the layout the SM90
        # grouped-contiguous kernels consume.
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )

        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=0,
            num_local_experts=E,
        )

        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.float32, device=device)

        aligned_counts = torch.tensor(
            aligned_counts_list,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).to(device, non_blocking=True)

        # (2) Triton ep_scatter: per-expert padded layout, FP32 row-major
        # scales (scale dtype follows the input scale on this branch).
        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.zeros(
            [all_tokens, D // FP8_BLOCK],
            device=device,
            dtype=torch.float32,
        )
        m_indices = torch.empty(all_tokens, dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_topk_ids)
        expert_start_loc = torch.empty_like(aligned_counts)
        ep_scatter(
            a_fp8,
            a_scale,
            adjusted_topk_ids,
            aligned_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=False,
        )
        # Defensive clamp against -1 leakage; matches GroupedFp4Executor.
        m_indices.clamp_(min=0, max=E - 1)
        del a_fp8, a_scale

        # GEMM 1: gate+up. No explicit recipe: SM90 default is (1, 128, 128)
        # with FP32 scales, which matches the prepared weight tensors.
        gate_up = torch.empty(
            all_tokens, 2 * inter, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (scatter_out, scatter_out_scale),
            (self._w13, self._s13),
            gate_up,
            m_indices,
        )
        del scatter_out, scatter_out_scale

        # (3) Fused SiLU+clamp+mul + per-token-group FP8 quant, FP32 scales.
        h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        del gate_up

        # GEMM 2: down
        down_out = torch.empty(all_tokens, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
        )
        del h_fp8, h_scale, m_indices

        # (4) Triton ep_gather with router-weight reduce.
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

    def _forward_capture_topk(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-capture path: per-(token, topk) dense FP8 GEMMs, no host sync.

        The eager path's ``.cpu().tolist()`` expert counting is illegal under
        stream capture; mirror :class:`GroupedFp4Executor`'s capture shape but
        run every GEMM through the SM90 dense ``fp8_gemm_nt`` (FP32 scales).
        Decode batches are small, so the per-token loop is acceptable for the
        captured region.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        cfg = self.cfg
        N, D = x.shape
        inter = cfg.moe_inter_dim
        K = indices.size(1)
        device = x.device

        x_2d = x.reshape(N, D).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)

        y = torch.zeros((N, D), dtype=torch.float32, device=device)
        for n in range(N):
            x_fp8, x_scale = sgl_per_token_group_quant_fp8(
                x_2d[n : n + 1].contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=False,
                scale_tma_aligned=False,
                scale_ue8m0=False,
            )
            for k in range(K):
                eid_t = indices[n, k : k + 1]
                router_w = weights[n, k : k + 1, None]

                w13 = torch.index_select(self._w13, 0, eid_t).squeeze(0)
                s13 = torch.index_select(self._s13, 0, eid_t).squeeze(0)
                gate_up = torch.empty(1, 2 * inter, device=device, dtype=torch.bfloat16)
                fp8_gemm_nt((x_fp8, x_scale), (w13, s13), gate_up)

                h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
                    gate_up,
                    clamp_limit=cfg.swiglu_limit,
                    group_size=FP8_BLOCK,
                )
                w2 = torch.index_select(self._w2, 0, eid_t).squeeze(0)
                s2 = torch.index_select(self._s2, 0, eid_t).squeeze(0)
                down_out = torch.empty(1, D, device=device, dtype=torch.bfloat16)
                fp8_gemm_nt((h_fp8, h_scale), (w2, s2), down_out)
                y[n : n + 1].add_(down_out.float() * router_w)

        return y

    # ------------------------------------------------------------------
    # EP dispatch (world_size == ep_size, e.g. DP8 x EP8): NCCL all-to-all
    # ------------------------------------------------------------------

    def _forward_ep(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Expert-parallel forward over SM90 ranks via NCCL all-to-all.

        Each rank owns ``self._e_local`` contiguous routed experts
        (``expert_start = ep_rank * e_local``). Per (token, topk) assignment:
        dispatch the token row to the expert-owner rank, compute the local
        grouped path there, all-to-all results back, and reduce with router
        weights on the originating rank. Collectives are entered by every
        rank (including empty local batches), matching the framework's
        DP/EP collective contract.

        Eager only: CUDA-graph capture with EP needs static-size dispatch
        buffers (DeepEP-style) and is not wired yet.
        """
        import torch.distributed as dist

        cfg = self.cfg
        N, D = x.shape
        K = indices.size(1)
        device = x.device
        world = int(cfg.ep_size)
        e_local = self._e_local
        expert_start = int(cfg.ep_rank) * e_local


        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("grouped_fp4_sm90 EP path requires torch.distributed")
        if dist.get_world_size() != world:
            raise RuntimeError(
                f"EP executor world mismatch: torch.distributed "
                f"{dist.get_world_size()} != cfg.ep_size {world}"
            )

        # NOTE: no early return on N == 0 — every rank must enter the
        # all-to-alls below (zero-row exchanges are well-formed) to keep the
        # collective symmetric across the DP/EP group.

        flat_ids = indices.reshape(-1).to(torch.int64)
        flat_w = weights.reshape(-1).to(torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < world * e_local)
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        flat_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))

        dest = flat_ids // e_local
        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=world).to(torch.int64)
        send_counts_cpu = send_counts.cpu().tolist()
        send_tokens = (
            torch.arange(N, device=device, dtype=torch.int64)
            .repeat_interleave(K)
            .index_select(0, order)
        )
        send_eids = flat_ids.index_select(0, order)
        send_w = flat_w.index_select(0, order)

        # Exchange row counts so every rank knows its recv splits.
        counts_in = torch.tensor(send_counts_cpu, dtype=torch.int64, device=device)
        counts_out = torch.empty_like(counts_in)
        dist.all_to_all_single(counts_out, counts_in)
        recv_counts_cpu = counts_out.cpu().tolist()

        total_send = int(sum(send_counts_cpu))
        total_recv = int(sum(recv_counts_cpu))

        # Dispatch token rows (bf16) + assignment metadata.
        send_x = x.index_select(0, send_tokens)
        recv_x = torch.empty((total_recv, D), dtype=x.dtype, device=device)
        dist.all_to_all_single(
            recv_x, send_x.contiguous(), recv_counts_cpu, send_counts_cpu
        )
        send_meta = torch.stack(
            (send_eids, torch.arange(total_send, device=device, dtype=torch.int64)),
            dim=1,
        )
        recv_meta = torch.empty((total_recv, 2), dtype=torch.int64, device=device)
        dist.all_to_all_single(
            recv_meta, send_meta.contiguous(), recv_counts_cpu, send_counts_cpu
        )

        # Compute on the received rows with LOCAL expert indices; router
        # weight applied by the sender during the reduce.
        local_ids = (recv_meta[:, 0] - expert_start).clamp_(0, e_local - 1)
        ones = torch.ones(
            (total_recv, 1), dtype=torch.float32, device=device
        )
        recv_out = self._forward_local(recv_x, ones, local_ids.unsqueeze(1))

        # Return per-assignment outputs to the originating ranks.
        send_back = recv_out.to(x.dtype).contiguous()
        recv_back = torch.empty((total_send, D), dtype=x.dtype, device=device)
        dist.all_to_all_single(
            recv_back, send_back, send_counts_cpu, recv_counts_cpu
        )

        # Deterministic reduce: one write per assignment slot, then a fixed
        # topk-order sum per token.
        assign_out = torch.zeros((N * K, D), dtype=torch.float32, device=device)
        assign_out.index_copy_(0, order, recv_back.float())
        y = (assign_out.view(N, K, D) * flat_w.view(N, K, 1)).sum(dim=1)
        return y

    def _forward_ep_static(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Capacity-padded sync-free EP forward (decode / graph capture).

        Dispatch shape is independent of the routing content: every rank
        sends a fixed ``[world, cap, D+2]`` buffer (``cap = N*K`` slots per
        destination, worst case), so the two all-to-alls use equal splits —
        no count exchange, no host synchronization, CUDA-graph-capturable.
        Expert/assignment metadata is bit-cast into two extra BF16 columns
        (both are small non-negative integers, exact in BF16; padding slots
        carry eid = -1). The expert-side compute runs on the masked grouped
        GEMMs (fixed ``[E, cap_m, .]`` allocation, device-side ``masked_m``),
        so only real rows compute while every shape stays constant.
        """
        import torch.distributed as dist

        cfg = self.cfg
        N, D = x.shape
        K = indices.size(1)
        device = x.device
        world = int(cfg.ep_size)
        e_local = self._e_local
        expert_start = int(cfg.ep_rank) * e_local
        inter = cfg.moe_inter_dim
        cap = N * K
        r_total = world * cap
        bf16 = torch.bfloat16

        flat_ids = indices.reshape(-1).to(torch.int64)
        flat_w = weights.reshape(-1).to(torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < world * e_local)
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        flat_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))

        # ---- sender-side packing (shape-stable) ----
        dest = flat_ids // e_local
        order = torch.argsort(dest, stable=True)
        dest_sorted = flat_ids.index_select(0, order) // e_local
        # scatter_add (not bincount): bincount synchronizes with the
        # host, which is illegal under graph capture.
        counts = torch.zeros(
            world, dtype=torch.int64, device=device
        ).scatter_add_(0, dest, torch.ones_like(dest))
        starts = torch.cumsum(counts, 0) - counts  # exclusive, per rank
        pos = torch.arange(cap, device=device, dtype=torch.int64)
        within = pos - starts.index_select(0, dest_sorted)
        assign_idx = order  # original assignment index per sorted position

        x_rows = x.repeat_interleave(K, dim=0).index_select(0, order).to(bf16)
        send_buf = torch.zeros((world, cap, D + 2), dtype=bf16, device=device)
        send_buf[..., D] = -1.0  # padding sentinel; valid slots overwrite
        send_buf[dest_sorted, within, :D] = x_rows
        send_buf[dest_sorted, within, D] = flat_ids.index_select(0, order).to(bf16)
        send_buf[dest_sorted, within, D + 1] = assign_idx.to(bf16)

        # ---- equal-split all-to-alls (no splits, no host sync) ----
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf)

        recv_eid = recv_buf[..., D].to(torch.int64)
        local_eid = (recv_eid - expert_start).clamp_(0, e_local - 1)
        recv_valid = recv_eid >= 0

        flat_rows = recv_buf[..., :D].reshape(r_total, D).contiguous()
        flat_local = local_eid.reshape(r_total)
        flat_valid = recv_valid.reshape(r_total)

        # ---- quantize received rows once (row-major FP32 scales) ----
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            flat_rows,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )

        # ---- masked grouped layout: [E, cap_m, .], constant shapes ----
        cap_m = align(cap * world, _GROUPED_ALIGNMENT)
        masked_in = torch.zeros(
            (e_local, cap_m, D), dtype=torch.float8_e4m3fn, device=device
        )
        masked_scale = torch.zeros(
            (e_local, cap_m, D // FP8_BLOCK), dtype=torch.float32, device=device
        )
        expert_start_loc = torch.arange(
            e_local, device=device, dtype=torch.int32
        ) * cap_m
        output_index = torch.zeros(
            (r_total, 1), dtype=torch.int32, device=device
        )
        ep_scatter_v2(
            a_fp8,
            a_scale,
            torch.where(
                flat_valid, flat_local, torch.full_like(flat_local, -1)
            ).to(torch.int32).unsqueeze(1),
            cap_m,
            expert_start_loc,
            masked_in.view(e_local * cap_m, D),
            masked_scale,
            output_index,
            scale_ue8m0=False,
        )
        count_ids = torch.where(
            flat_valid, flat_local, torch.full_like(flat_local, e_local)
        )
        masked_m = torch.zeros(
            e_local + 1, dtype=torch.int64, device=device
        ).scatter_add_(
            0, count_ids, torch.ones_like(count_ids)
        )[:e_local].to(torch.int32)
        expected_m = max(1, min(cap_m, r_total // max(e_local, 1)))

        # GEMM 1 (masked): gate+up
        gate_up = torch.empty(
            (e_local, cap_m, 2 * inter), device=device, dtype=bf16
        )
        m_grouped_fp8_gemm_nt_masked(
            (masked_in, masked_scale),
            (self._w13, self._s13),
            gate_up,
            masked_m,
            expected_m,
        )
        h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
            gate_up.view(e_local * cap_m, 2 * inter),
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        down = torch.empty((e_local, cap_m, D), device=device, dtype=bf16)
        m_grouped_fp8_gemm_nt_masked(
            (h_fp8.view(e_local, cap_m, inter), h_scale.view(e_local, cap_m, inter // FP8_BLOCK)),
            (self._w2, self._s2),
            down,
            masked_m,
            expected_m,
        )

        out_rows = down.view(e_local * cap_m, D).index_select(
            0, output_index.reshape(-1).clamp_(min=0)
        )

        # ---- equal-split all-to-all back ----
        send_back = out_rows.reshape(world, cap, D).contiguous()
        recv_back = torch.empty_like(send_back)
        dist.all_to_all_single(recv_back, send_back)

        # ---- deterministic sender-side reduce ----
        assign_out = torch.zeros((N * K, D), dtype=torch.float32, device=device)
        flat_back = recv_back.reshape(r_total, D)
        got = flat_back[dest_sorted * cap + within]
        assign_out.index_copy_(0, assign_idx, got.float())
        y = (assign_out.view(N, K, D) * flat_w.view(N, K, 1)).sum(dim=1)
        return y

    # ------------------------------------------------------------------
    # DeepEP low-latency EP (production decode + CUDA-graph pattern)
    # ------------------------------------------------------------------

    def _deepep_ll_enabled(self) -> bool:
        """Resolve the DeepEP LL binding once (env-gated).

        The engine creates the low-latency buffer at startup when
        ``--use_deepep_moe 1 --use_deepep_low_latency 1`` is set; this
        executor adopts it when ``DSV4_EP_DEEPEP_LL=1`` (the decode-role
        switch). The binding is validated against this layer's geometry so
        a mismatched buffer fails loudly instead of corrupting dispatch.
        """
        if getattr(self, "_deepep_ll", False) is not False:
            return bool(self._deepep_ll)
        if os.environ.get("DSV4_EP_DEEPEP_LL", "0") != "1":
            self._deepep_ll = False
            return False
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        wrapper = DeepEPWrapper.get_initialized_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "DSV4_EP_DEEPEP_LL=1 requires the DeepEP low-latency buffer; "
                "start the server with --use_deepep_moe 1 "
                "--use_deepep_low_latency 1"
            )
        if wrapper.mode != DeepEPMode.LOW_LATENCY:
            raise RuntimeError(
                f"DSV4_EP_DEEPEP_LL=1 but DeepEP wrapper mode is {wrapper.mode}"
            )
        cfg = self.cfg
        if wrapper.hidden_size != cfg.dim:
            raise RuntimeError(
                f"DeepEP buffer hidden {wrapper.hidden_size} != layer dim {cfg.dim}"
            )
        if wrapper.num_experts != cfg.n_routed_experts:
            raise RuntimeError(
                f"DeepEP buffer experts {wrapper.num_experts} != "
                f"layer experts {cfg.n_routed_experts}"
            )
        if wrapper.ep_size != int(cfg.ep_size):
            raise RuntimeError(
                f"DeepEP buffer ep_size {wrapper.ep_size} != layer ep_size "
                f"{cfg.ep_size}"
            )
        self._deepep_ll = wrapper
        return True

    def _forward_ep_deepep(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP low-latency dispatch → masked grouped GEMMs → combine.

        ``low_latency_dispatch`` returns the per-local-expert capacity
        layout ``[E, cap*world, D]`` with device-side per-expert counts —
        fixed shapes, collective-symmetric on empty local batches, and
        CUDA-graph-capturable. ``low_latency_combine`` applies the router
        weights and reduces over top-k on the originating rank.
        """
        cfg = self.cfg
        N, D = x.shape
        K = indices.size(1)
        device = x.device
        e_local = self._e_local
        world = int(cfg.ep_size)
        inter = cfg.moe_inter_dim

        wrapper = self._deepep_ll
        buffer = wrapper.buffer
        cap = wrapper.ll_num_max_token_per_rank

        flat_ids = indices.reshape(-1).to(torch.int64)
        flat_w = weights.reshape(-1).to(torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < world * e_local)
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        flat_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))
        topk_idx = flat_ids.view(N, K)
        topk_w = flat_w.view(N, K)

        recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_max_dispatch_tokens_per_rank=cap,
            num_experts=cfg.n_routed_experts,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=False,
        )
        # recv_x: [E, cap*world, D] bf16 (bf16 dispatch branch)
        E, M_tot, D_r = recv_x.shape
        assert E == e_local and D_r == D

        # Masked activation quant: only the rows each expert actually
        # received (recv_count). The capacity buffer's padded rows are
        # garbage the masked GEMM never reads, and at decode shapes the
        # valid fraction is tiny (N*K*world valid rows of E*M_tot), so the
        # unmasked quant burned ~1000x redundant work at bs=1 (measured
        # 2.7 ms/step in the PD decode trace). The v2 masked path was
        # initially blamed for corrupted output, but that was the wrapper's
        # 2D stride(-1) bug (fixed); validated since by the full-model PD
        # golden.
        recv_x_c = recv_x if recv_x.is_contiguous() else recv_x.contiguous()
        recv_count_i32 = recv_count.to(torch.int32)
        # Triton masked quant: the framework sgl wrapper's row-major branch
        # silently ignores masked_m (its MaskedLayoutScheduler only exists in
        # the column-major/UE8M0/fused-silu instantiation), so it quantized
        # the full capacity buffer; this kernel honors the mask and emits
        # the identical [E, M, G] FP32-scale layout.
        a_fp8, a_scale = per_token_group_quant_f32_scale_masked(
            recv_x_c,
            group_size=FP8_BLOCK,
            eps=1e-10,
            masked_m=recv_count_i32,
        )
        expected_m = max(1, min(M_tot, (N * K * world) // max(e_local, 1)))

        gate_up = torch.empty((E, M_tot, 2 * inter), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (a_fp8.view(E, M_tot, D), a_scale.view(E, M_tot, D // FP8_BLOCK)),
            (self._w13, self._s13),
            gate_up,
            recv_count,
            expected_m,
        )
        h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
            masked_m=recv_count_i32,
        )
        down = torch.empty((E, M_tot, D), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (
                h_fp8.view(E, M_tot, inter),
                h_scale.view(E, M_tot, inter // FP8_BLOCK),
            ),
            (self._w2, self._s2),
            down,
            recv_count,
            expected_m,
        )

        combined, _, _ = buffer.low_latency_combine(
            down,
            topk_idx,
            topk_w,
            handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
        )
        return combined.float()

    # ------------------------------------------------------------------
    # DeepGEMM SM90 MegaMoE (fused dispatch/GEMM/SwiGLU/combine)
    # ------------------------------------------------------------------

    def _mega_moe_enabled(self) -> bool:
        """Resolve the SM90 MegaMoE binding once (env-gated).

        Requires a deep_gemm build carrying the SM90 mega entry points
        (the nv_dev lineage / DeepGEMM PR #383; the currently pinned
        2.8.0 wheel is SM100-only for mega). Falls back with a clear
        error when the installed build lacks them.
        """
        cached = getattr(self, "_mega_moe", False)
        if cached is not False:
            return bool(cached)
        if os.environ.get("DSV4_EP_MEGA_MOE", "0") != "1":
            self._mega_moe = False
            return False
        import deep_gemm

        if not hasattr(deep_gemm, "get_symm_buffer_for_sm90_mega_moe"):
            raise RuntimeError(
                "DSV4_EP_MEGA_MOE=1 requires a deep_gemm build with the SM90 "
                "MegaMoE entry points (DeepGEMM PR #383 / nv_dev lineage); "
                f"installed deep_gemm {getattr(deep_gemm, '__version__', '?')} "
                "does not provide them"
            )
        cfg = self.cfg
        wrapper = None
        if int(cfg.dim) % 256 != 0 or int(cfg.moe_inter_dim) % 128 != 0:
            raise RuntimeError(
                "SM90 MegaMoE requires hidden % 256 == 0 and "
                f"moe_inter_dim % 128 == 0, got dim={cfg.dim} "
                f"inter={cfg.moe_inter_dim}"
            )
        self._mega_moe = deep_gemm
        return True

    def _init_mega_moe(self) -> None:
        """Create the symmetric buffer and transformed weights (collective).

        Must be called by every rank in the same layer order — model
        construction is already collective-symmetric, so binding here is
        safe. Idempotent.
        """
        if getattr(self, "_mega_buffer", None) is not None:
            return
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("SM90 MegaMoE requires torch.distributed")
        deep_gemm = self._mega_moe
        cfg = self.cfg
        cap = int(os.environ.get("DSV4_EP_MEGA_MOE_CAP", "128"))
        self._mega_cap = cap
        self._mega_buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
            dist.group.WORLD,
            cfg.n_routed_experts,
            cap,
            cfg.moe_k,
            cfg.dim,
            cfg.moe_inter_dim,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        (self._mega_l1, self._mega_l2) = deep_gemm.transform_weights_for_mega_moe_sm90(
            (self._w13, self._s13), (self._w2, self._s2)
        )

    def _forward_ep_mega(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """SM90 MegaMoE fused forward; returns ``[N, D] fp32``."""
        if getattr(self, "_mega_buffer", None) is None:
            self._init_mega_moe()
        cfg = self.cfg
        N, D = x.shape
        K = indices.size(1)
        device = x.device
        if N > self._mega_cap:
            raise RuntimeError(
                f"SM90 MegaMoE capacity {self._mega_cap} < batch {N}; raise "
                "DSV4_EP_MEGA_MOE_CAP or route this phase to another backend"
            )

        flat_ids = indices.reshape(-1).to(torch.int64)
        flat_w = weights.reshape(-1).to(torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < int(cfg.n_routed_experts))
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        flat_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))
        topk_idx = flat_ids.view(N, K)
        topk_w = flat_w.view(N, K)

        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        buf = self._mega_buffer
        buf.x[:N].copy_(a_fp8)
        buf.x_sf[:N].copy_(a_scale)
        buf.topk_idx[:N].copy_(topk_idx)
        buf.topk_weights[:N].copy_(topk_w)

        y = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        self._mega_moe.fp8_mega_moe(
            y,
            self._mega_l1,
            self._mega_l2,
            buf,
            recipe=(FP8_BLOCK, FP8_BLOCK, FP8_BLOCK),
            activation="swiglu",
            activation_clamp=float(cfg.swiglu_limit),
        )
        return y.float()
