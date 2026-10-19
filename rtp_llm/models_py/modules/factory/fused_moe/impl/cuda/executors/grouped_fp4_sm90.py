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
from rtp_llm.models_py.kernels.cuda.quant_layouts import _FP4_LUT, FP4_BLOCK, FP8_BLOCK
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    per_token_group_quant_f32_scale_masked,
    silu_mul_fp8_quant_f32_scale,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on both SM90 and SM100).
_GROUPED_ALIGNMENT = 128

# Opt-in persistent masked workspace (team/perf-log.md R36/R42): the per-
# forward torch.zeros of masked_in re-fills a ~33 MB fp8 buffer every decode
# step (the FillFunctor family, ~0.47 ms/step on the 4x H20 dp4ep4 decode
# trace). ep_scatter_v2 is a pure-STORE scatter and every downstream
# consumer (masked GEMM, silu+quant, ep_gather) is row-independent and only
# reads rows below masked_m, so uninitialized capacity rows never reach the
# output. Default off pending the A/B + semantic gate.
_MASKED_WS_CACHE: Dict[Any, Any] = {}


def _masked_ws_reuse_enabled() -> bool:
    return os.environ.get("DSV4_MASKED_WS_REUSE", "0") == "1"


def _static_pack_fused_enabled() -> bool:
    """M-R1 fused sender-side packing (R473): swap the aten
    argsort/count/cumsum/index_put chain in ``_forward_ep_static`` for the
    three Triton kernels in ``triton_kernels/moe/ep_static_pack.py``.
    Capture path only (global_cap is None checked at the call site); the
    aten chain stays the default and the eager/sidecar variant's fallback."""
    return os.environ.get("DSV4_EP_STATIC_PACK_FUSED", "0") == "1"


def _static_eager_enabled() -> bool:
    """Route eager (non-captured) EP forwards through the capacity-padded
    static path (INBOX user note 2026-10-15 11:25: fix the eager prefill
    collective-spin root cause with the _forward_ep_static recipe —
    preallocated capacity buffers, equal-split all-to-alls, scatter_add,
    zero host sync — on a variable-length-capable eager path)."""
    return os.environ.get("DSV4_EP_STATIC_EAGER", "0") == "1"


def _allgather_ep_enabled() -> bool:
    """Route eager (non-captured) EP forwards through the all-gather +
    reduce-scatter combine (INBOX user note 2026-10-15 16:20: borrow the
    grouped_fp8 Hopper transport — one host sync per layer, no host
    rendezvous, no per-assignment all-to-alls — onto the FP4 grouped
    compute; "borrow, don't copy PR #1508")."""
    return os.environ.get("DSV4_EP_ALLGATHER", "0") == "1"


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16} (intranode.cu asserts false on
# others; ported from dsv41 strategies/deepep.py). V4-Flash routes K=6, so
# the normal-mode forward pads to 8 — see ``_pad_topk_for_deepep``.
_DEEPEP_NORMAL_SUPPORTED_TOPK = (2, 4, 8, 16)


def _pad_topk_for_deepep(
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``(indices, weights)`` to a DeepEP-supported topk width.

    Padding slots carry the -1/0.0 sentinel pair: get_dispatch_layout drops
    them from the rank/expert maps and ``recompute_topk_ids_sum_expert_count``
    drops them from the local expert counts, so they never reach a GEMM.
    Whether the pinned upstream wheel carries the same topk constraint is
    unverified (team/perf-log.md R85), so the pad is unconditional — it is
    a no-op for topk values the kernel already accepts.
    """
    n_act = indices.size(-1)
    if n_act in _DEEPEP_NORMAL_SUPPORTED_TOPK:
        return indices, weights
    pad_to = next((k for k in _DEEPEP_NORMAL_SUPPORTED_TOPK if k > n_act), None)
    if pad_to is None:
        raise RuntimeError(
            "DeepEP normal dispatch supports topk in "
            f"{sorted(_DEEPEP_NORMAL_SUPPORTED_TOPK)}, got {n_act}"
        )
    N = indices.size(0)
    pad_n = pad_to - n_act
    pad_idx = torch.full((N, pad_n), -1, dtype=indices.dtype, device=indices.device)
    pad_w = torch.zeros((N, pad_n), dtype=weights.dtype, device=weights.device)
    return (
        torch.cat([indices, pad_idx], dim=-1),
        torch.cat([weights, pad_w], dim=-1),
    )


def _static_eager_min_tokens() -> int:
    """Only route prefill-sized forwards through the static-eager path.

    Decode-sized eager forwards (engine warmup, tiny batches) must stay on
    the dynamic path: the capacity buffers (~13 GB at the 16384-token
    bucket) would otherwise be allocated and allocator-retained right
    before the decode CUDA-graph capture and OOM it (R59.4).
    """
    return int(os.environ.get("DSV4_EP_STATIC_EAGER_MIN_TOKENS", "1024"))


_CHUNKED_TOKENS_VAR = None


def _static_eager_chunked_tokens() -> int:
    """Rank-consistent prefill signal for the static-eager gate.

    The chunker synchronizes the prompt token count across DP ranks
    (all_reduce MAX) before chunking, so this value is IDENTICAL on every
    rank — unlike the local batch size N, which is 0 on the idle ranks of
    a single-request prefill. Gating on the local N alone splits the
    ranks across two different collective sequences (static: equal-split
    all-to-alls; dynamic: count-based all-to-alls) — the mismatched first
    collective corrupts the NCCL send/recv pairings and manifests as
    downstream memory corruption (v5/v6 crash root cause,
    team/perf-log R63.1). Returns 0 when no chunked prefill is in flight
    (warmup / decode), preserving the R59.4 small-N guard.
    """
    global _CHUNKED_TOKENS_VAR
    if _CHUNKED_TOKENS_VAR is None:
        # Deferred import: chunked_layer pulls in the strategy registry,
        # which imports this module.
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            _SYNCHRONIZED_CHUNK_TOKENS,
        )

        _CHUNKED_TOKENS_VAR = _SYNCHRONIZED_CHUNK_TOKENS
    planned = _CHUNKED_TOKENS_VAR.get()
    return planned if planned is not None else 0


def _static_eager_cap_tokens() -> int:
    """Dispatch capacity in tokens: the MoE chunk upper bound (prompts are
    chunked to <= this many tokens upstream, so N*K <= cap always holds)."""
    raw = os.environ.get("DSV4_EP_STATIC_EAGER_CAP_TOKENS")
    if raw:
        return int(raw)
    raw = os.environ.get("DSV4_MOE_CHUNK_TOKENS")
    if raw:
        return int(raw)
    return 16384


_STATIC_EAGER_OVERFLOW: dict = {}


def _static_eager_overflow_add(delta: torch.Tensor) -> None:
    key = str(delta.device)
    total = _STATIC_EAGER_OVERFLOW.get(key)
    if total is None:
        total = torch.zeros((), dtype=torch.int64, device=delta.device)
        _STATIC_EAGER_OVERFLOW[key] = total
    total.add_(delta)


def _static_eager_overflow_check() -> int:
    """Sum of per-expert capacity overflows seen so far (syncs once)."""
    return int(sum(int(t.item()) for t in _STATIC_EAGER_OVERFLOW.values()))


def _masked_ws_buffers(e_local: int, cap_m: int, D: int, device: torch.device) -> tuple:
    """Persistent (masked_in, masked_scale) buffers, torch.empty (no fill)."""
    key = (e_local, cap_m, D, str(device))
    bufs = _MASKED_WS_CACHE.get(key)
    if bufs is None:
        masked_in = torch.empty(
            (e_local, cap_m, D), dtype=torch.float8_e4m3fn, device=device
        )
        masked_scale = torch.empty(
            (e_local, cap_m, D // FP8_BLOCK), dtype=torch.float32, device=device
        )
        bufs = (masked_in, masked_scale)
        _MASKED_WS_CACHE[key] = bufs
    return bufs


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
    s_fp32 = torch.empty(
        (E, n_blocks_n, n_blocks_k), dtype=torch.float32, device=device
    )

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
        w_fp8[start:end] = q.view(end - start, out_dim, in_dim).to(torch.float8_e4m3fn)
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
            if _static_eager_enabled() and (
                N >= _static_eager_min_tokens()
                or _static_eager_chunked_tokens() >= _static_eager_min_tokens()
            ):
                # Eager + opt-in + prefill-sized: the capacity-padded
                # equal-split path with a CONFIG-static global capacity
                # (chunk upper bound), which keeps the shapes symmetric
                # across heterogeneous-N ranks without any host sync (see
                # _forward_ep_eager_static). Small-N eager forwards (engine
                # warmup before graph capture) stay dynamic: the capacity
                # buffers would OOM the capture (R59.4). The gate reads the
                # chunker's RANK-SYNCHRONIZED token count alongside the
                # local N so every rank picks the same collective sequence —
                # idle DP ranks of a single-request prefill must enter the
                # static path too, or the first all-to-all pairs a static
                # equal-split sender with a dynamic count-based receiver
                # (NCCL mismatch, v5/v6 corruption root cause, R63.1).
                return self._forward_ep_eager_static(x, weights, indices)
            if _allgather_ep_enabled():
                # Eager + opt-in: the all-gather/reduce-scatter combine —
                # one host sync per layer, shape-stable collectives, the
                # stock _forward_local compute over the gathered rows.
                # Lower priority than the static-eager gate above on
                # purpose: the two are alternative eager transports and
                # only one should ever be enabled.
                return self._forward_ep_allgather(x, weights, indices)
            if self._deepep_normal_enabled():
                # Eager + opt-in: DeepEP NORMAL-mode dispatch/combine (the
                # INBOX 2026-10-15 16:20 direction (a); NOT the dead
                # low-latency line — normal mode was never tried on this
                # lane). Third alternative eager transport; mutually
                # exclusive with the two gates above in practice.
                return self._forward_ep_deepep_normal(x, weights, indices)
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
        ones = torch.ones((total_recv, 1), dtype=torch.float32, device=device)
        recv_out = self._forward_local(recv_x, ones, local_ids.unsqueeze(1))

        # Return per-assignment outputs to the originating ranks.
        send_back = recv_out.to(x.dtype).contiguous()
        recv_back = torch.empty((total_send, D), dtype=x.dtype, device=device)
        dist.all_to_all_single(recv_back, send_back, send_counts_cpu, recv_counts_cpu)

        # Deterministic reduce: one write per assignment slot, then a fixed
        # topk-order sum per token.
        assign_out = torch.zeros((N * K, D), dtype=torch.float32, device=device)
        assign_out.index_copy_(0, order, recv_back.float())
        y = (assign_out.view(N, K, D) * flat_w.view(N, K, 1)).sum(dim=1)
        return y

    def _forward_ep_allgather(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather + reduce-scatter EP combine over the FP4 grouped compute.

        The transport of the upstream grouped_fp8 Hopper path (PR #1508)
        borrowed onto this executor's FP4 compute, per the 2026-10-15 16:20
        direction note: keep the masked/contiguous grouped GEMMs, replace
        only dispatch/combine. Per layer: one MAX all-reduce on the local
        token count (the only COLLECTIVE host sync — _forward_local's
        per-expert count readback is a second, local D2H), three
        all_gather_into_tensor calls (activations, router weights, expert
        ids), the stock _forward_local compute over the gathered rows with
        ids remapped to the local expert range, then one
        reduce_scatter_tensor whose cross-rank sum rebuilds each rank's
        own token rows.

        Versus _forward_ep's per-assignment all-to-alls (N*K rows each way,
        a counts exchange, and two host syncs per layer), this moves ~K
        times fewer activation bytes (K=6, ep=4) through shape-stable
        NCCL collectives and drops the argsort/bincount host work
        entirely. Numerics: the router weight is applied inside ep_gather
        per local assignment and each rank's partial is summed in FP32
        there, then rounded to BF16 once for the reduce-scatter payload
        (the same BF16 wire format _forward_ep uses per assignment) — no
        bit-parity claim against the all-to-all path is made or needed
        (output-semantics obligation only).

        Eager only: the .item() on the count sync forbids graph capture
        (capture keeps taking _forward_ep_static). Empty local batches
        still participate: exchange_n is floored at 1 and zero-padded rows
        are inert (-1 ids are skipped by ep_scatter and leave ep_gather's
        accumulator at zero).
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

        # Rank-uniform exchange size — the one host sync per layer.
        # Independent DP schedulers may hold unequal batches (including
        # zero); the MAX keeps every rank's collectives shape-symmetric.
        count = torch.tensor(N, dtype=torch.int64, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        exchange_n = max(int(count.item()), 1)

        if N < exchange_n:
            pad = exchange_n - N
            x = torch.nn.functional.pad(x, (0, 0, 0, pad), value=0.0)
            weights = torch.nn.functional.pad(weights, (0, 0, 0, pad), value=0.0)
            indices = torch.nn.functional.pad(indices, (0, 0, 0, pad), value=-1)

        # All ranks' tokens, router weights, and expert ids. The gathered
        # activation block is world*exchange_n rows — the only per-layer
        # transient beyond _forward_local's own (chunk-aware via the
        # caller's chunking; see the R84 memory budget note).
        x_full = torch.empty((world * exchange_n, D), dtype=x.dtype, device=device)
        dist.all_gather_into_tensor(x_full, x.contiguous())
        w_full = torch.empty(
            (world * exchange_n, K), dtype=torch.float32, device=device
        )
        dist.all_gather_into_tensor(
            w_full, weights.reshape(x.shape[0], -1).to(torch.float32).contiguous()
        )
        i_full = torch.empty((world * exchange_n, K), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(
            i_full, indices.reshape(x.shape[0], -1).to(torch.int64).contiguous()
        )

        # Remap ids to the local expert range. Non-local, invalid, and
        # padding slots become -1: the recompute kernel's sentinel contract
        # (ep_kernels.recompute_topk_ids_sum_expert_count) drops them from
        # the expert counts, ep_scatter skips them, and ep_gather leaves
        # those tokens' accumulator at zero — so this rank's partial is
        # exactly its local experts' weighted contribution.
        flat = i_full.reshape(-1)
        valid = (flat >= 0) & (flat < world * e_local)
        local = torch.where(valid, flat - expert_start, torch.full_like(flat, -1))
        keep = (local >= 0) & (local < e_local)
        i_remapped = torch.where(keep, local, torch.full_like(local, -1)).view(
            world * exchange_n, K
        )

        partial = self._forward_local(x_full, w_full, i_remapped)

        # reduce_scatter_tensor sums element-wise across ranks and hands
        # rank r the r-th chunk of rows; summed over ranks, each row is the
        # full top-k total for that token. BF16 payload halves the wire
        # bytes (the MoE output is BF16 downstream regardless).
        out = torch.empty((exchange_n, D), dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(out, partial.to(torch.bfloat16).contiguous())
        return out[:N].float()

    # ------------------------------------------------------------------
    # DeepEP NORMAL-mode EP (prefill transport; INBOX 2026-10-15 16:20 (a))
    # ------------------------------------------------------------------

    def _deepep_normal_enabled(self) -> bool:
        """Resolve the DeepEP normal-mode binding once (env-gated).

        The engine creates the normal buffer at startup when the server is
        started with ``--use_deepep_moe 1 --use_deepep_low_latency 0
        --use_deepep_internode 0`` (backend_manager -> init_deepep_wrapper;
        with tp_size == 1 ``use_deepep_moe=1`` itself flips
        ``use_all_gather`` off in server_config_setup, so no extra flag is
        needed); this executor adopts it when ``DSV4_EP_DEEPEP_NORMAL=1``.
        The binding is validated against this layer's geometry so a
        mismatched buffer fails loudly instead of corrupting dispatch.
        """
        if getattr(self, "_deepep_normal", False) is not False:
            return bool(self._deepep_normal)
        if os.environ.get("DSV4_EP_DEEPEP_NORMAL", "0") != "1":
            self._deepep_normal = False
            return False
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        wrapper = DeepEPWrapper.get_initialized_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "DSV4_EP_DEEPEP_NORMAL=1 requires the DeepEP normal buffer; "
                "start the server with --use_deepep_moe 1 "
                "--use_deepep_low_latency 0 --use_deepep_internode 0"
            )
        if wrapper.mode != DeepEPMode.NORMAL:
            raise RuntimeError(
                f"DSV4_EP_DEEPEP_NORMAL=1 but DeepEP wrapper mode is {wrapper.mode}"
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
        self._deepep_normal = wrapper
        return True

    def _forward_ep_deepep_normal(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP normal dispatch → stock ``_forward_local`` → combine.

        The INBOX 2026-10-15 16:20 direction (a) transport: keep the FP4
        grouped compute untouched, replace only dispatch/combine with
        DeepEP's normal-mode kernels (device-side layout, one dispatch, one
        combine — the same prefill pattern the framework's
        DeepepNormalRouter family and dsv41's DeepEPStrategy use). Versus
        ``_forward_ep``'s per-assignment all-to-alls + counts exchange +
        host rendezvous, the expert routing happens inside the dispatch
        kernel and no host-side argsort/bincount runs at all. This is NOT
        the rejected low-latency line (DSV4_EP_DEEPEP_LL): that was the
        fixed-capacity decode variant with the upstream dispatch bug;
        normal mode was never run on this lane.

        Numerics: DeepEP's combine is a WEIGHTLESS cross-rank addition, so
        the router weights ride the dispatched payload and are applied
        inline by ``_forward_local``'s ep_gather on the recv rows — the
        same weighting the single-rank path already does. The fp32 partial
        is rounded to BF16 once for the combine payload (the same wire
        format class ``_forward_ep`` uses per assignment); no bit-parity
        claim against the all-to-all path is made or needed
        (output-semantics obligation only).

        ``recv_topk_idx`` comes back in the LOCAL expert index space with
        -1 rows for non-destined slots (dsv41 field note: force int64 +
        contiguous — the ACCL tensor occasionally carries a non-standard
        dtype). K=6 is padded to 8 for the intranode dispatch kernel's
        topk switch; padded slots are -1/0.0 and drop out through the
        sentinel contract.

        Eager only: the dispatch layout and combine synchronize with the
        host, so capture keeps taking ``_forward_ep_static``. Every rank
        enters the collectives symmetrically, including empty local batches
        (no early return on N == 0 — zero-row dispatches are well-formed).
        """
        cfg = self.cfg
        N, D = x.shape
        K = indices.size(1)
        device = x.device

        wrapper = self._deepep_normal
        buffer = wrapper.buffer

        topk_idx = indices.to(torch.int64).contiguous()
        topk_w = weights.reshape(N, K).to(torch.float32).contiguous()
        topk_idx, topk_w = _pad_topk_for_deepep(topk_idx, topk_w)

        # 1. Device-side dispatch layout.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, cfg.n_routed_experts)

        # 2. Dispatch the BF16 token rows + topk scaffolding (bf16 branch:
        # activation quant stays inside _forward_local, matching the
        # DeepEPNormalRouter's non-fp8 _prepare_dispatch_input).
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            _num_recv_tokens_per_expert_list,
            handle,
            _,
        ) = buffer.dispatch(
            x.contiguous(),
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            topk_idx,
            topk_w,
            expert_alignment=1,
        )

        # 3. Local compute over the recv rows, LOCAL expert ids as-is.
        recv_idx = recv_topk_idx.to(torch.int64).contiguous()
        M = recv_x.size(0)
        if M > 0:
            y_local = self._forward_local(
                recv_x.contiguous(),
                recv_topk_weights.to(torch.float32).contiguous(),
                recv_idx,
            )
        else:
            # No recv rows this rank — an empty fp32 partial keeps combine
            # well-formed (dsv41 DeepEPStrategy's M == 0 guard).
            y_local = torch.zeros(M, D, dtype=torch.float32, device=device)

        # 4. Combine back to the source ranks (weightless addition; the
        # per-assignment weights were already applied in ep_gather). The
        # wire dtype must match the dispatched activation's.
        y_combined, _, _ = buffer.combine(y_local.to(x.dtype), handle)
        return y_combined.float()

    def _forward_ep_static(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        global_cap: "int | None" = None,
    ) -> torch.Tensor:
        """Capacity-padded sync-free EP forward (decode / graph capture).

        Dispatch shape is independent of the routing content: every rank
        sends a fixed ``[world, cap, .]`` buffer (``cap = N*K`` slots per
        destination, worst case), so the two all-to-alls use equal splits —
        no count exchange, no host synchronization, CUDA-graph-capturable.
        Expert/assignment metadata is bit-cast into two extra BF16 columns
        on the capture path (small non-negative integers, exact in BF16 at
        decode scale; padding slots carry eid = -1); the eager path carries
        it in an FP32 sidecar instead — assign_idx reaches cap-1 at chunk
        scale, beyond BF16's exact-integer range (R62.1). The expert-side
        compute runs on the masked grouped GEMMs (fixed ``[E, cap_m, .]``
        allocation, device-side ``masked_m``), so only real rows compute
        while every shape stays constant.

        ``global_cap`` (eager variant): use a config-static slot count
        instead of this rank's ``N*K`` so heterogeneous-N ranks stay
        shape-symmetric (see ``_forward_ep_eager_static``); ``None`` keeps
        the capture path byte-identical to the historical behavior.
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
        cap = N * K if global_cap is None else int(global_cap)
        r_total = world * cap
        bf16 = torch.bfloat16

        flat_ids = indices.reshape(-1).to(torch.int64)
        flat_w = weights.reshape(-1).to(torch.float32)
        valid = (flat_ids >= 0) & (flat_ids < world * e_local)
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        flat_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))

        pack_fused = global_cap is None and _static_pack_fused_enabled()
        if pack_fused:
            # M-R1 fused sender-side packing (R473): three Triton kernels
            # replace the argsort/count/cumsum/index_put chain below. Slot
            # layout is bitwise-identical (parity gate:
            # modules/dsv4/test/test_ep_static_pack_parity.py); the combine
            # reads slots back by assignment index -- the same bijection the
            # aten form materialized via argsort + index_copy.
            from rtp_llm.models_py.triton_kernels.moe.ep_static_pack import (
                ep_static_pack_send,
            )

            x_bf = (
                x
                if (x.dtype == bf16 and x.is_contiguous())
                else (x.to(bf16).contiguous())
            )
            send_buf, within = ep_static_pack_send(flat_ids, x_bf, world, e_local, cap)
            send_meta = None
            use_sidecar = False
        else:
            # ---- sender-side packing (shape-stable) ----
            dest = flat_ids // e_local
            order = torch.argsort(dest, stable=True)
            dest_sorted = flat_ids.index_select(0, order) // e_local
            # scatter_add (not bincount): bincount synchronizes with the
            # host, which is illegal under graph capture.
            counts = torch.zeros(world, dtype=torch.int64, device=device).scatter_add_(
                0, dest, torch.ones_like(dest)
            )
            starts = torch.cumsum(counts, 0) - counts  # exclusive, per rank
            pos = torch.arange(N * K, device=device, dtype=torch.int64)
            within = pos - starts.index_select(0, dest_sorted)
            assign_idx = order  # original assignment index per sorted position

            x_rows = x.repeat_interleave(K, dim=0).index_select(0, order).to(bf16)
            # Eager path (global_cap): metadata travels in an FP32 sidecar
            # instead of the two BF16 columns. BF16 integers are exact only
            # below 256: eid qualifies at any chunk size, but assign_idx
            # reaches cap-1 (24575 at the 4096-token bucket), so the
            # "small non-negative integers, exact in BF16" assumption fails
            # at prefill scale (R62.1). The columns are write-only today,
            # but carry exact integers so no future consumer can silently
            # read corrupted indices back. FP32 is integer-exact to 2^24.
            use_sidecar = global_cap is not None
            send_buf = torch.zeros(
                (world, cap, D + (0 if use_sidecar else 2)), dtype=bf16, device=device
            )
            send_buf[dest_sorted, within, :D] = x_rows
            send_meta = None
            if use_sidecar:
                send_meta = torch.zeros(
                    (world, cap, 2), dtype=torch.float32, device=device
                )
                send_meta[..., 0] = -1.0  # padding sentinel; valid slots overwrite
                send_meta[dest_sorted, within, 0] = flat_ids.index_select(0, order).to(
                    torch.float32
                )
                send_meta[dest_sorted, within, 1] = assign_idx.to(torch.float32)
            else:
                send_buf[..., D] = -1.0  # padding sentinel; valid slots overwrite
                send_buf[dest_sorted, within, D] = flat_ids.index_select(0, order).to(
                    bf16
                )
                send_buf[dest_sorted, within, D + 1] = assign_idx.to(bf16)

        # ---- equal-split all-to-alls (no splits, no host sync) ----
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all_single(recv_buf, send_buf)
        if use_sidecar:
            recv_meta = torch.empty_like(send_meta)
            dist.all_to_all_single(recv_meta, send_meta)
            recv_eid = recv_meta[..., 0].to(torch.int64)
        else:
            recv_meta = None
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
        # Capture path (global_cap is None): the historical worst-case-safe
        # formula (cap is tiny at decode, so the 64x over-provision is free).
        # Eager path: cap is the global chunk-bucket capacity, so scale the
        # per-expert capacity to slack x the equal-split share instead (the
        # historical formula would demand ~100 GB at a 16k-token chunk).
        if global_cap is None:
            cap_m = align(cap * world, _GROUPED_ALIGNMENT)
        else:
            # Default 0.5 (R60 budget memo): slack-scaled equal-share mean,
            # keeping the phase-peak transient footprint low against the
            # ~16 GiB post-capture headroom (slack=2 demanded ~21 GB and
            # OOMed the decode capture when the buffers were allocated
            # early).
            raw_slack = os.environ.get("DSV4_EP_STATIC_EAGER_SLACK", "0.5")
            slack = max(1, int(float(raw_slack) * 2))
            share = max(1, slack * cap * world // (e_local * 2))
            # Structural floor (v5 crash root cause, team/perf-log R62.2):
            # topk routes are per-token DISTINCT, so a single active DP
            # rank contributes at most cap//K rows to any one expert. The
            # perf harness's degenerate repeated "hello" prompt routes
            # ~every token of a chunk to the same top-6 experts (~cap//K
            # rows each), which blows past the 8x-mean share (3072 < 4096
            # at the 4096-token bucket) and drives the unguarded
            # ep_scatter_v2 atomic cursor past the expert segment — the
            # OOB write surfaces as CUBLAS_STATUS_EXECUTION_FAILED at the
            # next gate GEMM. With the floor, single-active-rank overflow
            # is impossible by construction; multi-rank skew beyond it
            # still trips the device-side overflow counter.
            bound = cap // K
            cap_m = align(max(share, bound), _GROUPED_ALIGNMENT)
        if _masked_ws_reuse_enabled():
            masked_in, masked_scale = _masked_ws_buffers(e_local, cap_m, D, device)
        else:
            masked_in = torch.zeros(
                (e_local, cap_m, D), dtype=torch.float8_e4m3fn, device=device
            )
            masked_scale = torch.zeros(
                (e_local, cap_m, D // FP8_BLOCK), dtype=torch.float32, device=device
            )
        expert_start_loc = (
            torch.arange(e_local, device=device, dtype=torch.int32) * cap_m
        )
        output_index = torch.zeros((r_total, 1), dtype=torch.int32, device=device)
        ep_scatter_v2(
            a_fp8,
            a_scale,
            torch.where(flat_valid, flat_local, torch.full_like(flat_local, -1))
            .to(torch.int32)
            .unsqueeze(1),
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
        masked_m = (
            torch.zeros(e_local + 1, dtype=torch.int64, device=device)
            .scatter_add_(0, count_ids, torch.ones_like(count_ids))[:e_local]
            .to(torch.int32)
        )
        if global_cap is not None:
            # Track per-expert capacity overflow device-side (zero host
            # sync): an expert whose real count exceeds cap_m would already
            # have been scattered out of bounds — the counter makes the
            # corruption loud at the next natural sync point instead of
            # silently producing garbage. Check via
            # _static_eager_overflow_check() (or STRICT mode per layer).
            _static_eager_overflow_add((masked_m > cap_m).sum())
            if os.environ.get("DSV4_EP_STATIC_EAGER_STRICT", "0") == "1":
                if _static_eager_overflow_check() > 0:
                    raise RuntimeError(
                        "static-eager EP overflow: per-expert recv count "
                        "exceeded cap_m; raise DSV4_EP_STATIC_EAGER_SLACK"
                    )
        expected_m = max(1, min(cap_m, r_total // max(e_local, 1)))

        # GEMM 1 (masked): gate+up
        gate_up = torch.empty((e_local, cap_m, 2 * inter), device=device, dtype=bf16)
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
            (
                h_fp8.view(e_local, cap_m, inter),
                h_scale.view(e_local, cap_m, inter // FP8_BLOCK),
            ),
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
        flat_back = recv_back.reshape(r_total, D)
        if pack_fused:
            # Assignment-indexed slot lookup: dest[i] * cap + within[i] is
            # the slot assignment i was written to, so the gather lands rows
            # directly in assignment order -- the aten form's index_copy on
            # the argsort permutation, folded away (R483 equivalence).
            slot = (flat_ids // e_local).to(torch.int64) * cap + within.to(torch.int64)
            assign_out = flat_back[slot].float()
        else:
            assign_out = torch.zeros((N * K, D), dtype=torch.float32, device=device)
            got = flat_back[dest_sorted * cap + within]
            assign_out.index_copy_(0, assign_idx, got.float())
        y = (assign_out.view(N, K, D) * flat_w.view(N, K, 1)).sum(dim=1)
        return y

    def _forward_ep_eager_static(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length-capable static EP forward for eager prefill.

        Same recipe as ``_forward_ep_static`` (capacity-padded buffers,
        equal-split all-to-alls, scatter_add, zero host sync) but with a
        CONFIG-static global capacity (the MoE chunk upper bound) so
        heterogeneous-N ranks — e.g. a single 16k prompt on one DP rank
        while the others are empty — stay shape-symmetric. Kills the
        per-layer count-exchange D2H rendezvous and halves the collectives
        (4 -> 2), which is the 72.2% NCCL-spin TTFT root cause
        (team/perf-log.md R53.2; INBOX user note 2026-10-15 11:25).
        Wire cost: idle ranks ship capacity-sized padding (~2x aggregate
        transfer vs the dynamic path), bounded by the chunk bucket.
        """
        cap_tokens = _static_eager_cap_tokens()
        N = x.shape[0]
        if N > cap_tokens:
            raise RuntimeError(
                f"DSV4_EP_STATIC_EAGER: N={N} exceeds the capacity bucket "
                f"{cap_tokens}; raise DSV4_EP_STATIC_EAGER_CAP_TOKENS"
            )
        K = indices.size(1)
        return self._forward_ep_static(x, weights, indices, global_cap=cap_tokens * K)

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

        gate_up = torch.empty(
            (E, M_tot, 2 * inter), device=device, dtype=torch.bfloat16
        )
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
