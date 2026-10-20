"""Hopper MoE with FP4 checkpoint weights converted to FP8 at load time."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.quant_layouts import _FP4_LUT, FP4_BLOCK, FP8_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_f32_scale,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

_GROUPED_ALIGNMENT = 128


def _has_sm90_grouped_fp8_kernel() -> bool:
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked"):
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
    # Scales are recomputed per 128x128 block after FP4 dequantization.
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

        blocks = w_f32.view(end - start, n_blocks_n, FP8_BLOCK, n_blocks_k, FP8_BLOCK)
        amax = blocks.abs().amax(dim=(2, 4))
        scale = (amax / fp8_max).clamp_(min=1e-12)
        q = (blocks / scale[:, :, None, :, None]).clamp_(-fp8_max, fp8_max)
        w_fp8[start:end] = q.view(end - start, out_dim, in_dim).to(torch.float8_e4m3fn)
        s_fp32[start:end] = scale
        del w_f32, blocks, amax, scale, q
    return w_fp8, s_fp32


class GroupedFp4Sm90Executor(Fp8Fp4ExecutorBase):
    """DeepEP normal prefill and low-latency graph decode on SM90."""

    # Bound the temporary FP32 storage during weight conversion.
    _DEQUANT_EXPERT_SLICE = 16

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.dim % FP8_BLOCK == 0)
        checker.check(config.moe_inter_dim % FP8_BLOCK == 0)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(_has_sm90_grouped_fp8_kernel())

    def setup_weights(self, layer_weights: Dict) -> None:
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

        torch.cuda.empty_cache()

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation.lower() not in ("silu", "siglu", "swiglu"):
            raise ValueError("SM90 MoE requires SiLU activation")
        if (
            expert_map is not None
            or a2_scale is not None
            or apply_router_weight_on_input
        ):
            raise ValueError(
                "SM90 MoE requires output routing weights and internal scales"
            )
        if payload.expert_x_scale is None:
            raise ValueError("SM90 MoE requires FP8 dispatch scales")
        if payload.expert_x.ndim == 3:
            output = self._forward_masked(payload)
        else:
            if not payload.expert_ids_are_local:
                raise ValueError("SM90 contiguous MoE requires rank-local expert IDs")
            output = self._forward_local(
                payload.expert_x,
                payload.expert_x_scale,
                payload.expert_topk_weights,
                payload.expert_topk_ids,
            )
        return CombineForwardPayload(fused_expert_output=output)

    def _forward_local(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        N, D = x.shape
        E = self._e_local
        inter = cfg.moe_inter_dim
        device = x.device

        if x.dtype != torch.float8_e4m3fn or x_scale.dtype != torch.float32:
            raise ValueError("SM90 expert input requires FP8 data and FP32 scales")
        if x_scale.shape != (N, D // FP8_BLOCK):
            raise ValueError("SM90 expert input requires one scale per 128 elements")

        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=0,
            num_local_experts=E,
        )

        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.bfloat16, device=device)

        aligned_counts = torch.tensor(
            aligned_counts_list,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).to(device, non_blocking=True)

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
            x,
            x_scale,
            adjusted_topk_ids,
            aligned_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=False,
        )

        m_indices.clamp_(min=0, max=E - 1)

        gate_up = torch.empty(
            all_tokens, 2 * inter, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_gemm_nt_contiguous(
            (scatter_out, scatter_out_scale),
            (self._w13, self._s13),
            gate_up,
            m_indices,
            disable_ue8m0_cast=True,
        )
        del scatter_out, scatter_out_scale

        h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        del gate_up

        down_out = torch.empty(all_tokens, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
            disable_ue8m0_cast=True,
        )
        del h_fp8, h_scale, m_indices

        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out

    def _forward_masked(self, payload: ExpertForwardPayload) -> torch.Tensor:
        cfg = self.cfg
        a_fp8, a_scale = payload.expert_x, payload.expert_x_scale
        D = cfg.dim
        E, M, hidden = a_fp8.shape
        if E != self._e_local or hidden != D:
            raise ValueError("DeepEP LL received an incompatible expert layout")
        if a_fp8.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
            raise ValueError("DeepEP LL must return FP8 data and unpacked FP32 scales")
        if a_scale.shape != (E, M, D // FP8_BLOCK):
            raise ValueError(
                "DeepEP LL requires block-128 scales; AcclEP needs ACCL_FP8_CAST_LEVEL=1"
            )
        counts = payload.expert_tokens_meta.expert_num_tokens
        expected_m = max(1, min(M, payload.expert_tokens_meta.expected_m))
        inter = cfg.moe_inter_dim
        gate_up = torch.empty(
            (E, M, 2 * inter), dtype=torch.bfloat16, device=a_fp8.device
        )
        m_grouped_fp8_gemm_nt_masked(
            (a_fp8, a_scale),
            (self._w13, self._s13),
            gate_up,
            counts,
            expected_m,
            disable_ue8m0_cast=True,
        )
        h_fp8, h_scale = silu_mul_fp8_quant_f32_scale(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
            masked_m=counts,
        )
        down = torch.empty((E, M, D), dtype=torch.bfloat16, device=a_fp8.device)
        m_grouped_fp8_gemm_nt_masked(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down,
            counts,
            expected_m,
            disable_ue8m0_cast=True,
        )
        return down
