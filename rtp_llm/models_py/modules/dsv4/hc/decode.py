"""HC preparation and boundary fusion for DSv4 decode."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4._profiler import hc_record_function_range
from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.mhc_prenorm_backend import MHC_FMA_MAX_TOKENS


@dataclass(frozen=True)
class HCStage:
    hc: HCUnitBase
    norm: nn.Module


class PreparedHCInput(NamedTuple):
    normalized_input: torch.Tensor
    post_mix: torch.Tensor
    comb_mix: torch.Tensor


def _profile_prefix(hc: HCUnitBase) -> str:
    layer = f"L{hc.layer_id:02d}" if hc.layer_id >= 0 else "Lxx"
    return f"dsv4.hc.{layer}.{hc.name or 'unit'}"


def _can_fuse_norm(stage: HCStage, residual: torch.Tensor) -> bool:
    from .mhc_tilelang import _can_use_tk
    from .tilelang_impl import TileLangHCUnit

    return (
        isinstance(stage.hc, TileLangHCUnit)
        and type(stage.norm) is RMSNorm
        and stage.norm.weight.dtype == torch.bfloat16
        and stage.norm.weight.is_contiguous()
        and _can_use_tk(residual, stage.hc.hc_mult)
    )


def prepare_input(
    stage: HCStage, residual: torch.Tensor, dbg_tag: str | None = None
) -> PreparedHCInput:
    from .mhc_tilelang import tk_mhc_pre

    hc, norm = stage.hc, stage.norm
    if dbg_tag is None and _can_fuse_norm(stage, residual):
        with hc_record_function_range(f"{_profile_prefix(hc)}.pre_norm"):
            result = tk_mhc_pre(
                residual,
                hc.fn,
                hc.scale,
                hc.base,
                norm_eps=hc.norm_eps,
                pre_eps=hc.hc_eps,
                sinkhorn_eps=hc.hc_eps,
                sinkhorn_iters=hc.hc_sinkhorn_iters,
                hc_mult=hc.hc_mult,
                output_norm_weight=norm.weight.data,
                output_norm_eps=norm.variance_epsilon,
            )
            assert result is not None
            return PreparedHCInput(*result)
    x, post, comb = hc.pre(residual, dbg_tag=dbg_tag)
    normalized = norm(x.reshape(-1, x.shape[-1])).view(x.shape)
    return PreparedHCInput(normalized, post, comb)


def try_post_prepare(
    source: HCStage,
    target: HCStage,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, PreparedHCInput] | None:
    from .mhc_tilelang import tk_mhc_fused_post_pre
    from .tilelang_impl import TileLangHCUnit

    if not isinstance(source.hc, TileLangHCUnit) or not _can_fuse_norm(
        target, residual
    ):
        return None
    num_tokens = residual.numel() // (target.hc.hc_mult * target.hc.dim)
    if num_tokens > MHC_FMA_MAX_TOKENS:
        return None
    name = (
        f"{_profile_prefix(source.hc)}.post_pre_norm->"
        f"{_profile_prefix(target.hc).removeprefix('dsv4.hc.')}"
    )
    with hc_record_function_range(name):
        result = tk_mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            target.hc,
            output_norm_weight=target.norm.weight.data,
            output_norm_eps=target.norm.variance_epsilon,
        )
    updated_residual, prepared = result
    return updated_residual, PreparedHCInput(*prepared)
