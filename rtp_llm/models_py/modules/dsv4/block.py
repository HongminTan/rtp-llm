"""DeepSeek-V4 Transformer Block: mHC + Attention + MoE.

Mirrors `inference/model.py:Block`. Each call applies hc_pre/F/hc_post
twice — once for Attention and once for MoE FFN.
"""

import logging
import os
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.chunk_env import chunked_moe_enabled
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    CommitOnlyAttentionFP8,
)
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.dsv4.hc.decode import (
    HCStage,
    PreparedHCInput,
    prepare_input,
    try_post_prepare,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.utils.model_weight import W

_PrefillFastHCImpls = Tuple[Callable, Callable, Callable, Callable]

_MOE_WEIGHT_NAMES = {
    "routed_gate": W.v4_routed_w1_w,
    "routed_gate_scale": W.v4_routed_w1_s,
    "routed_up": W.v4_routed_w3_w,
    "routed_up_scale": W.v4_routed_w3_s,
    "routed_down": W.v4_routed_w2_w,
    "routed_down_scale": W.v4_routed_w2_s,
    "router": W.v4_router_w,
    "router_bias": W.v4_router_bias,
    "router_tid2eid": W.v4_router_tid2eid,
    "shared_gate_up": W.v4_shared_w13_w,
    "shared_gate_up_scale": W.v4_shared_w13_s,
    "shared_down": W.v4_shared_w2_w,
    "shared_down_scale": W.v4_shared_w2_s,
}


def _prefill_fast_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(norm, RMSNorm)
        and norm.__class__.__module__ == "rtp_llm.models_py.modules.base.cuda.norm"
    ):
        return norm(x, output=x)
    return norm(x)


def forward_decode_layer(
    layer,
    next_layer,
    hidden,
    attn_metadata,
    input_ids,
    kv_cache,
    *,
    prepared_input: PreparedHCInput | None = None,
) -> tuple[torch.Tensor, PreparedHCInput | None]:
    from rtp_llm.models_py.modules.dsv4 import _record_tensor

    next_stage = None
    if (
        next_layer is not None
        and next_layer._mega_csa_adapter is None
        and next_layer._mega_hca_adapter is None
        and not _record_tensor.should_record_layer(next_layer.layer_id)
    ):
        next_stage = HCStage(next_layer.attn_hc, next_layer.attn_norm)
    return layer.forward_decode_with_hc(
        hidden,
        attn_metadata,
        input_ids,
        kv_cache=kv_cache,
        prepared_input=prepared_input,
        next_stage=next_stage,
        fuse_boundaries=True,
    )


class Block(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        moe_config: MoEConfigAdapter,
        moe_inter_dim: int,
        score_func: str,
        n_hash_layers: int,
        vocab_size: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        max_tokens_per_rank: int = 8192,
        is_decode_role: bool = False,
        fp8_kv_cache: bool = False,
        commit_only: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.fp8_kv_cache = fp8_kv_cache

        attn_cls = CommitOnlyAttentionFP8 if commit_only else AttentionFP8
        self.attn = attn_cls(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            window_size=window_size,
            compress_ratio=compress_ratio,
            compress_rope_theta=compress_rope_theta,
            rope_theta=rope_theta,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_seq_len=original_seq_len,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            norm_eps=norm_eps,
            layer_weights=layer_weights,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self._cp_sync_after_attn_done = False
        if commit_only:
            self.ffn = None
            self.attn_norm = None
            self.ffn_norm = None
            self.attn_hc = None
            self.ffn_hc = None
            self._prefill_fast_hc_impls_cached = None
            return

        if layer_weights is None:
            raise ValueError("Block requires per-layer weights")
        adapt_split_moe_weights(
            layer_weights, moe_inter_dim, moe_config.n_shared_experts, _MOE_WEIGHT_NAMES
        )
        chunking_enabled = chunked_moe_enabled()
        layer_config = moe_config.for_layer(
            layer_id=layer_id,
            moe_inter_dim=moe_inter_dim,
            moe_w1_layout="gate_up",
            max_tokens_per_rank=max_tokens_per_rank,
            moe_quant_method="FP8_FP4",
            warmup_include_capacity=chunking_enabled and not is_decode_role,
        )
        self.ffn = ChunkedFp8Fp4MoeLayer(
            config=layer_config,
            score_func=score_func,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            layer_weights=layer_weights,
            is_decode_role=is_decode_role,
            chunking_enabled=chunking_enabled,
            observer_factory=self._moe_observer,
            record_function_scope=_profiler.moe_record_function_scope,
        )
        # Framework loader already casts norms to bf16 (compute_dtype) and
        # hc_* tensors to fp32 (descriptor data_type); pass refs straight
        # into ``RMSNorm`` at construction time.  Norms here see 2D inputs
        # ``[T, dim]`` from the hc_pre output, so framework ``RMSNorm``
        # (which expects 2D) drops in directly.
        self.attn_norm = RMSNorm(layer_weights[W.v4_attn_norm], norm_eps)
        self.ffn_norm = RMSNorm(layer_weights[W.v4_ffn_norm], norm_eps)

        self.attn_hc = build_hc_unit(
            layer_weights[W.v4_hc_attn_fn],
            layer_weights[W.v4_hc_attn_base],
            layer_weights[W.v4_hc_attn_scale],
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name="attn",
        )
        self.ffn_hc = build_hc_unit(
            layer_weights[W.v4_hc_ffn_fn],
            layer_weights[W.v4_hc_ffn_base],
            layer_weights[W.v4_hc_ffn_scale],
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name="ffn",
        )
        self._prefill_fast_hc_impls_cached = self._resolve_prefill_fast_hc_impls()
        self._mega_csa_adapter = None
        self._mega_hca_adapter = None

    def enable_mega_csa(self, runtime, layer_weights: Dict[str, torch.Tensor]) -> None:
        """Attach the TP1 CSA adapter only to compress-ratio-4 layers."""
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
            COMPRESS_RATIO,
        )

        if int(self.attn.compress_ratio) != COMPRESS_RATIO:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import (
            MegaCSAAdapter,
        )

        # Packed Mega weights add GPU memory; keep the original weights for
        # prefill and decode shapes that use the ordinary attention path.
        self._mega_csa_adapter = MegaCSAAdapter(self, layer_weights, runtime)

    def _moe_observer(self, positions: Optional[torch.Tensor]):
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        if not _rt.should_record_layer(self.layer_id):
            return None
        names = {
            "input": "moe_x_in",
            "topk_weights": "moe_topk_weights",
            "topk_indices": "moe_topk_indices",
            "routed_y": "moe_routed_y",
            "shared_y": "moe_shared_y",
            "final_y": "moe_y",
        }
        global_pos = int(_rt._DBG_GLOBAL_POS)

        def observe(kind: str, tensor: torch.Tensor) -> None:
            suffix = names.get(kind)
            if suffix is None:
                return
            name = f"L{self.layer_id:02d}_{suffix}"
            _rt.record_if_level(2, name, tensor)
            if (
                global_pos < 0
                or positions is None
                or positions.numel() != tensor.size(0)
            ):
                return
            mask = positions.to(device=tensor.device, dtype=torch.long).reshape(-1)
            mask = mask == global_pos
            _rt.record_if_level(2, f"{name}_pos{global_pos}", tensor[mask].contiguous())

        return observe

    def enable_mega_hca(self, runtime, layer_weights: Dict[str, torch.Tensor]) -> None:
        """Attach the TP1 HCA adapter only to compress-ratio-128 layers."""
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
            HCA_COMPRESS_RATIO,
        )

        if int(self.attn.compress_ratio) != HCA_COMPRESS_RATIO:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import (
            MegaHCAAdapter,
        )

        self._mega_hca_adapter = MegaHCAAdapter(self, layer_weights, runtime)

    def _sync_after_first_cp_prefill_attention(self) -> None:
        if self._cp_sync_after_attn_done:
            return
        if os.environ.get("DSV4_CP_SYNC_AFTER_ATTN_ONCE", "1") == "0":
            return
        if self.ffn.strategy_name not in (
            "mega_moe",
            "mega_moe_se",
        ):
            return
        if getattr(self.attn, "_cp_ctx", None) is None:
            return

        import torch.distributed as dist

        if not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        ):
            return

        torch.cuda.synchronize()
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        except TypeError:
            dist.barrier()
        self._cp_sync_after_attn_done = True
        logging.info(
            "[DeepSeekV4Block] CP first-prefill sync after attention done layer=%d",
            self.layer_id,
        )

    def _resolve_prefill_fast_hc_impls(self) -> _PrefillFastHCImpls:
        attn_hc_pre = getattr(self.attn_hc, "_pre_impl_prefill_fast", None)
        if attn_hc_pre is None:
            attn_hc_pre = self.attn_hc._pre_impl
        ffn_hc_pre = getattr(self.ffn_hc, "_pre_impl_prefill_fast", None)
        if ffn_hc_pre is None:
            ffn_hc_pre = self.ffn_hc._pre_impl
        attn_hc_post = getattr(self.attn_hc, "_post_impl_prefill_fast", None)
        if attn_hc_post is None:
            attn_hc_post = self.attn_hc._post_impl
        ffn_hc_post = getattr(self.ffn_hc, "_post_impl_prefill_fast", None)
        if ffn_hc_post is None:
            ffn_hc_post = self.ffn_hc._post_impl
        return attn_hc_pre, ffn_hc_pre, attn_hc_post, ffn_hc_post

    def _prefill_fast_hc_impls(self) -> _PrefillFastHCImpls:
        impls = getattr(self, "_prefill_fast_hc_impls_cached", None)
        if impls is None:
            impls = self._resolve_prefill_fast_hc_impls()
            self._prefill_fast_hc_impls_cached = impls
        return impls

    def prefill_fast_callable(self):
        if not isinstance(self.attn, AttentionFP8):
            return None
        # ``prefill.forward_layers`` caches this private callable only after the
        # layer has passed the production FP8 support matrix. Keep the public
        # ``forward_prefill_fast`` wrapper defensive for tests/manual calls.
        return self._forward_prefill_fast_fp8

    def prefill_fast_attn_pre(self, x: torch.Tensor):
        attn_hc_pre, _, _, _ = self._prefill_fast_hc_impls()
        residual = x
        x_pre, post, comb = attn_hc_pre(x)
        x_pre = _prefill_fast_norm(self.attn_norm, x_pre)
        return residual, x_pre, post, comb

    def prefill_fast_attn_body(
        self,
        x_pre: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_cache=None,
        block_tables_by_type=None,
    ) -> torch.Tensor:
        return self.attn(
            x_pre,
            positions,
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
        )

    def prefill_fast_attn_post(
        self,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        _, _, attn_hc_post, _ = self._prefill_fast_hc_impls()
        x = attn_hc_post(attn_out, residual, post, comb)
        self._sync_after_first_cp_prefill_attention()
        return x

    def prefill_fast_ffn_pre(self, x: torch.Tensor):
        _, ffn_hc_pre, _, _ = self._prefill_fast_hc_impls()
        residual = x
        x_pre, post, comb = ffn_hc_pre(x)
        x_pre = _prefill_fast_norm(self.ffn_norm, x_pre)
        return residual, x_pre, post, comb

    def prefill_fast_ffn_body(
        self, x_pre: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.ffn(x_pre, input_ids)

    def prefill_fast_ffn_post(
        self,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, ffn_hc_post = self._prefill_fast_hc_impls()
        return ffn_hc_post(ffn_out, residual, post, comb)

    def forward_decode(
        self,
        x: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
        input_ids: torch.Tensor,
        kv_cache=None,
        attn_fn=None,
    ) -> torch.Tensor:
        """Decode a block, optionally replacing its attention call with attn_fn."""
        output, _ = self.forward_decode_with_hc(
            x, attn_metadata, input_ids, kv_cache=kv_cache, attn_fn=attn_fn
        )
        return output

    def forward_decode_with_hc(
        self,
        x: torch.Tensor,  # [B, 1, hc, dim]
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
        input_ids: torch.Tensor,  # [B, 1]
        kv_cache=None,
        attn_fn=None,
        *,
        prepared_input: PreparedHCInput | None = None,
        next_stage: HCStage | None = None,
        fuse_boundaries: bool = False,
    ) -> tuple[torch.Tensor, PreparedHCInput | None]:
        """Return the residual and optional normalized HC inputs for the next block."""
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        _dbg_layer = _rt.should_record_layer(self.layer_id)
        fuse_boundaries = fuse_boundaries and not _dbg_layer and attn_fn is None
        attn_stage = HCStage(self.attn_hc, self.attn_norm)
        ffn_stage = HCStage(self.ffn_hc, self.ffn_norm)
        ffn_pre = None
        mega_adapter = self._mega_csa_adapter or self._mega_hca_adapter
        use_mega_attention = (
            attn_fn is None
            and mega_adapter is not None
            and mega_adapter.supports_decode_shape(x, attn_metadata)
        )
        if use_mega_attention:
            # Fused HC/norm intermediates are not exposed by Mega. The shared
            # residual record below remains available; DSV4_MEGA=0 restores
            # the ordinary attention intermediate records for debugging.
            x = mega_adapter.forward_attention_sublayer(
                self, x, attn_metadata, kv_cache=kv_cache
            )
        else:
            # Attention path
            residual = x
            if prepared_input is None:
                prepared_input = prepare_input(
                    attn_stage,
                    x,
                    dbg_tag=(
                        f"L{self.layer_id:02d}_decode_attn_hc_pre"
                        if _dbg_layer
                        else None
                    ),
                )
            x_pre = prepared_input.normalized_input
            post, comb = prepared_input.post_mix, prepared_input.comb_mix
            if _dbg_layer:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_in", x_pre)
            if attn_fn is not None:
                attn_out = attn_fn(x_pre)
            else:
                attn_out = self.attn.forward_decode(
                    x_pre, attn_metadata, kv_cache=kv_cache
                )
            if _dbg_layer:
                _rt.record_if_level(
                    2, f"L{self.layer_id:02d}_decode_attn_out", attn_out
                )
            boundary = (
                try_post_prepare(
                    attn_stage,
                    ffn_stage,
                    attn_out,
                    residual,
                    post,
                    comb,
                )
                if fuse_boundaries
                else None
            )
            if boundary is None:
                x = self.attn_hc.post(attn_out, residual, post, comb)
            else:
                x, ffn_pre = boundary
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_residual", x)

        # FFN path — MoE has no per-step state, reuse existing forward
        residual = x
        if ffn_pre is None:
            ffn_pre = prepare_input(
                ffn_stage,
                x,
                dbg_tag=(
                    f"L{self.layer_id:02d}_decode_ffn_hc_pre" if _dbg_layer else None
                ),
            )
        x_pre = ffn_pre.normalized_input
        post, comb = ffn_pre.post_mix, ffn_pre.comb_mix
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_in", x_pre)
        ffn_out = self.ffn(x_pre, input_ids, is_decode_forward=True)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_out", ffn_out)
        if fuse_boundaries and next_stage is not None:
            boundary = try_post_prepare(
                ffn_stage, next_stage, ffn_out, residual, post, comb
            )
            if boundary is not None:
                return boundary
        return self.ffn_hc.post(ffn_out, residual, post, comb), None

    def forward_prefill_fast(
        self,
        x: torch.Tensor,  # [T, hc, dim]
        input_ids: Optional[torch.Tensor],  # [T]
        positions: torch.Tensor,  # [T] int64
        cu_seqlens: torch.Tensor,  # [B+1] int64
        kv_cache=None,
        block_tables_by_type=None,
    ) -> torch.Tensor:
        """Prefill fast path for the FP8 production path.

        This preserves the normal prefill operator order, but calls the HC
        implementations directly and skips the debug/shape-check wrapper work
        in :meth:`forward`.  The generic attention fallback is deliberately
        kept on the default path because it has extra layout handling that the
        SM103 DSV4-Pro FP8 path does not need.
        """
        fast_call = self.prefill_fast_callable()
        if fast_call is None:
            return self.forward(
                x,
                input_ids,
                positions,
                cu_seqlens,
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )
        return fast_call(
            x,
            input_ids,
            positions,
            cu_seqlens,
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
        )

    def _forward_prefill_fast_fp8(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kv_cache=None,
        block_tables_by_type=None,
    ) -> torch.Tensor:
        """Validated FP8 prefill fast body.

        This private method is cached by ``prefill.forward_layers`` after the
        layer is proven to be ``Block + AttentionFP8``. Keep it semantically
        paired with ``forward``: production runs may switch between the normal
        and fast paths via env/debug gates.
        """
        if input_ids is None:
            raise RuntimeError("DSV4 prefill fast path requires input_ids")

        (
            attn_hc_pre,
            ffn_hc_pre,
            attn_hc_post,
            ffn_hc_post,
        ) = self._prefill_fast_hc_impls()

        residual = x
        x_pre, post, comb = attn_hc_pre(x)
        if self.attn.can_fuse_prefill_attn_norm_input_quant(
            x_pre, self.attn_norm.weight.data
        ):
            x_pre, shared_input_quant = self.attn.prefill_fused_attn_norm_input_quant(
                x_pre,
                self.attn_norm.weight.data,
                self.attn_norm.variance_epsilon,
            )
            attn_out = self.attn.forward_with_shared_input_quant(
                x_pre,
                positions,
                shared_input_quant,
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )
        else:
            x_pre = _prefill_fast_norm(self.attn_norm, x_pre)
            attn_out = self.attn(
                x_pre,
                positions,
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )
        x = attn_hc_post(attn_out, residual, post, comb)
        self._sync_after_first_cp_prefill_attention()

        residual = x
        x_pre, post, comb = ffn_hc_pre(x)
        x_pre = _prefill_fast_norm(self.ffn_norm, x_pre)
        ffn_out = self.ffn(x_pre, input_ids)
        return ffn_hc_post(ffn_out, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,  # [T, hc, dim]
        input_ids: Optional[torch.Tensor],  # [T]
        positions: torch.Tensor,  # [T] int64
        cu_seqlens: torch.Tensor,  # [B+1] int64
        kv_cache=None,
        block_tables_by_type=None,
    ) -> torch.Tensor:
        """Flat per-block forward — accepts ``[T, hc, dim]`` hidden and 1D
        ``input_ids`` / ``positions`` / ``cu_seqlens``, matching the vLLM
        DSV4 layer layout.  HC pre/post are flat-native;
        ``self.ffn`` (MoE) reshapes to 2D internally so any shape flows.

        Attention is called with a padded ``[B, max_S, dim]`` tensor and
        per-row ``start_pos [B]`` / ``sequence_lengths [B]`` derived from
        ``cu_seqlens`` + ``positions``.  For B==1 the scatter/gather
        collapses to an identity copy and Attention's internal scalar
        path kicks in (start_pos tensor with ``numel()==1`` falls through
        to ``int(start_pos)``), so the single-request case is bit-equal
        to the pre-batched behavior.
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        # By default keep the narrow trace from _record_tensor; use
        # MOEDBG_LAYER/MOEDBG_ALL_LAYERS for wider bisection.
        _dbg_layer = _rt.should_record_layer(self.layer_id)
        dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
        dbg_pos_mask = None
        dbg_pos_name = None
        dbg_positions = positions
        cp_ctx = getattr(self.attn, "_cp_ctx", None)
        if cp_ctx is not None and getattr(cp_ctx, "global_positions", None) is not None:
            cp_positions = cp_ctx.global_positions
            if cp_positions.numel() == positions.numel():
                dbg_positions = cp_positions
        if _dbg_layer and dbg_pos >= 0:
            dbg_pos_mask = dbg_positions.to(torch.long) == int(dbg_pos)
            dbg_pos_name = f"pos{dbg_pos}"
        # Attention path
        residual = x
        x_pre, post, comb = self.attn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_attn_hc_pre" if _dbg_layer else None,
        )  # [T, dim], [T, hc, 1], [T, hc, hc]
        x_pre = self.attn_norm(x_pre)  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_in", x_pre)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_in_{dbg_pos_name}",
                    x_pre[dbg_pos_mask].contiguous(),
                )

        # AttentionFP8 is flat-native: takes [T, dim] + per-token positions,
        # no padding / per-request scalars. The runtime kv-cache dtype flag can
        # be false while this branch still constructs AttentionFP8, so dispatch
        # on the module type instead of only the cache config.
        if isinstance(self.attn, AttentionFP8):
            attn_out = self.attn(
                x_pre,  # [T, dim]
                positions,  # [T] int64 absolute positions
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )  # [T, dim]
        else:
            # Present flat [T, dim] as [B, S, dim] for Attention.  The common
            # prefill layouts (single request, or dense equal-length batch) are
            # already request-major, so a view is enough.  Ragged batches keep
            # the padded scatter/gather fallback.
            cu_long = cu_seqlens.to(torch.long)
            batch_size = int(cu_long.numel() - 1)
            seqlens = cu_long[1:] - cu_long[:-1]  # [B]
            T = int(x_pre.size(0))
            D = int(x_pre.size(-1))
            device = x_pre.device
            max_S = (
                int(seqlens.max().item())
                if batch_size > 0 and seqlens.device.type == "cpu"
                else T
            )
            # Per-row start_pos = absolute position of each request's first
            # token, pulled directly off the flat positions tensor.
            start_pos_per_req = positions[cu_long[:-1]].to(torch.long)  # [B]

            equal_len_dense = (
                T == batch_size * max_S
                and seqlens.device.type == "cpu"
                and bool((seqlens == max_S).all().item())
            )
            dense_layout = batch_size == 1 or (batch_size > 1 and equal_len_dense)
            if dense_layout:
                x_padded = x_pre.view(batch_size, max_S, D)
                b_idx = None
                s_idx = None
            else:
                t_idx = torch.arange(T, device=device, dtype=torch.long)
                # right=True so tokens at cu_seqlens[b] land in request b
                # (cu_long[1:] are exclusive ends).
                b_idx = torch.searchsorted(cu_long[1:].contiguous(), t_idx, right=True)
                s_idx = t_idx - cu_long[:-1][b_idx]
                x_padded = torch.zeros(
                    batch_size, max_S, D, dtype=x_pre.dtype, device=device
                )
                x_padded[b_idx, s_idx] = x_pre

            attn_out_padded = self.attn(
                x_padded,  # [B, max_S, dim]
                start_pos_per_req,
                sequence_lengths=seqlens,
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )  # [B, max_S, dim]
            if dense_layout:
                attn_out = attn_out_padded.reshape(T, D)
            else:
                attn_out = attn_out_padded[b_idx, s_idx]  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_out", attn_out)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_out_{dbg_pos_name}",
                    attn_out[dbg_pos_mask].contiguous(),
                )
        x = self.attn_hc.post(attn_out, residual, post, comb)  # [T, hc, dim]
        self._sync_after_first_cp_prefill_attention()
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_residual", x)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_residual_{dbg_pos_name}",
                    x[dbg_pos_mask].contiguous(),
                )

        # FFN path
        residual = x
        x_pre, post, comb = self.ffn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_ffn_hc_pre" if _dbg_layer else None,
        )  # [T, dim], ...
        x_pre = self.ffn_norm(x_pre)  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_ffn_in", x_pre)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_ffn_in_{dbg_pos_name}",
                    x_pre[dbg_pos_mask].contiguous(),
                )
        ffn_in_ids = (
            input_ids
            if input_ids is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        if _dbg_layer and dbg_pos_mask is not None:
            ffn_out = self.ffn(x_pre, ffn_in_ids, positions=dbg_positions)
        else:
            ffn_out = self.ffn(x_pre, ffn_in_ids)  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_ffn_out", ffn_out)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_ffn_out_{dbg_pos_name}",
                    ffn_out[dbg_pos_mask].contiguous(),
                )
        x = self.ffn_hc.post(ffn_out, residual, post, comb)  # [T, hc, dim]
        if _dbg_layer and dbg_pos_mask is not None:
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_ffn_residual_{dbg_pos_name}",
                x[dbg_pos_mask].contiguous(),
            )
        return x  # [T, hc, dim]
