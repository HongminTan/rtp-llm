"""Hopper MegaMoE with native MXFP4 routed weights and FP8 activations."""

import inspect
import logging
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_sm90_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.sm90_buffer import (
    get_sm90_buffer,
    get_sm90_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)
from rtp_llm.models_py.utils.arch import is_sm90
from rtp_llm.utils.model_weight import W

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

_MEGA_MOE_SM90_JIT_WARMED_KEYS: set[tuple] = set()


def sm90_mega_available(*, shared: bool = False) -> bool:
    if not torch.cuda.is_available() or not is_sm90():
        return False
    try:
        import deep_gemm
        import torch.distributed as dist
        from torch.distributed import _symmetric_memory  # noqa: F401

        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return False
        for name in (
            "get_symm_buffer_for_mega_moe_sm90",
            "transform_weights_for_mega_moe_sm90",
            "sm90_fp8_fp4_mega_moe",
        ):
            if not callable(getattr(deep_gemm, name, None)):
                return False
        if shared:
            parameters = inspect.signature(deep_gemm.sm90_fp8_fp4_mega_moe).parameters
            if not {"shared_l1_weights", "shared_l2_weights"} <= parameters.keys():
                return False
        return True
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return False


class MegaMoeSm90Executor(Fp8Fp4ExecutorBase):
    execute_empty_inputs = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        super().check_conditions(checker, config)
        checker.check(1 < config.ep_size <= 32)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(config.max_tokens_per_rank > 0)
        checker.check(0 < config.dim <= 8192 and config.dim % 512 == 0)
        checker.check(
            0 < config.moe_inter_dim <= 4096 and config.moe_inter_dim % 256 == 0
        )
        checker.check(
            config.n_routed_experts > 0
            and config.n_routed_experts % config.ep_size == 0
        )
        checker.check(
            1 <= config.n_activated_experts <= min(32, config.n_routed_experts)
        )
        # Decode combine needs at least one 512-byte BF16 slice per route.
        checker.check(config.n_activated_experts * 256 <= config.dim)
        checker.check(sm90_mega_available(shared=cls.includes_shared_expert))

    def setup_weights(self, weights) -> None:
        import deep_gemm

        cfg = self.cfg
        w13, s13 = normalize_moe_w13_gate_up(
            weights[W.moe_w1], weights[W.moe_s1], cfg.moe_inter_dim, cfg.moe_w1_layout
        )
        w2, s2 = weights[W.moe_w2], weights[W.moe_s2]
        local_experts = cfg.n_routed_experts // cfg.ep_size
        if w13.shape[0] != local_experts or w2.shape[0] != local_experts:
            raise ValueError("SM90 MegaMoE requires EP-local expert weight stacks")
        self._shared_weights = self._prepare_shared_weights(weights)
        # The SM90 transform consumes natural K32 UE8M0 scales, not SM100's
        # packed scale layout. It may requantize weights with wide scale ranges.
        self._l1, self._l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (w13.contiguous(), s13.contiguous()), (w2.contiguous(), s2.contiguous())
        )
        self._buffer = get_sm90_buffer(
            cfg, w13.device, shared=self.includes_shared_expert
        )
        self._output = get_sm90_output(
            self._buffer.num_max_tokens_per_rank,
            cfg.dim,
            torch.bfloat16,
            w13.device,
            shared=self.includes_shared_expert,
        )
        for key in (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2):
            weights.pop(key)
        if self.includes_shared_expert:
            for key in (W.ffn_w13, W.ffn_s13, W.ffn_w2, W.ffn_s2):
                weights.pop(key)
        self._maybe_warmup_jit_once()

    def _prepare_shared_weights(self, weights):
        return {}

    def _run(self, x, weights, indices, num_tokens_bound: Optional[int]):
        import deep_gemm

        tokens = x.size(0)
        buf = self._buffer
        if tokens > buf.num_max_tokens_per_rank:
            raise ValueError(
                f"SM90 MegaMoE tokens={tokens} exceed capacity={buf.num_max_tokens_per_rank}"
            )
        if tokens:
            quantized, scales = sgl_per_token_group_quant_fp8(
                x.contiguous(),
                group_size=128,
                eps=1e-4,
                column_major_scales=False,
                scale_tma_aligned=False,
                scale_ue8m0=False,
            )
            buf.x[:tokens].copy_(quantized)
            buf.x_sf[:tokens].copy_(scales)
            buf.topk_idx[:tokens].copy_(indices)
            buf.topk_weights[:tokens].copy_(weights)
        output = self._output[:tokens]
        sync_cuda_graph_warmup_ranks(
            f"moe.mega_moe.layer{self.cfg.layer_id}.before_deepgemm", x.device
        )
        deep_gemm.sm90_fp8_fp4_mega_moe(
            output,
            self._l1,
            self._l2,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            num_tokens_bound=num_tokens_bound,
            **self._shared_weights,
        )
        return output

    def forward(self, x, weights, indices):
        # Eager DP ranks can have different token counts. Only graph capture
        # has a common padded token bound; empty ranks still enter the kernel.
        bound = x.size(0) if torch.cuda.is_current_stream_capturing() else None
        return self._run(x, weights, indices, bound)

    def _resolve_jit_warmup_token_counts(self) -> list[int]:
        """Limit warmup inputs to the logical cap, not the aligned buffer size."""
        max_tokens = max(int(self.cfg.max_tokens_per_rank), 1)
        override = parse_mega_moe_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens)
        return generate_mega_moe_sm90_jit_token_counts(
            num_ranks=self.cfg.ep_size,
            num_experts=self.cfg.n_routed_experts,
            num_topk=self.cfg.n_activated_experts,
            max_tokens_per_rank=max_tokens,
            include_cap=self.cfg.warmup_include_capacity,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM90 MegaMoE JIT warmup must not run inside CUDA graph capture"
            )

        import torch.distributed as dist

        from .mega_moe import _activate_mega_moe_rank_nvcc_tmpdir, _restore_tmpdir

        cfg = self.cfg
        token_counts = self._resolve_jit_warmup_token_counts()
        if not token_counts:
            return
        warmup_key = (
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            int(cfg.max_tokens_per_rank),
            cfg.swiglu_limit,
            self.includes_shared_expert,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_SM90_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[MegaMoE-SM90] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d shared=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                int(cfg.max_tokens_per_rank),
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                int(self.includes_shared_expert),
            )
        tmpdir, previous_tmpdir = _activate_mega_moe_rank_nvcc_tmpdir(rank)
        try:
            if rank == 0:
                logging.info("[MegaMoE-SM90] rank-local nvcc TMPDIR=%s", tmpdir)
            self.warmup_jit(token_counts)
        finally:
            _restore_tmpdir(previous_tmpdir)
        _MEGA_MOE_SM90_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[MegaMoE-SM90] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        """Warm eager and graph-bound variants of the supplied token counts.

        Unbounded calls use buffer capacity for wave/lag scheduling; graph
        bounds may select a different kernel. All ranks must use identical
        configuration and token overrides to keep collective launches aligned.
        """
        if not mega_moe_jit_warmup_enabled():
            return
        import torch.distributed as dist

        cfg = self.cfg
        device = self._output.device
        max_tokens = max(token_counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        weights = torch.zeros(
            (max_tokens, cfg.n_activated_experts), dtype=torch.float32, device=device
        )
        num_local_experts = max(cfg.n_local_experts, 1)
        local_expert_ids = (
            cfg.local_expert_start
            + torch.arange(cfg.n_activated_experts, dtype=torch.long, device=device)
            % num_local_experts
        )
        indices = local_expert_ids.view(1, -1).expand(max_tokens, -1).contiguous()

        for token_count in token_counts:
            for bound in (None, token_count):
                dist.barrier()
                self._run(
                    x[:token_count],
                    weights[:token_count],
                    indices[:token_count],
                    bound,
                )
                torch.cuda.synchronize(device)
        dist.barrier()
