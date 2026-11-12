"""SM90 MegaMoE with native SE or standalone wider shared experts."""

import torch

from rtp_llm.utils.model_weight import W

from .mega_moe_sm90 import MegaMoeSm90Executor


class MegaMoeSm90SEExecutor(MegaMoeSm90Executor):
    includes_shared_expert = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        if config.n_shared_experts == 1:
            super().check_conditions(checker, config)
        else:
            MegaMoeSm90Executor.check_conditions(checker, config)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)

    def setup_weights(self, weights) -> None:
        # Native SE uses the routed expert width. Wider shared weights remain
        # in the layer for its existing standalone shared executor.
        self.includes_shared_expert = self.cfg.n_shared_experts == 1
        super().setup_weights(weights)

    def _prepare_shared_weights(self, weights):
        if not self.includes_shared_expert:
            return {}
        cfg = self.cfg
        result = {}
        for name, wk, sk, n, k in (
            ("shared_l1_weights", W.ffn_w13, W.ffn_s13, 2 * cfg.moe_inter_dim, cfg.dim),
            ("shared_l2_weights", W.ffn_w2, W.ffn_s2, cfg.dim, cfg.moe_inter_dim),
        ):
            weight, scale = weights[wk], weights[sk]
            if weight.dtype != torch.float8_e4m3fn or weight.shape != (n, k):
                raise ValueError(f"{name} must be FP8 E4M3 with shape {(n, k)}")
            if scale.shape != (n // 128, k // 128) or scale.dtype not in (
                torch.float8_e8m0fnu,
                torch.float32,
            ):
                raise ValueError(f"{name} requires block-128 FP32 or UE8M0 scales")
            if (
                weight.device != weights[W.moe_w1].device
                or scale.device != weight.device
            ):
                raise ValueError(f"{name} must share the routed weights' device")
            result[name] = (weight.contiguous(), scale.float().contiguous())
        return result
