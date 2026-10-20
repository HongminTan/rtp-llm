import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.deepep_wrapper import DeepepWrapperConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
    DeepepNormalRouterBase,
    DeepepNormalRouterFp8Fp4,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
    CudaGroupedFp4Strategy,
    CudaLocalLoopStrategy,
    CudaMegaMoeSEStrategy,
    CudaMegaMoeStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.ops import CPRotateMethod, MoeConfig, ParallelismConfig


def _config(
    ep_size=1,
    n_shared_experts=0,
    strategy="auto",
    world_size=None,
    has_shared_expert_gate=False,
):
    model_config = ModelConfig()
    model_config.hidden_size = 512
    model_config.moe_inter_size = 256
    model_config.expert_num = 16
    model_config.moe_k = 4
    model_config.n_shared_experts = n_shared_experts
    model_config.swiglu_limit = 10.0
    model_config.routed_scaling_factor = 1.0
    model_config.moe_w1_layout = "up_gate"
    parallelism = ParallelismConfig()
    parallelism.ep_size = ep_size
    parallelism.ep_rank = 0
    parallelism.world_size = world_size if world_size is not None else ep_size
    parallelism.world_rank = 0
    parallelism.dp_size = ep_size
    parallelism.dp_rank = 0
    moe_config = MoeConfig()
    moe_config.moe_strategy = strategy
    return MoEConfigAdapter(model_config, parallelism, moe_config).for_layer(
        layer_id=0,
        moe_inter_dim=256,
        moe_w1_layout="up_gate",
        max_tokens_per_rank=128,
        moe_quant_method="FP8_FP4",
        has_shared_expert_gate=has_shared_expert_gate,
    )


def _registry():
    registry = StrategyRegistry()
    registry.register(CudaMegaMoeSEStrategy())
    registry.register(CudaMegaMoeStrategy())
    registry.register(CudaGroupedFp4Strategy())
    registry.register(CudaLocalLoopStrategy())
    return registry


class Fp8Fp4StrategySelectionTest(unittest.TestCase):
    def test_ll_capacity_follows_input_tp_view(self):
        for cp, expected in ((False, 64), (True, 128)):
            with self.subTest(cp=cp):
                source = _config(ep_size=4)
                parallelism = source.parallelism_config
                parallelism.tp_size = 4
                parallelism.tp_rank = 2
                parallelism.prefill_cp_config.method = (
                    CPRotateMethod.ALL_GATHER if cp else CPRotateMethod.DISABLED
                )
                source.moe_config.use_deepep_low_latency = True
                source.moe_config.ll_num_max_token = 128
                config = MoEConfigAdapter(
                    source.model_config, parallelism, source.moe_config
                )
                wrapper_config = DeepepWrapperConfig.from_config_adapter(config)
                self.assertEqual(wrapper_config.tp_size, 4)
                self.assertEqual(wrapper_config.ll_num_max_token_per_rank, expected)
                self.assertEqual(config.tp_size, 1 if cp else 4)

    def test_fp8_fp4_normal_router_contract(self):
        config = _config(ep_size=4)
        valid = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
        )
        with patch.object(
            DeepepNormalRouterBase, "__init__", return_value=None
        ) as init:
            DeepepNormalRouterFp8Fp4(config, valid)
            init.assert_called_once_with(config, valid, expert_alignment=1)

        invalid = (
            FusedMoEQuantConfig(),
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, per_act_token_quant=True
            ),
            FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, block_shape=[128, 64]),
        )
        for quant_config in invalid:
            with self.subTest(quant_config=quant_config):
                with self.assertRaisesRegex(ValueError, "requires FP8"):
                    DeepepNormalRouterFp8Fp4(config, quant_config)

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4._has_fp8_fp4_grouped_kernel",
        return_value=True,
    )
    def test_auto_prefers_grouped_on_single_rank(self, _):
        self.assertIsInstance(
            _registry().get_strategy(_config()), CudaGroupedFp4Strategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4._has_fp8_fp4_grouped_kernel",
        return_value=False,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop._has_local_loop_kernel",
        return_value=True,
    )
    def test_auto_falls_back_to_local_loop(self, _, __):
        self.assertIsInstance(
            _registry().get_strategy(_config()), CudaLocalLoopStrategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    def test_auto_uses_mega_without_shared_experts(self, _):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2)), CudaMegaMoeStrategy
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_auto_prefers_mega_se_for_multiple_shared_experts(self, _, __):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2, n_shared_experts=3)),
            CudaMegaMoeSEStrategy,
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=False,
    )
    def test_auto_falls_back_to_mega_when_se_is_unavailable(self, _, __):
        self.assertIsInstance(
            _registry().get_strategy(_config(ep_size=2, n_shared_experts=2)),
            CudaMegaMoeStrategy,
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_auto_keeps_gated_shared_expert_outside_mega_se(self, _, __):
        config = _config(
            ep_size=2,
            n_shared_experts=2,
            has_shared_expert_gate=True,
        )
        self.assertIsInstance(
            _registry().get_strategy(config),
            CudaMegaMoeStrategy,
        )

    def test_grouped_capability_requires_both_deep_gemm_symbols(self):
        available = SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=object(),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

        for missing in (
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "get_mk_alignment_for_contiguous_layout",
        ):
            with self.subTest(missing=missing):
                unsupported = SimpleNamespace(
                    **{
                        name: getattr(available, name)
                        for name in vars(available)
                        if name != missing
                    }
                )
                with mock.patch.dict(
                    sys.modules, {"deep_gemm": unsupported}
                ), mock.patch.object(
                    torch.cuda, "is_available", return_value=True
                ), mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 0)
                ):
                    self.assertFalse(_has_fp8_fp4_grouped_kernel())

    def test_grouped_capability_requires_cuda_sm100(self):
        available = SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=object(),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())
        with mock.patch.dict(sys.modules, {"deep_gemm": available}), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=False,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=False,
    )
    def test_explicit_mega_se_rejects_unavailable_backend(self, _, __):
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe_se'"):
            _registry().get_strategy(
                _config(ep_size=2, n_shared_experts=2, strategy="mega_moe_se")
            )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=False,
    )
    def test_explicit_mega_se_uses_routed_and_standalone_shared(self, _, __):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            MegaMoeExecutor,
        )

        strategy = _registry().get_strategy(
            _config(ep_size=2, n_shared_experts=2, strategy="mega_moe_se")
        )
        self.assertIsInstance(strategy, CudaMegaMoeSEStrategy)
        executor = strategy.get_attributes().get_executor_class()
        self.assertIs(executor, MegaMoeExecutor)
        # This flag makes the layer retain FP8 shared weights and construct
        # the existing shared-expert executor plus FP32 combine epilogue.
        self.assertFalse(executor.includes_shared_expert)

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe._mega_moe_available",
        return_value=True,
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se._mega_moe_se_available",
        return_value=True,
    )
    def test_mega_strategies_reject_ep_group_different_from_world(self, _, __):
        config = _config(
            ep_size=2, n_shared_experts=2, strategy="mega_moe_se", world_size=4
        )
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe_se'"):
            _registry().get_strategy(config)
        config.moe_strategy = "mega_moe"
        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='mega_moe'"):
            _registry().get_strategy(config)


if __name__ == "__main__":
    unittest.main()
