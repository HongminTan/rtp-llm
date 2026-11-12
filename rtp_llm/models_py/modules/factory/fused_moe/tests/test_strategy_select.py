import sys
import unittest
from contextlib import contextmanager
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
    dim=512,
):
    model_config = ModelConfig()
    model_config.hidden_size = dim
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


@contextmanager
def _backend(sm, *, routed=True, shared=True):
    prefix = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda."
    native_shared = shared
    with patch(prefix + "strategy.fp8_fp4.is_sm90", return_value=sm == 9), patch(
        prefix + "executors.mega_moe._mega_moe_available", return_value=routed
    ), patch(
        prefix + "executors.mega_moe_se._mega_moe_se_available", return_value=shared
    ), patch(
        prefix + "executors.mega_moe_sm90.sm90_mega_available",
        side_effect=lambda shared=False: routed and (native_shared if shared else True),
    ):
        yield


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


class MegaStrategySelectionTest(unittest.TestCase):
    def test_auto_selection(self):
        for sm in (9, 10):
            for count, gated, native in (
                (0, False, True),
                (1, False, True),
                (3, False, True),
                (1, True, True),
                (1, False, False),
            ):
                with self.subTest(
                    sm=sm, count=count, gated=gated, native=native
                ), _backend(sm, shared=native):
                    config = _config(
                        ep_size=2,
                        n_shared_experts=count,
                        has_shared_expert_gate=gated,
                        dim=1024,
                    )
                    strategy = _registry().get_strategy(config)
                    fused = (
                        native and count > 0 and not gated and (sm == 10 or count == 1)
                    )
                    self.assertIsInstance(
                        strategy,
                        CudaMegaMoeSEStrategy if fused else CudaMegaMoeStrategy,
                    )
                    executor = strategy.get_executor_class()
                    self.assertEqual(executor.includes_shared_expert, fused)
                    self.assertEqual("sm90" in executor.__module__, sm == 9)

    def test_explicit_selection_returns_backend_executor_class(self):
        for sm in (9, 10):
            for name in ("mega_moe", "mega_moe_se"):
                for count, gated, native in (
                    (0, False, True),
                    (1, False, True),
                    (3, False, True),
                    (1, True, True),
                    (1, False, False),
                ):
                    with self.subTest(
                        sm=sm, name=name, count=count, gated=gated, native=native
                    ), _backend(sm, shared=native):
                        config = _config(
                            ep_size=2,
                            n_shared_experts=count,
                            has_shared_expert_gate=gated,
                            strategy=name,
                            dim=1024,
                        )
                        rejected = name == "mega_moe_se" and (count == 0 or gated)
                        if rejected:
                            with self.assertRaises(ValueError):
                                _registry().get_strategy(config)
                        else:
                            strategy = _registry().get_strategy(config)
                            self.assertEqual(strategy.strategy_name, name)
                            # This checks class selection only. SM90 SE with
                            # wider shared experts changes its instance flag in
                            # setup_weights; execution tests cover that fallback.
                            executor_name = (
                                "MegaMoeSm90" if sm == 9 else "MegaMoe"
                            ) + (
                                "SEExecutor"
                                if name == "mega_moe_se" and native
                                else "Executor"
                            )
                            self.assertEqual(
                                strategy.get_executor_class().__name__, executor_name
                            )

    def test_unavailable_backend_rejected(self):
        for sm in (9, 10):
            with self.subTest(sm=sm), _backend(
                sm, routed=False, shared=False
            ), self.assertRaises(ValueError):
                _registry().get_strategy(
                    _config(
                        ep_size=2, n_shared_experts=1, strategy="mega_moe_se", dim=1024
                    )
                )

    def test_mismatched_world_rejected(self):
        for sm in (9, 10):
            for name in ("mega_moe", "mega_moe_se"):
                with self.subTest(sm=sm, name=name), _backend(sm), self.assertRaises(
                    ValueError
                ):
                    _registry().get_strategy(
                        _config(
                            ep_size=2,
                            n_shared_experts=1,
                            strategy=name,
                            world_size=4,
                            dim=1024,
                        )
                    )

    def test_grouped_storage_does_not_restrict_mega_factory(self):
        from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory

        for sm in (9, 10):
            for storage in ("fp8", "fp4"):
                for name in ("mega_moe", "mega_moe_se"):
                    with self.subTest(sm=sm, storage=storage, name=name), _backend(sm):
                        config = _config(
                            ep_size=2, n_shared_experts=1, strategy=name, dim=1024
                        )
                        config.dsv4_moe_sm90_storage = storage
                        strategy = _registry().get_strategy(config)
                        with patch.object(FusedMoeFactory, "_registry", _registry()):
                            factory = FusedMoeFactory()
                        with patch.object(
                            factory.registry, "get_strategy", return_value=strategy
                        ), patch.object(strategy, "create_router"), patch.object(
                            strategy, "create_executor"
                        ), patch(
                            "rtp_llm.models_py.modules.factory.fused_moe.factory.FusedMoe"
                        ):
                            factory.create_fused_moe(config, {})


class Sm90CapabilityTest(unittest.TestCase):
    def test_capability_probes_native_shared_api(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_sm90 import (
            sm90_mega_available,
        )

        def kernel(*, shared_l1_weights=None, shared_l2_weights=None):
            pass

        api = SimpleNamespace(
            get_symm_buffer_for_mega_moe_sm90=lambda: None,
            transform_weights_for_mega_moe_sm90=lambda: None,
            sm90_fp8_fp4_mega_moe=kernel,
        )
        with patch.dict(sys.modules, {"deep_gemm": api}), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_sm90.is_sm90",
            return_value=True,
        ), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch(
            "torch.distributed.get_world_size", return_value=2
        ):
            self.assertTrue(sm90_mega_available())
            self.assertTrue(sm90_mega_available(shared=True))
            api.sm90_fp8_fp4_mega_moe = lambda: None
            self.assertTrue(sm90_mega_available())
            self.assertFalse(sm90_mega_available(shared=True))


class BackendSpecificSelectionTest(unittest.TestCase):

    def test_explicit_deepep_is_not_overridden_by_mega(self):
        with _backend(9):
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
                CudaGroupedFp4Sm90EpStrategy,
            )

            registry = _registry()
            registry.register(CudaGroupedFp4Sm90EpStrategy())
            config = _config(
                ep_size=2, n_shared_experts=1, strategy="grouped_fp4_sm90_ep"
            )
            config.use_deepep_moe = True
            config.use_deepep_low_latency = False
            config.moe_config.use_deepep_moe = True
            config.moe_config.use_deepep_low_latency = False
            with patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4_sm90._has_sm90_grouped_fp8_kernel",
                return_value=True,
            ), patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm",
                return_value=(9, 0),
            ), patch(
                "rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported",
                return_value=True,
            ):
                self.assertIsInstance(
                    registry.get_strategy(config), CudaGroupedFp4Sm90EpStrategy
                )

    def test_decode_route_slices_must_fit_hidden_size(self):
        with _backend(9):
            config = _config(ep_size=2, strategy="mega_moe")
            config.dim = 512
            with self.assertRaises(ValueError):
                _registry().get_strategy(config)


if __name__ == "__main__":
    unittest.main()
