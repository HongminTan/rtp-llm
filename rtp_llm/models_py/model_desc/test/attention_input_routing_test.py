import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.model_desc.block_map import get_group_tags_for_layers
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextMetadata,
    _maybe_write_cp_cache_store,
    _write_cp_cache_store,
)


class FakeKVCache:
    def __init__(self, layer_tags: list[list[str]]):
        self.layer_tags = layer_tags

    def get_layer_cache_groups(self, layer_idx: int):
        return [SimpleNamespace(tag=tag) for tag in self.layer_tags[layer_idx]]


class RoutingModel(GptModelBase):
    def __init__(self, fmha_group_tags: list[str] | None):
        nn.Module.__init__(self)
        self.config = object()
        self.parallelism_config = object()
        self.weight = object()
        self.fmha_config = object()
        self.fmha_group_tags = fmha_group_tags

    def _get_fmha_group_tags(self) -> list[str] | None:
        return self.fmha_group_tags


class DecodeImplA:
    pass


class DecodeImplB:
    pass


class PrefillImplA:
    pass


class PrefillImplB:
    pass


class WinnerImpl:
    pass


def _dynamic_routing_model():
    model = RoutingModel(None)
    model.py_hw_kernel_config = SimpleNamespace(enable_dynamic_decode_backend=True)
    model.backend_plan = {}
    model._logged_decode_backend = {}
    model._attention_backend_gate_context = None
    model.device_type = DeviceType.Cuda
    model.kv_cache = object()
    model.fmha_config = object()
    model.parallelism_config = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        dp_rank=0,
        get_attn_tp_size=lambda: 1,
    )
    model.config = SimpleNamespace(
        headwise_config=None,
        hybrid_attention_config=None,
        getAttentionConfigs=lambda _tp: SimpleNamespace(use_mla=False),
    )
    return model


def _install_decode_gate(model, *impl_classes):
    with patch(
        "rtp_llm.models_py.model_desc.module_base.DECODE_MHA_IMPS",
        list(impl_classes),
    ):
        model.set_attention_backend_gate(
            decode_passed_names=[impl_cls.__name__ for impl_cls in impl_classes],
            prefill_passed_names=None,
        )


def _decode_inputs(bs=2, *, is_prefill=False):
    return SimpleNamespace(
        input_lengths=torch.ones(bs, dtype=torch.int32),
        is_prefill=is_prefill,
    )


class AttentionInputRoutingTest(unittest.TestCase):
    def test_cp_cache_store_uses_each_layer_tag_metadata(self):
        expected = {}
        layer_inputs = {}
        for index, tag in enumerate(("full", "linear0", "linear1"), start=1):
            actual_lengths = torch.tensor([index], dtype=torch.int32)
            prefix_lengths = torch.tensor([index + 10], dtype=torch.int32)
            block_ids = torch.tensor([[index + 20]], dtype=torch.int32)
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            layer_inputs[tag] = (
                SimpleNamespace(
                    context_parallel_info=SimpleNamespace(
                        prefill_actual_input_lengths_cpu=actual_lengths
                    ),
                    prefix_lengths=prefix_lengths,
                    kv_cache_block_id=block_ids,
                    cache_store_inputs=cache_store_inputs,
                ),
                kv_cache,
            )
            expected[tag] = (
                actual_lengths,
                prefix_lengths,
                block_ids,
                cache_store_inputs,
                kv_cache,
            )

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            for tag in ("full", "linear0", "linear1"):
                _write_cp_cache_store(*layer_inputs[tag])

        self.assertEqual(write_cache_store.call_count, 3)
        for call, tag in zip(
            write_cache_store.call_args_list, ("full", "linear0", "linear1")
        ):
            for actual, wanted in zip(call.args, expected[tag]):
                self.assertIs(actual, wanted)

    def test_cp_cache_store_skips_layer_without_store_inputs(self):
        attention_inputs = SimpleNamespace(cache_store_inputs=None)

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

        write_cache_store.assert_not_called()

    def test_cp_cache_store_requires_context_parallel_metadata(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            context_parallel_info=None,
        )

        with self.assertRaisesRegex(
            RuntimeError, "CP cache store requires context_parallel_info"
        ):
            _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor([1], dtype=torch.int32)
            ),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_block_id=torch.tensor([[1]], dtype=torch.int32),
        )

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            _maybe_write_cp_cache_store(
                attention_inputs,
                SimpleNamespace(tag="linear0"),
                Qwen3NextMetadata(),
            )

        write_cache_store.assert_not_called()

    def test_get_group_tags_for_model_selected_layers(self):
        kv_cache = FakeKVCache([["full"], ["linear0"], ["linear1"], ["full", "aux"]])

        self.assertEqual(get_group_tags_for_layers(kv_cache, [0, 3]), ["full", "aux"])

    def test_prepare_fmha_impl_only_for_model_selected_tags(self):
        inputs_by_tag = {
            "full": object(),
            "linear0": object(),
            "linear1": object(),
        }
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(fmha_impl, {"full": inputs_by_tag["full"]})
        factory.assert_called_once()

    def test_default_model_prepares_every_tag(self):
        inputs_by_tag = {"group0": object(), "group1": object()}
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(None)

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs)

        self.assertEqual(fmha_impl, inputs_by_tag)
        self.assertEqual(factory.call_count, 2)

    def test_dynamic_output_probe_does_not_log_or_mark_backend(self):
        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()
        static_impl = object()
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
                return_value=static_impl,
            ) as factory,
            patch.object(model, "_log_decode_backend_once") as log_backend,
        ):
            actual = model.prepare_fmha_impl(object(), is_cuda_graph=True)

        self.assertIs(actual, static_impl)
        factory.assert_called_once()
        log_backend.assert_not_called()
        self.assertEqual(model._logged_decode_backend, {})

    def test_completed_plan_miss_logs_fixed_priority_once(self):
        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()

        class StaticImpl:
            pass

        model.backend_plan[2] = None
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
                return_value=StaticImpl(),
            ) as factory,
            patch("rtp_llm.models_py.model_desc.module_base.logging.info") as info,
        ):
            model.prepare_fmha_impl(object(), is_cuda_graph=True)
            model.prepare_fmha_impl(object(), is_cuda_graph=True)

        self.assertEqual(factory.call_count, 2)
        info.assert_called_once_with(
            "[dispatcher] decode backend in use: bs=%d -> %s (%s)",
            2,
            "StaticImpl",
            "fixed-priority",
        )
        self.assertEqual(model._logged_decode_backend, {2: "StaticImpl"})

    def test_completed_winner_applies_and_logs_once_without_static_factory(self):
        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()
        winner_impl = object()
        model.backend_plan[2] = WinnerImpl
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector.instantiate_decode_impl",
                return_value=winner_impl,
            ) as instantiate,
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl"
            ) as factory,
            patch("rtp_llm.models_py.model_desc.module_base.logging.info") as info,
        ):
            first = model.prepare_fmha_impl(object(), is_cuda_graph=True)
            second = model.prepare_fmha_impl(object(), is_cuda_graph=True)

        self.assertIs(first, winner_impl)
        self.assertIs(second, winner_impl)
        self.assertEqual(instantiate.call_count, 2)
        factory.assert_not_called()
        info.assert_called_once_with(
            "dynamic_decode_plan_applied bs=%d backend=%s tp_rank=%d dp_rank=%d",
            2,
            "WinnerImpl",
            0,
            0,
        )

    def test_winner_apply_failure_never_uses_static_factory(self):
        from rtp_llm.models_py.modules.factory.attention.dispatch import (
            backend_selector,
        )

        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()
        model.backend_plan[2] = WinnerImpl
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch.object(
                backend_selector,
                "instantiate_decode_impl",
                side_effect=backend_selector.DynamicDecodeFatalError("fatal"),
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl"
            ) as factory,
        ):
            with self.assertRaises(backend_selector.DynamicDecodeFatalError):
                model.prepare_fmha_impl(object(), is_cuda_graph=True)

        factory.assert_not_called()
        self.assertEqual(model._logged_decode_backend, {})

    def test_decode_selection_before_gate_installation_is_wiring_error(self):
        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()
        from rtp_llm.models_py.modules.factory.attention.dispatch import (
            backend_selector,
        )

        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch.object(backend_selector, "run_backend_selection") as selector,
        ):
            with self.assertRaisesRegex(RuntimeError, "has not been installed"):
                model.select_decode_backend(object())

        self.assertEqual(model.backend_plan, {})
        selector.assert_not_called()

    def test_enabled_gate_rejects_empty_allowlist(self):
        model = _dynamic_routing_model()
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            model.set_attention_backend_gate(
                decode_passed_names=[], prefill_passed_names=None
            )

    def test_prefill_gate_context_resolves_classes_once(self):
        model = _dynamic_routing_model()
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.DECODE_MHA_IMPS",
                [DecodeImplA, DecodeImplB],
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.PREFILL_MHA_IMPS",
                [PrefillImplA, PrefillImplB],
            ),
        ):
            model.set_attention_backend_gate(
                decode_passed_names=["DecodeImplA"],
                prefill_passed_names=["PrefillImplB", "PrefillImplA"],
            )

        self.assertEqual(model.get_decode_gate_passed(), frozenset({DecodeImplA}))
        self.assertEqual(
            model.get_prefill_gate_passed(),
            frozenset({PrefillImplA, PrefillImplB}),
        )

    def test_prefill_getter_requires_enabled_domain_and_installation(self):
        model = _dynamic_routing_model()
        with self.assertRaisesRegex(RuntimeError, "has not been installed"):
            model.get_prefill_gate_passed()

        _install_decode_gate(model, DecodeImplA)
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            model.get_prefill_gate_passed()
        self.assertEqual(model.get_decode_gate_passed(), frozenset({DecodeImplA}))

    def test_prefill_gate_rejects_duplicate_and_unknown_names(self):
        with patch(
            "rtp_llm.models_py.model_desc.module_base.PREFILL_MHA_IMPS",
            [PrefillImplA, PrefillImplB],
        ):
            model = _dynamic_routing_model()
            with self.assertRaisesRegex(ValueError, "duplicates"):
                model.set_attention_backend_gate(
                    decode_passed_names=None,
                    prefill_passed_names=["PrefillImplA", "PrefillImplA"],
                )
            self.assertIsNone(model._attention_backend_gate_context)
            with self.assertRaisesRegex(ValueError, "unknown backend names"):
                model.set_attention_backend_gate(
                    decode_passed_names=None,
                    prefill_passed_names=["NoSuchPrefillImpl"],
                )
            self.assertIsNone(model._attention_backend_gate_context)

    def test_ready_gate_filters_selector_and_records_normal_plan_miss(self):
        from rtp_llm.models_py.modules.factory.attention.dispatch import (
            backend_selector,
        )

        model = _dynamic_routing_model()
        _install_decode_gate(model, DecodeImplB)
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=_decode_inputs(),
            ),
            patch.object(
                backend_selector, "run_backend_selection", return_value=None
            ) as selector,
        ):
            model.select_decode_backend(object())

        selector.assert_called_once_with(
            model, ANY, gate_passed=frozenset({DecodeImplB})
        )
        self.assertEqual(model.backend_plan, {2: None})

    def test_gate_setter_clears_plans_and_rejects_updates_after_selection(self):
        model = _dynamic_routing_model()
        model.backend_plan[2] = DecodeImplA
        model._logged_decode_backend[2] = "OldImpl"

        _install_decode_gate(model, DecodeImplA)
        self.assertEqual(model.backend_plan, {})
        self.assertEqual(model._logged_decode_backend, {})
        with self.assertRaisesRegex(RuntimeError, "already been installed"):
            model.set_attention_backend_gate(
                decode_passed_names=[DecodeImplA.__name__],
                prefill_passed_names=None,
            )

        with patch(
            "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
            return_value=_decode_inputs(),
        ):
            from rtp_llm.models_py.modules.factory.attention.dispatch import (
                backend_selector,
            )

            with patch.object(
                backend_selector, "run_backend_selection", return_value=None
            ):
                model.select_decode_backend(object())

    def test_unexpected_selector_exception_propagates_without_plan(self):
        from rtp_llm.models_py.modules.factory.attention.dispatch import (
            backend_selector,
        )

        model = _dynamic_routing_model()
        _install_decode_gate(model, DecodeImplA)
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=_decode_inputs(),
            ),
            patch.object(
                backend_selector,
                "run_backend_selection",
                side_effect=RuntimeError("unexpected"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                model.select_decode_backend(object())

        self.assertEqual(model.backend_plan, {})

    def test_capability_guard_and_nonfinal_paths_keep_plan_and_logs_empty(self):
        model = _dynamic_routing_model()
        attention_inputs = _decode_inputs()
        model.py_hw_kernel_config.enable_dynamic_decode_backend = False
        with patch(
            "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
            return_value=attention_inputs,
        ):
            model.select_decode_backend(object())
        self.assertEqual(model.backend_plan, {})
        self.assertIsNone(model._attention_backend_gate_context)

        model.py_hw_kernel_config.enable_dynamic_decode_backend = True
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=attention_inputs,
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
                return_value=object(),
            ),
            patch.object(model, "_log_decode_backend_once") as log_backend,
        ):
            model.prepare_fmha_impl(object(), is_cuda_graph=False)
        log_backend.assert_not_called()

        prefill_inputs = _decode_inputs(is_prefill=True)
        model.backend_plan[2] = None
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
                return_value=prefill_inputs,
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
                return_value=object(),
            ),
            patch.object(model, "_log_decode_backend_once") as log_backend,
        ):
            model.prepare_fmha_impl(object(), is_cuda_graph=True)
        log_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
