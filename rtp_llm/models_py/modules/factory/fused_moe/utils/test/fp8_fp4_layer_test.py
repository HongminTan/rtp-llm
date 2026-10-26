import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertGatePayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    layer as layer_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)


class _FakeGate:
    def __init__(self, events, gate_pack_enabled):
        self.events = events
        self.gate_pack_enabled = gate_pack_enabled

    def can_prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.can_prepare")
        return self.gate_pack_enabled

    def prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.prepare")
        return object()

    def __call__(self, x, input_ids):
        self.events.append("gate.forward")
        tokens = x.size(0)
        return (
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class _FakeFusedMoe:
    supports_gate_pack = True

    def __init__(self, events):
        self.events = events

    def forward_gate_pack(
        self,
        *,
        hidden_states,
        gate_payload,
        activation,
        after_dispatch=None,
        before_combine=None
    ):
        self.events.append("fused_moe.forward_gate_pack")
        if after_dispatch is not None:
            after_dispatch()
        if before_combine is not None:
            before_combine()
        return torch.ones_like(hidden_states)

    def __call__(
        self,
        *,
        hidden_states,
        topk_weights,
        topk_ids,
        activation,
        after_dispatch=None,
        before_combine=None
    ):
        self.events.append("fused_moe.forward")
        if after_dispatch is not None:
            after_dispatch()
        if before_combine is not None:
            before_combine()
        return torch.ones_like(hidden_states)


class _FakeSharedExecutor:
    def after_dispatch(self):
        pass

    def before_combine(self):
        pass

    def __init__(self, events):
        self.events = events
        self.x = None

    def begin(self, x):
        self.events.append("shared.begin")

    def abort(self):
        self.events.append("shared.abort")

    def run_shared(self, x):
        self.events.append("shared.run_shared")
        self.x = x

    def finish(self):
        self.events.append("shared.finish")
        return torch.ones_like(self.x)


class _Fp32Executor(Fp8Fp4ExecutorBase):
    def setup_weights(self, weights):
        pass

    def forward(self, x, topk_weights, topk_ids):
        return torch.ones_like(x, dtype=torch.float32)

    def forward_gate_pack(self, x, gate_payload):
        tokens = x.size(0)
        return (
            torch.ones_like(x, dtype=torch.float32),
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class Fp8Fp4MoeLayerSchedulingTest(unittest.TestCase):
    def _layer(self, events, gate_pack_enabled, with_shared=True):
        layer = Fp8Fp4MoeLayer.__new__(Fp8Fp4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.dim = 4
        layer.gate = _FakeGate(events, gate_pack_enabled)
        layer.fused_moe = _FakeFusedMoe(events)
        layer.shared_experts = object() if with_shared else None
        layer._shared_executor = _FakeSharedExecutor(events) if with_shared else None
        return layer

    @staticmethod
    def _combine(events):
        def combine(routed, shared, out_dtype, out=None):
            events.append("combine")
            result = (routed.float() + shared.float()).to(out_dtype)
            if out is None:
                return result
            out.copy_(result)
            return out

        return combine

    def test_gate_pack_starts_shared_expert_before_gate(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            output = layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "shared.begin",
                "shared.run_shared",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
                "shared.finish",
                "combine",
            ],
        )
        torch.testing.assert_close(output, torch.full_like(x, 2))

    def test_ordinary_gate_stays_before_shared_expert(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "shared.begin",
                "gate.forward",
                "shared.run_shared",
                "fused_moe.forward",
                "shared.finish",
                "combine",
            ],
        )

    def test_stage_callbacks_are_forwarded_on_both_routing_paths(self):
        for gate_pack_enabled in (False, True):
            with self.subTest(gate_pack_enabled=gate_pack_enabled):
                events = []
                layer = self._layer(events, gate_pack_enabled)
                layer._shared_executor.after_dispatch = lambda: events.append(
                    "after_dispatch"
                )
                layer._shared_executor.before_combine = lambda: events.append(
                    "before_combine"
                )
                with patch.object(
                    layer_module,
                    "combine_routed_and_shared",
                    side_effect=self._combine(events),
                ):
                    layer(torch.ones((2, 4), dtype=torch.bfloat16), None)
                self.assertEqual(
                    events[-4:],
                    ["after_dispatch", "before_combine", "shared.finish", "combine"],
                )

    def test_schedule_failures_abort_without_finishing(self):
        for phase in ("begin", "run_shared", "routed", "finish"):
            with self.subTest(phase=phase):
                events = []
                layer = self._layer(events, gate_pack_enabled=True)
                target = (
                    layer.fused_moe if phase == "routed" else layer._shared_executor
                )
                method = "forward_gate_pack" if phase == "routed" else phase
                with patch.object(target, method, side_effect=RuntimeError(phase)):
                    with self.assertRaisesRegex(RuntimeError, phase):
                        layer(torch.ones((2, 4), dtype=torch.bfloat16), None)
                self.assertEqual(events[-1], "shared.abort")
                self.assertNotIn("shared.finish", events)

    def test_fused_shared_expert_keeps_gate_first(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True, with_shared=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
            ],
        )


class Fp8Fp4ExecutorOutputDtypeTest(unittest.TestCase):
    def setUp(self):
        self.executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(self.executor)
        self.x = torch.ones((2, 4), dtype=torch.bfloat16)

    def _execute(self, payload):
        return self.executor.execute(
            payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        ).fused_expert_output

    def test_ordinary_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                expert_topk_weights=torch.ones((2, 1), dtype=torch.float32),
                expert_topk_ids=torch.zeros((2, 1), dtype=torch.long),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual((output + self.x).dtype, torch.bfloat16)

    def test_gate_pack_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                gate_payload=ExpertGatePayload(
                    scores=torch.ones((2, 1), dtype=torch.float32),
                    topk=1,
                    score_func="sigmoid",
                    route_scale=1.0,
                ),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
