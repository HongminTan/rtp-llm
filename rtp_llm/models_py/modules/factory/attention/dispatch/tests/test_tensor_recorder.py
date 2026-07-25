"""Focused recorder contract tests inside the established attention aggregate."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_kv_history import (
    GoldenKVHistory,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_sdpa_impl import (
    GoldenSDPAImpl,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder import (
    TensorRecorder,
)


class _Model:
    layer_num = 2

    def __init__(self):
        self.config = SimpleNamespace(getAttentionConfigs=lambda _tp_size: _configs())
        self.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        self.fmha_config = SimpleNamespace()

    def prepare_fmha_impl(self, *_args, **_kwargs):
        return None


class _Inner:
    fmha_params = None

    def forward(self, qkv, _kv_cache, _layer_idx):
        return qkv[:, :4]


class _CaptureInner(_Inner):
    def forward(self, qkv, kv_cache, layer_idx):
        if not hasattr(self, "qkvs"):
            self.qkvs = {}
        self.qkvs[layer_idx] = qkv.detach().clone()
        return super().forward(qkv, kv_cache, layer_idx)


def _inputs():
    return SimpleNamespace(
        is_prefill=False,
        sequence_lengths=torch.tensor([8, 5], dtype=torch.int32),
        input_lengths=torch.tensor([1, 1], dtype=torch.int32),
        prefix_lengths=torch.empty(0, dtype=torch.int32),
        cu_seqlens=torch.zeros(3, dtype=torch.int32),
    )


def _configs():
    return SimpleNamespace(
        head_num=2,
        kv_head_num=1,
        size_per_head=2,
        dtype=torch.bfloat16,
        rope_config=SimpleNamespace(style=None),
        use_logn_attn=True,
    )


def _layers(value=1.0):
    return {
        i: AttentionLayerRecord(
            layer_idx=i,
            output=torch.full((2, 4), value, dtype=torch.bfloat16),
        )
        for i in range(2)
    }


def _install_fake_wrapper(recorder, active_ids=(0, 1), value=1.0):
    recorder._wrapper = SimpleNamespace(
        layer_records=_layers(value),
        head_num=2,
        kv_head_num=1,
        head_dim=2,
        dtype=torch.bfloat16,
        is_prefill=False,
        sequence_lengths=torch.tensor([8, 5], dtype=torch.int32),
        input_lengths=torch.tensor([1, 1], dtype=torch.int32),
        prefix_lengths=torch.empty(0, dtype=torch.int32),
        cu_seqlens=torch.zeros(3, dtype=torch.int32),
        active_sequence_ids=tuple(active_ids),
    )


def _install_fake_prefill_wrapper(recorder, phase, active_ids=(0, 1), value=1.0):
    prefix = (
        torch.zeros(len(active_ids), dtype=torch.int32)
        if phase == "plain"
        else torch.full((len(active_ids),), 4, dtype=torch.int32)
    )
    recorder._wrapper = SimpleNamespace(
        layer_records=_layers(value),
        head_num=2,
        kv_head_num=1,
        head_dim=2,
        dtype=torch.bfloat16,
        is_prefill=True,
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        input_lengths=torch.ones(len(active_ids), dtype=torch.int32),
        prefix_lengths=prefix,
        cu_seqlens=torch.arange(len(active_ids) + 1, dtype=torch.int32),
        active_sequence_ids=tuple(active_ids),
    )


def _record(
    recorder,
    scenario,
    impl,
    phase,
    active_ids=(0, 1),
    value=1.0,
    gate_qualifying=True,
):
    recorder.start_run(
        scenario,
        impl,
        phase,
        active_ids,
        gate_qualifying=gate_qualifying,
    )
    _install_fake_wrapper(recorder, active_ids, value)
    recorder.stop_run(f"{scenario}::{impl}::{phase}")


def _record_prefill(
    recorder,
    scenario,
    impl,
    phase,
    active_ids=(0, 1),
    value=1.0,
    gate_qualifying=True,
):
    recorder.start_run(
        scenario,
        impl,
        phase,
        active_ids,
        gate_qualifying=gate_qualifying,
    )
    _install_fake_prefill_wrapper(recorder, phase, active_ids, value)
    recorder.stop_run(f"{scenario}::{impl}::{phase}")


def test_recording_wrapper_snapshots_metadata_and_output_only():
    inputs = _inputs()
    wrapper = RecordingWrapper(
        _Inner(), _configs(), inputs, active_sequence_ids=(3, 7), record_qkv=False
    )
    inputs.sequence_lengths.fill_(99)
    wrapper.forward(torch.arange(16, dtype=torch.float32).reshape(2, 8), None, 0)
    layer = wrapper.layer_records[0]
    assert layer.q is None and layer.k is None and layer.v is None
    assert wrapper.sequence_lengths.tolist() == [8, 5]
    assert wrapper.active_sequence_ids == (3, 7)


def test_recording_wrapper_full_qkv_and_duplicate_layer_rejected():
    wrapper = RecordingWrapper(
        _Inner(), _configs(), _inputs(), active_sequence_ids=(0, 1), record_qkv=True
    )
    qkv = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    wrapper.forward(qkv, None, 0)
    layer = wrapper.layer_records[0]
    assert layer.q is not None and layer.k is not None and layer.v is not None
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "duplicate layer"):
        wrapper.forward(qkv, None, 0)


def test_recording_wrapper_records_candidate_but_returns_golden_output():
    inner = _CaptureInner()
    golden = _layers(value=7.0)
    wrapper = RecordingWrapper(
        inner,
        _configs(),
        _inputs(),
        active_sequence_ids=(0, 1),
        record_qkv=False,
        golden_layer_records=golden,
    )
    candidate_qkv = torch.zeros((2, 8), dtype=torch.bfloat16)
    returned = wrapper.forward(candidate_qkv, None, 0)
    assert torch.equal(inner.qkvs[0], candidate_qkv)
    assert torch.equal(returned, golden[0].output)
    layer = wrapper.layer_records[0]
    assert layer.q is None and layer.k is None and layer.v is None
    assert torch.equal(layer.output, torch.zeros_like(golden[0].output))

    next_qkv = torch.cat([returned, returned], dim=-1)
    returned_next = wrapper.forward(next_qkv, None, 1)
    assert torch.equal(inner.qkvs[1], next_qkv)
    assert torch.equal(returned_next, golden[1].output)


def test_recorder_wires_phase_golden_through_prepare_and_stop():
    model = _Model()
    original_prepare = model.prepare_fmha_impl
    recorder = TensorRecorder(model, record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0", value=7.0)
    candidate = recorder.candidate_snapshot("decode_0")["registry"][0]
    recorder.set_expected_candidates("decode_0", [candidate])
    recorder.start_run("scenario", candidate, "decode_0", [0, 1])

    inner = _CaptureInner()
    recorder._factory = lambda *_args: inner
    model_inputs = SimpleNamespace(attention_inputs=_inputs())
    with mock.patch(
        "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_all_supported_impls",
        return_value=[],
    ):
        wrapper = recorder._recording_prepare(model_inputs)
    candidate_qkv = torch.zeros((2, 8), dtype=torch.bfloat16)
    returned = wrapper.forward(candidate_qkv, None, 0)
    returned_next = wrapper.forward(torch.cat([returned, returned], dim=-1), None, 1)
    recorder.stop_run(f"scenario::{candidate}::decode_0")

    assert torch.equal(returned, _layers(value=7.0)[0].output)
    assert torch.equal(returned_next, _layers(value=7.0)[1].output)
    candidate_record = recorder.records["scenario"][candidate][0]
    assert torch.equal(candidate_record.layer_records[0].output, torch.zeros((2, 4)))
    assert recorder._state == "IDLE"
    assert model.prepare_fmha_impl == original_prepare


def test_golden_history_appends_chunked_prefill_without_replacing_history():
    history = GoldenKVHistory().bind_batch([7])
    first_k = torch.tensor([[1.0], [2.0]])
    first_v = torch.tensor([[3.0], [4.0]])
    next_k = torch.tensor([[5.0]])
    next_v = torch.tensor([[6.0]])
    history.store_plain(0, 0, first_k, first_v)
    history.append_prefill(0, 0, next_k, next_v)
    stored_k, stored_v = history.get_decode_history(0, 0)
    assert torch.equal(stored_k, torch.cat([first_k, next_k]))
    assert torch.equal(stored_v, torch.cat([first_v, next_v]))


def test_golden_sdpa_bootstrap_forward_stores_appends_and_checks_prefix_length():
    class _CacheWriter:
        def write(self, *_args):
            pass

    def _bootstrap_impl(history, prefix_len, input_len, q, k, v):
        impl = GoldenSDPAImpl.__new__(GoldenSDPAImpl)
        impl.history = history
        impl.append_prefill_history = True
        impl.head_num = 1
        impl.kv_head_num = 1
        impl.attn_inputs = SimpleNamespace(
            is_prefill=True,
            cu_seqlens=torch.tensor([0, input_len], dtype=torch.int32),
            prefix_lengths=torch.tensor([prefix_len], dtype=torch.int32),
            input_lengths=torch.tensor([input_len], dtype=torch.int32),
        )
        impl.cache_writer = _CacheWriter()
        impl._split_qkv = lambda _qkv: (q, k, v)
        impl._sdpa = lambda query, *_args, **_kwargs: query
        return impl

    history = GoldenKVHistory().bind_batch([7])
    first = torch.tensor([[[1.0]], [[2.0]]])
    _bootstrap_impl(history, 0, 2, first, first + 10, first + 20).forward(
        torch.zeros((2, 3)), None, 0
    )
    second = torch.tensor([[[3.0]]])
    _bootstrap_impl(history, 2, 1, second, second + 10, second + 20).forward(
        torch.zeros((1, 3)), None, 0
    )
    stored_k, stored_v = history.get_decode_history(0, 0)
    assert stored_k.shape[0] == 3
    assert stored_v.shape[0] == 3
    assert torch.equal(stored_k[-1], second[0] + 10)

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "history length"):
        _bootstrap_impl(history, 2, 1, second, second + 10, second + 20).forward(
            torch.zeros((1, 3)), None, 0
        )


def test_golden_bootstrap_is_silent_and_restores_recorder_state():
    model = _Model()
    original_prepare = model.prepare_fmha_impl
    recorder = TensorRecorder(model, record_qkv=True)
    recorder.start_golden_bootstrap("scenario", "bootstrap_s0_chunk_0", [3])
    with mock.patch(
        "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.GoldenSDPAImpl"
    ) as golden_impl:
        recorder._factory(None, None, None)
        assert golden_impl.call_args.kwargs["append_prefill_history"] is True
    _install_fake_wrapper(recorder, active_ids=(3,))
    recorder.stop_golden_bootstrap("scenario::golden_bootstrap::bootstrap_s0_chunk_0")

    assert recorder.records == {}
    assert recorder._manifest == {}
    assert recorder._state == "IDLE"
    assert model.prepare_fmha_impl == original_prepare


def test_golden_bootstrap_prepare_skips_candidates_qkv_and_failed_run_cleans_up():
    model = _Model()
    recorder = TensorRecorder(model, record_qkv=True)
    recorder.start_golden_bootstrap("scenario", "bootstrap_s0_chunk_0", [3])
    recorder._factory = lambda *_args: _Inner()
    with mock.patch(
        "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_all_supported_impls",
        side_effect=AssertionError("bootstrap must not enumerate candidates"),
    ):
        wrapper = recorder._recording_prepare(
            SimpleNamespace(attention_inputs=_inputs())
        )
    assert wrapper.record_qkv is False
    assert recorder._candidates == {}
    recorder.stop_golden_bootstrap(
        "scenario::golden_bootstrap::bootstrap_s0_chunk_0", completed=False
    )
    assert recorder._state == "IDLE"
    assert model.prepare_fmha_impl == recorder._orig_prepare


def test_decode_candidate_constructor_error_is_fatal():
    class _BrokenCandidate:
        __name__ = "BrokenCandidate"

        def __init__(self, *_args):
            raise RuntimeError("decode constructor failed")

    recorder = TensorRecorder(_Model(), record_qkv=False)
    recorder.start_run("scenario", "golden", "decode_0", [0, 1])
    model_inputs = SimpleNamespace(attention_inputs=_inputs())
    with mock.patch(
        "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_all_supported_impls",
        return_value=[_BrokenCandidate],
    ), unittest.TestCase().assertRaisesRegex(RuntimeError, "constructor failed"):
        recorder._recording_prepare(model_inputs)
    recorder.close()


def test_state_machine_identity_and_close_are_strict():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    recorder.start_run("scenario", "golden", "decode_0", [0, 1])
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already running"):
        recorder.start_run("scenario", "golden", "decode_0", [0, 1])
    _install_fake_wrapper(recorder)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "identity mismatch"):
        recorder.stop_run("wrong")
    recorder.close()
    recorder.close()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "closed"):
        recorder.start_run("scenario", "golden", "decode_0", [0, 1])


def test_candidate_snapshot_and_manifest_finalize_have_no_collective():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    snapshot = recorder.candidate_snapshot("decode_0")
    assert len(snapshot["registry"]) == len(snapshot["viable_mask"])
    candidate = snapshot["registry"][0]
    recorder._candidates["decode_0"] = [candidate]
    snapshot = recorder.candidate_snapshot("decode_0")
    assert snapshot["viable_mask"][0] == 1
    recorder.set_expected_candidates("decode_0", [candidate])
    _record(recorder, "scenario", candidate, "decode_0")

    result = recorder.finalize_decode_gate("BASE")
    assert result["valid"]
    assert result["applicable_mask"][0] == 1
    assert result["passed_mask"][0] == 1
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already finalized"):
        recorder.finalize_decode_gate("BASE")


def test_manifest_gate_qualifying_is_frozen_and_controls_applicability():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(
        recorder,
        "diagnostic",
        "golden",
        "decode_0",
        gate_qualifying=False,
    )
    candidate = recorder.candidate_snapshot("decode_0")["registry"][0]
    recorder.set_expected_candidates("decode_0", [candidate])
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "gate_qualifying"):
        recorder.start_run(
            "diagnostic",
            candidate,
            "decode_0",
            [0, 1],
            gate_qualifying=True,
        )
    _record(
        recorder,
        "diagnostic",
        candidate,
        "decode_0",
        gate_qualifying=False,
    )
    assert not recorder._manifest[("diagnostic", "decode_0")].gate_qualifying
    diagnostic = recorder.finalize_decode_gate("BASE")
    assert diagnostic["valid"]
    assert diagnostic["applicable_mask"][0] == 0
    assert diagnostic["passed_mask"][0] == 0

    qualifying = TensorRecorder(_Model(), record_qkv=False)
    _record(qualifying, "diagnostic", "golden", "decode_0")
    candidate = qualifying.candidate_snapshot("decode_0")["registry"][0]
    qualifying.set_expected_candidates("decode_0", [candidate])
    _record(qualifying, "diagnostic", candidate, "decode_0")
    qualifying_result = qualifying.finalize_decode_gate("BASE")
    assert qualifying_result["valid"]
    assert qualifying_result["applicable_mask"][0] == 1
    assert qualifying_result["passed_mask"][0] == 1


def test_finalize_rejects_single_fp8_threshold_failure():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    candidate = recorder.candidate_snapshot("decode_0")["registry"][0]
    recorder.set_expected_candidates("decode_0", [candidate])
    recorder.start_run("scenario", candidate, "decode_0", [0, 1])
    _install_fake_wrapper(recorder)
    # 1.30x scale: nrmse 0.259 trips the hard 0.20 fence (cos is SNR-gated and
    # mean_ulp sits below the artifact-backed 25.0 threshold at this scale).
    recorder._wrapper.layer_records[0].output.mul_(1.30)
    recorder.stop_run(f"scenario::{candidate}::decode_0")

    result = recorder.finalize_decode_gate("FP8")
    assert result["valid"]
    assert result["passed_mask"][0] == 0


def test_candidate_missing_is_ready_but_not_verified_or_passed():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    candidate = recorder.candidate_snapshot("decode_0")["registry"][0]
    recorder.set_expected_candidates("decode_0", [candidate])
    result = recorder.finalize_decode_gate("BASE")
    assert result["valid"]
    assert result["applicable_mask"][0] == 1
    assert result["passed_mask"][0] == 0


def test_finalize_prefill_gate_uses_prefill_registry_and_records():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record_prefill(recorder, "scenario", "golden", "plain")
    snapshot = recorder.candidate_snapshot("plain")
    from rtp_llm.models_py.modules.factory.attention.attn_factory import (
        PREFILL_MHA_IMPS,
    )

    assert snapshot["registry"] == [cls.__name__ for cls in PREFILL_MHA_IMPS]
    candidate = snapshot["registry"][0]
    recorder.set_expected_candidates("plain", [candidate])
    _record_prefill(recorder, "scenario", candidate, "plain")
    _record_prefill(recorder, "scenario", "golden", "prefix")
    recorder.set_expected_candidates("prefix", [candidate])
    _record_prefill(recorder, "scenario", candidate, "prefix")

    result = recorder.finalize_prefill_gate("BASE")
    assert result["valid"]
    assert result["registry"] == snapshot["registry"]
    assert result["applicable_mask"][0] == 1
    assert result["passed_mask"][0] == 1
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already finalized"):
        recorder.finalize_prefill_gate("BASE")


def test_each_domain_finalizes_once_on_the_same_recorder():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    decode_candidate = recorder.candidate_snapshot("decode_0")["registry"][0]
    recorder.set_expected_candidates("decode_0", [decode_candidate])
    _record(recorder, "scenario", decode_candidate, "decode_0")
    _record_prefill(recorder, "scenario", "golden", "plain")
    prefill_candidate = recorder.candidate_snapshot("plain")["registry"][0]
    recorder.set_expected_candidates("plain", [prefill_candidate])
    _record_prefill(recorder, "scenario", prefill_candidate, "plain")

    decode_result = recorder.finalize_decode_gate("BASE")
    prefill_result = recorder.finalize_prefill_gate("BASE")
    assert decode_result["valid"] and prefill_result["valid"]
    assert decode_result["passed_mask"][0] == 1
    assert prefill_result["passed_mask"][0] == 1
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already finalized"):
        recorder.finalize_decode_gate("BASE")
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already finalized"):
        recorder.start_run("scenario", "golden", "decode_1", [0, 1])


def test_finalize_prefill_gate_without_prefill_manifest_is_invalid():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    recorder.set_expected_candidates("decode_0", [])
    result = recorder.finalize_prefill_gate("BASE")
    assert not result["valid"]


def test_invalid_golden_or_nonapplicable_record_is_unavailable():
    recorder = TensorRecorder(_Model(), record_qkv=False)
    recorder.start_run("scenario", "golden", "decode_0", [0, 1])
    _install_fake_wrapper(recorder)
    del recorder._wrapper.layer_records[1]
    recorder.stop_run("scenario::golden::decode_0")
    recorder.set_expected_candidates("decode_0", [])
    result = recorder.finalize_decode_gate("BASE")
    assert not result["valid"]

    recorder = TensorRecorder(_Model(), record_qkv=False)
    _record(recorder, "scenario", "golden", "decode_0")
    recorder.set_expected_candidates("decode_0", [])
    golden = recorder.records["scenario"]["golden"][0]
    recorder.records["scenario"]["Unexpected"] = [
        AttentionForwardRecord(
            impl_name="Unexpected",
            phase=golden.phase,
            layer_records=_layers(),
            head_num=golden.head_num,
            kv_head_num=golden.kv_head_num,
            head_dim=golden.head_dim,
            dtype=golden.dtype,
            is_prefill=golden.is_prefill,
            sequence_lengths=golden.sequence_lengths.clone(),
            input_lengths=golden.input_lengths.clone(),
            prefix_lengths=golden.prefix_lengths.clone(),
            cu_seqlens=golden.cu_seqlens.clone(),
            active_sequence_ids=golden.active_sequence_ids,
        )
    ]
    result = recorder.finalize_decode_gate("BASE")
    assert not result["valid"]


class TensorRecorderTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(TensorRecorderTest, _name, staticmethod(_fn))
del _name, _fn


if __name__ == "__main__":
    unittest.main()
