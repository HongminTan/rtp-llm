"""Unit tests for the pre-landed decode gate using synthetic CPU tensors.

These tests hand-build AttentionForwardRecord objects and cover the gate's
judgment and aggregation logic. They do not exercise the production warmup or
dispatcher wiring, which is not connected in this change. Tensor calculations
use CPU tensors; the Bazel target still runs on a driver-equipped node because
normal package imports link CUDA-dependent rtp_llm libraries.
"""

import unittest
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    GOLDEN,
    AttentionForwardRecord,
    AttentionLayerRecord,
    _build_decode_gate_strict,
    _build_prefill_gate_strict,
    _normalize_kv_dtype,
    build_decode_gate,
    gate_to_mask,
    mask_to_gate,
    merge_tp_gates,
)


@contextmanager
def _assert_raises(exc):
    """Minimal stand-in for pytest.raises (bazel py_test has no pytest)."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


H, KVH, D = 8, 8, 128
HD = H * D
NLAYERS = 4


def _seed():
    torch.manual_seed(1234)


def _layer(layer_idx, output, kv_len):
    return AttentionLayerRecord(
        layer_idx=layer_idx,
        output=output,
    )


def _fwd(impl, layer_outputs, kv_len, scenario_base, phase="decode_0"):
    lrs = {li: _layer(li, out, kv_len) for li, out in layer_outputs.items()}
    if isinstance(kv_len, torch.Tensor):
        sequence_lengths = kv_len.to(dtype=torch.int32)
    elif isinstance(kv_len, (list, tuple)):
        sequence_lengths = torch.tensor(kv_len, dtype=torch.int32)
    else:
        sequence_lengths = torch.tensor([kv_len], dtype=torch.int32)
    return AttentionForwardRecord(
        impl_name=impl,
        phase=phase,
        layer_records=lrs,
        head_num=H,
        kv_head_num=KVH,
        head_dim=D,
        dtype=torch.bfloat16,
        is_prefill=False,
        sequence_lengths=sequence_lengths,
        input_lengths=torch.ones_like(sequence_lengths),
        prefix_lengths=torch.empty(0, dtype=torch.int32),
        cu_seqlens=torch.zeros(sequence_lengths.numel() + 1, dtype=torch.int32),
        active_sequence_ids=tuple(range(sequence_lengths.numel())),
    )


def _golden_layers(batch_size=1):
    out = {}
    for li in range(NLAYERS):
        t = torch.randn(batch_size, HD)
        out[li] = (t + torch.sign(t) * 0.5).to(torch.bfloat16)
    return out


def _add_noise(t, rel):
    f = t.float()
    rms = f.pow(2).mean().sqrt()
    return (f + torch.randn_like(f) * rms * rel).to(t.dtype)


def _scale(t, factor):
    return (t.float() * factor).to(t.dtype)


def _page_corrupt(t, n=64):
    out = t.clone().float()
    out[0, :n] = torch.randn(n) * 5.0
    return out.to(t.dtype)


def _head_collapse(t, head=3):
    out = t.clone().float().reshape(1, H, D)
    rms = t.float().pow(2).mean().sqrt()
    out[0, head] = torch.randn(D) * rms * 3.0
    return out.reshape(1, HD).to(t.dtype)


def _ulp_heavy_golden():
    """1/8 large elements carry the signal; 7/8 tiny elements carry the ULP."""
    t = torch.full((1, HD), 1e-3)
    t[0, : HD // 8] = 1.0
    return t.to(torch.bfloat16)


def _ulp_spike(t, factor=1.4):
    """Relative error on tiny elements only: mean_ulp blows past 25 while the
    signal-weighted metrics (cos/nrmse) and the absolute error stay clean."""
    out = t.clone().float()
    tiny = out.abs() < 0.5
    out[tiny] = out[tiny] * factor
    return out.to(t.dtype)


def _low_snr_golden_layers(magnitude=0.05):
    """Uniform tiny-magnitude outputs (V cancellation shape from SM100 logs)."""
    out = {}
    for li in range(NLAYERS):
        out[li] = (torch.sign(torch.randn(1, HD)) * magnitude).to(torch.bfloat16)
    return out


def _add_abs_noise(t, abs_rms):
    f = t.float()
    return (f + torch.randn_like(f) * abs_rms).to(t.dtype)


def _clean_candidate(
    golden_layers, kv_len, scenario, impl, rel=0.003, phase="decode_0"
):
    return _fwd(
        impl,
        {li: _add_noise(g, rel) for li, g in golden_layers.items()},
        kv_len,
        scenario,
        phase=phase,
    )


def _records_one_scenario(golden_layers, kv_len, scenario, cand_map):
    bucket = {GOLDEN: [_fwd(GOLDEN, golden_layers, kv_len, scenario)]}
    for impl, rec in cand_map.items():
        bucket[impl] = [rec]
    return {scenario: bucket}


def _manifest(records, qualifying_by_scenario=None):
    qualifying_by_scenario = qualifying_by_scenario or {}
    manifest = {}
    for scenario, bucket in records.items():
        candidates = tuple(name for name in bucket if name != GOLDEN)
        for golden in bucket[GOLDEN]:
            manifest[(scenario, golden.phase)] = {
                "active_sequence_ids": golden.active_sequence_ids,
                "expected_candidates": candidates,
                "gate_qualifying": qualifying_by_scenario.get(scenario, True),
            }
    return manifest


def test_clean_candidate_passes():
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "p0_c512_d1", "XQADecodeImpl")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"XQADecodeImpl": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"XQADecodeImpl"})
    assert "XQADecodeImpl" in res.verified
    assert isinstance(res.passed, frozenset)


def test_identical_candidate_passes():
    _seed()
    gl = _golden_layers()
    cand = _fwd("XQAImpl", {li: g.clone() for li, g in gl.items()}, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"XQAImpl": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"XQAImpl"})


def test_decode_fork_input_length_may_differ_from_golden():
    _seed()
    gl = _golden_layers()
    golden = _fwd(GOLDEN, gl, 514, "scenario", phase="decode_0")
    candidate = _fwd(
        "XQAImpl",
        {layer_idx: output.clone() for layer_idx, output in gl.items()},
        514,
        "scenario",
        phase="decode_0",
    )
    golden.input_lengths = torch.tensor([512], dtype=torch.int32)
    candidate.input_lengths = torch.tensor([514], dtype=torch.int32)
    records = {"scenario": {GOLDEN: [golden], "XQAImpl": [candidate]}}
    result = build_decode_gate(records, "BASE")
    assert result.verified == frozenset({"XQAImpl"})
    assert result.passed == frozenset({"XQAImpl"})


def test_scale_error_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _scale(g, 1.05) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert "Buggy" in res.verified
    assert "Buggy" in res.failures()


def test_page_corruption_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _page_corrupt(g) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    assert build_decode_gate(recs, "BASE").passed == frozenset()


def test_single_head_collapse_fails():
    _seed()
    gl = _golden_layers()
    cand = _fwd(
        "Buggy", {li: _head_collapse(g) for li, g in gl.items()}, 512, "p0_c512_d1"
    )
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    assert build_decode_gate(recs, "BASE").passed == frozenset()


def test_and_across_layers():
    _seed()
    gl = _golden_layers()
    layers = {li: _add_noise(g, 0.003) for li, g in gl.items()}
    layers[2] = _scale(gl[2], 1.05)
    cand = _fwd("Buggy", layers, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Buggy": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    bad_layers = {v.layer_idx for v in res.failures()["Buggy"]}
    assert bad_layers == {2}


def test_and_across_kv_scenarios():
    _seed()
    gl_a = _golden_layers()
    gl_b = _golden_layers()
    cand_a = _clean_candidate(gl_a, 512, "p0_c512_d1", "X")
    cand_b = _fwd(
        "X", {li: _scale(g, 1.05) for li, g in gl_b.items()}, 32768, "p0_c32768_d1"
    )
    recs = {}
    recs.update(_records_one_scenario(gl_a, 512, "p0_c512_d1", {"X": cand_a}))
    recs.update(_records_one_scenario(gl_b, 32768, "p0_c32768_d1", {"X": cand_b}))
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    fail_scenarios = {v.scenario for v in res.failures()["X"]}
    assert "p0_c32768_d1" in fail_scenarios


def test_clean_across_two_kv_passes():
    _seed()
    gl_a, gl_b = _golden_layers(), _golden_layers()
    recs = {}
    recs.update(
        _records_one_scenario(
            gl_a,
            512,
            "p0_c512_d1",
            {"X": _clean_candidate(gl_a, 512, "p0_c512_d1", "X")},
        )
    )
    recs.update(
        _records_one_scenario(
            gl_b,
            32768,
            "p0_c32768_d1",
            {"X": _clean_candidate(gl_b, 32768, "p0_c32768_d1", "X")},
        )
    )
    assert build_decode_gate(recs, "BASE").passed == frozenset({"X"})


def test_decode_step_phases_are_all_judged():
    _seed()
    scenario = "b2__s0_p512c0d2__s1_p256c0d1"
    gl0, gl1 = _golden_layers(batch_size=2), _golden_layers(batch_size=1)
    good0 = _clean_candidate(gl0, [513, 257], scenario, "X", phase="decode_0")
    bad1 = _fwd(
        "X",
        {li: _scale(g, 1.05) for li, g in gl1.items()},
        [514],
        scenario,
        phase="decode_1",
    )
    recs = {
        scenario: {
            GOLDEN: [
                _fwd(GOLDEN, gl0, [513, 257], scenario, phase="decode_0"),
                _fwd(GOLDEN, gl1, [514], scenario, phase="decode_1"),
            ],
            "X": [good0, bad1],
        }
    }
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert "X" in res.verified
    assert {v.phase for v in res.detail["X"]} == {"decode_0", "decode_1"}
    assert {v.phase for v in res.failures()["X"]} == {"decode_1"}


def test_missing_decode_step_in_candidate_fails():
    _seed()
    scenario = "b2__s0_p512c0d2__s1_p256c0d1"
    gl0, gl1 = _golden_layers(batch_size=2), _golden_layers(batch_size=1)
    recs = {
        scenario: {
            GOLDEN: [
                _fwd(GOLDEN, gl0, [513, 257], scenario, phase="decode_0"),
                _fwd(GOLDEN, gl1, [514], scenario, phase="decode_1"),
            ],
            "X": [_clean_candidate(gl0, [513, 257], scenario, "X", phase="decode_0")],
        }
    }
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert "X" not in res.verified
    missing = [v for v in res.failures()["X"] if v.phase == "decode_1"]
    assert missing
    assert all("missing decode phase" in v.fail_reason for v in missing)


def test_multibatch_decode_record_uses_outer_scenario_and_max_kv_len():
    _seed()
    scenario = "b3__s0_p512c0d1__s1_p1024c0d1__s2_p768c0d1"
    kv_lens = [513, 1025, 769]
    gl = _golden_layers(batch_size=3)
    recs = {
        scenario: {
            GOLDEN: [_fwd(GOLDEN, gl, kv_lens, scenario, phase="decode_0")],
            "X": [_clean_candidate(gl, kv_lens, scenario, "X", phase="decode_0")],
        }
    }
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"X"})
    assert res.detail["X"][0].scenario == scenario
    assert res.detail["X"][0].kv_len == max(kv_lens)


def test_unverified_backend_excluded():
    _seed()
    gl = _golden_layers()
    plain_only = AttentionForwardRecord(
        impl_name="PlainOnly",
        phase="plain",
        layer_records={0: _layer(0, gl[0].clone(), 512)},
        head_num=H,
        kv_head_num=KVH,
        head_dim=D,
        dtype=torch.bfloat16,
    )
    recs = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _clean_candidate(gl, 512, "p0_c512_d1", "X")}
    )
    recs["p0_c512_d1"]["PlainOnly"] = [plain_only]
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"X"})
    assert "PlainOnly" not in res.verified
    assert "PlainOnly" not in res.detail


def test_legacy_bare_decode_phase_is_not_accepted():
    _seed()
    gl = _golden_layers()
    recs = {
        "p0_c512_d1": {
            GOLDEN: [_fwd(GOLDEN, gl, 512, "p0_c512_d1", phase="decode")],
            "X": [_clean_candidate(gl, 512, "p0_c512_d1", "X", phase="decode")],
        }
    }
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert res.verified == frozenset()
    assert res.detail == {}


def test_missing_layer_in_candidate_fails():
    _seed()
    gl = _golden_layers()
    layers = {li: _add_noise(g, 0.003) for li, g in gl.items()}
    del layers[3]
    cand = _fwd("X", layers, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"X": cand})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    reasons = [v.fail_reason for v in res.failures()["X"] if v.layer_idx == 3]
    assert reasons and "missing layer" in reasons[0]
    assert "X" not in res.verified


def test_extra_layer_and_layer_idx_mismatch_are_not_verified():
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "p0_c512_d1", "X")
    cand.layer_records[NLAYERS] = _layer(
        NLAYERS, cand.layer_records[0].output.clone(), 512
    )
    assert (
        "X"
        not in build_decode_gate(
            _records_one_scenario(gl, 512, "p0_c512_d1", {"X": cand}), "BASE"
        ).verified
    )

    cand = _clean_candidate(gl, 512, "p0_c512_d1", "X")
    cand.layer_records[2].layer_idx = 99
    assert (
        "X"
        not in build_decode_gate(
            _records_one_scenario(gl, 512, "p0_c512_d1", {"X": cand}), "BASE"
        ).verified
    )


def test_shape_dtype_and_nonfinite_mismatch_fail_closed():
    _seed()
    gl = _golden_layers()
    mutations = [
        lambda t: t[:, :-1],
        lambda t: t.float(),
        lambda t: t.clone().fill_(float("nan")),
        lambda t: t.clone().fill_(float("inf")),
    ]
    for mutate in mutations:
        cand = _clean_candidate(gl, 512, "p0_c512_d1", "X")
        cand.layer_records[1].output = mutate(cand.layer_records[1].output)
        res = build_decode_gate(
            _records_one_scenario(gl, 512, "p0_c512_d1", {"X": cand}), "BASE"
        )
        assert res.passed == frozenset()
        assert "X" not in res.verified


def _assert_absolute_schema_rejected(records, reason):
    result, error = _build_decode_gate_strict(records, "BASE")
    assert result.passed == frozenset()
    assert result.verified == frozenset()
    assert error is not None and reason in error


def test_identically_wrong_output_width_is_rejected():
    _seed()
    scenario = "p0_c512_d1"
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, scenario, "X")
    records = _records_one_scenario(gl, 512, scenario, {"X": cand})
    for impl in (GOLDEN, "X"):
        for layer in records[scenario][impl][0].layer_records.values():
            layer.output = layer.output[:, :-1]
    _assert_absolute_schema_rejected(records, "output width differs from head metadata")


def test_identically_wrong_output_row_count_is_rejected():
    _seed()
    scenario = "b2__s0_p512c0d1__s1_p256c0d1"
    gl = _golden_layers(batch_size=2)
    cand = _clean_candidate(gl, [513, 257], scenario, "X")
    records = _records_one_scenario(gl, [513, 257], scenario, {"X": cand})
    for impl in (GOLDEN, "X"):
        for layer in records[scenario][impl][0].layer_records.values():
            layer.output = layer.output[:1]
    _assert_absolute_schema_rejected(
        records, "output row count differs from active batch size"
    )


def test_zero_head_metadata_is_rejected():
    _seed()
    scenario = "p0_c512_d1"
    for field_name in ("head_num", "kv_head_num", "head_dim"):
        gl = _golden_layers()
        cand = _clean_candidate(gl, 512, scenario, "X")
        records = _records_one_scenario(gl, 512, scenario, {"X": cand})
        for impl in (GOLDEN, "X"):
            setattr(records[scenario][impl][0], field_name, 0)
        _assert_absolute_schema_rejected(records, "head metadata must be positive")


def test_decode_length_tensor_layout_is_strict():
    _seed()
    scenario = "b2__s0_p512c0d1__s1_p256c0d1"
    malformed = {
        "sequence_lengths": torch.tensor([513], dtype=torch.int32),
        "input_lengths": torch.tensor([1], dtype=torch.int32),
        "prefix_lengths": torch.tensor([0], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
    }
    expected_reasons = {
        "sequence_lengths": "sequence_lengths count differs",
        "input_lengths": "input_lengths count differs",
        "prefix_lengths": "prefix_lengths must be empty for decode",
        "cu_seqlens": "cu_seqlens count differs",
    }
    for field_name, value in malformed.items():
        gl = _golden_layers(batch_size=2)
        cand = _clean_candidate(gl, [513, 257], scenario, "X")
        records = _records_one_scenario(gl, [513, 257], scenario, {"X": cand})
        for impl in (GOLDEN, "X"):
            setattr(records[scenario][impl][0], field_name, value.clone())
        _assert_absolute_schema_rejected(records, expected_reasons[field_name])


def test_active_sequence_ids_mismatch_is_not_verified():
    _seed()
    gl = _golden_layers(batch_size=2)
    cand = _clean_candidate(gl, [513, 257], "scenario", "X")
    cand.active_sequence_ids = (1, 0)
    res = build_decode_gate(
        _records_one_scenario(gl, [513, 257], "scenario", {"X": cand}), "BASE"
    )
    assert res.passed == frozenset()
    assert "X" not in res.verified


def test_duplicate_candidate_phase_and_decode_gap_are_not_verified():
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "scenario", "X")
    recs = _records_one_scenario(gl, 512, "scenario", {"X": cand})
    recs["scenario"]["X"].append(cand)
    assert "X" not in build_decode_gate(recs, "BASE").verified

    recs = {
        "scenario": {
            GOLDEN: [_fwd(GOLDEN, gl, 512, "scenario", phase="decode_1")],
            "X": [_fwd("X", gl, 512, "scenario", phase="decode_1")],
        }
    }
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert res.verified == frozenset()


def test_fp8_dtype_selects_loose_threshold():
    _seed()
    gl = _golden_layers()
    fp8_noise = {li: _add_noise(g, 0.05) for li, g in gl.items()}

    def fresh_recs():
        return _records_one_scenario(
            gl, 512, "p0_c512_d1", {"X": _fwd("X", dict(fp8_noise), 512, "p0_c512_d1")}
        )

    assert build_decode_gate(fresh_recs(), "BASE").passed == frozenset()
    assert build_decode_gate(fresh_recs(), "FP8").passed == frozenset({"X"})


def test_failed_verdict_reports_absolute_scale_diagnostics():
    _seed()
    gl = _golden_layers()
    bad = {li: _scale(g, 2.0) for li, g in gl.items()}
    records = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _fwd("X", bad, 512, "p0_c512_d1")}
    )

    failures = build_decode_gate(records, "FP8").failures()["X"]
    assert failures
    # A 2x scale error lands in LOW_SNR (snr=1) where cos is exempted, yet the
    # scale-invariant nrmse fence keeps it out of the allowlist.
    assert all(v.nrmse > v.nrmse_threshold for v in failures)
    for field in (
        "rms_abs=",
        "ref_rms=",
        "snr=",
        "regime=",
        "pass_abs=",
    ):
        assert all(field in verdict.fail_reason for verdict in failures)


def test_fp8_ulp_anomaly_is_observational_not_gating():
    """Cross-precision mean ULP tracks the platform quantization scheme, so it
    is recorded on the verdict but never gates the FP8 judgment."""
    _seed()
    scenario = "qualifying"
    gl = _golden_layers()
    gl[0] = _ulp_heavy_golden()
    layers = {li: output.clone() for li, output in gl.items()}
    layers[0] = _ulp_spike(gl[0])
    records = _records_one_scenario(
        gl, 512, scenario, {"X": _fwd("X", layers, 512, scenario)}
    )

    result, error = _build_decode_gate_strict(
        records, "FP8", manifest=_manifest(records)
    )
    assert error is None
    assert result.verified == frozenset({"X"})
    assert result.passed == frozenset({"X"})
    anomaly = [v for v in result.detail["X"] if v.layer_idx == 0][0]
    assert anomaly.overall_pass
    assert anomaly.mean_ulp > anomaly.mean_ulp_threshold


def test_base_rejects_single_metric_ulp_failure_without_soft_outlier_budget():
    """Same-precision BF16 keeps mean_ulp as a hard per-element gate."""
    _seed()
    scenario = "qualifying"
    gl = _golden_layers()
    gl[0] = _ulp_heavy_golden()
    layers = {li: output.clone() for li, output in gl.items()}
    layers[0] = _ulp_spike(gl[0], factor=1.04)
    records = _records_one_scenario(
        gl, 512, scenario, {"X": _fwd("X", layers, 512, scenario)}
    )

    result, error = _build_decode_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert error is None
    assert result.verified == frozenset({"X"})
    assert result.passed == frozenset()
    failures = result.failures()["X"]
    assert len(failures) == 1
    assert failures[0].gate_qualifying
    assert failures[0].snr_regime == "HIGH_SNR"
    assert failures[0].cos_sim >= failures[0].cos_threshold
    assert failures[0].nrmse <= failures[0].nrmse_threshold
    assert failures[0].mean_ulp > failures[0].mean_ulp_threshold


def test_fp8_rejects_all_qualifying_metric_failures():
    _seed()
    scenario = "qualifying"
    gl = _golden_layers()
    two_soft = {li: output.clone() for li, output in gl.items()}
    two_soft[0] = _scale(gl[0], 1.30)
    two_soft[1] = _scale(gl[1], 1.30)
    records = _records_one_scenario(
        gl, 512, scenario, {"X": _fwd("X", two_soft, 512, scenario)}
    )
    result, error = _build_decode_gate_strict(
        records, "FP8", manifest=_manifest(records)
    )
    assert error is None
    assert result.passed == frozenset()
    assert len(result.failures()["X"]) == 2

    hard = {li: output.clone() for li, output in gl.items()}
    hard[0] = _scale(gl[0], 1.40)
    records = _records_one_scenario(
        gl, 512, scenario, {"X": _fwd("X", hard, 512, scenario)}
    )
    result, error = _build_decode_gate_strict(
        records, "FP8", manifest=_manifest(records)
    )
    assert error is None
    assert result.passed == frozenset()
    hard_failure = result.failures()["X"][0]
    assert hard_failure.nrmse > hard_failure.nrmse_threshold

    for corrupt in (
        lambda output: torch.zeros_like(output),
        lambda output: -output,
        _page_corrupt,
        lambda output: output.clone().fill_(float("nan")),
        lambda output: output.clone().fill_(float("inf")),
    ):
        corrupted = {li: output.clone() for li, output in gl.items()}
        corrupted[0] = corrupt(gl[0])
        records = _records_one_scenario(
            gl,
            512,
            scenario,
            {"X": _fwd("X", corrupted, 512, scenario)},
        )
        result, error = _build_decode_gate_strict(
            records, "FP8", manifest=_manifest(records)
        )
        assert error is None
        assert result.passed == frozenset()


def test_fp8_low_snr_cos_shortfall_with_clean_abs_passes():
    """SM100 replay shape: tiny-norm layers where cos<0.998 is
    mathematically forced (cos ~ snr/sqrt(1+snr^2), snr~8) must pass when the
    absolute error, nrmse and mean_ulp are all clean."""
    _seed()
    gl = _low_snr_golden_layers()
    noisy = {li: _add_abs_noise(g, 0.006) for li, g in gl.items()}
    records = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _fwd("X", noisy, 512, "p0_c512_d1")}
    )
    res = build_decode_gate(records, "FP8")
    assert res.passed == frozenset({"X"})
    verdicts = res.detail["X"]
    assert all(v.snr_regime == "LOW_SNR" for v in verdicts)
    assert any(v.cos_sim < v.cos_threshold for v in verdicts)


def test_fp8_low_snr_abs_error_above_floor_still_fails():
    _seed()
    gl = _low_snr_golden_layers()
    blown = {li: _add_abs_noise(g, 0.9) for li, g in gl.items()}
    records = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _fwd("X", blown, 512, "p0_c512_d1")}
    )
    res = build_decode_gate(records, "FP8")
    assert res.passed == frozenset()
    failures = res.failures()["X"]
    assert failures
    assert all("in LOW_SNR" in v.fail_reason for v in failures)


def test_nonqualifying_failure_is_diagnostic_only():
    _seed()
    qualifying_scenario = "qualifying"
    diagnostic_scenario = "diagnostic"
    gl_good = _golden_layers()
    gl_bad = _golden_layers()
    good = _fwd(
        "X",
        {li: output.clone() for li, output in gl_good.items()},
        512,
        qualifying_scenario,
    )
    bad = _fwd(
        "X",
        {li: _scale(output, 1.40) for li, output in gl_bad.items()},
        1024,
        diagnostic_scenario,
    )
    records = {}
    records.update(
        _records_one_scenario(gl_good, 512, qualifying_scenario, {"X": good})
    )
    records.update(_records_one_scenario(gl_bad, 1024, diagnostic_scenario, {"X": bad}))
    result, error = _build_decode_gate_strict(
        records,
        "FP8",
        manifest=_manifest(records, {diagnostic_scenario: False}),
    )
    assert error is None
    assert result.verified == frozenset({"X"})
    assert result.passed == frozenset({"X"})
    diagnostics = result.failures()["X"]
    assert diagnostics
    assert all(not verdict.gate_qualifying for verdict in diagnostics)


def test_multiple_candidates_partition():
    _seed()
    gl = _golden_layers()
    good = _clean_candidate(gl, 512, "p0_c512_d1", "Good")
    bad = _fwd("Bad", {li: _scale(g, 1.05) for li, g in gl.items()}, 512, "p0_c512_d1")
    recs = _records_one_scenario(gl, 512, "p0_c512_d1", {"Good": good, "Bad": bad})
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset({"Good"})
    assert res.verified == frozenset({"Good", "Bad"})


def test_merge_tp_gates_intersection():
    assert merge_tp_gates([frozenset({"A", "B"}), frozenset({"B", "C"})]) == frozenset(
        {"B"}
    )
    assert merge_tp_gates([frozenset({"A", "B"}), frozenset({"A", "B"})]) == frozenset(
        {"A", "B"}
    )
    assert merge_tp_gates([frozenset({"A"}), frozenset()]) == frozenset()
    assert merge_tp_gates([]) == frozenset()
    assert merge_tp_gates([frozenset({"A", "B"})]) == frozenset({"A", "B"})


def test_empty_and_golden_missing():
    assert build_decode_gate({}, "BASE").passed == frozenset()
    _seed()
    gl = _golden_layers()
    cand = _clean_candidate(gl, 512, "p0_c512_d1", "X")
    recs = {"p0_c512_d1": {"X": [cand]}}
    res = build_decode_gate(recs, "BASE")
    assert res.passed == frozenset()
    assert res.verified == frozenset()


def test_normalize_kv_dtype():
    assert _normalize_kv_dtype("BASE") == "BASE"
    assert _normalize_kv_dtype("BF16") == "BASE"
    assert _normalize_kv_dtype("FP8") == "FP8"
    assert _normalize_kv_dtype("fp8_e4m3") == "FP8"
    with _assert_raises(NotImplementedError):
        _normalize_kv_dtype("INT8")
    with _assert_raises(ValueError):
        _normalize_kv_dtype("weird")

    class _Enum:
        name = "FP8"

    assert _normalize_kv_dtype(_Enum()) == "FP8"


def test_detail_collected_for_all_layers_even_on_pass():
    _seed()
    gl = _golden_layers()
    recs = _records_one_scenario(
        gl, 512, "p0_c512_d1", {"X": _clean_candidate(gl, 512, "p0_c512_d1", "X")}
    )
    res = build_decode_gate(recs, "BASE")
    assert len(res.detail["X"]) == NLAYERS
    assert all(v.overall_pass for v in res.detail["X"])
    assert res.failures() == {}


# ─── Prefill gate: plain/prefix strict judgment on the shared builder ───


def _prefill_fwd(impl, layer_outputs, scenario, phase, input_lengths, prefix_lengths):
    input_lengths_t = torch.tensor(input_lengths, dtype=torch.int32)
    prefix_lengths_t = torch.tensor(prefix_lengths, dtype=torch.int32)
    cu_seqlens = torch.zeros(len(input_lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = input_lengths_t.cumsum(0)
    lrs = {li: _layer(li, out, 0) for li, out in layer_outputs.items()}
    return AttentionForwardRecord(
        impl_name=impl,
        phase=phase,
        layer_records=lrs,
        head_num=H,
        kv_head_num=KVH,
        head_dim=D,
        dtype=torch.bfloat16,
        is_prefill=True,
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        input_lengths=input_lengths_t,
        prefix_lengths=prefix_lengths_t,
        cu_seqlens=cu_seqlens,
        active_sequence_ids=tuple(range(len(input_lengths))),
    )


def _prefill_records(scenario, phases, impl="P", corrupt_phase=None):
    """phases: {phase: (input_lengths, prefix_lengths)}; corrupt_phase scales the candidate."""
    _seed()
    bucket = {GOLDEN: [], impl: []}
    goldens = {}
    for phase, (input_lengths, prefix_lengths) in phases.items():
        total_tokens = int(sum(input_lengths))
        gl = _golden_layers(batch_size=total_tokens)
        goldens[phase] = gl
        bucket[GOLDEN].append(
            _prefill_fwd(GOLDEN, gl, scenario, phase, input_lengths, prefix_lengths)
        )
        if corrupt_phase == phase:
            outputs = {li: _scale(g, 1.05) for li, g in gl.items()}
        else:
            outputs = {li: _add_noise(g, 0.003) for li, g in gl.items()}
        bucket[impl].append(
            _prefill_fwd(impl, outputs, scenario, phase, input_lengths, prefix_lengths)
        )
    return {scenario: bucket}, goldens


_PLAIN_PREFIX_PHASES = {
    "plain": ([128, 64], [0, 0]),
    "prefix": ([64, 32], [512, 256]),
}


def test_prefill_plain_and_prefix_all_pass():
    records, _ = _prefill_records("scenario", _PLAIN_PREFIX_PHASES)
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert error is None
    assert result.verified == frozenset({"P"})
    assert result.passed == frozenset({"P"})
    assert {v.phase for v in result.detail["P"]} == {"plain", "prefix"}


def test_prefill_single_phase_failure_excludes_backend():
    records, _ = _prefill_records(
        "scenario", _PLAIN_PREFIX_PHASES, corrupt_phase="prefix"
    )
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert error is None
    assert result.verified == frozenset({"P"})
    assert result.passed == frozenset()
    assert {v.phase for v in result.failures()["P"]} == {"prefix"}


def test_prefill_missing_applicable_phase_blocks_pass():
    records, _ = _prefill_records("scenario", _PLAIN_PREFIX_PHASES)
    records["scenario"]["P"] = [
        record for record in records["scenario"]["P"] if record.phase == "plain"
    ]
    manifest = _manifest(records)
    manifest[("scenario", "prefix")]["expected_candidates"] = ("P",)
    result, error = _build_prefill_gate_strict(records, "BASE", manifest=manifest)
    assert error is None
    assert result.verified == frozenset()
    assert result.passed == frozenset()
    missing = [v for v in result.failures()["P"] if v.phase == "prefix"]
    assert missing and "missing prefill phase" in missing[0].fail_reason


def test_prefill_plain_only_scenario_passes():
    records, _ = _prefill_records("scenario", {"plain": ([128, 64], [0, 0])})
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert error is None
    assert result.passed == frozenset({"P"})


def test_prefill_low_snr_scale_error_still_rejected_by_nrmse():
    records, goldens = _prefill_records("scenario", {"plain": ([128], [0])})
    layers = {li: g.clone() for li, g in goldens["plain"].items()}
    layers[0] = _scale(goldens["plain"][0], 1.30)
    records["scenario"]["P"] = [
        _prefill_fwd("P", layers, "scenario", "plain", [128], [0])
    ]
    result, error = _build_prefill_gate_strict(
        records, "FP8", manifest=_manifest(records)
    )
    assert error is None
    assert result.passed == frozenset()
    failure = result.failures()["P"][0]
    assert failure.snr_regime == "LOW_SNR"
    assert failure.nrmse > failure.nrmse_threshold
    assert "cos_sim=" not in failure.fail_reason


def test_prefill_gate_ignores_decode_records_and_vice_versa():
    _seed()
    records, _ = _prefill_records("scenario", {"plain": ([128], [0])})
    gl_decode = _golden_layers()
    records["scenario"][GOLDEN].append(
        _fwd(GOLDEN, gl_decode, 512, "scenario", phase="decode_0")
    )
    records["scenario"]["D"] = [
        _clean_candidate(gl_decode, 512, "scenario", "D", phase="decode_0")
    ]
    prefill_manifest = {
        ("scenario", "plain"): {
            "active_sequence_ids": (0,),
            "expected_candidates": ("P",),
            "gate_qualifying": True,
        }
    }
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=prefill_manifest
    )
    assert error is None
    assert result.passed == frozenset({"P"})
    assert "D" not in result.detail

    decode_manifest = {
        ("scenario", "decode_0"): {
            "active_sequence_ids": (0,),
            "expected_candidates": ("D",),
            "gate_qualifying": True,
        }
    }
    decode_result, decode_error = _build_decode_gate_strict(
        records, "BASE", manifest=decode_manifest
    )
    assert decode_error is None
    assert decode_result.passed == frozenset({"D"})
    assert "P" not in decode_result.detail


def test_prefill_nonqualifying_only_is_not_verified_or_passed():
    records, _ = _prefill_records("diagnostic", {"plain": ([128], [0])})
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records, {"diagnostic": False})
    )
    assert error is None
    assert result.verified == frozenset()
    assert result.passed == frozenset()
    assert all(not v.gate_qualifying for v in result.detail["P"])


def test_prefill_manifest_rejects_decode_phase_key():
    records, _ = _prefill_records("scenario", {"plain": ([128], [0])})
    manifest = _manifest(records)
    manifest[("scenario", "decode_0")] = {
        "active_sequence_ids": (0,),
        "expected_candidates": (),
        "gate_qualifying": True,
    }
    result, error = _build_prefill_gate_strict(records, "BASE", manifest=manifest)
    assert result.passed == frozenset()
    assert error is not None and "outside prefill domain" in error


def test_prefill_structure_contract_is_strict():
    def fresh():
        return _prefill_records("scenario", _PLAIN_PREFIX_PHASES)

    # golden+candidate violations reject the whole schema
    records, _ = fresh()
    for impl in (GOLDEN, "P"):
        for record in records["scenario"][impl]:
            record.is_prefill = False
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert result.passed == frozenset()
    assert error is not None and "marked as decode" in error

    records, _ = fresh()
    for impl in (GOLDEN, "P"):
        for record in records["scenario"][impl]:
            if record.phase == "prefix":
                record.prefix_lengths = torch.zeros(2, dtype=torch.int32)
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert result.passed == frozenset()
    assert error is not None and "prefix_lengths must be positive" in error

    records, _ = fresh()
    for impl in (GOLDEN, "P"):
        for record in records["scenario"][impl]:
            if record.phase == "plain":
                record.cu_seqlens = record.cu_seqlens.flip(0)
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert result.passed == frozenset()
    assert error is not None and "prefix sum of input_lengths" in error

    # candidate-only violation stays local: not verified, golden intact
    records, _ = fresh()
    for record in records["scenario"]["P"]:
        if record.phase == "plain":
            for layer in record.layer_records.values():
                layer.output = layer.output[:-1]
    result, error = _build_prefill_gate_strict(
        records, "BASE", manifest=_manifest(records)
    )
    assert error is None
    assert result.verified == frozenset()
    assert result.passed == frozenset()
    reasons = [v.fail_reason for v in result.failures()["P"] if v.phase == "plain"]
    assert reasons and "total token count" in reasons[0]


# ─── CPU-only bitmask helpers for the pre-landed TP integration ───


def test_gate_to_mask_position_is_identity():
    reg = ["XQAImpl", "XQADecodeImpl", "PyFlashinferDecodeImpl"]
    assert gate_to_mask(frozenset({"XQAImpl", "PyFlashinferDecodeImpl"}), reg) == [
        1,
        0,
        1,
    ]
    assert gate_to_mask(frozenset(), reg) == [0, 0, 0]


def test_mask_to_gate_intersection_and_asym():
    reg = ["A", "B", "C"]
    tp = 2
    # A: passed both ranks (2,2) -> selected; B: only 1 rank passed but both ranks verified (1,2) -> not in intersection, no alert (verified is complete);
    # C: only 1 rank verified (0,1) -> asymmetric verification, alerted and excluded.
    merged, asym = mask_to_gate([2, 1, 0], [2, 2, 1], reg, tp)
    assert merged == frozenset({"A"})
    assert asym == ["C"]


def test_mask_to_gate_all_pass():
    reg = ["A", "B"]
    merged, asym = mask_to_gate([2, 2], [2, 2], reg, 2)
    assert merged == frozenset({"A", "B"})
    assert asym == []


# Bind the module-level test_* functions onto a TestCase so bazel's unittest
# runner (no pytest available) discovers and runs them.
class DecodeGateTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(DecodeGateTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()
