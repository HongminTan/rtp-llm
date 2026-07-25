"""Strict attention-backend precision gates over canonical attention records.

Both the decode domain (phase decode_N over DECODE_MHA_IMPS) and the prefill
domain (phase plain/prefix over PREFILL_MHA_IMPS) share one strict builder.
This module is intentionally collective-free. The recorder builds a local
result and C++ performs the fixed-shape WORLD protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    FP8_QUALITY_THRESHOLDS,
    THRESHOLDS,
    evaluate_fp8_quality,
    evaluate_precision,
)

GOLDEN = "golden"
DECODE = "decode"
PREFILL = "prefill"
PREFILL_PHASES = ("plain", "prefix")
_DECODE_PHASE = re.compile(r"^decode_([0-9]+)$")

# records: scenario_base -> impl_name -> [AttentionForwardRecord]
Records = Dict[str, Dict[str, List[AttentionForwardRecord]]]


@dataclass
class LayerVerdict:
    impl: str
    scenario: str
    phase: str
    layer_idx: int
    kv_len: int
    overall_pass: bool
    cos_sim: float
    nrmse: float
    mean_ulp: float
    snr: float
    snr_regime: str
    cos_threshold: float
    nrmse_threshold: float
    mean_ulp_threshold: float
    fail_reason: str
    gate_qualifying: bool = True


@dataclass
class GateResult:
    passed: frozenset
    detail: Dict[str, List[LayerVerdict]]
    # verified means complete structural coverage of every applicable key.
    verified: frozenset = field(default_factory=frozenset)

    def failures(self) -> Dict[str, List[LayerVerdict]]:
        out: Dict[str, List[LayerVerdict]] = {}
        for impl, verdicts in self.detail.items():
            bad = [verdict for verdict in verdicts if not verdict.overall_pass]
            if bad:
                out[impl] = bad
        return out


@dataclass(frozen=True)
class _ManifestView:
    active_sequence_ids: Tuple[int, ...]
    expected_candidates: Tuple[str, ...]
    gate_qualifying: bool = True


def _normalize_kv_dtype(kv_cache_dtype) -> str:
    name = getattr(kv_cache_dtype, "name", kv_cache_dtype)
    value = str(name).upper()
    if "FP8" in value:
        return "FP8"
    if "INT8" in value:
        raise NotImplementedError(
            "decode gate: INT8 KV cache not supported (only BASE/FP8)"
        )
    if "BASE" in value or "BF16" in value or "BFLOAT16" in value:
        return "BASE"
    raise ValueError(f"decode gate: unrecognized kv_cache_dtype={kv_cache_dtype!r}")


def precision_thresholds(kv_cache_dtype) -> Mapping[str, float]:
    kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
    return FP8_QUALITY_THRESHOLDS if kv_dtype == "FP8" else THRESHOLDS["BF16"]


def _phase_index(phase: str) -> Optional[int]:
    match = _DECODE_PHASE.fullmatch(phase)
    return int(match.group(1)) if match else None


def _is_decode_phase(phase: str) -> bool:
    return _phase_index(phase) is not None


def _phase_in_domain(phase: str, domain: str) -> bool:
    if domain == DECODE:
        return _is_decode_phase(phase)
    return phase in PREFILL_PHASES


def _decode_phase_sort_key(phase: str):
    index = _phase_index(phase)
    return (index if index is not None else 10**9, phase)


def _records_for_phase(
    recs: Sequence[AttentionForwardRecord], phase: str
) -> List[AttentionForwardRecord]:
    return [record for record in recs if record.phase == phase]


def _decode_records_by_phase(
    recs: Sequence[AttentionForwardRecord],
) -> Dict[str, AttentionForwardRecord]:
    """Compatibility helper that excludes duplicate or malformed decode phases."""
    grouped: Dict[str, List[AttentionForwardRecord]] = {}
    for record in recs:
        if _is_decode_phase(record.phase):
            grouped.setdefault(record.phase, []).append(record)
    return {
        phase: records[0]
        for phase, records in sorted(
            grouped.items(), key=lambda item: _decode_phase_sort_key(item[0])
        )
        if len(records) == 1
    }


def _decode_candidate_names(records: Records) -> List[str]:
    names = {
        impl
        for bucket in records.values()
        for impl, recs in bucket.items()
        if impl != GOLDEN and any(_is_decode_phase(record.phase) for record in recs)
    }
    return sorted(names)


def _infer_kv_len(record: AttentionForwardRecord) -> int:
    lengths = record.sequence_lengths
    if lengths is not None and lengths.numel() > 0:
        return int(lengths.flatten().max().item())
    # Prefill records carry an empty sequence_lengths; derive the KV extent
    # from per-sequence input (+prefix) lengths for failure diagnostics.
    inputs = record.input_lengths
    if inputs is not None and inputs.numel() > 0:
        totals = inputs.flatten().to(torch.int64)
        prefixes = record.prefix_lengths
        if prefixes is not None and prefixes.numel() == totals.numel():
            totals = totals + prefixes.flatten().to(torch.int64)
        return int(totals.max().item())
    return 0


def _failure_verdict(
    impl: str,
    scenario: str,
    phase: str,
    reason: str,
    layer_idx: int = -1,
    kv_len: int = 0,
    gate_qualifying: bool = True,
) -> LayerVerdict:
    return LayerVerdict(
        impl=impl,
        scenario=scenario,
        phase=phase,
        layer_idx=layer_idx,
        kv_len=kv_len,
        overall_pass=False,
        cos_sim=float("nan"),
        nrmse=float("nan"),
        mean_ulp=float("nan"),
        snr=float("nan"),
        snr_regime="N/A",
        cos_threshold=float("nan"),
        nrmse_threshold=float("nan"),
        mean_ulp_threshold=float("nan"),
        fail_reason=reason,
        gate_qualifying=gate_qualifying,
    )


def _same_optional_tensor(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and bool(torch.equal(left, right))
    )


def _forward_record_structure_reason(
    record: AttentionForwardRecord,
    expected_layers: Sequence[int],
    expected_active_ids: Sequence[int],
    role: str,
) -> Optional[str]:
    active_ids = tuple(expected_active_ids)
    if record.active_sequence_ids != active_ids:
        return f"{role} active_sequence_ids mismatch manifest"
    if not active_ids:
        return f"{role} active_sequence_ids is empty"
    if len(set(active_ids)) != len(active_ids):
        return f"{role} active_sequence_ids contains duplicates"
    is_decode = _is_decode_phase(record.phase)
    if is_decode:
        if record.is_prefill:
            return f"{role} decode record marked as prefill"
    elif record.phase in PREFILL_PHASES:
        if not record.is_prefill:
            return f"{role} prefill record marked as decode"
    else:
        return f"{role} phase {record.phase!r} is outside the gate domains"
    if record.head_num <= 0 or record.kv_head_num <= 0 or record.head_dim <= 0:
        return f"{role} head metadata must be positive"

    batch_size = len(active_ids)
    if is_decode:
        if record.sequence_lengths is None:
            return f"{role} sequence_lengths is missing"
        for field_name in ("sequence_lengths", "input_lengths"):
            value = getattr(record, field_name)
            if value is None:
                return f"{role} {field_name} is missing"
            if value.numel() != batch_size:
                return f"{role} {field_name} count differs from active batch size"
        if record.prefix_lengths is None:
            return f"{role} prefix_lengths is missing"
        if record.prefix_lengths.numel() != 0:
            return f"{role} prefix_lengths must be empty for decode"
        if record.cu_seqlens is None:
            return f"{role} cu_seqlens is missing"
        if record.cu_seqlens.numel() != batch_size + 1:
            return f"{role} cu_seqlens count differs from active batch size"
        expected_rows = batch_size
        rows_desc = "active batch size"
    else:
        # Prefill layout contract (see PyWrappedModel::buildPyAttentionInputs):
        # sequence_lengths is empty, input/prefix lengths are per-sequence, and
        # cu_seqlens is the query-token prefix sum over the batch.
        if record.sequence_lengths is not None and record.sequence_lengths.numel() != 0:
            return f"{role} sequence_lengths must be empty for prefill"
        if record.input_lengths is None:
            return f"{role} input_lengths is missing"
        if record.input_lengths.numel() != batch_size:
            return f"{role} input_lengths count differs from active batch size"
        input_lengths = record.input_lengths.flatten().to(torch.int64)
        if bool((input_lengths <= 0).any().item()):
            return f"{role} input_lengths must be positive for prefill"
        if record.prefix_lengths is None:
            return f"{role} prefix_lengths is missing"
        if record.prefix_lengths.numel() != batch_size:
            return f"{role} prefix_lengths count differs from active batch size"
        prefix_lengths = record.prefix_lengths.flatten().to(torch.int64)
        if record.phase == "plain" and bool((prefix_lengths != 0).any().item()):
            return f"{role} prefix_lengths must be zero for plain prefill"
        if record.phase == "prefix" and bool((prefix_lengths <= 0).any().item()):
            return f"{role} prefix_lengths must be positive for prefix prefill"
        if record.cu_seqlens is None:
            return f"{role} cu_seqlens is missing"
        if record.cu_seqlens.numel() != batch_size + 1:
            return f"{role} cu_seqlens count differs from active batch size"
        cu_seqlens = record.cu_seqlens.flatten().to(torch.int64)
        if int(cu_seqlens[0].item()) != 0 or not bool(
            torch.equal(cu_seqlens[1:] - cu_seqlens[:-1], input_lengths)
        ):
            return f"{role} cu_seqlens is not the prefix sum of input_lengths"
        expected_rows = int(cu_seqlens[-1].item())
        rows_desc = "total token count"

    expected = set(expected_layers)
    actual = set(record.layer_records)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        return f"{role} missing layer(s)={missing} extra layer(s)={extra}"
    expected_width = record.head_num * record.head_dim
    for layer_idx in expected_layers:
        layer = record.layer_records[layer_idx]
        if layer.layer_idx != layer_idx:
            return f"{role} layer key/index mismatch at layer {layer_idx}"
        if not isinstance(layer.output, torch.Tensor):
            return f"{role} output missing at layer {layer_idx}"
        if layer.output.ndim != 2:
            return f"{role} output rank must be 2 at layer {layer_idx}"
        if layer.output.shape[0] != expected_rows:
            return (
                f"{role} output row count differs from {rows_desc} at layer {layer_idx}"
            )
        if layer.output.shape[1] != expected_width:
            return (
                f"{role} output width differs from head metadata at layer {layer_idx}"
            )
        if record.dtype is not None and layer.output.dtype != record.dtype:
            return f"{role} forward/output dtype mismatch at layer {layer_idx}"
        if not bool(torch.isfinite(layer.output).all().item()):
            return f"{role} output contains non-finite values at layer {layer_idx}"
    return None


def _forward_pair_structure_reason(
    golden: AttentionForwardRecord,
    candidate: AttentionForwardRecord,
    expected_layers: Sequence[int],
    expected_active_ids: Sequence[int],
) -> Optional[str]:
    reason = _forward_record_structure_reason(
        golden, expected_layers, expected_active_ids, "golden"
    )
    if reason is not None:
        return reason
    reason = _forward_record_structure_reason(
        candidate, expected_layers, expected_active_ids, "candidate"
    )
    if reason is not None:
        return reason
    if (
        golden.head_num,
        golden.kv_head_num,
        golden.head_dim,
        golden.dtype,
    ) != (
        candidate.head_num,
        candidate.kv_head_num,
        candidate.head_dim,
        candidate.dtype,
    ):
        return "candidate forward metadata differs from golden"
    compare_fields = ["sequence_lengths", "prefix_lengths", "cu_seqlens"]
    if golden.phase in PREFILL_PHASES:
        # Decode forks may legally carry a different input_lengths than golden
        # (fork history vs prompt); prefill token layout must match exactly.
        compare_fields.insert(1, "input_lengths")
    for field_name in compare_fields:
        if not _same_optional_tensor(
            getattr(golden, field_name), getattr(candidate, field_name)
        ):
            return f"candidate {field_name} differs from golden"

    for layer_idx in expected_layers:
        golden_layer = golden.layer_records[layer_idx]
        candidate_layer = candidate.layer_records[layer_idx]
        if golden_layer.output.shape != candidate_layer.output.shape:
            return f"candidate output shape mismatch at layer {layer_idx}"
        if golden_layer.output.dtype != candidate_layer.output.dtype:
            return f"candidate output dtype mismatch at layer {layer_idx}"
    return None


def judge_layer(
    golden: AttentionForwardRecord,
    cand: AttentionForwardRecord,
    layer_idx: int,
    kv_dtype: str,
    scenario: str,
    gate_qualifying: bool = True,
) -> LayerVerdict:
    candidate_output = cand.layer_records[layer_idx].output
    golden_output = golden.layer_records[layer_idx].output
    if kv_dtype == "FP8":
        metrics = evaluate_fp8_quality(candidate_output, golden_output)
    else:
        metrics = evaluate_precision(candidate_output, golden_output, "BF16")
    fail_reason = metrics["fail_reason"]
    if not metrics["overall_pass"]:
        diagnostics = (
            f"rms_abs={metrics['rms_abs_err']:.6e},"
            f"ref_rms={metrics['ref_rms']:.6e},"
            f"snr={metrics['snr']:.6f},"
            f"regime={metrics['snr_regime']},"
            f"pass_abs={metrics['pass_abs']}"
        )
        fail_reason = f"{fail_reason}; {diagnostics}" if fail_reason else diagnostics
    return LayerVerdict(
        impl=cand.impl_name,
        scenario=scenario,
        phase=cand.phase,
        layer_idx=layer_idx,
        kv_len=_infer_kv_len(golden),
        overall_pass=metrics["overall_pass"],
        cos_sim=metrics["cos_sim"],
        nrmse=metrics["nrmse"],
        mean_ulp=metrics["mean_ulp"],
        snr=metrics["snr"],
        snr_regime=metrics["snr_regime"],
        cos_threshold=metrics["cos_threshold"],
        nrmse_threshold=metrics["nrmse_threshold"],
        mean_ulp_threshold=metrics["mean_ulp_threshold"],
        fail_reason=fail_reason,
        gate_qualifying=gate_qualifying,
    )


def _manifest_value(entry) -> _ManifestView:
    if isinstance(entry, Mapping):
        active_ids = entry.get("active_sequence_ids", ())
        candidates = entry.get("expected_candidates", ())
        gate_qualifying = entry.get("gate_qualifying", True)
    else:
        active_ids = getattr(entry, "active_sequence_ids")
        candidates = getattr(entry, "expected_candidates")
        gate_qualifying = getattr(entry, "gate_qualifying", True)
    return _ManifestView(
        tuple(int(value) for value in active_ids),
        tuple(str(value) for value in (candidates or ())),
        bool(gate_qualifying),
    )


def _infer_manifest(
    records: Records,
) -> Tuple[Optional[Dict[Tuple[str, str], _ManifestView]], Optional[str]]:
    candidate_names = tuple(_decode_candidate_names(records))
    manifest: Dict[Tuple[str, str], _ManifestView] = {}
    for scenario, bucket in records.items():
        grouped: Dict[str, List[AttentionForwardRecord]] = {}
        for record in bucket.get(GOLDEN, []):
            if _is_decode_phase(record.phase):
                grouped.setdefault(record.phase, []).append(record)
        if not grouped:
            continue
        phase_indices = sorted(_phase_index(phase) for phase in grouped)
        if phase_indices != list(range(len(phase_indices))):
            return None, f"golden decode phases are not continuous in {scenario}"
        for phase, phase_records in grouped.items():
            if len(phase_records) != 1:
                return None, f"golden duplicate phase {scenario}/{phase}"
            manifest[(scenario, phase)] = _ManifestView(
                phase_records[0].active_sequence_ids, candidate_names, True
            )
    return manifest, None


def _build_gate_strict(
    records: Records,
    kv_dtype: str,
    manifest_view: Mapping[Tuple[str, str], _ManifestView],
    expected_layer_ids: Optional[Sequence[int]],
    domain: str,
) -> Tuple[GateResult, Optional[str]]:
    for _scenario, phase in manifest_view:
        if not _phase_in_domain(phase, domain):
            return (
                GateResult(frozenset(), {}, frozenset()),
                f"manifest phase outside {domain} domain: {phase}",
            )

    expected_keys = set(manifest_view)
    for scenario, bucket in records.items():
        for impl, recs in bucket.items():
            for record in recs:
                if not _phase_in_domain(record.phase, domain):
                    continue
                key = (scenario, record.phase)
                if key not in expected_keys:
                    return (
                        GateResult(frozenset(), {}, frozenset()),
                        f"{domain} record outside manifest: {scenario}/{record.phase}/{impl}",
                    )
                expected_impls = manifest_view[key].expected_candidates
                if impl != GOLDEN and impl not in expected_impls:
                    return (
                        GateResult(frozenset(), {}, frozenset()),
                        f"non-applicable candidate record: {scenario}/{record.phase}/{impl}",
                    )

    if domain == DECODE:
        for scenario in {scenario for scenario, _phase in expected_keys}:
            indices = sorted(
                _phase_index(phase)
                for key_scenario, phase in expected_keys
                if key_scenario == scenario
            )
            if indices != list(range(len(indices))):
                return (
                    GateResult(frozenset(), {}, frozenset()),
                    f"manifest decode phases are not continuous in {scenario}",
                )

    golden_by_key: Dict[Tuple[str, str], AttentionForwardRecord] = {}
    for key, entry in manifest_view.items():
        scenario, phase = key
        golden_records = _records_for_phase(
            records.get(scenario, {}).get(GOLDEN, []), phase
        )
        if len(golden_records) != 1:
            return (
                GateResult(frozenset(), {}, frozenset()),
                f"golden record count is {len(golden_records)} for {scenario}/{phase}",
            )
        golden = golden_records[0]
        expected_layers = (
            tuple(expected_layer_ids)
            if expected_layer_ids is not None
            else tuple(sorted(golden.layer_records))
        )
        reason = _forward_pair_structure_reason(
            golden, golden, expected_layers, entry.active_sequence_ids
        )
        if reason is not None:
            return (
                GateResult(frozenset(), {}, frozenset()),
                f"golden invalid at {scenario}/{phase}: {reason}",
            )
        golden_by_key[key] = golden

    applicable_keys: Dict[str, List[Tuple[str, str]]] = {}
    for key, entry in manifest_view.items():
        for impl in entry.expected_candidates:
            applicable_keys.setdefault(impl, []).append(key)

    passed = set()
    verified = set()
    detail: Dict[str, List[LayerVerdict]] = {}
    for impl, keys in sorted(applicable_keys.items()):
        complete = True
        precision_failure = False
        qualifying_layers = 0
        for scenario, phase in sorted(keys):
            golden = golden_by_key[(scenario, phase)]
            gate_qualifying = manifest_view[(scenario, phase)].gate_qualifying
            candidates = _records_for_phase(
                records.get(scenario, {}).get(impl, []), phase
            )
            if len(candidates) != 1:
                complete = False
                reason = (
                    f"candidate missing {domain} phase present in manifest"
                    if not candidates
                    else f"candidate duplicate {domain} phase count={len(candidates)}"
                )
                detail.setdefault(impl, []).append(
                    _failure_verdict(
                        impl,
                        scenario,
                        phase,
                        reason,
                        kv_len=_infer_kv_len(golden),
                        gate_qualifying=gate_qualifying,
                    )
                )
                continue
            candidate = candidates[0]
            expected_layers = (
                tuple(expected_layer_ids)
                if expected_layer_ids is not None
                else tuple(sorted(golden.layer_records))
            )
            reason = _forward_pair_structure_reason(
                golden,
                candidate,
                expected_layers,
                manifest_view[(scenario, phase)].active_sequence_ids,
            )
            if reason is not None:
                complete = False
                missing_layers = set(expected_layers) - set(candidate.layer_records)
                detail.setdefault(impl, []).append(
                    _failure_verdict(
                        impl,
                        scenario,
                        phase,
                        reason,
                        layer_idx=min(missing_layers) if missing_layers else -1,
                        kv_len=_infer_kv_len(golden),
                        gate_qualifying=gate_qualifying,
                    )
                )
                continue
            for layer_idx in expected_layers:
                verdict = judge_layer(
                    golden,
                    candidate,
                    layer_idx,
                    kv_dtype,
                    scenario,
                    gate_qualifying,
                )
                detail.setdefault(impl, []).append(verdict)
                if gate_qualifying:
                    qualifying_layers += 1
                    if not verdict.overall_pass:
                        precision_failure = True
        if complete and qualifying_layers > 0:
            verified.add(impl)
            if not precision_failure:
                passed.add(impl)

    return (
        GateResult(
            passed=frozenset(passed),
            detail=detail,
            verified=frozenset(verified),
        ),
        None,
    )


def _build_decode_gate_strict(
    records: Records,
    kv_cache_dtype,
    manifest: Optional[Mapping[Tuple[str, str], object]] = None,
    expected_layer_ids: Optional[Sequence[int]] = None,
) -> Tuple[GateResult, Optional[str]]:
    kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
    if manifest is None:
        normalized_manifest, error = _infer_manifest(records)
        if error is not None:
            return GateResult(frozenset(), {}, frozenset()), error
        manifest_view = normalized_manifest or {}
    else:
        manifest_view = {key: _manifest_value(value) for key, value in manifest.items()}
    return _build_gate_strict(
        records, kv_dtype, manifest_view, expected_layer_ids, DECODE
    )


def _build_prefill_gate_strict(
    records: Records,
    kv_cache_dtype,
    manifest: Mapping[Tuple[str, str], object],
    expected_layer_ids: Optional[Sequence[int]] = None,
) -> Tuple[GateResult, Optional[str]]:
    """Prefill gate over plain/prefix records; the manifest is never inferred."""
    kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
    manifest_view = {key: _manifest_value(value) for key, value in manifest.items()}
    return _build_gate_strict(
        records, kv_dtype, manifest_view, expected_layer_ids, PREFILL
    )


def build_decode_gate(records: Records, kv_cache_dtype) -> GateResult:
    """Compatibility API with strict complete-coverage verified semantics."""
    result, _error = _build_decode_gate_strict(records, kv_cache_dtype)
    return result


def merge_tp_gates(per_rank: Sequence[frozenset]) -> frozenset:
    if not per_rank:
        return frozenset()
    merged = set(per_rank[0])
    for rank_gate in per_rank[1:]:
        merged &= set(rank_gate)
    return frozenset(merged)


def gate_to_mask(values: frozenset, registry: Sequence[str]) -> List[int]:
    return [1 if name in values else 0 for name in registry]


def mask_to_gate(
    passed_sum: Sequence[int],
    verified_sum: Sequence[int],
    registry: Sequence[str],
    tp: int,
):
    merged, asymmetric = set(), []
    for index, name in enumerate(registry):
        if passed_sum[index] == tp and verified_sum[index] == tp:
            merged.add(name)
        elif 0 < verified_sum[index] < tp:
            asymmetric.append(name)
    return frozenset(merged), asymmetric
