"""Pre-landed decode precision-gate logic for dynamic backend dispatch.

build_decode_gate is designed to consume warmup attention records and produce an
allowlist by comparing each candidate with golden outputs across the recorded
scenario / batch / KV coverage and all layers. The warmup recorder and
production dispatcher integration are not connected in this change: current
callers leave gate_passed=None, so runtime selection still uses support-only
filtering.

Intended semantics once the integration is connected:
  - Only decode phases are judged; prefill keeps fixed priority and does not
    enter the gate.
  - Each backend is a global AND over (scenario x decode step x layer); failing
    any recorded decode step at any layer eliminates the backend overall.
  - A backend must be verified at least once to be eligible to PASS; a backend
    with no decode record cannot pass the gate just by "not failing".
  - The output is (passed, detail, verified); the gate only produces an
    allowlist, it does not select a backend or fall back, and the empty set is
    handled by the dispatcher's fallback.
  - Under TP, each rank will run locally and the intersection will be taken
    across ranks.
    The intersection's bitmask encode/decode (gate_to_mask / mask_to_gate) lives
    in this module; the pre-landed NCCL integration helper
    reduce_gate_across_tp lives in backend_selector.

This module makes no direct CUDA or NCCL calls. Tensor calculations run on the
input records' device. Unit tests use CPU tensors, but normal package imports
still require a driver-equipped node because of the surrounding rtp_llm import
chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    evaluate_fp8_quality,
    evaluate_precision,
)

GOLDEN = "golden"
DECODE = "decode"


# ─── Input contract mirror (aligned with the warmup framework's fields) ────────
# Only mirror the fields the consumer actually uses; q/k/v do not participate in
# the gate's judgment (which looks at output), kept to match the contract.


@dataclass
class AttentionLayerRecord:
    layer_idx: int
    output: torch.Tensor  # [T, H*D] (already flattened by the wrapper)
    q: Optional[torch.Tensor] = None  # [T, H, D]
    k: Optional[torch.Tensor] = None  # [T, KVH, D]
    v: Optional[torch.Tensor] = None  # [T, KVH, D]
    is_prefill: bool = False
    sequence_lengths: Optional[torch.Tensor] = None
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None


@dataclass
class AttentionForwardRecord:
    impl_name: str
    phase: str  # "plain" | "prefix" | "decode_{st}"
    layer_records: Dict[int, AttentionLayerRecord]
    head_num: int = 0
    kv_head_num: int = 0
    head_dim: int = 0
    dtype: Optional[torch.dtype] = None


# records: scenario_base -> impl_name -> [AttentionForwardRecord]
Records = Dict[str, Dict[str, List["AttentionForwardRecord"]]]


# ─── Per-layer judgment ────────────────────────────────────────────────────────


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
    fail_reason: str


def _normalize_kv_dtype(kv_cache_dtype) -> str:
    """Normalize to 'BASE' (strict bf16 bucket) or 'FP8' (loose quantization-quality bucket).

    Accepts a string ('BASE'/'BF16'/'FP8'/...) or an enum with a .name
    (rtp_llm KvCacheDataType). INT8 explicitly raises NotImplementedError
    (consistent with GoldenCacheWriter).
    """
    name = getattr(kv_cache_dtype, "name", kv_cache_dtype)
    s = str(name).upper()
    if "FP8" in s:
        return "FP8"
    if "INT8" in s:
        raise NotImplementedError(
            "decode gate: INT8 KV cache not supported (only BASE/FP8)"
        )
    if "BASE" in s or "BF16" in s or "BFLOAT16" in s:
        return "BASE"
    raise ValueError(f"decode gate: unrecognized kv_cache_dtype={kv_cache_dtype!r}")


def _infer_kv_len(lr: AttentionLayerRecord) -> int:
    """Max kv_len for diagnostics; decode records may contain multiple active batch rows."""
    sl = lr.sequence_lengths
    if sl is not None and hasattr(sl, "numel") and sl.numel() > 0:
        return int(sl.flatten().max().item())
    return 0


def judge_layer(
    golden: AttentionForwardRecord,
    cand: AttentionForwardRecord,
    layer_idx: int,
    kv_dtype: str,
    scenario: str,
) -> LayerVerdict:
    """Single-layer golden vs candidate judgment. Judged as evaluate(candidate, golden):

    golden is the reference (b), so the SNR's ref_rms = RMS(golden). BASE uses the
    three-tier metrics (cos+nrmse+mean_ulp + SNR gate); FP8 uses cos+nrmse
    (+SNR gate, mean_ulp does not participate).
    """
    g_lr = golden.layer_records[layer_idx]
    c_lr = cand.layer_records[layer_idx]
    a = c_lr.output  # candidate
    b = g_lr.output  # golden = reference

    if kv_dtype == "FP8":
        m = evaluate_fp8_quality(a, b)
    else:
        m = evaluate_precision(a, b, "BF16")

    return LayerVerdict(
        impl=cand.impl_name,
        scenario=scenario,
        phase=cand.phase,
        layer_idx=layer_idx,
        kv_len=_infer_kv_len(g_lr),
        overall_pass=m["overall_pass"],
        cos_sim=m["cos_sim"],
        nrmse=m["nrmse"],
        mean_ulp=m["mean_ulp"],
        snr=m["snr"],
        snr_regime=m["snr_regime"],
        fail_reason=m["fail_reason"],
    )


def _is_decode_phase(phase: str) -> bool:
    return phase.startswith(f"{DECODE}_")


def _decode_phase_sort_key(phase: str):
    suffix = phase[len(DECODE) + 1 :]
    try:
        return (int(suffix), phase)
    except ValueError:
        return (10**9, phase)


def _decode_records_by_phase(
    recs: Sequence[AttentionForwardRecord],
) -> Dict[str, AttentionForwardRecord]:
    """Return decode step records keyed by phase; valid decode phases are 'decode_{st}'."""
    out: Dict[str, AttentionForwardRecord] = {}
    for r in recs:
        if _is_decode_phase(r.phase):
            out[r.phase] = r
    return dict(sorted(out.items(), key=lambda item: _decode_phase_sort_key(item[0])))


def _decode_candidate_names(records: Records) -> List[str]:
    """All non-golden impl names that have any decode record (deduplicated, stably sorted)."""
    names: List[str] = []
    seen = set()
    for bucket in records.values():
        for impl, recs in bucket.items():
            if impl == GOLDEN or impl in seen:
                continue
            if _decode_records_by_phase(recs):
                seen.add(impl)
                names.append(impl)
    return sorted(names)


# ─── Gate: build_decode_gate ───────────────────────────────────────────────────


@dataclass
class GateResult:
    passed: frozenset  # passing backends; intended dispatcher input once connected
    detail: Dict[
        str, List[LayerVerdict]
    ]  # per-impl per-layer verdict (diagnostics/alerts)
    verified: frozenset = field(
        default_factory=frozenset
    )  # backends actually verified (passed is a subset of verified)

    def failures(self) -> Dict[str, List[LayerVerdict]]:
        """Per-eliminated-backend failing-layer details (for alerting/locating)."""
        out: Dict[str, List[LayerVerdict]] = {}
        for impl, vs in self.detail.items():
            if impl in self.passed:
                continue
            bad = [v for v in vs if not v.overall_pass]
            if bad:
                out[impl] = bad
        return out


def build_decode_gate(records: Records, kv_cache_dtype) -> GateResult:
    """Build the pre-landed pass/fail result.

    Candidate results are combined via a global AND over scenarios, decode steps,
    and layers.

    This function is not invoked by the production selection path yet.

    Args:
        records: warmup output, scenario_base -> impl -> [AttentionForwardRecord].
        kv_cache_dtype: 'BASE'/'FP8' or an rtp_llm KvCacheDataType enum.
    Returns:
        GateResult(passed=frozenset, detail=..., verified=...).
    """
    kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
    passed = set()
    verified = set()
    detail: Dict[str, List[LayerVerdict]] = {}

    for impl in _decode_candidate_names(records):
        impl_ok = True
        impl_verified = False
        for scenario, bucket in records.items():
            golden_by_phase = _decode_records_by_phase(bucket.get(GOLDEN, []))
            cand_by_phase = _decode_records_by_phase(bucket.get(impl, []))
            if not golden_by_phase or not cand_by_phase:
                continue  # this scenario lacks golden or this backend -> not judged here
            for phase, golden in golden_by_phase.items():
                cand = cand_by_phase.get(phase)
                if cand is None:
                    for layer_idx in sorted(golden.layer_records):
                        v = LayerVerdict(
                            impl=impl,
                            scenario=scenario,
                            phase=phase,
                            layer_idx=layer_idx,
                            kv_len=_infer_kv_len(golden.layer_records[layer_idx]),
                            overall_pass=False,
                            cos_sim=float("nan"),
                            nrmse=float("nan"),
                            mean_ulp=float("nan"),
                            snr=float("nan"),
                            snr_regime="N/A",
                            fail_reason="candidate missing decode phase present in golden",
                        )
                        detail.setdefault(impl, []).append(v)
                        impl_ok = False
                    impl_verified = True
                    continue
                impl_verified = True
                for layer_idx in sorted(golden.layer_records):
                    if layer_idx not in cand.layer_records:
                        # structural missing layer: golden has it, candidate does not -> judged as failure
                        v = LayerVerdict(
                            impl=impl,
                            scenario=scenario,
                            phase=phase,
                            layer_idx=layer_idx,
                            kv_len=_infer_kv_len(golden.layer_records[layer_idx]),
                            overall_pass=False,
                            cos_sim=float("nan"),
                            nrmse=float("nan"),
                            mean_ulp=float("nan"),
                            snr=float("nan"),
                            snr_regime="N/A",
                            fail_reason="candidate missing layer present in golden",
                        )
                    else:
                        v = judge_layer(golden, cand, layer_idx, kv_dtype, scenario)
                    detail.setdefault(impl, []).append(v)
                    impl_ok = (
                        impl_ok and v.overall_pass
                    )  # global AND (no break, collect full detail for diagnostics)
        if impl_verified:
            verified.add(impl)
            if impl_ok:
                passed.add(impl)

    return GateResult(
        passed=frozenset(passed), detail=detail, verified=frozenset(verified)
    )


def merge_tp_gates(per_rank: Sequence[frozenset]) -> frozenset:
    """Cross-rank merge: take the intersection of each rank's passed set (failing on any rank fails overall).

    This is a CPU-only semantic reference for unit tests. Once the gate is
    connected, the intended production merge is reduce_gate_across_tp in
    backend_selector, which uses bitmask all_reduce(SUM).
    """
    if not per_rank:
        return frozenset()
    out = set(per_rank[0])
    for s in per_rank[1:]:
        out &= set(s)
    return frozenset(out)


# ─── CPU-only bitmask helpers for the pre-landed TP integration ─────────────────────
# The candidate list (DECODE_MHA_IMPS) is identical and ordered across all ranks,
# so position is identity: encode the frozenset into a fixed-length bitmask, do one
# all_reduce(SUM) on the mask, and the bits == tp are the intersection. A few dozen
# bytes, no pickle/variable-length/gloo.


def gate_to_mask(s: frozenset, registry: Sequence[str]) -> List[int]:
    """frozenset -> fixed-length 0/1 mask (position is identity; registry is the all-rank ordered registry class names)."""
    return [1 if n in s else 0 for n in registry]


def mask_to_gate(
    passed_sum: Sequence[int],
    verified_sum: Sequence[int],
    registry: Sequence[str],
    tp: int,
):
    """Decode after all_reduce(SUM): reduced==tp means all ranks passed = the intersection; 0<v<tp marks asymmetric verification (alert and exclude).

    Returns (merged: frozenset, asym: List[str]).
    """
    out, asym = set(), []
    for i, n in enumerate(registry):
        if passed_sum[i] == tp and verified_sum[i] == tp:
            out.add(n)
        elif 0 < verified_sum[i] < tp:
            asym.append(n)
    return frozenset(out), asym
