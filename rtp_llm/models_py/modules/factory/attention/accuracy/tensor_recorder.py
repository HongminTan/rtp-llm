import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
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
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    PREFILL_MHA_IMPS,
    get_all_supported_impls,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import PyModelInputs

_DECODE_PHASE = re.compile(r"^decode_[0-9]+$")
_PREFILL_PHASES = ("plain", "prefix")


@dataclass
class _ManifestEntry:
    active_sequence_ids: Tuple[int, ...]
    gate_qualifying: bool = True
    expected_candidates: Optional[Tuple[str, ...]] = None


class TensorRecorder:
    def __init__(self, model: Any, record_qkv: bool = True) -> None:
        self.model = model
        self.record_qkv = bool(record_qkv)
        # scenario -> impl -> forward records
        self.records: Dict[str, Dict[str, List[AttentionForwardRecord]]] = {}
        self._manifest: Dict[Tuple[str, str], _ManifestEntry] = {}
        self._current_bucket: Optional[Dict[str, List[AttentionForwardRecord]]] = None
        self._current_scenario_base_name: Optional[str] = None
        self._current_active_sequence_ids: Tuple[int, ...] = ()
        self._golden_history: Optional[GoldenKVHistory] = None
        self._wrapper: Optional[RecordingWrapper] = None
        self._impl_name: Optional[str] = None
        self._phase: Optional[str] = None
        self._factory: Optional[Callable[..., FMHAImplBase]] = None
        self._golden_layer_records: Optional[Dict[int, Any]] = None
        self._candidates: Dict[str, List[str]] = {}
        self._orig_prepare: Callable[..., FMHAImplBase] = model.prepare_fmha_impl
        self._patched = False
        self._golden_bootstrap = False
        self._state = "IDLE"
        self._finalized_domains: set = set()

    @staticmethod
    def _registry_classes(phase: str):
        raw_registry = (
            DECODE_MHA_IMPS if _DECODE_PHASE.fullmatch(phase) else PREFILL_MHA_IMPS
        )
        classes = list(raw_registry)
        names = [impl_class.__name__ for impl_class in classes]
        if any(not name for name in names):
            raise ValueError(
                f"attention registry for {phase} contains an empty class name"
            )
        if len(names) != len(set(names)):
            raise ValueError(
                f"attention registry for {phase} contains duplicate class names"
            )
        return classes

    @classmethod
    def _registry(cls, phase: str) -> List[str]:
        return [impl_class.__name__ for impl_class in cls._registry_classes(phase)]

    def _require_idle(self, action: str) -> None:
        if self._state == "CLOSED":
            raise RuntimeError(f"tensor recorder is closed; cannot {action}")
        if self._state == "FINALIZED":
            raise RuntimeError(f"tensor recorder is already finalized; cannot {action}")
        if self._state == "RUNNING":
            raise RuntimeError(f"tensor recorder is already running; cannot {action}")

    def candidate_snapshot(self, phase: str) -> Dict[str, object]:
        self._require_idle("snapshot candidates")
        if self._current_scenario_base_name is None:
            raise RuntimeError("candidate snapshot requires an active scenario")
        key = (self._current_scenario_base_name, phase)
        if key not in self._manifest:
            raise RuntimeError(f"candidate snapshot has no golden manifest key: {key}")
        registry = self._registry(phase)
        viable = set(self._candidates.get(phase, ()))
        unknown = sorted(viable - set(registry))
        if unknown:
            raise RuntimeError(
                f"candidate snapshot contains implementations outside registry: {unknown}"
            )
        return {
            "registry": registry,
            "viable_mask": [1 if name in viable else 0 for name in registry],
        }

    def set_expected_candidates(self, phase: str, names: Sequence[str]) -> None:
        self._require_idle("set expected candidates")
        if self._current_scenario_base_name is None:
            raise RuntimeError("expected candidates require an active scenario")
        key = (self._current_scenario_base_name, phase)
        entry = self._manifest.get(key)
        if entry is None:
            raise RuntimeError(
                f"expected candidates have no golden manifest key: {key}"
            )
        if entry.expected_candidates is not None:
            raise RuntimeError(f"expected candidates already frozen for {key}")
        normalized = tuple(str(name) for name in names)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate expected candidate for {key}: {normalized}")
        registry = self._registry(phase)
        unknown = sorted(set(normalized) - set(registry))
        if unknown:
            raise ValueError(f"unknown expected candidates for {key}: {unknown}")
        # Preserve registry identity even if a caller provides a different order.
        expected = set(normalized)
        entry.expected_candidates = tuple(name for name in registry if name in expected)

    def start_run(
        self,
        scenario_base_name: str,
        impl_name: str,
        phase: str,
        batch_sequence_ids: Sequence[int],
        gate_qualifying: bool = True,
    ) -> None:
        self._require_idle("start run")
        if phase.startswith("decode") and not _DECODE_PHASE.fullmatch(phase):
            raise ValueError(f"invalid decode phase: {phase}")
        active_ids = tuple(int(seq_id) for seq_id in batch_sequence_ids)
        if len(set(active_ids)) != len(active_ids):
            raise ValueError(f"duplicate active sequence ids: {active_ids}")

        if scenario_base_name != self._current_scenario_base_name:
            self._golden_history = GoldenKVHistory()
            self._candidates = {}
            self._current_scenario_base_name = scenario_base_name
        self._current_bucket = self.records.setdefault(scenario_base_name, {})
        key = (scenario_base_name, phase)
        existing = [
            record
            for record in self._current_bucket.get(impl_name, ())
            if record.phase == phase
        ]
        if existing:
            raise RuntimeError(
                f"duplicate forward record scheduled for {scenario_base_name}/{phase}/{impl_name}"
            )

        if impl_name == "golden":
            if key in self._manifest:
                raise RuntimeError(f"duplicate golden manifest key: {key}")
            self._manifest[key] = _ManifestEntry(
                active_sequence_ids=active_ids,
                gate_qualifying=bool(gate_qualifying),
            )
            if self._golden_history is None:
                raise RuntimeError("golden K/V history is not initialized")
            history_view = self._golden_history.bind_batch(active_ids)
            self._factory = lambda attn_configs, attention_inputs, _parallelism_config: GoldenSDPAImpl(
                attn_configs, attention_inputs, history_view
            )
        else:
            entry = self._manifest.get(key)
            if entry is None:
                raise RuntimeError(f"candidate has no golden manifest key: {key}")
            if entry.expected_candidates is None:
                raise RuntimeError(f"candidate list is not frozen for {key}")
            if impl_name not in entry.expected_candidates:
                raise RuntimeError(f"candidate {impl_name} is not applicable to {key}")
            if active_ids != entry.active_sequence_ids:
                raise RuntimeError(
                    f"candidate active ids {active_ids} differ from manifest {entry.active_sequence_ids}"
                )
            if bool(gate_qualifying) != entry.gate_qualifying:
                raise RuntimeError(
                    f"candidate gate_qualifying={bool(gate_qualifying)} differs from "
                    f"manifest {entry.gate_qualifying} for {key}"
                )
            impl_class = next(
                (
                    candidate_class
                    for candidate_class in self._registry_classes(phase)
                    if candidate_class.__name__ == impl_name
                ),
                None,
            )
            if impl_class is None:
                raise ValueError(f"unknown attention implementation: {impl_name}")
            if _DECODE_PHASE.fullmatch(phase):
                golden_records = [
                    record
                    for record in self._current_bucket.get("golden", ())
                    if record.phase == phase
                ]
                if len(golden_records) != 1:
                    raise RuntimeError(
                        f"decode candidate requires one golden record for {key}, "
                        f"got {len(golden_records)}"
                    )
                self._golden_layer_records = golden_records[0].layer_records
            self._factory = (
                lambda attn_configs, attention_inputs, parallelism_config: impl_class(
                    attn_configs, attention_inputs, parallelism_config
                )
            )

        self._impl_name = impl_name
        self._phase = phase
        self._current_active_sequence_ids = active_ids
        self._wrapper = None
        self._golden_bootstrap = False
        self.model.prepare_fmha_impl = self._recording_prepare
        self._patched = True
        self._state = "RUNNING"

    def start_golden_bootstrap(
        self,
        scenario_base_name: str,
        phase: str,
        batch_sequence_ids: Sequence[int],
    ) -> None:
        self._require_idle("start golden bootstrap")
        active_ids = tuple(int(seq_id) for seq_id in batch_sequence_ids)
        if not active_ids or len(set(active_ids)) != len(active_ids):
            raise ValueError(f"invalid golden bootstrap sequence ids: {active_ids}")
        if scenario_base_name != self._current_scenario_base_name:
            self._golden_history = GoldenKVHistory()
            self._candidates = {}
            self._current_scenario_base_name = scenario_base_name
        if self._golden_history is None:
            raise RuntimeError("golden bootstrap history is not initialized")

        history_view = self._golden_history.bind_batch(active_ids)
        self._factory = (
            lambda attn_configs, attention_inputs, _parallelism_config: GoldenSDPAImpl(
                attn_configs,
                attention_inputs,
                history_view,
                append_prefill_history=True,
            )
        )
        self._impl_name = "golden_bootstrap"
        self._phase = str(phase)
        self._current_active_sequence_ids = active_ids
        self._wrapper = None
        self._golden_bootstrap = True
        self.model.prepare_fmha_impl = self._recording_prepare
        self._patched = True
        self._state = "RUNNING"

    def stop_golden_bootstrap(
        self, bootstrap_name: str, completed: bool = True
    ) -> None:
        if self._state != "RUNNING" or not self._golden_bootstrap:
            raise RuntimeError("golden bootstrap is not running")
        expected_identity = (
            f"{self._current_scenario_base_name}::golden_bootstrap::{self._phase}"
        )
        if bootstrap_name != expected_identity:
            raise RuntimeError(
                f"golden bootstrap identity mismatch: expected {expected_identity}, got {bootstrap_name}"
            )
        self.restore()
        try:
            if completed and (self._wrapper is None or not self._wrapper.layer_records):
                raise RuntimeError(f"no records for {bootstrap_name}")
        finally:
            if self._wrapper is not None:
                for layer_record in self._wrapper.layer_records.values():
                    layer_record.release()
                self._wrapper.layer_records.clear()
            self._wrapper = None
            self._factory = None
            self._impl_name = None
            self._phase = None
            self._current_active_sequence_ids = ()
            self._golden_bootstrap = False
            self._state = "IDLE"

    def stop_run(self, scenario_name: str) -> None:
        if self._state != "RUNNING" or self._golden_bootstrap:
            raise RuntimeError("tensor recorder is not running")
        expected_identity = (
            f"{self._current_scenario_base_name}::{self._impl_name}::{self._phase}"
        )
        if scenario_name != expected_identity:
            raise RuntimeError(
                f"recording identity mismatch: expected {expected_identity}, got {scenario_name}"
            )
        self.restore()
        try:
            if self._wrapper is None or not self._wrapper.layer_records:
                raise RuntimeError(f"no records for {scenario_name}")
            if (
                self._current_bucket is None
                or self._impl_name is None
                or self._phase is None
            ):
                raise RuntimeError(f"recording state is incomplete for {scenario_name}")
            record = AttentionForwardRecord(
                impl_name=self._impl_name,
                phase=self._phase,
                layer_records=self._wrapper.layer_records,
                head_num=self._wrapper.head_num,
                kv_head_num=self._wrapper.kv_head_num,
                head_dim=self._wrapper.head_dim,
                dtype=self._wrapper.dtype,
                is_prefill=self._wrapper.is_prefill,
                sequence_lengths=self._wrapper.sequence_lengths,
                input_lengths=self._wrapper.input_lengths,
                prefix_lengths=self._wrapper.prefix_lengths,
                cu_seqlens=self._wrapper.cu_seqlens,
                active_sequence_ids=self._wrapper.active_sequence_ids,
            )
            self._current_bucket.setdefault(self._impl_name, []).append(record)
            self._wrapper = None
        finally:
            self._factory = None
            self._golden_layer_records = None
            self._impl_name = None
            self._phase = None
            self._current_active_sequence_ids = ()
            self._state = "IDLE"

    def restore(self) -> None:
        if self._patched:
            self.model.prepare_fmha_impl = self._orig_prepare
            self._patched = False

    def finalize_decode_gate(self, kv_cache_dtype) -> Dict[str, object]:
        return self._finalize_gate("decode", kv_cache_dtype)

    def finalize_prefill_gate(self, kv_cache_dtype) -> Dict[str, object]:
        return self._finalize_gate("prefill", kv_cache_dtype)

    def _finalize_gate(self, domain: str, kv_cache_dtype) -> Dict[str, object]:
        if self._state == "CLOSED":
            raise RuntimeError(
                f"tensor recorder is closed; cannot finalize {domain} gate"
            )
        if self._state == "RUNNING":
            raise RuntimeError(
                f"tensor recorder is already running; cannot finalize {domain} gate"
            )
        if domain in self._finalized_domains:
            raise RuntimeError(
                f"tensor recorder is already finalized; cannot finalize {domain} gate"
            )
        # Delayed to avoid accuracy.__init__'s eager TensorRecorder export forming
        # attention_record -> accuracy package -> tensor_recorder -> decode_gate.
        from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
            _build_decode_gate_strict,
            _build_prefill_gate_strict,
            _normalize_kv_dtype,
            gate_to_mask,
            precision_thresholds,
        )

        is_decode = domain == "decode"
        registry = self._registry("decode_0" if is_decode else "plain")

        def in_domain(phase: str) -> bool:
            if is_decode:
                return bool(_DECODE_PHASE.fullmatch(phase))
            return phase in _PREFILL_PHASES

        domain_manifest = {
            key: entry for key, entry in self._manifest.items() if in_domain(key[1])
        }
        applicable = {
            name
            for entry in domain_manifest.values()
            if entry.gate_qualifying
            for name in (entry.expected_candidates or ())
        }

        unavailable_reason = None
        if not domain_manifest:
            unavailable_reason = f"no_{domain}_manifest"
        elif any(
            entry.expected_candidates is None for entry in domain_manifest.values()
        ):
            unavailable_reason = f"{domain}_manifest_candidates_not_frozen"
        layer_num = int(getattr(self.model, "layer_num", 0))
        if layer_num <= 0:
            unavailable_reason = "invalid_model_layer_num"

        result = None
        if unavailable_reason is None:
            try:
                normalized_kv_dtype = _normalize_kv_dtype(kv_cache_dtype)
            except (NotImplementedError, ValueError) as error:
                unavailable_reason = str(error)
        if unavailable_reason is None:
            # Structural validation is encoded as UNAVAILABLE by the strict
            # helper. Unexpected metric/device exceptions intentionally escape.
            builder = (
                _build_decode_gate_strict if is_decode else _build_prefill_gate_strict
            )
            result, unavailable_reason = builder(
                self.records,
                normalized_kv_dtype,
                manifest=domain_manifest,
                expected_layer_ids=tuple(range(layer_num)),
            )

        passed = result.passed if result is not None else frozenset()
        thresholds = precision_thresholds(kv_cache_dtype)
        rank = int(getattr(self.model.parallelism_config, "world_rank", 0))
        for impl_name in registry:
            verdicts = [] if result is None else result.detail.get(impl_name, [])
            qualifying = [verdict for verdict in verdicts if verdict.gate_qualifying]
            finite_cos = [
                verdict.cos_sim
                for verdict in qualifying
                if math.isfinite(verdict.cos_sim)
            ]
            finite_nrmse = [
                verdict.nrmse for verdict in qualifying if math.isfinite(verdict.nrmse)
            ]
            finite_ulp = [
                verdict.mean_ulp
                for verdict in qualifying
                if math.isfinite(verdict.mean_ulp)
            ]
            failures = [
                f"scenario={verdict.scenario},phase={verdict.phase},layer={verdict.layer_idx},rank={rank},"
                f"cos={verdict.cos_sim},nrmse={verdict.nrmse},mean_ulp={verdict.mean_ulp},"
                f"reason={verdict.fail_reason}"
                for verdict in qualifying
                if not verdict.overall_pass
            ]
            logging.info(
                "dynamic_%s_gate_metrics backend=%s rank=%d applicable=%d passed=%d "
                "min_cos=%s cos_threshold=%s max_nrmse=%s nrmse_threshold=%s "
                "max_mean_ulp=%s mean_ulp_threshold=%s failures=%s",
                domain,
                impl_name,
                rank,
                int(impl_name in applicable),
                int(impl_name in passed),
                min(finite_cos) if finite_cos else float("nan"),
                thresholds["cos_sim"],
                max(finite_nrmse) if finite_nrmse else float("nan"),
                thresholds["nrmse"],
                max(finite_ulp) if finite_ulp else float("nan"),
                thresholds["mean_ulp"],
                failures,
            )
        if unavailable_reason is not None:
            logging.error(
                "dynamic_%s_gate_invalid rank=%d reason=%s",
                domain,
                rank,
                unavailable_reason,
            )

        payload = {
            "valid": unavailable_reason is None,
            "registry": registry,
            "applicable_mask": gate_to_mask(frozenset(applicable), registry),
            "passed_mask": gate_to_mask(passed, registry),
        }
        self._finalized_domains.add(domain)
        self._state = "FINALIZED"
        return payload

    def close(self) -> None:
        if self._state == "CLOSED":
            return
        self.restore()
        self._state = "CLOSED"

        if self._wrapper is not None:
            for layer_record in self._wrapper.layer_records.values():
                layer_record.release()
            self._wrapper.layer_records.clear()

        records, self.records = self.records, {}
        self._manifest.clear()
        self._current_bucket = None
        self._current_scenario_base_name = None
        self._current_active_sequence_ids = ()
        self._golden_history = None
        self._wrapper = None
        self._impl_name = None
        self._phase = None
        self._factory = None
        self._golden_layer_records = None
        self._golden_bootstrap = False
        self._candidates.clear()

        for impl_map in records.values():
            for record_list in impl_map.values():
                for record in record_list:
                    record.release()

    def _recording_prepare(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> RecordingWrapper:
        if self._state != "RUNNING" or self._factory is None or self._phase is None:
            raise RuntimeError("accuracy recording has not been started")
        attn_configs = self.model.config.getAttentionConfigs(
            self.model.parallelism_config.get_attn_tp_size()
        )
        # Compare all implementations without RoPE or logn.
        attn_configs.rope_config.style = RopeStyle.No
        attn_configs.use_logn_attn = False
        attention_inputs = inputs.attention_inputs
        attention_inputs.is_cuda_graph = False
        attention_inputs.headwise_config = None
        parallelism_config = self.model.parallelism_config
        if not self._golden_bootstrap:
            supported = get_all_supported_impls(
                attn_configs,
                attention_inputs,
                self.model.fmha_config,
                parallelism_config,
            )
            viable = []
            for impl_class in supported:
                if _DECODE_PHASE.fullmatch(self._phase):
                    # Decode candidates feed the production precision allowlist. An
                    # unexpected constructor failure must abort the gate instead of
                    # being rewritten as "not applicable".
                    impl_class(attn_configs, attention_inputs, parallelism_config)
                    viable.append(impl_class.__name__)
                    continue
                try:
                    impl_class(attn_configs, attention_inputs, parallelism_config)
                    viable.append(impl_class.__name__)
                except Exception as error:
                    logging.warning(
                        "accuracy check: candidate %s dry-instantiate failed, excluded: %s",
                        impl_class.__name__,
                        error,
                    )
            self._candidates[self._phase] = viable
        self._wrapper = RecordingWrapper(
            self._factory(attn_configs, attention_inputs, parallelism_config),
            attn_configs,
            attention_inputs,
            self._current_active_sequence_ids,
            self.record_qkv and not self._golden_bootstrap,
            self._golden_layer_records,
        )
        return self._wrapper
