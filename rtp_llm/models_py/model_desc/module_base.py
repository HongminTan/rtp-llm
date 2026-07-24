import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttentionImpl
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class DecodeBackendGateStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class DecodeBackendGateState:
    status: DecodeBackendGateStatus
    passed: frozenset[str]
    verified: frozenset[str]
    reason: str
    registry_fingerprint: str
    manifest_fingerprint: str


def _backend_name_set(field: str, values: Any) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a collection of backend names")
    try:
        names = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field} must be a collection of backend names") from error
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{field} must contain non-empty backend names")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must not contain duplicate backend names")
    return frozenset(names)


class GptModelBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

        ## Dynamic decode backend dispatch result. Key absence means selection has
        ## not completed, None means a completed fixed-priority plan miss, and a
        ## class name is an already-broadcast winner.
        self.backend_plan: dict[int, Optional[str]] = {}

        ## Dedup for the "decode backend in use" log: capture bs -> last logged backend
        ## name, so prepare_fmha_impl emits one line per (bs, backend) at capture instead
        ## of once per replay step.
        self._logged_decode_backend: dict[int, str] = {}

        # The temporary accuracy model transfers one immutable snapshot to the real
        # model before CUDA Graph capture. NOT_REQUESTED is deliberately authoritative:
        # dynamic selection cannot silently fall back to support-only eligibility when
        # the snapshot was not injected.
        self._decode_backend_gate = DecodeBackendGateState(
            status=DecodeBackendGateStatus.NOT_REQUESTED,
            passed=frozenset(),
            verified=frozenset(),
            reason="",
            registry_fingerprint="",
            manifest_fingerprint="",
        )
        self._decode_backend_gate_injected = False
        self._decode_backend_gate_frozen = False
        self._logged_decode_gate_fallback: set[tuple[int, str, str]] = set()

    def set_decode_backend_gate(
        self,
        status: str,
        passed: Any,
        verified: Any,
        reason: str,
        registry_fingerprint: Any,
        manifest_fingerprint: Any,
    ) -> None:
        """Inject the one authoritative decode gate snapshot before selection."""
        if self._decode_backend_gate_frozen:
            raise RuntimeError("decode backend gate is frozen after selector use")
        if self._decode_backend_gate_injected:
            raise RuntimeError("decode backend gate has already been injected")

        try:
            gate_status = (
                status
                if isinstance(status, DecodeBackendGateStatus)
                else DecodeBackendGateStatus(str(status).upper())
            )
        except ValueError as error:
            raise ValueError(
                f"unsupported decode backend gate status: {status}"
            ) from error

        passed_names = _backend_name_set("passed", passed)
        verified_names = _backend_name_set("verified", verified)
        if gate_status != DecodeBackendGateStatus.READY and passed_names:
            raise ValueError(f"{gate_status.value} gate cannot pass a decode backend")
        if not passed_names.issubset(verified_names):
            raise ValueError("passed decode backends must be a subset of verified")

        self._decode_backend_gate = DecodeBackendGateState(
            status=gate_status,
            passed=passed_names,
            verified=verified_names,
            reason=str(reason or ""),
            registry_fingerprint=(
                "" if registry_fingerprint is None else str(registry_fingerprint)
            ),
            manifest_fingerprint=(
                "" if manifest_fingerprint is None else str(manifest_fingerprint)
            ),
        )
        self._decode_backend_gate_injected = True
        self.backend_plan.clear()
        self._logged_decode_backend.clear()
        self._logged_decode_gate_fallback.clear()
        logging.info(
            "dynamic_decode_gate_injected status=%s passed=%s verified=%s "
            "reason=%s registry_fp=%s manifest_fp=%s tp_rank=%d dp_rank=%d",
            gate_status.value,
            ",".join(sorted(passed_names)) or "-",
            ",".join(sorted(verified_names)) or "-",
            self._decode_backend_gate.reason or "-",
            self._decode_backend_gate.registry_fingerprint or "-",
            self._decode_backend_gate.manifest_fingerprint or "-",
            int(getattr(self.parallelism_config, "tp_rank", 0)),
            int(getattr(self.parallelism_config, "dp_rank", 0)),
        )

    def _freeze_decode_backend_gate(self) -> DecodeBackendGateState:
        self._decode_backend_gate_frozen = True
        return self._decode_backend_gate

    def _finish_decode_gate_miss(self, bs: int, gate: DecodeBackendGateState) -> None:
        self.backend_plan[bs] = None
        reason = (
            "ready_empty"
            if gate.status == DecodeBackendGateStatus.READY
            else gate.status.value.lower()
        )
        if gate.reason:
            detail = gate.reason
        elif reason == "ready_empty":
            detail = "empty_passed"
        elif (
            gate.status == DecodeBackendGateStatus.NOT_REQUESTED
            and not self._decode_backend_gate_injected
        ):
            detail = "not_injected"
        else:
            detail = "-"
        key = (bs, reason, detail)
        if key in self._logged_decode_gate_fallback:
            return
        self._logged_decode_gate_fallback.add(key)
        logging.warning(
            "dynamic_decode_fallback_static reason=gate_%s gate_reason=%s bs=%d "
            "tp_rank=%d dp_rank=%d",
            reason,
            detail,
            bs,
            int(getattr(self.parallelism_config, "tp_rank", 0)),
            int(getattr(self.parallelism_config, "dp_rank", 0)),
        )

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = self.kv_cache.layer_count
            layer0_caches = (
                self.kv_cache.get_layer_cache_groups(0) if num_layers > 0 else []
            )
            layer0_shapes = [cache.kv_cache_base.shape for cache in layer0_caches]
            layer0_scale_count = sum(
                cache.kv_scale_base is not None and cache.kv_scale_base.numel() > 0
                for cache in layer0_caches
            )
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shapes={layer0_shapes}, "
                f"layer0_scale_groups={layer0_scale_count}, "
            )
        return True

    def select_decode_backend(
        self, inputs: PyModelInputs, is_cuda_graph: bool = True
    ) -> None:
        """Called per bs by the engine at capture time: benchmark on-device and pick a
        decode backend, writing the winner into self.backend_plan[bs].

        Capability guards leave the key absent. Plan misses and recoverable pre-probe
        errors write None, so prepare_fmha_impl uses fixed priority. An exception after an
        on-device probe starts terminates the worker instead of returning here. Under
        TP all ranks must call in lockstep (the C++ side drives this per bs in the same
        order on every rank); internally only rank0 benchmarks and the winner is
        broadcast. Active only when enable_dynamic_decode_backend is on and the kv
        cache is ready.
        """
        if not getattr(
            self.py_hw_kernel_config, "enable_dynamic_decode_backend", False
        ):
            return
        gate = self._freeze_decode_backend_gate()
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            if gate.status != DecodeBackendGateStatus.READY or not gate.passed:
                primary_inputs = next(iter(attention_inputs.values()))
                self._finish_decode_gate_miss(
                    int(primary_inputs.input_lengths.size(0)), gate
                )
            return
        bs = int(attention_inputs.input_lengths.size(0))
        if gate.status != DecodeBackendGateStatus.READY or not gate.passed:
            self._finish_decode_gate_miss(bs, gate)
            return
        # CUDA-only feature: the selectable backends are FlashInfer-based decode impls
        # that only exist on CUDA. On ROCm/PPU/CPU the on-device bench would drive the
        # platform's paged-attention (e.g. aiter pa on ROCm) through the CUDA-graph
        # capture path with synthetic bench inputs, which faults. Skip here so those
        # platforms keep their fixed-priority decode backend (no behavior change).
        if self.device_type != DeviceType.Cuda:
            return
        if self.kv_cache is None:
            return
        # MLA models use a dedicated backend path (get_mla_impl) with no
        # selectable alternatives; skip the MHA-oriented bench entirely.
        attn_configs = self.config.getAttentionConfigs(
            self.parallelism_config.get_attn_tp_size()
        )
        if attn_configs.use_mla:
            return
        # Hybrid models (linear + full attention) have multi-group KV caches
        # whose layout is incompatible with MHA-only bench infrastructure;
        # the bench cannot safely construct valid inputs for XQA/FlashInfer.
        # Their full_attention layers use fixed-priority backend selection.
        hybrid_cfg = getattr(self.config, "hybrid_attention_config", None)
        if (
            hybrid_cfg is not None
            and len(getattr(hybrid_cfg, "hybrid_attention_types", [])) > 0
        ):
            return
        from rtp_llm.models_py.modules.factory.attention.dispatch import (
            backend_selector,
        )

        winner = backend_selector.run_backend_selection(
            self, inputs, gate_passed=gate.passed
        )
        self.backend_plan[bs] = winner

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> AttentionImpl | dict[str, AttentionImpl]:
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            fmha_group_tags = self._get_fmha_group_tags()
            selected_group_inputs = (
                attention_inputs.items()
                if fmha_group_tags is None
                else (
                    (tag, select_attention_inputs_for_tag(attention_inputs, tag))
                    for tag in fmha_group_tags
                )
            )
            return {
                tag: AttnImplFactory.get_fmha_impl(
                    self.config,
                    self.parallelism_config,
                    self.weight,
                    group_inputs,
                    self.fmha_config,
                    is_cuda_graph,
                )
                for tag, group_inputs in selected_group_inputs
            }

        # Dynamic dispatch lookup: only on the cuda graph capture path, and only when the
        # selection has completed for this bs. A winner must be applied or fail-stop;
        # an explicit None falls through to fixed priority.
        # Guard: backend_plan only contains DECODE backends; skip when is_prefill
        # (target-model verify/score has is_prefill=True with num_tokens_per_bs > 1).
        selection_complete = False
        if is_cuda_graph and not attention_inputs.is_prefill:
            bs = int(attention_inputs.input_lengths.size(0))
            selection_complete = bs in self.backend_plan
            name = self.backend_plan.get(bs)
            if selection_complete and name is not None:
                from rtp_llm.models_py.modules.factory.attention.dispatch import (
                    backend_selector,
                )

                inst = backend_selector.instantiate_decode_impl(
                    self, attention_inputs, name, is_cuda_graph
                )
                if self._logged_decode_backend.get(bs) != name:
                    self._logged_decode_backend[bs] = name
                    logging.info(
                        "dynamic_decode_plan_applied bs=%d backend=%s "
                        "tp_rank=%d dp_rank=%d",
                        bs,
                        name,
                        int(self.parallelism_config.tp_rank),
                        int(self.parallelism_config.dp_rank),
                    )
                return inst

        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        # Authoritative single log of the decode backend actually in use: only on the
        # cuda graph capture path (once per bs bucket, not per replay step) and only when
        # dynamic dispatch is on, so the fixed-priority fallback (plan miss) is visible too.
        if (
            is_cuda_graph
            and not attention_inputs.is_prefill
            and selection_complete
            and getattr(
                self.py_hw_kernel_config, "enable_dynamic_decode_backend", False
            )
        ):
            bs = int(attention_inputs.input_lengths.size(0))
            self._log_decode_backend_once(
                bs, type(fmha_impl).__name__, "fixed-priority"
            )
        return fmha_impl

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def _log_decode_backend_once(self, bs: int, name: str, source: str) -> None:
        """Log a fixed-priority decode fallback once per capture bucket."""
        if self._logged_decode_backend.get(bs) == name:
            return
        self._logged_decode_backend[bs] = name
        logging.info(
            "[dispatcher] decode backend in use: bs=%d -> %s (%s)", bs, name, source
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
