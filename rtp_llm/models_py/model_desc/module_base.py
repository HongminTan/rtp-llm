import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    PREFILL_MHA_IMPS,
    AttentionImpl,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

BackendClass = type[FMHAImplBase]


@dataclass(frozen=True)
class AttentionBackendGateContext:
    decode_gate_passed: frozenset[BackendClass] | None = None
    prefill_gate_passed: frozenset[BackendClass] | None = None


def _resolve_optional_gate_classes(
    domain: str,
    names: Sequence[str] | None,
    registry: Sequence[BackendClass],
) -> frozenset[BackendClass] | None:
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{domain} gate names must be a sequence of backend names")
    try:
        passed_names = tuple(names)
    except TypeError as error:
        raise TypeError(
            f"{domain} gate names must be a sequence of backend names"
        ) from error
    if not passed_names:
        raise ValueError(f"{domain} gate names must be non-empty when enabled")
    if any(not isinstance(name, str) or not name for name in passed_names):
        raise ValueError(f"{domain} gate names must contain non-empty strings")
    if len(passed_names) != len(set(passed_names)):
        raise ValueError(f"{domain} gate names must not contain duplicates")

    registry_by_name: dict[str, BackendClass] = {}
    for impl_cls in registry:
        impl_name = impl_cls.__name__
        if not impl_name:
            raise ValueError(f"{domain} registry contains an empty class name")
        if impl_name in registry_by_name:
            raise ValueError(
                f"{domain} registry contains duplicate class name {impl_name!r}"
            )
        registry_by_name[impl_name] = impl_cls
    unknown = [name for name in passed_names if name not in registry_by_name]
    if unknown:
        raise ValueError(f"{domain} gate contains unknown backend names: {unknown}")
    return frozenset(registry_by_name[name] for name in passed_names)


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
        ## class is an already-broadcast winner.
        self.backend_plan: dict[int, Optional[BackendClass]] = {}

        ## Dedup for the "decode backend in use" log: capture bs -> last logged backend
        ## name, so prepare_fmha_impl emits one line per (bs, backend) at capture instead
        ## of once per replay step.
        self._logged_decode_backend: dict[int, str] = {}

        self._attention_backend_gate_context: AttentionBackendGateContext | None = None

    def set_attention_backend_gate(
        self,
        *,
        decode_passed_names: Sequence[str] | None,
        prefill_passed_names: Sequence[str] | None,
    ) -> None:
        """Install the immutable per-model gate allowlists exactly once."""
        if self._attention_backend_gate_context is not None:
            raise RuntimeError(
                "attention backend gate context has already been installed"
            )
        if decode_passed_names is None and prefill_passed_names is None:
            raise ValueError("at least one attention backend gate must be enabled")
        context = AttentionBackendGateContext(
            decode_gate_passed=_resolve_optional_gate_classes(
                "decode", decode_passed_names, DECODE_MHA_IMPS
            ),
            prefill_gate_passed=_resolve_optional_gate_classes(
                "prefill", prefill_passed_names, PREFILL_MHA_IMPS
            ),
        )
        self._attention_backend_gate_context = context
        self.backend_plan.clear()
        self._logged_decode_backend.clear()
        logging.info(
            "attention_backend_gate_installed decode=%s prefill=%s tp_rank=%d dp_rank=%d",
            sorted(cls.__name__ for cls in context.decode_gate_passed or ()),
            sorted(cls.__name__ for cls in context.prefill_gate_passed or ()),
            int(getattr(self.parallelism_config, "tp_rank", 0)),
            int(getattr(self.parallelism_config, "dp_rank", 0)),
        )

    def _get_gate_passed(self, domain: str) -> frozenset[BackendClass]:
        context = self._attention_backend_gate_context
        if context is None:
            raise RuntimeError("attention backend gate context has not been installed")
        passed = (
            context.decode_gate_passed
            if domain == "decode"
            else context.prefill_gate_passed
        )
        if passed is None:
            raise RuntimeError(f"{domain} attention backend gate is not enabled")
        if not passed:
            raise RuntimeError(f"{domain} attention backend gate is unexpectedly empty")
        return passed

    def get_decode_gate_passed(self) -> frozenset[BackendClass]:
        return self._get_gate_passed("decode")

    def get_prefill_gate_passed(self) -> frozenset[BackendClass]:
        return self._get_gate_passed("prefill")

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
        gate_passed = self.get_decode_gate_passed()
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            return
        bs = int(attention_inputs.input_lengths.size(0))
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
            self, inputs, gate_passed=gate_passed
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
            impl_cls = self.backend_plan.get(bs)
            if selection_complete and impl_cls is not None:
                from rtp_llm.models_py.modules.factory.attention.dispatch import (
                    backend_selector,
                )

                inst = backend_selector.instantiate_decode_impl(
                    self, attention_inputs, impl_cls, is_cuda_graph
                )
                name = impl_cls.__name__
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
