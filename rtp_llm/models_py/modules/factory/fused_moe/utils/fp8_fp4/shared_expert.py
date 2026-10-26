"""Shared-expert execution policies for fused MoE.

Open-source MoE stacks such as vLLM and SGLang commonly overlap shared experts
with routed MoE work on an auxiliary CUDA stream.  They do not rely on BF16
direct accumulation by default.  RTP keeps the existing FP32 accumulate contract
and only fuses the final add+cast when possible.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    shared_expert_mode,
    strict_fused_moe_enabled,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.profiler import record_function_range

from .expert import require_silu_mul_split
from .quantized_linear import create_fp8_linear

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4._green_context import GreenContextPartition

logger = logging.getLogger(__name__)
_GREEN_CONTEXT_FAILURES_LOGGED: set[tuple[int, type[Exception], str]] = set()


@dataclass(frozen=True)
class _SharedExpertWorkspaceViews:
    x_fp8: torch.Tensor
    x_scale: torch.Tensor
    gate_up_bf16: torch.Tensor
    hidden_fp8: torch.Tensor
    hidden_scale: torch.Tensor
    out_bf16: torch.Tensor


@dataclass
class _SharedExpertWorkspace:
    capacity: int
    device: torch.device
    use_ue8m0: bool
    x_fp8: torch.Tensor
    x_scale_storage: torch.Tensor
    gate_up_bf16: torch.Tensor
    hidden_fp8: torch.Tensor
    hidden_scale_storage: torch.Tensor
    out_bf16: torch.Tensor
    views: dict[int, _SharedExpertWorkspaceViews] = field(default_factory=dict)


_SHARED_EXPERT_WORKSPACE_CACHE: dict[tuple, _SharedExpertWorkspace] = {}
_SHARED_EXPERT_STREAM_CACHE: dict[int, torch.cuda.Stream] = {}


def _normalize_cuda_device(device: torch.device) -> torch.device | None:
    if not torch.cuda.is_available() or device.type != "cuda":
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.device("cuda", device_index)


def _ensure_shared_expert_stream(device: torch.device) -> torch.cuda.Stream | None:
    device = _normalize_cuda_device(device)
    if device is None:
        return None
    device_index = device.index
    assert device_index is not None
    stream = _SHARED_EXPERT_STREAM_CACHE.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SHARED_EXPERT_STREAM_CACHE[device_index] = stream
    return stream


def _get_shared_expert_stream(
    device: torch.device,
    *,
    allow_create: bool,
) -> torch.cuda.Stream:
    device = _normalize_cuda_device(device)
    if device is None:
        raise RuntimeError(f"shared expert overlap requires CUDA device, got {device}")
    device_index = device.index
    assert device_index is not None
    stream = _SHARED_EXPERT_STREAM_CACHE.get(device_index)
    if stream is not None:
        return stream
    if not allow_create:
        raise RuntimeError(
            "shared expert overlap stream was not created before CUDA graph "
            f"capture for device cuda:{device_index}"
        )
    stream = torch.cuda.Stream(device=device)
    _SHARED_EXPERT_STREAM_CACHE[device_index] = stream
    return stream


def _find_module_cuda_device(module: nn.Module) -> torch.device | None:
    for tensor in list(module.parameters(recurse=True)) + list(
        module.buffers(recurse=True)
    ):
        if tensor.is_cuda:
            return tensor.device

    for submodule in module.modules():
        for attr in ("weight", "weight_scales", "bias"):
            tensor = getattr(submodule, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                return tensor.device
    return None


class W13SharedExpert(nn.Module):
    """Merged shared experts with a combined gate/up projection.

    The checkpoint stores shared w1 and w3 separately, but the loader merges
    them into ``w13`` so inference never keeps duplicate split linears.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        expert_weights: dict[str, torch.Tensor],
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        w13_w = expert_weights["w13_w"]
        w13_s = expert_weights["w13_s"]
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_w.shape[0] != 2 * inter_dim or w13_w.shape[1] != dim:
            raise RuntimeError(
                "shared w13 weight shape mismatch: "
                f"got {tuple(w13_w.shape)}, expected {(2 * inter_dim, dim)}"
            )
        self.w13 = create_fp8_linear(w13_w, w13_s)
        self.w2 = create_fp8_linear(expert_weights["w2_w"], expert_weights["w2_s"])
        self.swiglu_limit = swiglu_limit

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        with record_function_range("moe.shared_expert.w13"):
            gate_up = self._apply_layer(self.w13, x).float()
            gate, up = gate_up.chunk(2, dim=-1)
        with record_function_range("moe.shared_expert.silu_mul"):
            hidden = require_silu_mul_split()(
                gate.contiguous(),
                up.contiguous(),
                clamp_limit=self.swiglu_limit,
            )
        if weights is not None:
            hidden = weights * hidden
        with record_function_range("moe.shared_expert.w2"):
            return self._apply_layer(self.w2, hidden.to(dtype))


class FusedSharedExpertFastPath:
    """Workspace-backed shared-expert path.

    It quantizes the BF16 input once, runs one merged w13 FP8 GEMM into a
    reusable BF16 gate_up buffer, fuses SwiGLU+FP8 quantization, then runs w2.
    The merged w13 weight is prepared outside the forward hot path.
    """

    def __init__(
        self,
        max_tokens_per_rank: int | None = None,
        dim: int | None = None,
        inter_dim: int | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dim = dim
        self.inter_dim = inter_dim
        self.swiglu_limit = swiglu_limit
        self._workspace: _SharedExpertWorkspace | None = None
        self._prepared_shared_experts: nn.Module | None = None

    @staticmethod
    def _linear_parts(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(linear, "weight", None)
        scale = getattr(linear, "weight_scales", None)
        if weight is None or scale is None:
            raise RuntimeError("shared expert FP8 linear does not expose weight/scale")
        return weight, scale

    def _require_prepared(self) -> nn.Module:
        if self._prepared_shared_experts is None:
            raise RuntimeError(
                "shared expert fast path must be prepared before execution"
            )
        return self._prepared_shared_experts

    def can_run(self, x: torch.Tensor) -> bool:
        self._require_prepared()
        return x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2

    def prepare(self, shared_experts: nn.Module) -> None:
        """Validate the loader-prepared merged w13; no runtime concatenation."""
        if not hasattr(shared_experts, "w13"):
            raise RuntimeError("shared expert requires loader-prepared w13")
        w13_w, w13_s = self._linear_parts(shared_experts.w13)
        self._linear_parts(shared_experts.w2)
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_s.dim() != 2:
            raise RuntimeError(f"shared w13 scale must be 2D, got {w13_s.dim()}D")
        if w13_w.shape[0] % 2 != 0:
            raise RuntimeError(f"shared w13 rows must be even, got {w13_w.shape[0]}")
        inferred_inter_dim = int(w13_w.shape[0]) // 2
        self.inter_dim = inferred_inter_dim
        self._prepared_shared_experts = shared_experts

    @staticmethod
    def _tma_aligned_rows(rows: int, element_size: int) -> int:
        import deep_gemm

        return deep_gemm.get_tma_aligned_size(rows, element_size)

    @staticmethod
    def _scale_storage(
        num_groups: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        aligned_capacity = FusedSharedExpertFastPath._tma_aligned_rows(
            max(capacity, 1),
            torch.empty((), dtype=dtype).element_size(),
        )
        return torch.empty((num_groups, aligned_capacity), dtype=dtype, device=device)

    @staticmethod
    def _scale_view(storage: torch.Tensor, tokens: int) -> torch.Tensor:
        """TMA-aligned MN-major ``[tokens, groups]`` view of the scale storage."""
        aligned_tokens = FusedSharedExpertFastPath._tma_aligned_rows(
            max(tokens, 1),
            storage.element_size(),
        )
        return storage.as_strided(
            (tokens, storage.size(0)),
            (1, aligned_tokens),
        )

    def _ensure_workspace(self, x: torch.Tensor) -> _SharedExpertWorkspace:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )

        use_ue8m0 = is_deep_gemm_e8m0_used()
        T, D = x.shape
        if self.dim is None:
            self.dim = D
        if D != self.dim:
            raise RuntimeError(
                f"shared expert dim mismatch: got {D}, expected {self.dim}"
            )
        inter = self.inter_dim
        assert inter is not None
        capacity = max(T, self.max_tokens_per_rank or 0, 1)
        workspace = self._workspace
        if (
            workspace is not None
            and workspace.device == x.device
            and workspace.capacity >= capacity
            and workspace.hidden_fp8.size(1) == inter
            and workspace.use_ue8m0 == use_ue8m0
        ):
            return workspace
        if D % 128 != 0 or inter % 128 != 0:
            raise RuntimeError(
                f"shared expert fused path requires D/inter divisible by 128, got {D}/{inter}"
            )
        key = (x.device, D, inter, use_ue8m0)
        workspace = _SHARED_EXPERT_WORKSPACE_CACHE.get(key)
        if workspace is not None and workspace.capacity >= capacity:
            self._workspace = workspace
            return workspace

        x_fp8 = torch.empty(
            (capacity, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        if use_ue8m0:
            x_scale_storage = self._scale_storage(
                (D // 128 + 3) // 4, capacity, x.device, torch.int32
            )
            hidden_scale_storage = self._scale_storage(
                (inter // 128 + 3) // 4, capacity, x.device, torch.int32
            )
        else:
            x_scale_storage = self._scale_storage(
                D // 128, capacity, x.device, torch.float32
            )
            hidden_scale_storage = self._scale_storage(
                inter // 128, capacity, x.device, torch.float32
            )
        gate_up_bf16 = torch.empty(
            (capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=x.device,
        )
        hidden_fp8 = torch.empty(
            (capacity, inter),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        out_bf16 = torch.empty(
            (capacity, D),
            dtype=torch.bfloat16,
            device=x.device,
        )
        # Replace the cache entry as one versioned unit. Executors or captured
        # graphs that still reference the prior workspace keep its buffers
        # alive; each executor adopts this version when it next needs capacity.
        workspace = _SharedExpertWorkspace(
            capacity=capacity,
            device=x.device,
            use_ue8m0=use_ue8m0,
            x_fp8=x_fp8,
            x_scale_storage=x_scale_storage,
            gate_up_bf16=gate_up_bf16,
            hidden_fp8=hidden_fp8,
            hidden_scale_storage=hidden_scale_storage,
            out_bf16=out_bf16,
        )
        _SHARED_EXPERT_WORKSPACE_CACHE[key] = workspace
        self._workspace = workspace
        return workspace

    def _workspace_views(
        self,
        workspace: _SharedExpertWorkspace,
        tokens: int,
    ) -> _SharedExpertWorkspaceViews:
        cached = workspace.views.get(tokens)
        if cached is not None:
            return cached
        if tokens < 0 or tokens > workspace.capacity:
            raise RuntimeError(
                f"shared expert workspace tokens={tokens} exceed capacity={workspace.capacity}"
            )
        # All MoE layers in one forward use the same token count. Keep only
        # that shape so variable-length traffic cannot accumulate view objects
        # for every historical T over the process lifetime.
        workspace.views.clear()
        cached = _SharedExpertWorkspaceViews(
            x_fp8=workspace.x_fp8[:tokens],
            x_scale=self._scale_view(workspace.x_scale_storage, tokens),
            gate_up_bf16=workspace.gate_up_bf16[:tokens],
            hidden_fp8=workspace.hidden_fp8[:tokens],
            hidden_scale=self._scale_view(workspace.hidden_scale_storage, tokens),
            out_bf16=workspace.out_bf16[:tokens],
        )
        workspace.views[tokens] = cached
        return cached

    def run(self, x: torch.Tensor) -> torch.Tensor:
        if not self.can_run(x):
            raise RuntimeError(
                "fused shared expert requires CUDA bf16 2D input and FP8 "
                "loader-merged shared w13/w2 weights"
            )
        return self.run_output(self.run_hidden(x))

    def _prepared_views(
        self, x: torch.Tensor
    ) -> tuple[_SharedExpertWorkspace, _SharedExpertWorkspaceViews]:
        workspace = self._ensure_workspace(x)
        return workspace, self._workspace_views(workspace, x.size(0))

    @staticmethod
    def _quantize_input(
        workspace: _SharedExpertWorkspace,
        views: _SharedExpertWorkspaceViews,
        x: torch.Tensor,
    ) -> None:
        if workspace.use_ue8m0:
            from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
                quant_bf16_fp8_packed_ue8m0,
            )

            quant_bf16_fp8_packed_ue8m0(
                x, views.x_fp8, views.x_scale, group_size=128, eps=1.0e-4
            )
        else:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            sgl_per_token_group_quant_fp8(
                x.contiguous(),
                group_size=128,
                eps=1.0e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                output_q=views.x_fp8,
                output_scale=views.x_scale,
            )

    def _swiglu(
        self, workspace: _SharedExpertWorkspace, views: _SharedExpertWorkspaceViews
    ) -> None:
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_f32_scale,
            silu_mul_fp8_quant_packed,
        )

        silu_mul_fp8_quant = (
            silu_mul_fp8_quant_packed
            if workspace.use_ue8m0
            else silu_mul_fp8_quant_f32_scale
        )
        silu_mul_fp8_quant(
            views.gate_up_bf16,
            clamp_limit=self.swiglu_limit,
            group_size=128,
            output_q=views.hidden_fp8,
            output_scale=views.hidden_scale,
        )

    def run_hidden(self, x: torch.Tensor) -> _SharedExpertWorkspaceViews:
        """Quantize ``x``, run w13 and quantized SwiGLU into the workspace.

        Returned views alias the shared workspace. Consume them with
        ``run_output`` before another invocation reuses that workspace;
        callers must order work across streams and must not run concurrently.
        """
        shared_experts = self._require_prepared()
        workspace, views = self._prepared_views(x)
        if x.size(0) == 0:
            return views
        self._quantize_input(workspace, views, x)
        shared_experts.w13.forward_quantized(
            views.x_fp8, views.x_scale, out=views.gate_up_bf16
        )
        self._swiglu(workspace, views)
        return views

    def run_output(self, views: _SharedExpertWorkspaceViews) -> torch.Tensor:
        """Run w2; the returned tensor aliases the same reusable workspace."""
        shared_experts = self._require_prepared()
        if views.out_bf16.size(0) == 0:
            return views.out_bf16
        return shared_experts.w2.forward_quantized(
            views.hidden_fp8, views.hidden_scale, out=views.out_bf16
        )


class SharedExpertExecutor(ABC):
    """One non-reentrant shared-expert invocation at a time.

    Prepare the module, then call begin before the gate, run_shared, after_dispatch,
    before_combine and finish. Gate packing may run after run_shared instead.
    begin and stage hooks may change the current CUDA stream. finish or abort
    restores the stream saved by begin; waits order GPU work without blocking
    the host. Workspace-backed outputs must be consumed before workspace reuse.
    """

    name: str

    def __init__(self) -> None:
        self._active = False
        self._entry_stream: torch.cuda.Stream | None = None
        self._prepared_shared_experts: nn.Module | None = None

    def prepare(
        self,
        shared_experts: nn.Module,
        *,
        comm_sm_counts: Callable[[int], tuple[int, int] | None] | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError(
                "cannot prepare a shared expert executor during execution"
            )
        self._prepared_shared_experts = shared_experts

    def _require_prepared(self) -> nn.Module:
        if self._prepared_shared_experts is None:
            raise RuntimeError(
                "shared expert executor must be prepared before execution"
            )
        return self._prepared_shared_experts

    def begin(self, x: torch.Tensor) -> None:
        """Begin before routing; subsequent work may use a partition stream."""
        if self._active:
            raise RuntimeError(
                "shared expert executor does not support reentrant calls"
            )
        self._require_prepared()
        self._active = True
        self._entry_stream = torch.cuda.current_stream(x.device) if x.is_cuda else None

    def _end(self) -> None:
        if self._entry_stream is not None:
            torch.cuda.set_stream(self._entry_stream)
        self._entry_stream = None
        self._active = False

    def after_dispatch(self) -> None:
        """Join shared input work before routed expert GEMMs are enqueued."""
        pass

    def before_combine(self) -> None:
        """Submit shared output work before combine is enqueued."""
        pass

    @abstractmethod
    def run_shared(self, x: torch.Tensor) -> None:
        """Submit shared computation according to this executor's schedule."""
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> torch.Tensor:
        """Join submitted work, restore the entry stream and return the result."""
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        """Order submitted work before reuse, restore the stream and discard state."""
        raise NotImplementedError


class SequentialSharedExpertExecutor(SharedExpertExecutor):
    name = "sequential"

    def __init__(
        self,
        fast_path: FusedSharedExpertFastPath | None = None,
    ) -> None:
        super().__init__()
        self._out: torch.Tensor | None = None
        self._fast_path = fast_path

    def prepare(
        self,
        shared_experts: nn.Module,
        *,
        comm_sm_counts: Callable[[int], tuple[int, int] | None] | None = None,
    ) -> None:
        super().prepare(shared_experts, comm_sm_counts=comm_sm_counts)
        if self._fast_path is not None:
            self._fast_path.prepare(shared_experts)

    def run_shared(self, x: torch.Tensor) -> None:
        shared_experts = self._require_prepared()
        with record_function_range("moe.shared_expert"):
            self._out = _run_shared_expert(shared_experts, x, self._fast_path)

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        out = self._out
        self._out = None
        self._end()
        return out

    def abort(self) -> None:
        self._out = None
        self._end()


class Sm100OverlapSharedExpertExecutor(SharedExpertExecutor):
    """Overlap the complete shared expert with routed MoE until finish."""

    name = "overlap"

    def __init__(self, fast_path: FusedSharedExpertFastPath | None = None) -> None:
        super().__init__()
        self._fast_path = fast_path
        self._stream: torch.cuda.Stream | None = None
        self._current: torch.cuda.Stream | None = None
        self._input: torch.Tensor | None = None
        self._out: torch.Tensor | None = None

    def prepare(
        self,
        shared_experts: nn.Module,
        *,
        comm_sm_counts: Callable[[int], tuple[int, int] | None] | None = None,
    ) -> None:
        super().prepare(shared_experts, comm_sm_counts=comm_sm_counts)
        if self._fast_path is not None:
            self._fast_path.prepare(shared_experts)
        device = _find_module_cuda_device(shared_experts)
        if device is not None:
            _ensure_shared_expert_stream(device)

    def _fork_stream(self, x: torch.Tensor) -> None:
        self._stream = _get_shared_expert_stream(
            x.device, allow_create=not torch.cuda.is_current_stream_capturing()
        )
        self._current = torch.cuda.current_stream(x.device)
        self._stream.wait_stream(self._current)
        self._input = x

    def _wait(self) -> None:
        if self._stream is not None:
            assert self._current is not None
            current = torch.cuda.current_stream(self._stream.device)
            current.wait_stream(self._stream)
            if current != self._current:
                self._current.wait_stream(self._stream)

    def _clear(self) -> None:
        self._out = None
        self._input = None
        self._stream = None
        self._current = None
        self._end()

    def finish(self) -> torch.Tensor:
        self._wait()
        assert self._out is not None
        out = self._out
        self._clear()
        return out

    def abort(self) -> None:
        self._wait()
        self._clear()

    def _can_overlap(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and torch.cuda.is_available()):
            return False
        threshold = int(
            os.environ.get("MOE_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096")
        )
        if x.shape[0] > threshold:
            return False
        return (
            not torch.cuda.is_current_stream_capturing()
            and not cuda_graph_warmup_forward_enabled()
            and os.environ.get("MOEDBG", "0") == "0"
        )

    def run_shared(self, x: torch.Tensor) -> None:
        shared_experts = self._require_prepared()
        if not self._can_overlap(x):
            with record_function_range("moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)
            return
        self._fork_stream(x)
        with torch.cuda.stream(self._stream), record_function_range(
            "moe.shared_expert"
        ):
            self._out = _run_shared_expert(shared_experts, x, self._fast_path)


class Sm90OverlapSharedExpertExecutor(SharedExpertExecutor):
    """Overlap the shared expert with routing and communication on SM90.

    quant, w13 and SwiGLU overlap the gate and dispatch; w2 overlaps combine.
    In each window the shared expert and the main-stream kernels run in
    complementary green-context partitions: the shared partition holds the
    SMs the communication kernel leaves free for this forward's tokens, so
    neither side can occupy the other's SMs. Routed expert GEMMs run on the
    original stream with every SM, after the dispatch window joins; the add
    waits for the combine window. A window without free SMs runs its
    shared-expert part on the current stream.
    """

    name = "overlap"

    def __init__(self, fast_path: FusedSharedExpertFastPath | None = None) -> None:
        super().__init__()
        self._fast_path = fast_path
        self._comm_sm_counts: Callable[[int], tuple[int, int] | None] | None = None
        self._windows: list[GreenContextPartition | None] = []
        self._dispatch_window: GreenContextPartition | None = None
        self._combine_window: GreenContextPartition | None = None
        self._overlap = False
        self._current: torch.cuda.Stream | None = None
        self._views: _SharedExpertWorkspaceViews | None = None
        self._out: torch.Tensor | None = None

    def prepare(
        self,
        shared_experts: nn.Module,
        *,
        comm_sm_counts: Callable[[int], tuple[int, int] | None] | None = None,
    ) -> None:
        super().prepare(shared_experts, comm_sm_counts=comm_sm_counts)
        self._comm_sm_counts = None
        self._windows = []
        if self._fast_path is not None:
            self._fast_path.prepare(shared_experts)
        device = _find_module_cuda_device(shared_experts)
        if comm_sm_counts is None or device is None or comm_sm_counts(1) is None:
            return
        try:
            from rtp_llm.models_py.modules.dsv4._green_context import (
                get_or_create_partition_pairs,
            )

            self._windows = get_or_create_partition_pairs(device)
        except (ImportError, RuntimeError) as exc:
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            key = (index, type(exc), str(exc))
            if key not in _GREEN_CONTEXT_FAILURES_LOGGED:
                logger.warning(
                    "Green context initialization failed on cuda:%s (%s: %s); "
                    "falling back to sequential shared-expert execution",
                    index,
                    type(exc).__name__,
                    exc,
                )
                _GREEN_CONTEXT_FAILURES_LOGGED.add(key)
            return
        self._comm_sm_counts = comm_sm_counts

    def _window(self, comm_sms: int) -> GreenContextPartition | None:
        """Largest window whose shared partition fits beside ``comm_sms``."""
        free_sms = get_num_device_sms() - comm_sms
        return next(
            (
                window
                for window in reversed(self._windows)
                if window is not None and window.compute_sms <= free_sms
            ),
            None,
        )

    @staticmethod
    def _enter(window: GreenContextPartition, current: torch.cuda.Stream) -> None:
        """Start a window after ``current`` and move the main stream into it."""
        window.compute_stream.wait_stream(current)
        window.comm_stream.wait_stream(current)
        torch.cuda.set_stream(window.comm_stream)

    @staticmethod
    def _leave(window: GreenContextPartition, current: torch.cuda.Stream) -> None:
        """Join both partitions of a window into ``current`` and return to it."""
        current.wait_stream(window.comm_stream)
        current.wait_stream(window.compute_stream)
        torch.cuda.set_stream(current)

    @staticmethod
    @contextmanager
    def _on_shared(window: GreenContextPartition | None) -> Iterator[None]:
        """Run shared-expert kernels in the window's shared partition."""
        if window is None:
            yield
            return
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            configure_deep_gemm_num_sms,
        )

        with torch.cuda.stream(window.compute_stream), configure_deep_gemm_num_sms(
            window.compute_sms
        ):
            yield

    def begin(self, x: torch.Tensor) -> None:
        super().begin(x)
        if (
            self._comm_sm_counts is None
            or self._fast_path is None
            or not self._fast_path.can_run(x)
            or os.environ.get("MOEDBG", "0") != "0"
        ):
            return
        comm_sms = self._comm_sm_counts(x.size(0))
        if comm_sms is None:
            return
        dispatch_sms, combine_sms = comm_sms
        self._dispatch_window = self._window(dispatch_sms)
        self._combine_window = self._window(combine_sms)
        self._overlap = (
            self._dispatch_window is not None or self._combine_window is not None
        )
        if not self._overlap:
            return
        self._current = torch.cuda.current_stream(x.device)
        if self._dispatch_window is not None:
            self._enter(self._dispatch_window, self._current)

    def run_shared(self, x: torch.Tensor) -> None:
        shared_experts = self._require_prepared()
        if not self._overlap:
            with record_function_range("moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)
            return
        assert self._fast_path is not None
        with record_function_range("moe.shared_expert"), self._on_shared(
            self._dispatch_window
        ):
            self._views = self._fast_path.run_hidden(x)

    def after_dispatch(self) -> None:
        if self._overlap and self._dispatch_window is not None:
            assert self._current is not None
            self._leave(self._dispatch_window, self._current)

    def before_combine(self) -> None:
        if not self._overlap:
            return
        assert self._fast_path is not None and self._current is not None
        assert self._views is not None
        window = self._combine_window
        if window is not None:
            self._enter(window, self._current)
        with record_function_range("moe.shared_expert_w2"), self._on_shared(window):
            self._out = self._fast_path.run_output(self._views)

    def finish(self) -> torch.Tensor:
        if self._overlap and self._combine_window is not None:
            assert self._current is not None
            self._leave(self._combine_window, self._current)
        assert self._out is not None
        out = self._out
        self._clear()
        return out

    def abort(self) -> None:
        if self._overlap:
            assert self._current is not None
            for window in (self._dispatch_window, self._combine_window):
                if window is not None:
                    self._current.wait_stream(window.comm_stream)
                    self._current.wait_stream(window.compute_stream)
            torch.cuda.set_stream(self._current)
        self._clear()

    def _clear(self) -> None:
        self._overlap = False
        self._dispatch_window = None
        self._combine_window = None
        self._current = None
        self._views = None
        self._out = None
        self._end()


def _run_shared_expert(
    shared_experts: nn.Module,
    x: torch.Tensor,
    fast_path: FusedSharedExpertFastPath | None,
) -> torch.Tensor:
    if fast_path is not None and fast_path.can_run(x):
        try:
            return fast_path.run(x)
        except Exception:
            if strict_fused_moe_enabled():
                raise
    if strict_fused_moe_enabled():
        raise RuntimeError(
            "MOE_STRICT_FUSED=1 forbids the generic Expert.forward "
            "shared-expert fallback"
        )
    return shared_experts(x).float()


def get_shared_expert_executor(
    max_tokens_per_rank: int | None = None,
    dim: int | None = None,
    inter_dim: int | None = None,
    swiglu_limit: float = 0.0,
) -> SharedExpertExecutor:
    mode = shared_expert_mode()
    sm = get_sm() if torch.cuda.is_available() else None
    sm90 = sm is not None and sm[0] == 9
    if mode == "auto":
        # Overlap is enabled on SM90, where its schedule was measured.
        mode = "overlap" if sm90 else "sequential"
    fast_path = FusedSharedExpertFastPath(
        max_tokens_per_rank=max_tokens_per_rank,
        dim=dim,
        inter_dim=inter_dim,
        swiglu_limit=swiglu_limit,
    )
    if mode == "sequential":
        return SequentialSharedExpertExecutor(fast_path)
    if mode == "overlap":
        if sm90:
            return Sm90OverlapSharedExpertExecutor(fast_path)
        if sm is not None and sm[0] == 10:
            return Sm100OverlapSharedExpertExecutor(fast_path)
        architecture = "no CUDA device" if sm is None else f"SM{sm[0]}{sm[1]}"
        raise ValueError(
            f"MOE_SHARED_EXPERT_MODE=overlap does not support {architecture}; "
            "supported architectures: SM90 (9.x), SM100 (10.x)"
        )
    raise ValueError(
        f"invalid MOE_SHARED_EXPERT_MODE={mode!r}; expected auto|sequential|overlap"
    )


def combine_routed_and_shared(
    routed: torch.Tensor,
    shared: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        if out.shape != routed.shape:
            raise ValueError(
                f"combined MoE output shape {tuple(out.shape)} does not match "
                f"routed output shape {tuple(routed.shape)}"
            )
        if out.dtype != out_dtype or out.device != routed.device:
            raise ValueError(
                "combined MoE output must use the requested dtype and routed device"
            )

    if os.environ.get("MOE_SHARED_EXPERT_BF16_ADD", "0") == "1":
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "MOE_STRICT_FUSED=1 forbids MOE_SHARED_EXPERT_BF16_ADD=1"
            )
        if out is None:
            return (routed.to(out_dtype) + shared.to(out_dtype)).to(out_dtype)
        torch.add(routed.to(out_dtype), shared.to(out_dtype), out=out)
        return out

    try:
        from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
            fused_moe_epilogue,
        )

        return fused_moe_epilogue(routed, shared, out_dtype, out=out)
    except Exception:
        if strict_fused_moe_enabled():
            raise
        combined = (routed.float() + shared.float()).to(out_dtype)
        if out is None:
            return combined
        out.copy_(combined)
        return out
