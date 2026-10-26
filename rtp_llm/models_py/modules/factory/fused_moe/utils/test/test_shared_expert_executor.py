import itertools
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.dsv4 import _green_context as green_context
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import Expert
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    _SHARED_EXPERT_STREAM_CACHE,
    _SHARED_EXPERT_WORKSPACE_CACHE,
    FusedSharedExpertFastPath,
    SequentialSharedExpertExecutor,
    Sm90OverlapSharedExpertExecutor,
    Sm100OverlapSharedExpertExecutor,
    W13SharedExpert,
    combine_routed_and_shared,
    get_shared_expert_executor,
)
from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
    quant_bf16_fp8_packed_ue8m0,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed_from_parts,
)
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import concat_0

_MODULE = "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert"


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class _Shared(nn.Module):
    def forward(self, x):
        return (x.float() * 0.25).to(x.dtype)


class _SharedWithCudaWeight(_Shared):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(1, device="cuda")


def _quant_weight(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight_bf16.shape
    if is_deep_gemm_e8m0_used():
        k_packed = (in_features + 511) // 512
        aligned_rows = FusedSharedExpertFastPath._tma_aligned_rows(
            out_features,
            torch.empty((), dtype=torch.int32).element_size(),
        )
        scale_storage = torch.empty(
            (k_packed, aligned_rows),
            dtype=torch.int32,
            device=weight_bf16.device,
        )
        scales = scale_storage.as_strided(
            (out_features, k_packed),
            (1, aligned_rows),
        )
        scales.fill_(0x7F7F7F7F)
        return weight_bf16.to(torch.float8_e4m3fn), scales
    return (
        weight_bf16.to(torch.float8_e4m3fn),
        torch.ones(
            (out_features + 127) // 128,
            (in_features + 127) // 128,
            dtype=torch.float32,
            device=weight_bf16.device,
        ),
    )


def _make_shared_expert(
    dim: int = 256,
    inter: int = 256,
    swiglu_limit: float = 0.0,
) -> tuple[W13SharedExpert, Expert]:
    torch.manual_seed(123)
    device = torch.device("cuda")
    w1_bf16 = torch.randn((inter, dim), device=device, dtype=torch.bfloat16) * 0.05
    w2_bf16 = torch.randn((dim, inter), device=device, dtype=torch.bfloat16) * 0.05
    w3_bf16 = torch.randn((inter, dim), device=device, dtype=torch.bfloat16) * 0.05
    w1_w, w1_s = _quant_weight(w1_bf16)
    w2_w, w2_s = _quant_weight(w2_bf16)
    w3_w, w3_s = _quant_weight(w3_bf16)
    w13_w, w13_s = _quant_weight(concat_0([w1_bf16, w3_bf16]))
    split_ref = Expert(
        dim,
        inter,
        swiglu_limit=swiglu_limit,
        storage="fp8",
        expert_weights={
            "w1_w": w1_w,
            "w1_s": w1_s,
            "w2_w": w2_w,
            "w2_s": w2_s,
            "w3_w": w3_w,
            "w3_s": w3_s,
        },
    )
    shared_w13 = W13SharedExpert(
        dim,
        inter,
        expert_weights={
            "w13_w": w13_w,
            "w13_s": w13_s,
            "w2_w": w2_w,
            "w2_s": w2_s,
        },
        swiglu_limit=swiglu_limit,
    )
    return shared_w13, split_ref


def _split_reference(
    shared: Expert,
    x: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

    T, D = x.shape
    inter = shared.w1.weight.shape[0]
    if not is_deep_gemm_e8m0_used():
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        out = torch.empty((T, D), dtype=torch.bfloat16, device=x.device)
        if T == 0:
            return out
        x_parts = sgl_per_token_group_quant_fp8(x, 128, eps=1e-4)
        gate = torch.empty((T, inter), dtype=torch.bfloat16, device=x.device)
        up = torch.empty_like(gate)
        fp8_gemm_nt(x_parts, (shared.w1.weight, shared.w1.weight_scales), gate)
        fp8_gemm_nt(x_parts, (shared.w3.weight, shared.w3.weight_scales), up)
        gate, up = gate.float(), up.float()
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        hidden_parts = sgl_per_token_group_quant_fp8(
            hidden, 128, eps=1e-10 * torch.finfo(torch.float8_e4m3fn).max
        )
        fp8_gemm_nt(hidden_parts, (shared.w2.weight, shared.w2.weight_scales), out)
        return out

    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_scale_storage = FusedSharedExpertFastPath._scale_storage(
        (D // 128 + 3) // 4,
        max(T, 1),
        x.device,
        torch.int32,
    )
    x_scale = FusedSharedExpertFastPath._scale_view(x_scale_storage, T)
    gate = torch.empty((T, inter), dtype=torch.bfloat16, device=x.device)
    up = torch.empty((T, inter), dtype=torch.bfloat16, device=x.device)
    hidden_fp8 = torch.empty((T, inter), dtype=torch.float8_e4m3fn, device=x.device)
    hidden_scale_storage = FusedSharedExpertFastPath._scale_storage(
        (inter // 128 + 3) // 4,
        max(T, 1),
        x.device,
        torch.int32,
    )
    hidden_scale = FusedSharedExpertFastPath._scale_view(hidden_scale_storage, T)
    out = torch.empty((T, D), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    quant_bf16_fp8_packed_ue8m0(x, x_fp8, x_scale, group_size=128, eps=1.0e-4)
    fp8_gemm_nt(
        (x_fp8, x_scale),
        (shared.w1.weight, shared.w1.weight_scales),
        gate,
        disable_ue8m0_cast=False,
    )
    fp8_gemm_nt(
        (x_fp8, x_scale),
        (shared.w3.weight, shared.w3.weight_scales),
        up,
        disable_ue8m0_cast=False,
    )
    silu_mul_fp8_quant_packed_from_parts(
        gate,
        up,
        clamp_limit=swiglu_limit,
        group_size=128,
        output_q=hidden_fp8,
        output_scale=hidden_scale,
    )
    fp8_gemm_nt(
        (hidden_fp8, hidden_scale),
        (shared.w2.weight, shared.w2.weight_scales),
        out,
        disable_ue8m0_cast=False,
    )
    return out


def _fake_fp8_gemm_nt(a, b, output, *args, **kwargs) -> None:
    del args, kwargs

    def dequantize(parts):
        values, scales = parts
        if scales.dtype == torch.int32:
            shifts = torch.arange(4, device=scales.device) * 8
            exponents = (scales.unsqueeze(-1) >> shifts) & 255
            scales = torch.exp2(exponents.float() - 127).flatten(-2)
            scales = scales[:, : values.shape[1] // 128]
        elif scales.shape[0] != values.shape[0]:
            scales = scales.repeat_interleave(128, dim=0)
        return values.float() * scales.repeat_interleave(128, dim=-1)

    output.copy_((dequantize(a) @ dequantize(b).t()).to(torch.bfloat16))


class TestSharedExpertExecutor(unittest.TestCase):
    @staticmethod
    def _prepare_only_shared(inter: int = 256, dim: int = 256) -> nn.Module:
        shared = nn.Module()
        shared.w13 = mock.Mock(
            weight=torch.empty((2 * inter, dim)),
            weight_scales=torch.empty((2 * inter, 1)),
        )
        shared.w2 = mock.Mock(
            weight=torch.empty((dim, inter)),
            weight_scales=torch.empty((dim, 1)),
        )
        return shared

    def test_fast_path_prepare_infers_inter_dim(self):
        fast_path = FusedSharedExpertFastPath(dim=256, inter_dim=None)

        fast_path.prepare(self._prepare_only_shared(inter=256))

        self.assertEqual(fast_path.inter_dim, 256)

    def test_fast_path_prepare_uses_weight_inter_dim(self):
        fast_path = FusedSharedExpertFastPath(dim=256, inter_dim=128)

        fast_path.prepare(self._prepare_only_shared(inter=256))

        self.assertEqual(fast_path.inter_dim, 256)

    def test_fast_path_requires_prepare(self):
        fast_path = FusedSharedExpertFastPath()
        for method in (
            fast_path.can_run,
            fast_path.run,
            fast_path.run_hidden,
            fast_path.run_output,
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(RuntimeError, "must be prepared"):
                    method(mock.Mock())

    def test_fast_path_runs_bound_module_without_repreparing(self):
        weight = mock.Mock(shape=(512, 256))
        weight.dim.return_value = 2
        scale = mock.Mock()
        scale.dim.return_value = 2
        shared = SimpleNamespace(
            w13=mock.Mock(weight=weight, weight_scales=scale),
            w2=mock.Mock(weight=weight, weight_scales=scale),
        )
        fast_path = FusedSharedExpertFastPath()
        fast_path.prepare(shared)
        x = mock.Mock(is_cuda=True, dtype=torch.bfloat16)
        x.dim.return_value = 2
        x.size.return_value = 7
        views = mock.Mock()
        views.out_bf16.size.return_value = 7
        shared.w2.forward_quantized.return_value = views.out_bf16
        with mock.patch.object(fast_path, "prepare") as prepare, mock.patch.object(
            fast_path, "_prepared_views", return_value=(mock.Mock(), views)
        ), mock.patch.object(fast_path, "_quantize_input"), mock.patch.object(
            fast_path, "_swiglu"
        ):
            self.assertIs(fast_path.run(x), views.out_bf16)
            prepare.assert_not_called()
        shared.w13.forward_quantized.assert_called_once_with(
            views.x_fp8, views.x_scale, out=views.gate_up_bf16
        )
        shared.w2.forward_quantized.assert_called_once_with(
            views.hidden_fp8, views.hidden_scale, out=views.out_bf16
        )

    def test_workspace_views_are_shared_and_invalidated_on_growth(self):
        for use_ue8m0 in (False, True):
            with self.subTest(use_ue8m0=use_ue8m0), mock.patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=use_ue8m0,
            ):
                _SHARED_EXPERT_WORKSPACE_CACHE.clear()
                self.addCleanup(_SHARED_EXPERT_WORKSPACE_CACHE.clear)
                first = FusedSharedExpertFastPath(
                    max_tokens_per_rank=4,
                    dim=128,
                    inter_dim=128,
                )
                second = FusedSharedExpertFastPath(
                    max_tokens_per_rank=4,
                    dim=128,
                    inter_dim=128,
                )
                grower = FusedSharedExpertFastPath(
                    max_tokens_per_rank=9,
                    dim=128,
                    inter_dim=128,
                )
                x3 = torch.empty((3, 128), dtype=torch.bfloat16)

                def aligned(rows: int, element_size: int) -> int:
                    del element_size
                    return ((rows + 7) // 8) * 8

                with mock.patch.object(
                    FusedSharedExpertFastPath,
                    "_tma_aligned_rows",
                    side_effect=aligned,
                ):
                    workspace = first._ensure_workspace(x3)
                    views = first._workspace_views(workspace, 3)
                    self.assertIs(first._workspace_views(workspace, 3), views)

                    second_workspace = second._ensure_workspace(x3)
                    self.assertIs(second_workspace, workspace)
                    self.assertIs(second._workspace_views(second_workspace, 3), views)
                    self.assertEqual(tuple(views.x_fp8.shape), (3, 128))
                    self.assertEqual(tuple(views.x_scale.shape), (3, 1))
                    expected_scale_dtype = torch.int32 if use_ue8m0 else torch.float32
                    self.assertEqual(views.x_scale.dtype, expected_scale_dtype)
                    self.assertEqual(views.hidden_scale.dtype, expected_scale_dtype)
                    self.assertEqual(tuple(views.gate_up_bf16.shape), (3, 256))
                    self.assertEqual(tuple(views.hidden_fp8.shape), (3, 128))
                    self.assertEqual(tuple(views.hidden_scale.shape), (3, 1))
                    self.assertEqual(tuple(views.out_bf16.shape), (3, 128))

                    old_x_fp8 = workspace.x_fp8
                    grown_workspace = grower._ensure_workspace(
                        torch.empty((9, 128), dtype=torch.bfloat16)
                    )
                    self.assertIsNot(grown_workspace, workspace)
                    self.assertEqual(grown_workspace.capacity, 9)
                    self.assertEqual(grown_workspace.views, {})
                    self.assertIs(workspace.x_fp8, old_x_fp8)
                    self.assertEqual(workspace.capacity, 4)
                    self.assertIs(first._workspace, workspace)

                    first_grown = first._ensure_workspace(
                        torch.empty((9, 128), dtype=torch.bfloat16)
                    )
                    self.assertIs(first_grown, grown_workspace)
                    refreshed = first._workspace_views(first_grown, 3)
                    self.assertIsNot(refreshed, views)
                    self.assertIs(refreshed.x_fp8._base, grown_workspace.x_fp8)

                    # Variable-length traffic retains one cached shape, while callers
                    # holding previous views keep their original buffer references.
                    other_shape = first._workspace_views(first_grown, 5)
                    self.assertEqual(set(first_grown.views), {5})
                    self.assertEqual(tuple(other_shape.x_fp8.shape), (5, 128))
                    self.assertEqual(tuple(refreshed.x_fp8.shape), (3, 128))

                    first.prepare(self._prepare_only_shared(inter=256, dim=128))
                    resized = first._ensure_workspace(x3)
                    self.assertEqual(resized.hidden_fp8.size(1), 256)
                    self.assertEqual(grown_workspace.hidden_fp8.size(1), 128)

    def test_combine_preserves_fp32_accumulate_semantics(self):
        routed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        shared = torch.tensor([[0.5, -0.25], [0.125, -0.5]], dtype=torch.float32)
        with _env("MOE_STRICT_FUSED", "0"):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.float() + shared.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_bf16_add_experimental_switch(self):
        routed = torch.randn(4, 8, dtype=torch.float32)
        shared = torch.randn(4, 8, dtype=torch.float32)
        with _env("MOE_STRICT_FUSED", "0"), _env("MOE_SHARED_EXPERT_BF16_ADD", "1"):
            got = combine_routed_and_shared(routed, shared, torch.bfloat16)
        ref = (routed.to(torch.bfloat16) + shared.to(torch.bfloat16)).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_strict_rejects_bf16_add_switch(self):
        routed = torch.randn(4, 8, dtype=torch.float32)
        shared = torch.randn(4, 8, dtype=torch.float32)
        with _env("MOE_SHARED_EXPERT_BF16_ADD", "1"):
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                combine_routed_and_shared(routed, shared, torch.bfloat16)

    def test_strict_rejects_generic_shared_path(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        executor.prepare(_Shared())
        with self.assertRaisesRegex(RuntimeError, "generic Expert.forward"):
            executor.begin(x)
            executor.run_shared(x)

    def test_executor_dispatch(self):
        cases = (
            ("auto", (9, 0), Sm90OverlapSharedExpertExecutor),
            ("auto", (10, 0), SequentialSharedExpertExecutor),
            ("auto", (12, 0), SequentialSharedExpertExecutor),
            ("auto", None, SequentialSharedExpertExecutor),
            ("sequential", (9, 0), SequentialSharedExpertExecutor),
            ("sequential", (12, 0), SequentialSharedExpertExecutor),
            ("sequential", None, SequentialSharedExpertExecutor),
            ("overlap", (9, 0), Sm90OverlapSharedExpertExecutor),
            ("overlap", (10, 0), Sm100OverlapSharedExpertExecutor),
            ("overlap", (8, 0), None),
            ("overlap", (12, 0), None),
            ("overlap", None, None),
        )
        for mode, sm, expected in cases:
            with self.subTest(mode=mode, sm=sm), _env(
                "MOE_SHARED_EXPERT_MODE", mode
            ), mock.patch(
                "torch.cuda.is_available", return_value=sm is not None
            ), mock.patch(
                f"{_MODULE}.get_sm", return_value=sm
            ):
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "supported architectures"):
                        get_shared_expert_executor()
                else:
                    self.assertIsInstance(get_shared_expert_executor(), expected)

    def test_sequential_executor(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        executor = SequentialSharedExpertExecutor()
        executor.prepare(_Shared())
        with _env("MOE_STRICT_FUSED", "0"):
            executor.begin(x)
            executor.run_shared(x)
        got = executor.finish()
        ref = _Shared()(x).float()
        self.assertTrue(torch.equal(got, ref))

    def test_only_sm90_binds_communication_budget(self):
        router = mock.Mock()
        router.comm_sm_counts.return_value = (64, 64)
        shared = mock.Mock()
        x = mock.Mock(is_cuda=False)
        x.size.return_value = 5
        with mock.patch(f"{_MODULE}._find_module_cuda_device", return_value=None):
            for executor_type in (
                SequentialSharedExpertExecutor,
                Sm100OverlapSharedExpertExecutor,
            ):
                executor = executor_type()
                executor.prepare(shared, comm_sm_counts=router.comm_sm_counts)
                executor.begin(x)
                executor.after_dispatch()
                executor.before_combine()
                executor.abort()
                router.comm_sm_counts.assert_not_called()

            executor = Sm90OverlapSharedExpertExecutor(mock.Mock())
            executor.prepare(shared, comm_sm_counts=router.comm_sm_counts)
            router.comm_sm_counts.assert_not_called()
        with mock.patch(
            f"{_MODULE}._find_module_cuda_device", return_value=torch.device("cuda")
        ), mock.patch.object(
            green_context, "get_or_create_partition_pairs", return_value=[None]
        ):
            executor.prepare(shared, comm_sm_counts=router.comm_sm_counts)
        router.comm_sm_counts.assert_called_once_with(1)

    def test_executor_binding_survives_finish_and_abort(self):
        from contextlib import nullcontext

        x = mock.Mock(is_cuda=False)
        for executor_type in (
            SequentialSharedExpertExecutor,
            Sm90OverlapSharedExpertExecutor,
            Sm100OverlapSharedExpertExecutor,
        ):
            with self.subTest(executor=executor_type):
                fast_path = mock.Mock()
                executor = executor_type(fast_path)
                for method in (executor.begin, executor.run_shared):
                    with self.assertRaisesRegex(RuntimeError, "must be prepared"):
                        method(x)
                shared = mock.Mock()
                with mock.patch(
                    f"{_MODULE}._find_module_cuda_device", return_value=None
                ):
                    executor.prepare(shared)
                fast_path.prepare.assert_called_once_with(shared)
                with mock.patch(f"{_MODULE}._run_shared_expert") as run, mock.patch(
                    f"{_MODULE}.record_function_range",
                    side_effect=lambda *args: nullcontext(),
                ):
                    for discard in (False, True, False):
                        executor.begin(x)
                        with self.assertRaisesRegex(RuntimeError, "during execution"):
                            executor.prepare(mock.Mock())
                        executor.run_shared(x)
                        run.assert_called_with(shared, x, fast_path)
                        if discard:
                            executor.abort()
                        else:
                            self.assertIs(executor.finish(), run.return_value)
                        self.assertIs(executor._prepared_shared_experts, shared)
                fast_path.prepare.assert_called_once_with(shared)

    def test_green_context_failure_logs_once_and_retries(self):
        budget = mock.Mock(return_value=(64, 64))
        shared = mock.Mock()
        with mock.patch(f"{_MODULE}._GREEN_CONTEXT_FAILURES_LOGGED", set()), mock.patch(
            f"{_MODULE}.logger"
        ) as logger, mock.patch(
            f"{_MODULE}._find_module_cuda_device", return_value=torch.device("cuda:0")
        ) as device, mock.patch.object(
            green_context,
            "get_or_create_partition_pairs",
            side_effect=RuntimeError("setup failed"),
        ) as setup:
            first, second = (
                Sm90OverlapSharedExpertExecutor(),
                Sm90OverlapSharedExpertExecutor(),
            )
            first.prepare(shared, comm_sm_counts=budget)
            second.prepare(shared, comm_sm_counts=budget)
            self.assertEqual(setup.call_count, 2)
            logger.warning.assert_called_once()
            self.assertIn(
                "falling back to sequential", logger.warning.call_args.args[0]
            )
            self.assertIsNone(second._comm_sm_counts)
            setup.side_effect = ImportError("bindings missing")
            second.prepare(shared, comm_sm_counts=budget)
            self.assertEqual(logger.warning.call_count, 2)
            device.return_value = torch.device("cuda:1")
            second.prepare(shared, comm_sm_counts=budget)
            self.assertEqual(logger.warning.call_count, 3)
            setup.side_effect = None
            setup.return_value = [None]
            second.prepare(shared, comm_sm_counts=budget)
            self.assertIs(second._comm_sm_counts, budget)
            self.assertEqual(logger.warning.call_count, 3)

    def test_window_selection_respects_communication_budget(self):
        executor = Sm90OverlapSharedExpertExecutor()
        windows = [None] + [SimpleNamespace(compute_sms=n) for n in (8, 16, 24)]
        executor._windows = windows
        with mock.patch(f"{_MODULE}.get_num_device_sms", return_value=28):
            for budget, expected in (
                (4, 3),
                (5, 2),
                (12, 2),
                (20, 1),
                (21, 0),
                (32, 0),
            ):
                with self.subTest(budget=budget):
                    self.assertIs(executor._window(budget), windows[expected])

    def test_executor_lifecycle_restores_entry_stream(self):
        from contextlib import nullcontext

        for executor_type in (
            SequentialSharedExpertExecutor,
            Sm90OverlapSharedExpertExecutor,
            Sm100OverlapSharedExpertExecutor,
        ):
            for fail in (False, True):
                with self.subTest(executor=executor_type, fail=fail):
                    executor = executor_type()
                    with mock.patch(
                        f"{_MODULE}._find_module_cuda_device", return_value=None
                    ):
                        executor.prepare(mock.Mock())
                    entry, other = mock.Mock(), mock.Mock()
                    current = [entry]
                    x = mock.Mock(is_cuda=True)
                    output = mock.Mock()
                    with mock.patch(
                        "torch.cuda.current_stream",
                        side_effect=lambda *args: current[0],
                    ), mock.patch(
                        "torch.cuda.set_stream",
                        side_effect=lambda stream: current.__setitem__(0, stream),
                    ), mock.patch(
                        f"{_MODULE}._run_shared_expert", return_value=output
                    ), mock.patch(
                        f"{_MODULE}.record_function_range",
                        side_effect=lambda *args: nullcontext(),
                    ), mock.patch.object(
                        Sm100OverlapSharedExpertExecutor,
                        "_can_overlap",
                        return_value=False,
                    ):
                        executor.begin(x)
                        with self.assertRaisesRegex(RuntimeError, "reentrant"):
                            executor.begin(x)
                        executor.run_shared(x)
                        executor.after_dispatch()
                        executor.before_combine()
                        current[0] = other
                        if fail:
                            executor.abort()
                        else:
                            self.assertIs(executor.finish(), output)
                        self.assertIs(current[0], entry)
                        self.assertFalse(executor._active)
                        executor.begin(x)
                        executor.abort()

    def test_sm90_stage_failure_joins_windows_and_restores_stream(self):
        from contextlib import nullcontext

        for fail in (False, True):
            with self.subTest(fail=fail):
                fast_path = mock.Mock()
                executor = Sm90OverlapSharedExpertExecutor(fast_path)
                entry = mock.Mock()
                current = [entry]
                window = SimpleNamespace(
                    compute_stream=mock.Mock(), comm_stream=mock.Mock()
                )
                shared, x = mock.Mock(), mock.Mock(is_cuda=True)
                executor._prepared_shared_experts = shared
                executor._comm_sm_counts = mock.Mock(return_value=(64, 64))
                with mock.patch(
                    "torch.cuda.current_stream", side_effect=lambda *args: current[0]
                ), mock.patch(
                    "torch.cuda.set_stream",
                    side_effect=lambda stream: current.__setitem__(0, stream),
                ), mock.patch.object(
                    executor, "_window", return_value=window
                ), mock.patch.object(
                    executor, "_on_shared", side_effect=lambda *args: nullcontext()
                ), mock.patch(
                    f"{_MODULE}.record_function_range",
                    side_effect=lambda *args: nullcontext(),
                ), _env(
                    "MOEDBG", "0"
                ):
                    executor.begin(x)
                    self.assertIs(current[0], window.comm_stream)
                    executor.run_shared(x)
                    executor.after_dispatch()
                    self.assertIs(current[0], entry)
                    if fail:
                        fast_path.run_output.side_effect = RuntimeError("w2 failed")
                        with self.assertRaisesRegex(RuntimeError, "w2 failed"):
                            executor.before_combine()
                        executor.abort()
                    else:
                        executor.before_combine()
                        self.assertIs(current[0], window.comm_stream)
                        self.assertIs(
                            executor.finish(), fast_path.run_output.return_value
                        )
                    self.assertIs(current[0], entry)
                    entry.wait_stream.assert_any_call(window.compute_stream)
                    entry.wait_stream.assert_any_call(window.comm_stream)
                    self.assertFalse(executor._active)
                    self.assertIsNone(executor._views)

    def test_sm90_prepare_clears_previous_router(self):
        executor = Sm90OverlapSharedExpertExecutor()
        shared = mock.Mock()
        supported = mock.Mock()
        supported.comm_sm_counts.return_value = (64, 64)
        unsupported = mock.Mock()
        unsupported.comm_sm_counts.return_value = None
        with mock.patch(
            f"{_MODULE}._find_module_cuda_device", return_value=torch.device("cuda:0")
        ), mock.patch.object(
            green_context, "get_or_create_partition_pairs", return_value=[None]
        ) as windows:
            for router in (None, unsupported, supported):
                executor.prepare(shared, comm_sm_counts=supported.comm_sm_counts)
                self.assertIsNotNone(executor._comm_sm_counts)
                windows.side_effect = RuntimeError("setup failed")
                executor.prepare(
                    shared,
                    comm_sm_counts=None if router is None else router.comm_sm_counts,
                )
                self.assertIsNone(executor._comm_sm_counts)
                self.assertEqual(executor._windows, [])
                windows.side_effect = None

    def test_sm100_overlap_waits_only_at_finish(self):
        from contextlib import nullcontext

        module = (
            "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert"
        )
        executor = Sm100OverlapSharedExpertExecutor()
        with mock.patch(f"{_MODULE}._find_module_cuda_device", return_value=None):
            executor.prepare(mock.Mock())
        current, auxiliary = mock.Mock(), mock.Mock()
        output = mock.Mock()
        x = mock.Mock()
        with mock.patch.object(executor, "_can_overlap", return_value=True), mock.patch(
            f"{module}._get_shared_expert_stream", return_value=auxiliary
        ), mock.patch("torch.cuda.current_stream", return_value=current), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        ), mock.patch(
            "torch.cuda.set_stream"
        ), mock.patch(
            "torch.cuda.stream", side_effect=lambda stream: nullcontext()
        ), mock.patch(
            f"{module}._run_shared_expert", return_value=output
        ):
            executor.begin(x)
            executor.run_shared(x)
            auxiliary.wait_stream.assert_called_once_with(current)
            current.wait_stream.assert_not_called()
            self.assertIs(executor._input, x)
            self.assertIs(executor.finish(), output)
            current.wait_stream.assert_called_once_with(auxiliary)
            self.assertIsNone(executor._input)

    def test_sm100_preserves_overlap_fallbacks(self):
        executor = Sm100OverlapSharedExpertExecutor()
        x = mock.Mock(is_cuda=True, shape=(5, 128))
        for tokens, capturing, warmup, debug, expected in (
            (5, False, "0", "0", True),
            (4097, False, "0", "0", False),
            (5, True, "0", "0", False),
            (5, False, "1", "0", False),
            (5, False, "0", "1", False),
        ):
            with self.subTest(
                tokens=tokens, capturing=capturing, warmup=warmup, debug=debug
            ), mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ), _env(
                "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", warmup
            ), _env(
                "MOEDBG", debug
            ), _env(
                "MOE_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096"
            ):
                x.shape = (tokens, 128)
                self.assertEqual(executor._can_overlap(x), expected)

    @staticmethod
    def _make_fused_executors(dim: int = 256, swiglu_limit: float = 1.5, comm_sms=None):
        def fast_path():
            return FusedSharedExpertFastPath(
                max_tokens_per_rank=64,
                dim=dim,
                inter_dim=256,
                swiglu_limit=swiglu_limit,
            )

        shared, _ = _make_shared_expert(dim=dim, swiglu_limit=swiglu_limit)
        sequential = SequentialSharedExpertExecutor(fast_path())
        overlap = Sm90OverlapSharedExpertExecutor(fast_path())
        sequential.prepare(shared)
        overlap.prepare(shared, comm_sm_counts=mock.Mock(return_value=comm_sms))
        return shared, sequential, overlap

    @staticmethod
    def _run_layer_schedule(executor, shared, x, routed_weight):
        """Issue the executor hooks in the MoE layer order around routed work."""
        executor.begin(x)
        routed = x @ routed_weight
        executor.run_shared(x)
        routed = routed @ routed_weight
        executor.after_dispatch()
        routed = routed @ routed_weight
        executor.before_combine()
        return executor.finish() + 0 * routed

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_matches_sequential(self):
        shared, sequential, overlap = self._make_fused_executors()
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        for tokens, comm_sms in itertools.product(
            (1, 16, 33, 64), self._comm_sms_cases()
        ):
            with self.subTest(tokens=tokens, comm_sms=comm_sms):
                overlap.prepare(shared, comm_sm_counts=mock.Mock(return_value=comm_sms))
                x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
                ref = self._run_layer_schedule(sequential, shared, x, weight)
                got = self._run_layer_schedule(overlap, shared, x, weight)
                self.assertTrue(torch.equal(got, ref))

    @staticmethod
    def _comm_sms_cases():
        """Router budgets giving both, one or no green partitions."""
        from rtp_llm.models_py.utils.arch import get_num_device_sms

        n = get_num_device_sms()
        return (None, (n - 16, n - 8), (n, n - 8), (n - 16, n), (n, n))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_replays_captured_schedule(self):
        from rtp_llm.models_py.utils.arch import get_num_device_sms

        n = get_num_device_sms()
        shared, sequential, overlap = self._make_fused_executors(dim=1024)
        # Combine needs more SMs for larger batches, as in ACCL-EP.
        overlap.prepare(
            shared,
            comm_sm_counts=mock.Mock(
                side_effect=lambda tokens: (
                    n - 16,
                    n - 24 if tokens < 32 else n - 8,
                )
            ),
        )
        weight = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        for tokens in (1, 33, 64):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 1024, device="cuda", dtype=torch.bfloat16)
                out = torch.empty(tokens, 1024, device="cuda", dtype=torch.bfloat16)

                def forward():
                    out.copy_(self._run_layer_schedule(overlap, shared, x, weight))

                forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()

                x.copy_(torch.randn_like(x))
                graph.replay()
                torch.cuda.synchronize()
                ref = self._run_layer_schedule(sequential, shared, x, weight)
                self.assertTrue(torch.equal(out, ref))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_sm100_whole_expert_overlap_and_capture_fallback(self):
        shared, sequential, _ = self._make_fused_executors()
        overlap = Sm100OverlapSharedExpertExecutor(sequential._fast_path)
        overlap.prepare(shared)
        x = torch.randn(33, 256, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        with _env("MOEDBG", "0"), _env("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", "0"), _env(
            "MOE_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096"
        ):
            overlap.begin(x)
            overlap.run_shared(x)
            self.assertIsNotNone(overlap._stream)
            out.copy_(overlap.finish())
            sequential.begin(x)
            sequential.run_shared(x)
            self.assertTrue(torch.equal(out, sequential.finish()))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                overlap.begin(x)
                overlap.run_shared(x)
                self.assertIsNone(overlap._stream)
                out.copy_(overlap.finish())
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.cuda.synchronize()
            sequential.begin(x)
            sequential.run_shared(x)
            self.assertTrue(torch.equal(out, sequential.finish()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_executor_runs_generic_shared_expert_sequentially(self):
        overlap = Sm90OverlapSharedExpertExecutor()
        generic = _Shared().cuda()
        overlap.prepare(generic)
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        with _env("MOE_STRICT_FUSED", "0"):
            overlap.begin(x)
            self.assertFalse(overlap._overlap)
            overlap.run_shared(x)
        self.assertTrue(torch.equal(overlap.finish(), generic(x).float()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_overlap_prepare_precreates_device_stream(self):
        _SHARED_EXPERT_STREAM_CACHE.clear()
        shared = _SharedWithCudaWeight()
        executor = Sm100OverlapSharedExpertExecutor()

        executor.prepare(shared)

        device_index = shared.weight.device.index
        assert device_index is not None
        self.assertIn(device_index, _SHARED_EXPERT_STREAM_CACHE)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_merged_w13_matches_split_reference(self):
        swiglu_limit = 1.5
        shared, split_ref = _make_shared_expert(swiglu_limit=swiglu_limit)
        executor = FusedSharedExpertFastPath(
            max_tokens_per_rank=64,
            dim=256,
            inter_dim=256,
            swiglu_limit=swiglu_limit,
        )
        executor.prepare(shared)
        w13_w, w13_s = FusedSharedExpertFastPath._linear_parts(shared.w13)
        self.assertEqual(
            w13_w.shape,
            (512, 256),
        )
        if w13_s.dtype == torch.int32:
            self.assertEqual(w13_s.shape[0], 512)
            self.assertEqual(w13_s.stride(0), 1)
        else:
            self.assertEqual(w13_s.dtype, torch.float32)
            self.assertEqual(w13_s.shape, (4, 2))
        self.assertFalse(hasattr(shared, "w1"))
        self.assertFalse(hasattr(shared, "w3"))

        for tokens in (0, 1, 33):
            with self.subTest(tokens=tokens):
                torch.manual_seed(1000 + tokens)
                x = torch.randn(
                    (tokens, 256),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                gemm_calls = []

                def fake_with_record(a, b, output, *args, **kwargs):
                    gemm_calls.append(
                        (tuple(a[0].shape), tuple(b[0].shape), tuple(output.shape))
                    )
                    _fake_fp8_gemm_nt(a, b, output, *args, **kwargs)

                with mock.patch(
                    "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.fp8_gemm_nt",
                    side_effect=_fake_fp8_gemm_nt,
                ), mock.patch(
                    "rtp_llm.models_py.modules.factory.linear.impl.cuda."
                    "fp8_deepgemm_linear.fp8_gemm_nt",
                    side_effect=fake_with_record,
                ), mock.patch(
                    "rtp_llm.models_py.modules.factory.linear.impl.cuda."
                    "fp8_gemm_linear.CudaFp8GEMMLinear._should_use_flashinfer",
                    return_value=False,
                ):
                    got = executor.run(x)
                    merged_calls = list(gemm_calls)
                    ref = _split_reference(split_ref, x, swiglu_limit)
                if tokens == 0:
                    self.assertEqual(tuple(got.shape), (0, 256))
                    self.assertEqual(merged_calls, [])
                    continue
                self.assertEqual(
                    merged_calls,
                    [
                        ((tokens, 256), (512, 256), (tokens, 512)),
                        ((tokens, 256), (256, 256), (tokens, 256)),
                    ],
                )
                self.assertLess(calc_diff(got, ref), 0.0011)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_w13_generic_forward_matches_split_expert(self):
        swiglu_limit = 1.5
        shared, split_ref = _make_shared_expert(swiglu_limit=swiglu_limit)
        x = torch.randn((5, 256), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(hasattr(shared, "w1"))
        self.assertFalse(hasattr(shared, "w3"))

        with mock.patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear.fp8_gemm_nt",
            side_effect=_fake_fp8_gemm_nt,
        ):
            got = shared(x)
            ref = split_ref(x)
        self.assertLess(calc_diff(got, ref), 0.0011)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_strict_fused_requires_prepared_w13(self):
        _, shared = _make_shared_expert()
        executor = FusedSharedExpertFastPath(
            max_tokens_per_rank=8,
            dim=256,
            inter_dim=256,
        )
        with self.assertRaisesRegex(RuntimeError, "loader-prepared w13"):
            executor.prepare(shared)


if __name__ == "__main__":
    unittest.main()
