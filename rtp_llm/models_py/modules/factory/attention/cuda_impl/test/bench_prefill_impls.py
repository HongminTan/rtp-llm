"""Cross-impl prefill MHA benchmark.

Iterates over PREFILL_MHA_IMPS and, for every (q_dtype, kv_cache_dtype, seq_len,
reuse_cache_ratio) combination, runs `impl.forward()` on each impl whose
`support()` returns True. Reports latency (mean / p50 / p95 / p99) and TFLOPs/s.

Usage:
    bazelisk test --config=cuda12_9 --config=sm9x \
        --run_under=//rtp_llm/test/utils:gpu_lock \
        //rtp_llm/models_py/modules/factory/attention/cuda_impl/test:bench_prefill_impls \
        --test_timeout=1800 \
        --test_output=streamed \
        --nocache_test_results
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention import PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench_utils import (
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time_with_cudagraph,
    set_seed,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
    init_exec_ctx,
)

# Input-layout dispatch:
#   These impls' inner FMHA ops only take Q [total_tokens, head_num, head_dim];
#   K/V already live in the paged KV cache and aren't passed via the input tensor.
#   The wrapper normally routes QKV -> Q via FusedRopeKVCachePrefillOpQOut, but
#   with need_rope_kv_cache=False that conversion is skipped, so the bench must
#   feed Q-only directly to match the kernel's expected layout.
#   Other impls (ragged / TRT non-paged) take a merged 2D QKV
#   [total_tokens, (head_num + 2*kv_head_num) * head_dim] and split internally.
_PAGED_Q_ONLY_IMPLS = {
    "PyFlashinferPagedPrefillImpl",
    "TRTPagedMHAImpl",
    "FlashInferTRTLLMFMHAv2PagedPrefillImpl",
}

_DEFAULT_SKIP_IMPLS: dict[str, str] = {}

_PRESETS = {
    "qwen3-8b": dict(head_num=32, kv_head_num=8, head_dim=128),
    "llama3-8b": dict(head_num=32, kv_head_num=8, head_dim=128),
}

_Q_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

_KV_CACHE_DTYPE_MAP = {
    "bf16": KvCacheDataType.BASE,
    "fp8": KvCacheDataType.FP8,
}


def _kv_torch_dtype(kv_dtype_name: str, q_dtype: torch.dtype) -> torch.dtype:
    return torch.float8_e4m3fn if kv_dtype_name == "fp8" else q_dtype


def _short_err(e: Exception, max_len: int = 200) -> str:
    """Return the first line of the exception, truncated to max_len chars.

    Keeps C++ stack traces from polluting the per-case log / final table.
    """
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg if len(msg) <= max_len else msg[:max_len] + "..."


@dataclass
class BenchCase:
    q_dtype_name: str
    kv_dtype_name: str
    batch_size: int
    seq_len: int  # total length: prefix + input
    reuse_cache_ratio: float  # [0, 1)
    head_num: int
    kv_head_num: int
    head_dim: int
    page_size: int

    @property
    def prefix_len(self) -> int:
        return int(self.seq_len * self.reuse_cache_ratio)

    @property
    def input_len(self) -> int:
        return self.seq_len - self.prefix_len

    @property
    def total_tokens(self) -> int:
        return self.input_len * self.batch_size

    @property
    def mode_tag(self) -> str:
        return (
            "plain"
            if self.prefix_len == 0
            else f"reuse{int(self.reuse_cache_ratio*100)}%"
        )


@dataclass
class BenchResult:
    impl_name: str
    case: BenchCase
    status: str  # PASS / SKIP / FAIL / MISMATCH
    mean_ms: float = float("nan")
    p50_ms: float = float("nan")
    p95_ms: float = float("nan")
    p99_ms: float = float("nan")
    tflops: float = float("nan")
    max_diff: float = float("nan")
    note: str = ""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _setup_exec_ctx() -> None:
    py_env_configs = PyEnvConfigs()
    py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = 64
    engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)
    model_config = ModelConfig()
    model_config.max_seq_len = 131072

    pc = engine_config.parallelism_config
    init_exec_ctx(
        device_id=pc.world_rank % pc.local_world_size,
        trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
        enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
        mla_ops_type=int(model_config.mla_ops_type),
    )


def _build_parallelism_config() -> ParallelismConfig:
    pc = ParallelismConfig()
    pc.tp_size = 1
    pc.tp_rank = 0
    pc.world_size = 1
    pc.world_rank = 0
    pc.local_world_size = 1
    pc.local_rank = 0
    return pc


def _build_attn_configs(case: BenchCase) -> AttentionConfigs:
    # need_rope_kv_cache=False: skip the RoPE kernel, only benchmark the Attention itself.
    cfg = AttentionConfigs()
    cfg.head_num = case.head_num
    cfg.kv_head_num = case.kv_head_num
    cfg.size_per_head = case.head_dim
    cfg.tokens_per_block = case.page_size
    cfg.kernel_tokens_per_block = case.page_size
    cfg.use_mla = False
    cfg.is_causal = True
    cfg.need_rope_kv_cache = False
    cfg.dtype = _Q_DTYPE_MAP[case.q_dtype_name]
    cfg.kv_cache_dtype = _KV_CACHE_DTYPE_MAP[case.kv_dtype_name]
    cfg.max_seq_len = max(case.input_len * 2, 8192)
    return cfg


def _build_kv_cache_block_ids(
    batch_size: int, seq_lens: List[int], page_size: int
) -> torch.Tensor:
    pages_per_batch = [math.ceil(s / page_size) for s in seq_lens]
    total_pages = sum(pages_per_batch)
    max_blocks = max(pages_per_batch)

    perm = torch.randperm(total_pages).to(torch.int32)
    block_ids = torch.zeros((batch_size, max_blocks), dtype=torch.int32)
    off = 0
    for i, nb in enumerate(pages_per_batch):
        block_ids[i, :nb] = perm[off : off + nb]
        off += nb
    return block_ids


def _build_prefill_inputs(
    case: BenchCase, device: torch.device
) -> Tuple[PyAttentionInputs, torch.Tensor]:
    input_lens = [case.input_len] * case.batch_size
    prefix_lens = [case.prefix_len] * case.batch_size
    total_kvs = [p + i for p, i in zip(prefix_lens, input_lens)]

    attn_inputs = PyAttentionInputs()
    attn_inputs.is_prefill = True
    attn_inputs.input_lengths = torch.tensor(
        input_lens, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.sequence_lengths = torch.tensor(
        total_kvs, dtype=torch.int32, device="cpu"
    ).pin_memory()
    attn_inputs.prefix_lengths = torch.tensor(
        prefix_lens, dtype=torch.int32, device="cpu"
    ).pin_memory()

    block_ids = _build_kv_cache_block_ids(case.batch_size, total_kvs, case.page_size)
    attn_inputs.kv_cache_block_id_host = block_ids
    attn_inputs.kv_cache_block_id_device = block_ids.to(device)
    attn_inputs.kv_cache_kernel_block_id_host = block_ids
    attn_inputs.kv_cache_kernel_block_id_device = block_ids.to(device)

    def _cumsum(lens: List[int]) -> torch.Tensor:
        out, acc = [0], 0
        for x in lens:
            acc += x
            out.append(acc)
        return torch.tensor(out, dtype=torch.int32, device=device)

    # Q-side cumulative offsets
    attn_inputs.cu_seqlens = _cumsum(input_lens)
    # KV-side cumulative offsets
    attn_inputs.cu_kv_seqlens = _cumsum(total_kvs)
    attn_inputs.total_tokens = case.total_tokens
    attn_inputs.context_total_kv_length = sum(total_kvs)

    attn_inputs.dtype = get_typemeta(
        torch.zeros([1], dtype=_Q_DTYPE_MAP[case.q_dtype_name])
    )
    return attn_inputs, block_ids


def _build_inputs(
    case: BenchCase, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the single source of truth Q/K/V tensors used by both the kernel
    inputs (qkv_2d / q_3d / kv_cache_base) and the PyTorch reference.

    Returns:
        q: [BS, input_len, H, D]
        k: [BS, prefix+input, Hk, D]
        v: [BS, prefix+input, Hk, D]
    """
    dtype = _Q_DTYPE_MAP[case.q_dtype_name]
    full_kv_len = case.prefix_len + case.input_len
    # Uniform[-1, 1] avoids FP8 overflow in attention kernels
    q = (
        torch.rand(
            case.batch_size,
            case.input_len,
            case.head_num,
            case.head_dim,
            dtype=dtype,
            device=device,
        )
        * 2
        - 1
    )
    k = (
        torch.rand(
            case.batch_size,
            full_kv_len,
            case.kv_head_num,
            case.head_dim,
            dtype=dtype,
            device=device,
        )
        * 2
        - 1
    )
    v = (
        torch.rand(
            case.batch_size,
            full_kv_len,
            case.kv_head_num,
            case.head_dim,
            dtype=dtype,
            device=device,
        )
        * 2
        - 1
    )
    return q, k, v


def _pack_kernel_inputs(
    case: BenchCase, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build qkv_2d (for ragged / TRT impls) and q_3d (for paged Q-only impl).

    Ragged path requires prefix==0, so K/V passed via qkv equals the input portion.
    """
    k_input = k[:, case.prefix_len :, :, :]
    v_input = v[:, case.prefix_len :, :, :]

    q_flat = q.reshape(case.total_tokens, case.head_num * case.head_dim)
    k_flat = k_input.reshape(case.total_tokens, case.kv_head_num * case.head_dim)
    v_flat = v_input.reshape(case.total_tokens, case.kv_head_num * case.head_dim)
    qkv_2d = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()
    q_3d = q.reshape(case.total_tokens, case.head_num, case.head_dim).contiguous()
    return qkv_2d, q_3d


def _build_kv_cache(
    case: BenchCase,
    block_ids: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
) -> LayerKVCache:
    """Scatter K/V (BF16) into HND paged buffer, casting to FP8 if requested."""
    total_pages = int(block_ids.max().item()) + 1
    q_dtype = _Q_DTYPE_MAP[case.q_dtype_name]
    storage_dtype = _kv_torch_dtype(case.kv_dtype_name, q_dtype)
    # HND: [num_pages, 2, kv_heads, page_size, head_dim]
    shape = (total_pages, 2, case.kv_head_num, case.page_size, case.head_dim)
    kv_buf = torch.zeros(*shape, dtype=q_dtype, device=device)

    full_kv_len = case.prefix_len + case.input_len
    pages_per_batch = math.ceil(full_kv_len / case.page_size)
    for b in range(case.batch_size):
        for p in range(pages_per_batch):
            start = p * case.page_size
            end = min(start + case.page_size, full_kv_len)
            n = end - start
            page_id = int(block_ids[b, p].item())
            # Source [n, Hk, D] -> HND slot [Hk, n, D]
            kv_buf[page_id, 0, :, :n, :] = k[b, start:end].transpose(0, 1)
            kv_buf[page_id, 1, :, :n, :] = v[b, start:end].transpose(0, 1)

    if storage_dtype == torch.float8_e4m3fn:
        kv_buf = kv_buf.to(torch.float8_e4m3fn)

    kv_cache = LayerKVCache()
    kv_cache.kv_cache_base = kv_buf
    return kv_cache


def _compute_reference(
    case: BenchCase, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Compute ground-truth attention output via SDPA.

    For FP8 KV cache, mirror the BF16 -> FP8 -> BF16 cast on K/V so the reference
    is a fair "ideal output given lossy KV" — kernel and ref see the same K/V.

    Returns: [total_tokens, H, D] in q's dtype.
    """
    if case.kv_dtype_name == "fp8":
        k = k.to(torch.float8_e4m3fn).to(q.dtype)
        v = v.to(torch.float8_e4m3fn).to(q.dtype)

    BS, qlen, H, D = q.shape
    _, klen, Hk, _ = k.shape

    # GQA: replicate KV heads up to Q heads.
    if H != Hk:
        repeat = H // Hk
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    # SDPA wants [BS, H, seq, D].
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    # Q_i sits at absolute position (prefix + i); it can attend to K_j iff prefix + i >= j.
    prefix = case.prefix_len
    i_idx = torch.arange(qlen, device=q.device).unsqueeze(1)
    j_idx = torch.arange(klen, device=q.device).unsqueeze(0)
    mask = (prefix + i_idx) >= j_idx  # [qlen, klen]

    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=mask, is_causal=False
    )  # [BS, H, qlen, D]
    out = out.transpose(1, 2).contiguous().reshape(BS * qlen, H, D)
    return out


def _run_one_impl(
    impl_cls,
    case: BenchCase,
    attn_configs: AttentionConfigs,
    attn_inputs: PyAttentionInputs,
    kv_cache: LayerKVCache,
    qkv_2d: torch.Tensor,
    q_3d: torch.Tensor,
    parallelism_config: ParallelismConfig,
    warmup_iters: int,
    repeat_iters: int,
    ref_output: Optional[torch.Tensor] = None,
    atol: float = 5e-2,
    rtol: float = 1e-2,
    profile: bool = False,
    profile_dir: str = "/tmp/bench_traces",
    profile_iters: int = 5,
) -> BenchResult:
    impl_name = impl_cls.__name__
    result = BenchResult(impl_name=impl_name, case=case, status="SKIP")

    if impl_name in _DEFAULT_SKIP_IMPLS:
        result.note = _DEFAULT_SKIP_IMPLS[impl_name]
        return result

    try:
        supported = impl_cls.support(attn_configs, attn_inputs)
    except Exception as e:
        result.note = f"support() raised: {_short_err(e)}"
        return result
    if not supported:
        result.note = "support()==False"
        return result

    if not impl_cls.support_parallelism_config(parallelism_config):
        result.note = "support_parallelism_config()==False"
        return result

    try:
        instance = impl_cls(attn_configs, attn_inputs, parallelism_config)
    except Exception as e:
        result.status = "FAIL"
        result.note = f"construction failed: {_short_err(e)}"
        return result

    input_tensor = q_3d if impl_name in _PAGED_Q_ONLY_IMPLS else qkv_2d

    captured = {"out": None}

    def forward_fn():
        captured["out"] = instance.forward(input_tensor, kv_cache, 0)

    # Trial run once for clean error classification (FAIL: forward vs FAIL: bench).
    try:
        forward_fn()
        torch.cuda.synchronize()
    except Exception as e:
        result.status = "FAIL"
        result.note = f"forward() failed: {_short_err(e)}"
        return result

    if ref_output is not None and captured["out"] is not None:
        impl_out = captured["out"]
        if impl_out.dim() == 2:
            impl_out = impl_out.reshape(case.total_tokens, case.head_num, case.head_dim)
        try:
            diff = (impl_out.float() - ref_output.float()).abs()
            result.max_diff = float(diff.max().item())
        except Exception as e:
            result.status = "FAIL"
            result.note = f"diff calc failed: {_short_err(e)}"
            return result
        if not torch.allclose(
            impl_out.float(), ref_output.float(), atol=atol, rtol=rtol
        ):
            result.status = "MISMATCH"
            result.note = f"max_diff={result.max_diff:.3e} (atol={atol}, rtol={rtol})"
            return result

    if profile:
        import os

        os.makedirs(profile_dir, exist_ok=True)
        trace_path = os.path.join(
            profile_dir,
            f"{impl_name}_{case.kv_dtype_name}_seq{case.seq_len}_{case.mode_tag}.json",
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(profile_iters):
                forward_fn()
                torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        result.status = "PROFILED"
        result.note = trace_path
        return result

    try:
        times = bench_gpu_time_with_cudagraph(
            forward_fn,
            dry_run_iters=warmup_iters,
            repeat_iters=repeat_iters,
            l2_flush=True,
        )
    except Exception as e:
        result.status = "FAIL"
        result.note = f"bench failed: {_short_err(e)}"
        return result

    sorted_times = sorted(times)
    n = len(sorted_times)
    mean_ms = statistics.mean(times)
    p50_ms = statistics.median(times)
    p95_ms = sorted_times[min(int(n * 0.95), n - 1)]
    p99_ms = sorted_times[min(int(n * 0.99), n - 1)]

    # Q covers only new tokens (input_len); KV covers prefix + input.
    actual_seq_lens_q = torch.tensor(
        [case.input_len] * case.batch_size, dtype=torch.int32, device="cuda"
    )
    actual_seq_lens_kv = torch.tensor(
        [case.prefix_len + case.input_len] * case.batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    tflops = attention_tflops_per_sec_with_actual_seq_lens(
        actual_seq_lens_q,
        actual_seq_lens_kv,
        case.head_dim,
        case.head_dim,
        case.head_num,
        causal=True,
        ms=mean_ms,
    )

    result.status = "PASS"
    result.mean_ms = mean_ms
    result.p50_ms = p50_ms
    result.p95_ms = p95_ms
    result.p99_ms = p99_ms
    result.tflops = tflops
    return result


def _print_table(results: List[BenchResult]) -> None:
    # Only PASS rows; MISMATCH/FAIL/SKIP shown live during the run.
    pass_results = [r for r in results if r.status == "PASS"]
    w = max((len(r.impl_name) for r in pass_results), default=20) + 2
    header = (
        f"{'impl':<{w}}{'kv':<5}  {'seq':>6}  {'prefix':>6}  {'input':>6}  "
        f"{'mode':<10}  {'mean_ms':>9}  {'p50_ms':>9}  {'p95_ms':>9}  "
        f"{'p99_ms':>9}  {'TFLOPs/s':>9}  {'max_diff':>10}"
    )
    _log("\n" + "=" * len(header))
    _log(header)
    _log("-" * len(header))
    if not pass_results:
        _log("(no PASS results)")
        _log("=" * len(header))
        return
    for r in pass_results:
        c = r.case
        diff_str = (
            f"{r.max_diff:>10.3e}" if not math.isnan(r.max_diff) else f"{'-':>10}"
        )
        _log(
            f"{r.impl_name:<{w}}{c.kv_dtype_name:<5}  "
            f"{c.seq_len:>6}  {c.prefix_len:>6}  {c.input_len:>6}  "
            f"{c.mode_tag:<10}  "
            f"{r.mean_ms:>9.3f}  {r.p50_ms:>9.3f}  "
            f"{r.p95_ms:>9.3f}  {r.p99_ms:>9.3f}  {r.tflops:>9.2f}  {diff_str}"
        )
    _log("=" * len(header))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--q-dtype", type=str, default="bf16", help="Q / compute dtype: bf16,fp16"
    )
    p.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="bf16,fp8",
        help="KV cache dtype: bf16,fp8",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--input-len",
        type=str,
        default="128,1024,4096",
        help="seq_len list (each value is the " "total prefix+input length)",
    )
    p.add_argument(
        "--reuse-cache-ratio",
        type=str,
        default="0,0.5",
        help="reuse_cache hit ratios in [0, 1). "
        "0 = plain prefill; "
        "0.5 = half of KV is cached and paged path is taken",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="qwen3-8b",
        choices=list(_PRESETS.keys()) + ["custom"],
    )
    p.add_argument("--head-num", type=int, default=None)
    p.add_argument("--kv-head-num", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument(
        "--impls",
        type=str,
        default="",
        help="case-insensitive substring filter on impl class names; "
        "empty = all impls",
    )
    p.add_argument(
        "--check-correctness",
        action="store_true",
        default=True,
        help="compare each impl's output against a PyTorch SDPA "
        "reference and flag MISMATCH on numerical drift",
    )
    p.add_argument(
        "--no-check-correctness",
        dest="check_correctness",
        action="store_false",
        help="disable correctness check (saves ref-compute time on " "long sequences)",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=5e-2,
        help="absolute tolerance for correctness check",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="relative tolerance for correctness check",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="capture torch.profiler chrome trace per impl " "instead of benchmarking",
    )
    p.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/bench_traces",
        help="directory to write chrome trace JSON files",
    )
    p.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="number of forward iterations to profile",
    )
    return p.parse_args()


def _resolve_head_shape(args: argparse.Namespace) -> Tuple[int, int, int]:
    if args.preset != "custom":
        preset = _PRESETS[args.preset]
        head_num = args.head_num if args.head_num is not None else preset["head_num"]
        kv_head_num = (
            args.kv_head_num if args.kv_head_num is not None else preset["kv_head_num"]
        )
        head_dim = args.head_dim if args.head_dim is not None else preset["head_dim"]
    else:
        assert (
            args.head_num and args.kv_head_num and args.head_dim
        ), "custom preset requires --head-num, --kv-head-num and --head-dim"
        head_num, kv_head_num, head_dim = args.head_num, args.kv_head_num, args.head_dim
    return head_num, kv_head_num, head_dim


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        _log("CUDA not available, skipping benchmark")
        return 0

    set_seed(42)
    device = torch.device("cuda")

    try:
        _setup_exec_ctx()
    except Exception as e:
        _log(f"warning: init_exec_ctx failed: {e}")

    head_num, kv_head_num, head_dim = _resolve_head_shape(args)
    q_dtypes = [d.strip() for d in args.q_dtype.split(",") if d.strip()]
    kv_dtypes = [d.strip() for d in args.kv_cache_dtype.split(",") if d.strip()]
    for q in q_dtypes:
        assert (
            q in _Q_DTYPE_MAP
        ), f"unknown --q-dtype '{q}', supported: {list(_Q_DTYPE_MAP)}"
    for kv in kv_dtypes:
        assert (
            kv in _KV_CACHE_DTYPE_MAP
        ), f"unknown --kv-cache-dtype '{kv}', supported: {list(_KV_CACHE_DTYPE_MAP)}"
    input_lens = [int(x) for x in args.input_len.split(",") if x.strip()]
    reuse_ratios = [float(x) for x in args.reuse_cache_ratio.split(",") if x.strip()]
    for r in reuse_ratios:
        assert 0.0 <= r < 1.0, f"--reuse-cache-ratio must be in [0, 1), got {r}"
    impl_filter = args.impls.strip().lower()

    selected_impls = [
        cls
        for cls in PREFILL_MHA_IMPS
        if not impl_filter or impl_filter in cls.__name__.lower()
    ]
    _log(
        f"GPU: {torch.cuda.get_device_name(0)}  CC: {torch.cuda.get_device_capability(0)}"
    )
    _log(
        f"Registered impls: {len(PREFILL_MHA_IMPS)}, "
        f"selected for benchmark: {len(selected_impls)}"
    )
    for cls in selected_impls:
        _log(f"  - {cls.__name__}")
    _log(
        f"head_num={head_num}, kv_head_num={kv_head_num}, head_dim={head_dim}, "
        f"page_size={args.page_size}, batch={args.batch_size}"
    )

    parallelism_config = _build_parallelism_config()
    all_results: List[BenchResult] = []
    impl_name_w = max(len(c.__name__) for c in selected_impls) + 2

    for q_dtype_name in q_dtypes:
        for kv_dtype_name in kv_dtypes:
            for seq_len in input_lens:
                for ratio in reuse_ratios:
                    case = BenchCase(
                        q_dtype_name=q_dtype_name,
                        kv_dtype_name=kv_dtype_name,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        reuse_cache_ratio=ratio,
                        head_num=head_num,
                        kv_head_num=kv_head_num,
                        head_dim=head_dim,
                        page_size=args.page_size,
                    )
                    case_line = (
                        f"[Case] q={q_dtype_name}  kv={kv_dtype_name}  "
                        f"seq={seq_len}  prefix={case.prefix_len}  "
                        f"input={case.input_len}  mode={case.mode_tag}"
                    )
                    _log("\n" + "-" * len(case_line))
                    _log(case_line)
                    _log("-" * len(case_line))
                    cfg = _build_attn_configs(case)
                    attn_inputs, block_ids = _build_prefill_inputs(case, device)
                    q, k, v = _build_inputs(case, device)
                    qkv_2d, q_3d = _pack_kernel_inputs(case, q, k, v)
                    kv_cache = _build_kv_cache(case, block_ids, k, v, device)

                    ref_output = None
                    if args.check_correctness:
                        try:
                            ref_output = _compute_reference(case, q, k, v)
                        except Exception as e:
                            _log(f"  [warn] reference compute failed: {_short_err(e)}")
                            ref_output = None

                    for impl_cls in selected_impls:
                        res = _run_one_impl(
                            impl_cls,
                            case,
                            cfg,
                            attn_inputs,
                            kv_cache,
                            qkv_2d,
                            q_3d,
                            parallelism_config,
                            warmup_iters=args.warmup,
                            repeat_iters=args.repeat,
                            ref_output=ref_output,
                            atol=args.atol,
                            rtol=args.rtol,
                            profile=args.profile,
                            profile_dir=args.profile_dir,
                            profile_iters=args.profile_iters,
                        )
                        all_results.append(res)
                        if res.status == "PASS":
                            diff_str = (
                                f" max_diff={res.max_diff:.2e}"
                                if not math.isnan(res.max_diff)
                                else ""
                            )
                            tail = (
                                f"mean={res.mean_ms:.3f}ms "
                                f"tflops={res.tflops:.2f}{diff_str}"
                            )
                        else:
                            tail = res.note
                        _log(f"  {res.impl_name:<{impl_name_w}}{res.status:<10}{tail}")
                    del kv_cache, qkv_2d, q_3d, q, k, v, attn_inputs, ref_output
                    torch.cuda.empty_cache()

    _print_table(all_results)

    has_pass = any(r.status == "PASS" for r in all_results)
    has_mismatch = any(r.status == "MISMATCH" for r in all_results)
    if has_mismatch:
        _log("\n[!] one or more impls failed correctness check (MISMATCH)")
        return 2
    return 0 if has_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
