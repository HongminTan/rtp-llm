package(default_visibility = ["//visibility:public"])

# Exported so the perf_test() macro (defined in defs.bzl, instantiated by
# internal_source/rtp_llm/test/perf_test/BUILD) can reference test_entry.py
# across packages.
exports_files([
    "test_entry.py",
])

py_library(
    name = "perf_test_lib",
    srcs = [
        "batch_decode_test.py",
        "batch_perf_impl.py",
        "cache_grid_runner.py",
        "dataclass.py",
        "dataset.py",
        "deepseek_v4_prefill_formula_fit.py",
        "distribution_runner.py",
        "generate_prefill_3d_chart.py",
        "grid_runner.py",
        "hub_download.py",
        "offline_bench_test.py",
        "offline_runner.py",
        "perf_config.py",
        "perf_utils.py",
        "tps_runner.py",
        "sampling.py",
        "server.py",
        "test_util.py",
        "visualization.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "batch_decode_test_test",
    srcs = ["batch_decode_test_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "test_perf_config_and_runners",
    srcs = ["test_perf_config_and_runners.py"],
    data = [
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
)

# Offline throughput benchmark (new entry point).
# Manual target: Qwen3.5 weights must be available locally or resolvable by perf path resolution.
py_test(
    name = "offline_bench_test",
    main = "offline_bench_test.py",
    srcs = ["offline_bench_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    args = [
        "--checkpoint_path", "Qwen/Qwen3.5-35B-A3B",
        "--model_type", "qwen35_moe",
        "--input_len_min", "128",
        "--input_len_max", "512",
        "--output_len_min", "64",
        "--output_len_max", "256",
        "--concurrency_limit", "256",
        "--seq_size_per_block", "1024",
    ],
    tags = ["manual"],
)


# Distribution mode: sampling from dataset using model's parameters.
py_test(
        name = "distribution_perf_test",
        main = "batch_decode_test.py",
        srcs = [
            "batch_decode_test.py",
        ],
        timeout = "eternal",
        deps = [
            "//rtp_llm:pyodps",
            "//rtp_llm:testlib",
            "//rtp_llm/test/perf_test:perf_test_lib",
        ],
        data = [
            "//rtp_llm:sdk",
        ],
        args = [
            "--checkpoint_path", "Qwen/Qwen3.5-35B-A3B",
            "--dataset_name", "sharegpt",
            "--max_seq_len", "8192",
            "--concurrency_limit", "64",
            "--seq_size_per_block", "1024"
        ],
        tags = ["manual"],
    )

# Grid mode: perf test with fixed batch_size × input_len.
py_test(
    name = "grid_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type",
        "qwen35_moe",
        "--checkpoint_path",
        "Qwen/Qwen3.5-35B-A3B",
        "--batch_size",
        "1",
        "--input_len",
        "128",
        "--partial",
        "1",
        "--decode_test_length",
        "8",
        "--seq_size_per_block",
        "1024",
        "--tp_size",
        "1",
        "--dp_size",
        "1",
    ],
    tags = ["manual"],
)

# Prefix-cache mode: measures an explicit total-sequence × cache-length grid.
# Invoke with --test_arg=--cache_grid_json=<file> and --test_arg=--partial=2.
# The runner seeds each requested prefix and verifies reuse_len on every
# measured request; cache_len is a workload dimension, not an engine argument.
py_test(
    name = "cache_grid_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    tags = ["manual"],
)

# Generic model-agnostic entrypoint. It intentionally has no model-specific
# defaults; pass model/workload/engine settings with --test_arg at invocation.
py_test(
    name = "model_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    tags = ["manual"],
)

# V4-Flash CP=4 + EP=4 prefill profile sweep from 64k to 1M input.
# Manual-only — drives batch_size=1 prefill requests through the perf
# test framework so per-input_len timeline JSONs land in TEST_UNDECLARED_
# OUTPUTS_DIR. Uses a local checkpoint path so the perf hub resolver does
# not try ModelScope.
py_test(
    name = "v4_flash_cp4_ep4_prefill_64k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "2",
        "--input_len", "65536",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "8192",
        "--kernel_seq_size_per_block", "128",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "65538",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "122880",
        "--max_context_batch_size", "8",
        "--frontend_server_count", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_GRID_WARMUP_RUNS": "3",
        "PERF_FORMAL_WARMUP_RUNS": "1",
        "PERF_MEASURE_RUNS": "3",
        "PERF_PROFILE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "ENABLE_FP32_LM_HEAD": "0",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
        "RTP_LLM_DEVICE_INPUT": "1",
    },
    # TODO(dsv4-merge): these 4-GPU DSV4-Flash perf targets are referenced by
    # //internal_source/rtp_llm/test/smoke:smoke_sm100_dsv4 (member of the CI
    # maga_model_smoke suite). Suite membership overrides tags=["manual"], and
    # without a gpu tag the sm8x/sm9x negative --test_tag_filters do not drop
    # them, so they got scheduled onto CPU/H20 pools that lack
    # /mnt/nas1/hf/DeepSeek-V4-Flash (startup dies with "dsv4 kv cache descs
    # require layer_num > 0"). The SM100_ARM tag + exec_properties gate them to
    # the intended cuda13 / SM100 runner, matching the smoke_test() convention.
    tags = [
        "manual",
        "SM100_ARM_CU13",
    ],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)

# V4-Flash CP=4 + EP=4 prefill profile sweep with CP page-RR KV sharding.
py_test(
    name = "v4_flash_cp4_ep4_prefill_64k_page_rr_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "2",
        "--input_len", "65536",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "8192",
        "--kernel_seq_size_per_block", "128",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "65538",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "122880",
        "--max_context_batch_size", "8",
        "--frontend_server_count", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_GRID_WARMUP_RUNS": "3",
        "PERF_FORMAL_WARMUP_RUNS": "1",
        "PERF_MEASURE_RUNS": "3",
        "PERF_PROFILE_RUNS": "1",
        "PREFILL_CP_KV_CACHE_SHARDED": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "ENABLE_FP32_LM_HEAD": "0",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
        "RTP_LLM_DEVICE_INPUT": "1",
    },
    tags = ["manual"],
)

# V4-Flash CP=4 + EP=4 prefill profile on 200k input.  Same topology as
# v4_flash_cp4_ep4_prefill_64k_perf, with a longer single query for
# long-context chunk-size experiments.
py_test(
    name = "v4_flash_cp4_ep4_prefill_200k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "200000",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "128",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "200064",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    tags = ["manual"],
)

# V4-Flash CP=4 + EP=4 + Mega MoE prefill profile.  Same topology as
# the baseline above but with MOE_STRATEGY=mega_moe_se + DG_JIT_CPP_STANDARD=20.
# Smaller input (8k vs baseline 64k) because Mega MoE's symm-mem buffer
# allocation is sized from max_tokens_per_rank=max_seq_len; at 64k the
# symm allocation aborts in CUDASymmetricMemory destructor during warmup.
# 8k still exercises the Mega MoE fused dispatch + L1/L2 GEMM + combine
# path on every MoE layer — enough kernel data to compare against the
# baseline's per-layer DeepEP dispatch+combine breakdown.
py_test(
    name = "v4_flash_cp4_ep4_prefill_8k_mega_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "8192",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "128",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "8200",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    tags = ["manual"],
)

# V4-Flash DP=4 + EP=4 captured-graph decode grid sweep.  Mirrors the
# topology of smoke target v4_flash_decode_dp4_ep4_graph_mega_concurrent_sm100
# (TP=1 / DP=4 / EP=4 / world=4, DeepEP non-low-latency, Mega MoE via
# CUDA graph capture at BS=1,2,4,8) and drives the perf framework over
# the same batch_size grid so per-BS decode timelines land in
# TEST_UNDECLARED_OUTPUTS_DIR.  Local checkpoint avoids ModelScope.
py_test(
    name = "v4_flash_dp4_decode_grid_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "4096,8192",
        "--partial", "1",
        "--decode_test_length", "8",
        "--seq_size_per_block", "128",
        "--max_seq_len", "16384",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    # TODO(dsv4-merge): these 4-GPU DSV4-Flash perf targets are referenced by
    # //internal_source/rtp_llm/test/smoke:smoke_sm100_dsv4 (member of the CI
    # maga_model_smoke suite). Suite membership overrides tags=["manual"], and
    # without a gpu tag the sm8x/sm9x negative --test_tag_filters do not drop
    # them, so they got scheduled onto CPU/H20 pools that lack
    # /mnt/nas1/hf/DeepSeek-V4-Flash (startup dies with "dsv4 kv cache descs
    # require layer_num > 0"). The SM100_ARM tag + exec_properties gate them to
    # the intended cuda13 / SM100 runner, matching the smoke_test() convention.
    tags = [
        "manual",
        "SM100_ARM_CU13",
    ],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)

# V4-Flash DP=4 + EP=4 captured-graph decode at fixed bs=128 / input_len=64K.
# 64K context x 128 batch (per DP rank) x DP=4 x FP8 KV x FP8_PER_BLOCK weights
# x Mega MoE x cuda graph capture. Canonical iter target for feat/dsv4_on_dev
# decode perf-tuning (see dpskv4-study/perf/dsv4_on_dev_64k/iterations.md).
# --batch_size is per-DP-rank: GridRunner.run() multiplies by --dp_size before
# handing off to BatchPerfImpl, which then divides back by dp_size on the
# engine side (BatchPerfImpl._set_concurrency).  Net: --batch_size 128 +
# --dp_size 4 = 512 total requests distributed 128/rank across 4 DP ranks.
# EngineServer._build_engine_cli also overrides --concurrency_limit to
# max(batch_size_list)=128 at engine startup, so the BUILD
# --concurrency_limit below is effectively ignored.
# DSV4 fixed pools are sized by --dsv4_fixed_pool_blocks.  Each request can hold
# up to 3 fixed-pool blocks, so 128 concurrent reqs need 384 request-visible
# blocks per rank.  The default 256 leaves the bs128 run short of admission
# headroom and surfaces as incrKVBlock LACK MEM; 512 keeps roughly 33% margin.
# Keep one frontend per DP rank for this smoke: the test readiness check probes a
# shared rank port once, and multiple SO_REUSEPORT frontends can otherwise route
# early scheduler updates to a process that is still completing startup.
py_test(
    name = "v4_flash_dp4_decode_bs128_64k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "100",
        "--seq_size_per_block", "128",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--frontend_server_count", "1",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "6",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "ENABLE_FP32_LM_HEAD": "0",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
        "RTP_LLM_DEVICE_INPUT": "1",
    },
    # TODO(dsv4-merge): these 4-GPU DSV4-Flash perf targets are referenced by
    # //internal_source/rtp_llm/test/smoke:smoke_sm100_dsv4 (member of the CI
    # maga_model_smoke suite). Suite membership overrides tags=["manual"], and
    # without a gpu tag the sm8x/sm9x negative --test_tag_filters do not drop
    # them, so they got scheduled onto CPU/H20 pools that lack
    # /mnt/nas1/hf/DeepSeek-V4-Flash (startup dies with "dsv4 kv cache descs
    # require layer_num > 0"). The SM100_ARM tag + exec_properties gate them to
    # the intended cuda13 / SM100 runner, matching the smoke_test() convention.
    tags = [
        "manual",
        "SM100_ARM_CU13",
    ],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)

# No-graph sibling for diagnosing cuda-graph capture failures.
py_test(
    name = "v4_flash_dp4_decode_bs128_64k_perf_nograph",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "128",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--frontend_server_count", "1",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    tags = ["manual"],
)

# Diagnostic probe: bs=16 / 64K with cuda graph off — smallest bs that still
# uses 4-rank DP/EP topology, to confirm whether decode warmup regression is
# bs-threshold specific (bs=128/64 both fail on feat/dsv4_on_dev HEAD).
py_test(
    name = "v4_flash_dp4_decode_bs16_64k_perf_probe",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "16",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "128",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
    },
    tags = ["manual"],
)

py_test(
    name = "v4_flash_dp4_decode_bs64_64k_perf_probe",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "64",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "128",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    tags = ["manual"],
)

# V4-Flash DP=4 + EP=4 captured-graph decode grid sweep with MTP
# speculative decoding + FP8 KV cache.  Mirrors the decode side of the
# smoke target ``v4_flash_pd_cp2ep2_dp2ep2_sm100_fp8kv1_mtp_batch4_cudagraph``
# (TP=1 / DP=4 / EP=4 / world=4, DeepEP low-latency, decode CUDA graph,
# MTP draft via deepseek_v4_mtp + gen_num_per_cycle=1, FP8 KV) but
# scaled out from DP=2 → DP=4 and pinned at input_len=65536 (64K).
# Sweeps ``--batch_size`` over 1, 8, 16, 32 so per-BS MTP+graph decode
# timelines land in TEST_UNDECLARED_OUTPUTS_DIR.  ``decode_capture_config``
# captures one graph per BS (matches the grid).
py_test(
    name = "v4_flash_dp4_decode_64k_mtp_grid_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1,8,64,128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "30",
        "--seq_size_per_block", "128",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
        "--sp_type", "mtp",
        "--gen_num_per_cycle", "3",
        "--sp_model_type", "deepseek_v4_mtp",
        "--sp_checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--sp_act_type", "bf16",
        "--warm_up", "0",
        "--dsv4_fixed_pool_blocks", "1024"
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
        "RTP_LLM_DEVICE_INPUT": "1"
    },
    tags = ["manual"],
)

# DeepSeek V4-Pro CP=8 + EP=8 prefill profile on 64k input.
py_test(
    name = "v4_pro_cp8_ep8_prefill_64k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "1",
        "--input_len", "65536,173921,349287,612734,887451,1048575",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "256",
        "--tp_size", "8",
        "--dp_size", "1",
        "--ep_size", "8",
        "--world_size", "8",
        "--max_seq_len", "1048576",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "81920",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "1",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "DSV4_CHUNK_TOKENS": "8192",
    },
    tags = ["manual"],
)

# DeepSeek V4-Pro DP=8 + EP=8 captured-graph decode grid sweep.
py_test(
    name = "v4_pro_dp8_decode_grid_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "1,32,64,128,256",
        "--input_len", "4096,8192",
        "--partial", "1",
        "--decode_test_length", "8",
        "--seq_size_per_block", "128",
        "--max_seq_len", "16384",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--fp8_kv_cache", "1",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
    },
    tags = ["manual"],
)

# DeepSeek V4-Pro DP=8 + EP=8 captured-graph decode at fixed bs=128 / input_len=64K.
py_test(
    name = "v4_pro_dp8_decode_bs128_64k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "128",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "PERF_MEASURE_RUNS": "1",
        "MOE_STRATEGY": "mega_moe_se",
        "ENABLE_FP32_LM_HEAD": "0",
    },
    tags = ["manual"],
)

test_suite(
    name = "perf_dsv4_v4_pro",
    tests = [
        ":v4_pro_cp8_ep8_prefill_64k_perf",
        ":v4_pro_dp8_decode_grid_perf",
        ":v4_pro_dp8_decode_bs128_64k_perf",
    ],
    tags = ["manual"],
)

# DeepSeek V4-Flash perf coverage on SM100/SM103 (GB200/GB300).
#   v4_flash_cp4_ep4_prefill_64k_perf       → CP=4 EP=4 64k prefill timeline
#   v4_flash_cp4_ep4_prefill_64k_page_rr_perf → CP=4 EP=4 64k prefill + page-RR KV sharding
#   v4_flash_cp4_ep4_prefill_200k_perf      → CP=4 EP=4 200k prefill timeline
#   v4_flash_dp4_decode_grid_perf           → DP=4 EP=4 captured-graph decode grid
#   v4_flash_dp4_decode_bs128_64k_perf      → DP=4 EP=4 fixed bs=128/input=64K (perf-tuning canonical)
#   v4_flash_dp4_decode_64k_mtp_grid_perf   → DP=4 EP=4 + MTP + cudagraph @ 64K, BS={1,8,16,32}
test_suite(
    name = "perf_dsv4_v4_flash",
    tests = [
        ":v4_flash_cp4_ep4_prefill_64k_perf",
        ":v4_flash_cp4_ep4_prefill_64k_page_rr_perf",
        ":v4_flash_cp4_ep4_prefill_200k_perf",
        ":v4_flash_dp4_decode_grid_perf",
        ":v4_flash_dp4_decode_bs128_64k_perf",
        ":v4_flash_dp4_decode_64k_mtp_grid_perf",
    ],
    tags = ["manual"],
)

# V4-Flash DP=8 + EP=8 eager decode baseline on SM90 (H20). Mirrors the
# engine args of smoke target v4_flash_full_dp8ep8_sm90_h20 (dynamic-split
# NCCL a2a, no graphs); the formal TPOT/TPS reference for the SM90
# adaptation ahead of the DeepEP-LL/PD and MegaMoE decode measurements.
py_test(
    name = "v4_flash_dp8_decode_eager_sm90_h20_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1,16,64,128",
        "--input_len", "512,4096",
        "--partial", "1",
        "--decode_test_length", "30",
        "--num_measures", "5",
        "--seq_size_per_block", "256",
        "--max_seq_len", "8192",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        "--enable_cuda_graph", "0",
        "--max_context_batch_size", "128",
        "--reserver_runtime_mem_mb", "12288",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash skill-standard perf lengths on SM90 (H20): the adapt-model
# performance standard's 64K/128K input definitions. Prefill TPM from the
# grid runner's prefill phase; decode TPOT at the long context (eager
# dp8ep8 engine config - the perf framework has no PD role support, so
# this measures the prefill-role-representative eager topology).
py_test(
    name = "v4_flash_dp8_skill_len_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "30",
        "--num_measures", "3",
        "--seq_size_per_block", "256",
        "--max_seq_len", "65824",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        "--enable_cuda_graph", "0",
        "--max_context_batch_size", "4",
        "--reserver_runtime_mem_mb", "24576",
        "--dsv4_fixed_pool_blocks", "4096",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash skill decode gate on SM90: binary-search the max concurrency
# satisfying TPOT<=20ms and report TPS at that point (adapt-model
# performance standard). Eager dp8ep8 engine config (no PD role support
# in the perf framework - the gate evaluates the measurable topology;
# the PD-optimized TPOT is tracked in the RUN perf report).
py_test(
    name = "v4_flash_dp8_tpot_gate_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1,8,32",
        "--input_len", "4096",
        "--partial", "1",
        "--target_tpot", "20",
        "--decode_test_length", "30",
        "--num_measures", "3",
        "--seq_size_per_block", "256",
        "--max_seq_len", "8192",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        "--enable_cuda_graph", "0",
        "--max_context_batch_size", "64",
        "--reserver_runtime_mem_mb", "12288",
        "--dsv4_fixed_pool_blocks", "256",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash 90%-prefix-cache prefill TPM on SM90 (H20): the adapt-model
# performance standard's cache-hit scenario. Case 1 = exact 90% cached
# continuation (input 10240, cache 9216, tail 1024), case 2 = cold control.
py_test(
    name = "v4_flash_dp8_cache_grid_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--cache_grid_json", "/data2/liukan.lk/runs/dsv4_flash_h20_20261013/build_logs/dsv4_cache_grid.json",
        "--partial", "2",
        "--cache_measure_runs", "3",
        "--cache_commit_tail_tokens", "1024",
        "--num_measures", "3",
        "--seq_size_per_block", "256",
        "--max_seq_len", "11264",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        "--enable_cuda_graph", "0",
        "--max_context_batch_size", "4",
        "--reserver_runtime_mem_mb", "12288",
        "--reuse_cache", "1",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash prefill-phase timeline capture on SM90 (H20): partial=2 with
# the engine timeline enabled, feeding the step-6 prefill share/Top-10/
# MFU analysis (gates_calc.py -P prefill).
py_test(
    name = "v4_flash_dp8_prefill_trace_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "4096",
        "--partial", "2",
        "--num_measures", "3",
        "--seq_size_per_block", "256",
        "--max_seq_len", "8192",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        "--enable_cuda_graph", "0",
        "--max_context_batch_size", "4",
        "--reserver_runtime_mem_mb", "12288",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
        "GEN_TIMELINE_SYNC": "1",
        "GEN_TIMELINE_START_STEP": "1",
        "GEN_TIMELINE_NUM_STEPS": "2",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash DP=4 + EP=4 eager decode on 4x H20 (SM90): the 4-free-card shape
# on a shared 8x H20 host. batch_size is per DP rank, so total concurrency
# is 4x the listed value (bs 1,2 -> conc 4,8). Conc-1 TPOT is measured as a
# documented single-request run (GridRunner requires total % dp_size == 0).
py_test(
    name = "v4_flash_dp4_decode_eager_sm90_h20_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/admin/models/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/admin/models/DeepSeek-V4-Flash",
        "--batch_size", "1,2",
        "--input_len", "512,4096",
        "--partial", "1",
        "--decode_test_length", "30",
        "--num_measures", "5",
        "--seq_size_per_block", "256",
        "--max_seq_len", "8192",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        # CUDA graph capture is NOT optional on this lane (team/perf-log.md R3/R4).
        # With enable_cuda_graph 0 the EP MoE takes grouped_fp4_sm90._forward_ep(),
        # which does 2 device->host syncs + 4 dist.all_to_all_single per layer;
        # the ranks then re-rendezvous once per layer and the host blocks ~90 ms
        # per layer in the DtoH sync that follows. Measured on 4x H20:
        #   0 -> 13,196 ms/tok  (avg_total_ms 57,781 / 30 decode tokens)
        #   1 ->     30.4 ms/tok (avg_total_ms    945 / 30 decode tokens)  434x
        # Capture diverts to _forward_ep_static(): 2 collectives/layer and no
        # host syncs (scatter_add instead of the synchronising bincount).
        # reserver_runtime_mem_mb must be raised: the graph memory pools are
        # carved out of the reserved runtime arena, 12288 OOMs at capture.
        # 16384 is the value R4 measured with; see 7610109e4 on the reference
        # branch for the same recipe.
        "--enable_cuda_graph", "1",
        "--max_context_batch_size", "128",
        "--reserver_runtime_mem_mb", "16384",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        # Allocator policy, MEASURED on this very target (team/perf-log.md R13).
        # That run carried zero --test_arg, so the only diff against the R5
        # table is this one env var:
        #   TPOT  512/1 33.14 -> 31.38, 512/2 35.19 -> 33.32,
        #         4096/1 33.65 -> 32.01, 4096/2 35.27 -> 33.55   (all -5%)
        # and bs1-512 avg_total_ms 998.65 -> 948.95 mean (-5.0%) against a
        # within-run spread of only 1.2%. Uniform across four independent
        # points, so it is a result and not drift. max_split_size_mb:256 caps
        # the block split size so the graph-capture pools come out of a less
        # fragmented arena; garbage_collection_threshold:0.6 lets the collector
        # run earlier instead of growing the reserved segment. Do NOT swap in
        # expandable_segments: the reference branch documents it corrupting
        # numerics at this occupancy
        # (issues/expandable-segments-corrupts-at-memory-edge.md), and their
        # 7610109e4 picked exactly these two knobs (team/perf-log.md R12b).
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256,garbage_collection_threshold:0.6",
        # The deepgemm split-K mHC pre-gemm is NOT reachable on this lane: the
        # cu129-pinned DeepGEMM 2.1.1 does not export tf32_hc_prenorm_gemm, so
        # deepgemm_wrapper._missing_deep_gemm() raises "DeepGEMM is not
        # available" during model init and the engine never starts. That is a
        # third 2.1.1 compat gap after fp8_einsum and max_seqlen_k. Until the
        # symbol lands (see team/perf-log.md R2) tilelang_single is the only
        # backend this build can actually run -- do not flip this to deepgemm
        # (the reference branch's target uses it because its DeepGEMM has it).
        "WORLD_SIZE": "4",
        "DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
        # torch 2.8 (cuda12_9) lacks float8_e8m0fnu: nogds copier loads
        # UE8M0 scales via the byte-mode U8 fallback (see smoke target).
        "FASTSAFETENSORS_NOGDS": "1",
        # .bazelrc test:cuda12_9 points CC/CXX at the CI image's
        # gcc-toolset-12; this container ships only /usr/bin/gcc (see smoke).
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDAHOSTCXX": "/usr/bin/g++",
        "NVCC_PREPEND_FLAGS": "-ccbin=/usr/bin/g++",
        # Task-private flashinfer JIT workspace: the shared
        # ~/.cache/flashinfer/0.6.9+e87d610f cached_ops were built against
        # tvm-ffi 0.1.8 (NDArray=72) and crash tvm-ffi 0.1.12 (Map=72) with
        # "Conflicting static index 72" at first forward. Fresh base dir
        # forces a rebuild against the current headers (see smoke target).
        "FLASHINFER_WORKSPACE_BASE": "/home/admin/dsv4-flashinfer-ws",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

# V4-Flash prefill TTFT on 4x H20 (SM90): single-request TTFT at 16k/32k
# prompt (GridRunner pins BS=1 in prefill mode). max_seq_len covers 32k
# prompt + generated tail. Prefix-cache reuse stays off (default) so every
# measure is a real prefill of the fixed "hello ..." query.
py_test(
    name = "v4_flash_dp4_prefill_eager_sm90_h20_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/admin/models/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/admin/models/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "16000,32000",
        "--partial", "2",
        "--decode_test_length", "30",
        "--num_measures", "5",
        "--seq_size_per_block", "256",
        "--max_seq_len", "40960",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--moe_strategy", "grouped_fp4_sm90_ep",
        # CUDA graph ON. This was flipped only after a same-n, single-variable
        # A/B at input_len 16000, num_measures 5 (team/perf-log.md R10):
        #   graph OFF: 16586.5 16261.7 12656.3 13175.1 13348.4  -> mean 14405.6
        #   graph ON :  8854.3  9799.1  9650.1  9409.2  9618.6  -> mean  9466.3
        # The two ranges do not even overlap (max ON 9799 < min OFF 12656), so
        # the 1.52x / -34.3% is a result and not the +/-15% noise the earlier
        # single graph-on datapoint could not beat (R7/R8).
        # Note what this does and does not say: 16k prefill is NOT captured.
        # The prefill capture range is generated as 1..160 tokens
        # (hw_kernel_group_args.py _parse_prefill_capture_config, default
        # "160:1", logs "Generated 160 sequence lengths from range (step=1,
        # max=160)"; the runner dump shows max_kernel_block_table_width_: 160
        # and is_prefill_cuda_graph_mode_: 0), so a 16k request hits
        # cuda_graph_runner.cc "prefill seq_len %d exceeds max captured %d,
        # fallback to normal run" and still runs the eager EP path. The 1.52x
        # is therefore a side effect of enabling the graph at all (graph
        # memory pools carved out up front, so the eager path stops doing
        # synchronising allocator work mid-layer), NOT of capture. Widening
        # the capture range to cover 16k is the next, much larger lever.
        "--enable_cuda_graph", "1",
        # max_context_batch_size MUST stay 1 here. The engine's prefill warmup
        # (NormalEngine::prefillWarmUp -> PyWrappedModel::forward with
        # kv_cache=None) allocates a buffer of
        #   max_context_batch_size x max_seq_len x hidden x 2 bytes
        # and at 128 x 40960 x 4096 x 2 that is exactly 40.00 GiB. Model + KV
        # already hold 113.47 GiB of the 139.80 GiB card, leaving ~12 GiB, so
        # the engine dies during init with
        #   "CUDA out of memory. Tried to allocate 40.00 GiB"
        # and the target never starts -- with or without CUDA graph (see
        # team/perf-log.md R6; 128x is the whole error). Prefill mode pins BS=1
        # anyway, so 1 is both sufficient and 128x smaller (0.31 GiB).
        "--max_context_batch_size", "1",
        # reserver_runtime_mem_mb 16384 is NOT optional here either, and it is
        # the one that was isolated: at 12288 the engine still brings the model
        # up cleanly (the init OOM above is gone) but the backend dies on the
        # first real 16k prefill -- 8205_CONNECTION_RESET_BY_PEER for all 4
        # requests, ranks gone, 0/4 on every measure. With 16384 the same
        # request returns 4/4 five times out of five, with the graph still off,
        # so the arena is the cause and not the capture. At the committed
        # setting (arena 16384 + graph on) TTFT@16k is 9466 ms mean over 5
        # (8854..9799); the prefill body still runs the eager EP path and pays
        # the same per-layer rendezvous stall the decode target had before its
        # graph was enabled (R3/R8/R10).
        "--reserver_runtime_mem_mb", "16384",
        "--dsv4_fixed_pool_blocks", "512",
    ],
    env = {
        # Do NOT set PYTORCH_CUDA_ALLOC_CONF here. It is set on the DECODE
        # target, where it was measured at -5% TPOT (team/perf-log.md R13),
        # but on this prefill target the exact same env kills the backend:
        # every request on the first 16k prefill comes back
        # 8205_CONNECTION_RESET_BY_PEER and then 8202_CONNECT_FAILED, 0/4 on
        # the pre-arm measure and 0/4 on all five measured runs -- the same
        # signature as R7b's under-sized arena, but now with the arena at the
        # known-good 16384 and the allocator policy as the only diff against
        # R10's 5/5-PASSED graph-on run. See R14: the knob is net-negative
        # here, so it stays decode-only until someone isolates which of the
        # two sub-knobs (max_split_size_mb / garbage_collection_threshold)
        # does it and why.
        # The deepgemm split-K mHC pre-gemm is NOT reachable on this lane: the
        # cu129-pinned DeepGEMM 2.1.1 does not export tf32_hc_prenorm_gemm, so
        # deepgemm_wrapper._missing_deep_gemm() raises "DeepGEMM is not
        # available" during model init and the engine never starts. That is a
        # third 2.1.1 compat gap after fp8_einsum and max_seqlen_k. Until the
        # symbol lands (see team/perf-log.md R2) tilelang_single is the only
        # backend this build can actually run -- do not flip this to deepgemm
        # (the reference branch's target uses it because its DeepGEMM has it).
        "WORLD_SIZE": "4",
        "DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single",
        "DSV4_TRAP_INVALID_KV_ACCESS": "0",
        # torch 2.8 (cuda12_9) lacks float8_e8m0fnu: nogds copier loads
        # UE8M0 scales via the byte-mode U8 fallback (see smoke target).
        "FASTSAFETENSORS_NOGDS": "1",
        # .bazelrc test:cuda12_9 points CC/CXX at the CI image's
        # gcc-toolset-12; this container ships only /usr/bin/gcc (see smoke).
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDAHOSTCXX": "/usr/bin/g++",
        "NVCC_PREPEND_FLAGS": "-ccbin=/usr/bin/g++",
        # Task-private flashinfer JIT workspace: the shared
        # ~/.cache/flashinfer/0.6.9+e87d610f cached_ops were built against
        # tvm-ffi 0.1.8 (NDArray=72) and crash tvm-ffi 0.1.12 (Map=72) with
        # "Conflicting static index 72" at first forward. Fresh base dir
        # forces a rebuild against the current headers (see smoke target).
        "FLASHINFER_WORKSPACE_BASE": "/home/admin/dsv4-flashinfer-ws",
    },
    tags = ["manual", "H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)
