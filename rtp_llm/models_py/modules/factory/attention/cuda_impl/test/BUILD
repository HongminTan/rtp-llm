# Test configuration for PyFlashinferDecodeAttnOp

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

# Library for attention reference implementation
py_library(
    name = "atten_test_util",
    srcs = ["atten_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "trtllm_gen_test",
    srcs = ["trtllm_gen_test.py"],
    deps = [
        ":atten_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'}
)

# Test for PyFlashinferDecodeAttnOp decode attention implementation
py_test(
    name = "test_py_flashinfer_mha_decode",
    srcs = [
        "test_py_flashinfer_mha_decode.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_xqa",
    srcs = [
        "test_xqa.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# TRT attention tests - organized in trt_tests/ subdirectory
py_test(
    name = "test_trt_nonpadded",
    srcs = [
        "trt_tests/test_trt_nonpadded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trt_padded",
    srcs = [
        "trt_tests/test_trt_padded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trt_paged_nonpadded",
    srcs = [
        "trt_tests/test_trt_paged_nonpadded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Benchmark for FlashInfer tensor cores performance comparison
py_test(
    name = "bench_flashinfer_tensor_cores",
    srcs = [
        "bench_flashinfer_tensor_cores.py",
        "bench_utils.py",
    ],
    deps = py_test_deps,
    tags = ["manual", "H20"],  # manual tag for benchmark tests
    exec_properties = {
        'gpu': 'H20',
    },
)

# PrefillImpl BenchTest
py_test(
    name = "bench_prefill_impls",
    srcs = [
        "bench_prefill_impls.py",
        "bench_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    args = [
        "--q-dtype=bf16",
        "--kv-cache-dtype=bf16",
        "--input-len=15,128,1024,4096,16384,65536",
        "--reuse-cache-ratio=0,0.3,0.7",
        "--batch-size=1",
        "--preset=qwen3-8b",
        "--page-size=128",
        "--warmup=10",
        "--repeat=50",
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["manual", "H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# FlashInfer Prefill Tests - organized in test_flashinfer_prefill/ subdirectory

# Test for PyFlashinferPrefillAttnOp ragged prefill attention implementation
py_test(
    name = "test_py_flashinfer_ragged_mha_prefill",
    srcs = [
        "test_flashinfer_prefill/test_py_flashinfer_ragged_mha_prefill.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Test for PyFlashinferPrefillPagedAttnOp paged prefill attention implementation
py_test(
    name = "test_py_flashinfer_paged_mha_prefill",
    srcs = [
        "test_flashinfer_prefill/test_py_flashinfer_paged_mha_prefill.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Test for PyFlashinferPrefillPagedAttnOp CUDA graph path vs normal path
py_test(
    name = "test_py_flashinfer_paged_prefill_cuda_graph",
    srcs = [
        "test_flashinfer_prefill/test_py_flashinfer_paged_prefill_cuda_graph.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Test for MhaRotaryEmbeddingOp - RoPE implementation for MHA
py_test(
    name = "test_mha_rotary_emb",
    srcs = [
        "test_flashinfer_prefill/test_mha_rotary_emb.py",
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Flashinfer TRT attention tests
py_test(
    name = "test_trtllm_fmha_v2_paged_prefill",
    srcs = [
        "test_trtllm_fmha_v2_paged_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trtllm_fmha_v2_prefill",
    srcs = [
        "test_trtllm_fmha_v2_prefill.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py",
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

