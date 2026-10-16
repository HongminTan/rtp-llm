"""Diagnostics targets (R182-class: microbenches discriminating engine-level
phenomena with minimal standalone carriers)."""

py_test(
    name = "a2a_first_collective_bench",
    srcs = ["a2a_first_collective_bench.py"],
    main = "a2a_first_collective_bench.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
        "//rtp_llm:jit_cache_env",
    ],
    env = {"WORLD_SIZE": "4"},
    size = "large",
    tags = ["cu129_local_carrier", "exclusive"],
)
