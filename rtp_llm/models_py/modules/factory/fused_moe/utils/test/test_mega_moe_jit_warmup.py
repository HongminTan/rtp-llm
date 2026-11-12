import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _MEGA_MOE_JIT_WARMED_KEYS,
    MegaMoeExecutor,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _mega_moe_rank_nvcc_tmpdir,
    _restore_tmpdir,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_sm90 import (
    MegaMoeSm90Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts,
    generate_mega_moe_jit_token_counts,
    generate_mega_moe_sm90_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    mega_moe_sm90_config_signature,
    parse_mega_moe_jit_warmup_tokens_override,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


class MegaMoeJitWarmupTest(unittest.TestCase):
    def test_model_warmup_switch(self):
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_rank_local_nvcc_directory(self):
        with mock.patch.dict(
            os.environ, {"DG_JIT_CACHE_DIR": "/tmp/dg-cache"}, clear=True
        ):
            scratch = _mega_moe_rank_nvcc_tmpdir(7)
            self.assertNotEqual(scratch, _mega_moe_rank_nvcc_tmpdir(8))
            self.assertEqual(scratch, _mega_moe_rank_nvcc_tmpdir(7))
            self.assertLess(len(os.fsencode(scratch)) + 40, 108)
            self.assertEqual(os.environ["DG_JIT_CACHE_DIR"], "/tmp/dg-cache")

    def test_tmpdir_is_restored_after_warmup_failure(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        model_config = ModelConfig()
        model_config.hidden_size = 7168
        model_config.moe_inter_size = 2048
        model_config.expert_num = 256
        model_config.moe_k = 6
        model_config.n_shared_experts = 1
        model_config.swiglu_limit = 7.0
        model_config.routed_scaling_factor = 1.0
        model_config.moe_w1_layout = "up_gate"
        parallelism = ParallelismConfig()
        parallelism.ep_size = 8
        parallelism.ep_rank = 5
        parallelism.world_size = 8
        parallelism.world_rank = 5
        parallelism.dp_size = 8
        parallelism.dp_rank = 5
        moe_config = MoeConfig()
        moe_config.moe_strategy = "mega_moe"
        executor.cfg = MoEConfigAdapter(
            model_config, parallelism, moe_config
        ).for_layer(
            layer_id=1,
            moe_inter_dim=2048,
            moe_w1_layout="up_gate",
            max_tokens_per_rank=4096,
            moe_quant_method="FP8_FP4",
        )
        executor._input_packer = FusedMegaMoEInputPacker()
        executor.warmup_jit = mock.Mock(side_effect=RuntimeError("compile failed"))
        fake_deep_gemm = types.SimpleNamespace(get_num_sms=lambda: 148)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MEGA_MOE_NVCC_TMPDIR": tmpdir,
                "MODEL_WARM_UP": "1",
                "TMPDIR": "/old/tmp",
            },
            clear=True,
        ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        ), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.get_rank", return_value=5
        ):
            _MEGA_MOE_JIT_WARMED_KEYS.clear()
            self.addCleanup(_MEGA_MOE_JIT_WARMED_KEYS.clear)
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                executor._maybe_warmup_jit_once()
            self.assertFalse(_MEGA_MOE_JIT_WARMED_KEYS)
            self.assertEqual(os.environ["TMPDIR"], "/old/tmp")
            self.assertTrue(
                os.path.isdir(os.path.join(tmpdir, "rtp_llm_mega_moe_nvcc", "rank_5"))
            )

    def test_warmup_compiles_ordinary_and_both_gate_pack_variants(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        model_config = ModelConfig()
        model_config.hidden_size = 128
        model_config.moe_inter_size = 128
        model_config.expert_num = 16
        model_config.moe_k = 4
        model_config.n_shared_experts = 0
        model_config.swiglu_limit = 7.0
        model_config.routed_scaling_factor = 2.5
        model_config.moe_w1_layout = "up_gate"
        parallelism = ParallelismConfig()
        parallelism.ep_size = 2
        parallelism.ep_rank = 0
        parallelism.world_size = 2
        parallelism.world_rank = 0
        parallelism.dp_size = 2
        parallelism.dp_rank = 0
        moe_config = MoeConfig()
        moe_config.moe_strategy = "mega_moe"
        executor.cfg = MoEConfigAdapter(
            model_config, parallelism, moe_config
        ).for_layer(
            layer_id=1,
            moe_inter_dim=128,
            moe_w1_layout="up_gate",
            max_tokens_per_rank=4,
            moe_quant_method="FP8_FP4",
        )
        executor._mega_l1_w = torch.empty(1)
        executor._input_packer = FusedMegaMoEInputPacker()

        with mock.patch.dict(
            os.environ,
            {
                "MODEL_WARM_UP": "1",
                "MEGA_MOE_INPUT_PACKER_IMPL": "optimized",
            },
            clear=True,
        ), mock.patch.object(executor, "forward") as forward, mock.patch.object(
            executor, "forward_gate_pack"
        ) as forward_gate_pack, mock.patch(
            "torch.distributed.barrier"
        ) as barrier, mock.patch(
            "torch.cuda.synchronize"
        ):
            executor.warmup_jit([1, 4])

        self.assertEqual(forward.call_count, 2)
        self.assertEqual(forward_gate_pack.call_count, 4)
        payloads = [call.args[1] for call in forward_gate_pack.call_args_list]
        nonhash = [payload for payload in payloads if payload.bias is not None]
        hashed = [payload for payload in payloads if payload.tid2eid is not None]
        self.assertEqual(len(nonhash), 2)
        self.assertEqual(len(hashed), 2)
        self.assertTrue(all(payload.route_scale == 2.5 for payload in payloads))
        self.assertTrue(all(payload.input_ids is not None for payload in hashed))
        self.assertEqual(barrier.call_count, 3)

    def test_unchunked_generated_counts_use_bucket_representative(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
        )
        self.assertLess(tokens[-1], 16384)

    def test_chunked_generated_counts_cover_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
            include_cap=True,
        )
        self.assertEqual(tokens[-1], 16384)

    def test_generated_counts_cover_every_unique_bucket_for_ep_sizes(self):
        for ep_size in (2, 4, 8):
            with self.subTest(ep_size=ep_size):
                params = dict(
                    num_ranks=ep_size,
                    num_experts=256,
                    num_experts_per_rank=256 // ep_size,
                    num_topk=6,
                    intermediate_hidden=2048,
                    num_sms=148,
                )
                tokens = generate_mega_moe_jit_token_counts(
                    **params,
                    max_tokens_per_rank=4096,
                )
                self.assertTrue(any(1 < token < 4096 for token in tokens))
                signatures = [
                    mega_moe_config_signature(**params, num_tokens=token)
                    for token in tokens
                ]
                expected_signatures = []
                previous = None
                for token in range(1, 4097):
                    signature = mega_moe_config_signature(**params, num_tokens=token)
                    if signature != previous:
                        expected_signatures.append(signature)
                        previous = signature
                self.assertEqual(signatures, expected_signatures)

    def test_generated_counts_never_exceed_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_experts_per_rank=32,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=257,
        )
        self.assertTrue(tokens)
        self.assertLessEqual(max(tokens), 257)

    def test_override_tokens_are_generic_sorted_and_clamped(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "4098,2,2,999999,0,-1"},
        ):
            tokens = parse_mega_moe_jit_warmup_tokens_override()
        self.assertEqual(tokens, [2, 4098, 999999])
        self.assertEqual(clamp_token_counts(tokens or [], 65536), [2, 4098, 65536])

    def test_invalid_override_falls_back_to_generated_counts(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "2,not-a-token"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("invalid MEGA_MOE_JIT_WARMUP_TOKENS", "\n".join(logs.output))

    def test_override_without_positive_tokens_falls_back(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "0,-1,-20"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("contains no positive token counts", "\n".join(logs.output))

    def test_generic_adapter_supports_default_warmup_for_both_executors(self):
        model_config = ModelConfig()
        model_config.expert_num = 8
        model_config.moe_k = 2
        model_config.moe_inter_size = 128
        model_config.n_shared_experts = 1
        model_config.max_seq_len = 256
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 2
        parallelism_config.ep_rank = 0
        parallelism_config.world_size = 2
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

        cases = (
            (
                MegaMoeExecutor,
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "mega_moe.generate_mega_moe_jit_token_counts",
            ),
            (
                MegaMoeSEExecutor,
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "mega_moe_se.generate_mega_moe_se_jit_token_counts",
            ),
        )
        for executor_type, generator_path in cases:
            with self.subTest(executor=executor_type.__name__):
                executor = executor_type.__new__(executor_type)
                torch.nn.Module.__init__(executor)
                executor.cfg = config
                with mock.patch(generator_path, return_value=[1]) as generator:
                    self.assertEqual(
                        executor._resolve_jit_warmup_token_counts(128), [1]
                    )
                self.assertFalse(generator.call_args.kwargs["include_cap"])


class MegaMoeSm90JitWarmupTest(unittest.TestCase):
    def test_generated_counts_cover_row_layout_boundaries(self):
        # Independent boundary fixtures: arrival counter, occupancy, decode
        # policy, bound-derived lag and pull batching.
        expected = {
            2: [1, 4, 22, 86, 129, 257, 8192],
            4: [1, 4, 11, 43, 129, 257, 8192, 10881],
            8: [1, 4, 6, 22, 129, 257, 5377, 8192, 10881],
        }
        for ep_size, counts in expected.items():
            with self.subTest(ep_size=ep_size):
                self.assertEqual(
                    generate_mega_moe_sm90_jit_token_counts(
                        num_ranks=ep_size,
                        num_experts=256,
                        num_topk=6,
                        max_tokens_per_rank=16384,
                    ),
                    counts,
                )

    def test_row_layout_tile_and_bound_lag(self):
        # Row-layout MXFP4 retains the 64x256 tile above 256 tokens.
        for tokens in (1, 4, 128, 129, 256, 257, 683, 8192, 16384):
            with self.subTest(tokens=tokens):
                signature = mega_moe_sm90_config_signature(
                    num_experts_per_rank=32,
                    num_topk=6,
                    num_tokens=tokens,
                )
                self.assertEqual(signature[0], 64)
                self.assertEqual(signature[1], 4 <= tokens <= 128)
        # The bound is rounded up to 128 before computing the L2 lag.
        for tokens, lag in ((5376, 14), (5377, 29), (10880, 29), (10881, 43)):
            with self.subTest(tokens=tokens):
                signature = mega_moe_sm90_config_signature(
                    num_experts_per_rank=32,
                    num_topk=6,
                    num_tokens=tokens,
                )
                self.assertEqual(signature[6], lag)

    def test_generated_counts_never_exceed_runtime_cap(self):
        tokens = generate_mega_moe_sm90_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_topk=6,
            max_tokens_per_rank=257,
        )
        self.assertTrue(tokens)
        self.assertLessEqual(max(tokens), 257)

    def test_unchunked_generated_counts_use_bucket_representative(self):
        tokens = generate_mega_moe_sm90_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_topk=6,
            max_tokens_per_rank=16384,
        )
        self.assertLess(tokens[-1], 16384)

    def test_chunked_generated_counts_cover_runtime_cap(self):
        tokens = generate_mega_moe_sm90_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_topk=6,
            max_tokens_per_rank=16384,
            include_cap=True,
        )
        self.assertEqual(tokens[-1], 16384)

    def test_warmup_runs_eager_and_bound_variants_per_bucket(self):
        executor = MegaMoeSm90Executor.__new__(MegaMoeSm90Executor)
        torch.nn.Module.__init__(executor)
        model_config = ModelConfig()
        model_config.hidden_size = 128
        model_config.moe_inter_size = 128
        model_config.expert_num = 16
        model_config.moe_k = 4
        model_config.n_shared_experts = 0
        model_config.swiglu_limit = 7.0
        parallelism = ParallelismConfig()
        parallelism.ep_size = 2
        parallelism.ep_rank = 0
        parallelism.world_size = 2
        parallelism.world_rank = 0
        parallelism.dp_size = 2
        parallelism.dp_rank = 0
        executor.cfg = MoEConfigAdapter(
            model_config, parallelism, MoeConfig()
        ).for_layer(
            layer_id=1,
            moe_inter_dim=128,
            moe_w1_layout="gate_up",
            max_tokens_per_rank=4,
            moe_quant_method="FP8_FP4",
        )
        executor._output = torch.empty(4, 128)

        with mock.patch.dict(
            os.environ, {"MODEL_WARM_UP": "1"}, clear=True
        ), mock.patch.object(executor, "_run") as run, mock.patch(
            "torch.distributed.barrier"
        ) as barrier, mock.patch(
            "torch.cuda.synchronize"
        ):
            executor.warmup_jit([1, 4])

        shapes = [call.args[0].size(0) for call in run.call_args_list]
        bounds = [call.args[3] for call in run.call_args_list]
        self.assertEqual(shapes, [1, 1, 4, 4])
        self.assertEqual(bounds, [None, 1, None, 4])
        self.assertEqual(barrier.call_count, 5)


if __name__ == "__main__":
    unittest.main()
