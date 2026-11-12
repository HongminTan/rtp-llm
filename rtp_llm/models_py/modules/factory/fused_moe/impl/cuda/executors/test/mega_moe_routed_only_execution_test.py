"""Two-rank production coverage for routed-only, standalone, and fused MegaMoE."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.mega_moe_execution_test_util import (
    _DIM,
    _EXPERTS,
    _MAX_TOKENS,
    _WORLD_SIZE,
    _assert_outputs_do_not_alias,
    _check_eager_layer,
    _check_graph_replays,
    _init_process_group,
    _make_global_weights,
    _make_moe,
    _sync_ranks_for_mega,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    resolve_moe_max_tokens_per_rank,
    synchronized_moe_chunk_plan,
)
from rtp_llm.utils.model_weight import W


def _worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    tokens_per_rank: tuple[int, int],
    run_graph: bool,
    include_shared: bool = False,
    hash_routing: bool = False,
    num_layers: int = 1,
    cp_request_lengths: tuple[int, ...] = (),
) -> None:
    device = _init_process_group(rank, world_size, rendezvous_path)
    try:
        capacity = _MAX_TOKENS
        if cp_request_lengths:
            capacity = resolve_moe_max_tokens_per_rank(
                max(cp_request_lengths),
                sum(cp_request_lengths),
                world_size,
                8,
                max_context_batch_size=len(cp_request_lengths),
                chunking_enabled=False,
            )
            assert capacity == tokens_per_rank[rank] > 4096
        torch.manual_seed(20260901)
        global_weights = _make_global_weights(
            device,
            include_shared=include_shared,
            hash_routing=hash_routing,
        )
        local_start = rank * (_EXPERTS // world_size)
        local_end = local_start + (_EXPERTS // world_size)
        mega_weights = {
            W.moe_gate: global_weights[W.moe_gate],
            W.moe_gate_bias: global_weights[W.moe_gate_bias],
        }
        if hash_routing:
            mega_weights[W.moe_gate_tid2eid] = global_weights[W.moe_gate_tid2eid]
        for key in (
            W.moe_w1,
            W.moe_s1,
            W.moe_w2,
            W.moe_s2,
        ):
            mega_weights[key] = global_weights[key][local_start:local_end].clone()
        if include_shared:
            for key in (
                W.ffn_w13,
                W.ffn_s13,
                W.ffn_w2,
                W.ffn_s2,
            ):
                mega_weights[key] = global_weights[key].clone()

        # Routed-only uses the production default. The shared-expert variant
        # explicitly selects ordinary MegaMoE to exercise the production
        # fallback used when MegaMoE-SE is unavailable.
        def clone_weights(weights: dict) -> dict:
            return {key: value.clone() for key, value in weights.items()}

        mega_layers = [
            _make_moe(
                clone_weights(mega_weights),
                ep_size=world_size,
                ep_rank=rank,
                strategy="mega_moe" if include_shared else None,
                n_shared_experts=int(include_shared),
                hash_routing=hash_routing,
                layer_id=7 + layer,
                max_tokens_per_rank=capacity,
                chunking_enabled=not cp_request_lengths,
            )
            for layer in range(num_layers)
        ]
        reference_layers = [
            _make_moe(
                clone_weights(global_weights),
                ep_size=1,
                ep_rank=0,
                strategy="local_loop",
                n_shared_experts=int(include_shared),
                hash_routing=hash_routing,
                layer_id=7 + layer,
                max_tokens_per_rank=capacity,
                chunking_enabled=not cp_request_lengths,
            )
            for layer in range(num_layers)
        ]
        mega = mega_layers[0]
        reference = reference_layers[0]
        assert isinstance(mega.fused_moe.fused_experts, MegaMoeExecutor)
        assert isinstance(reference.fused_moe.fused_experts, LocalLoopExecutor)
        if include_shared:
            assert not mega.fused_moe.includes_shared_expert
            assert mega._moe.shared_experts is not None
            assert mega._moe._shared_executor is not None
        else:
            assert not mega.fused_moe.includes_shared_expert
            assert mega._moe.shared_experts is None
            assert mega._moe._shared_executor is None

        tokens = tokens_per_rank[rank]
        torch.manual_seed(20261000 + rank)
        x = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
        input_ids = torch.arange(tokens, dtype=torch.long, device=device)
        with synchronized_moe_chunk_plan(mega_layers, tokens, device):
            with torch.inference_mode():
                actual = x
                expected = x
                for reference_layer, mega_layer in zip(reference_layers, mega_layers):
                    # Compare each discontinuous top-k layer from exactly the
                    # same input. The MegaMoE output still feeds the next
                    # layer, so this remains a real multi-layer stack while
                    # avoiding meaningless error amplification when tiny FP4
                    # differences select a different next-layer expert.
                    layer_input = actual
                    expected = reference_layer(layer_input, input_ids).clone()
                    _sync_ranks_for_mega(device)
                    actual = mega_layer(layer_input, input_ids).clone()
        torch.cuda.synchronize(device)
        dist.barrier()

        _assert_outputs_do_not_alias(expected, actual)
        assert actual.shape == x.shape
        assert torch.isfinite(actual).all()
        if tokens:
            error = (actual.float() - expected.float()).abs()
            baseline = expected.float().abs().mean().item() + 1e-6
            relative_error = error.mean().item() / baseline
            assert relative_error < 0.05, (
                f"rank={rank} MegaMoE relative error {relative_error:.6f} "
                "exceeds the local-loop baseline"
            )

        # Capture and replay the routed-only production path at a fixed
        # decode shape. This catches allocations or host-side decisions that
        # would make the n_shared_experts=0 branch graph-unsafe.
        if not run_graph:
            return
        assert num_layers == 1, "CUDA Graph coverage uses one production layer"
        torch.manual_seed(20261100 + rank)
        graph_x = torch.randn(2, _DIM, dtype=torch.bfloat16, device=device)
        graph_ids = torch.arange(2, dtype=torch.long, device=device)
        _check_graph_replays(
            mega, reference, graph_x, graph_ids, device, seed=20261200 + rank
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _se_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    tokens_per_rank: tuple[int, int],
    run_graph: bool,
    cp_request_lengths: tuple[int, ...] = (),
) -> None:
    device = _init_process_group(rank, world_size, rendezvous_path)
    try:
        capacity = _MAX_TOKENS
        if cp_request_lengths:
            capacity = resolve_moe_max_tokens_per_rank(
                max(cp_request_lengths),
                sum(cp_request_lengths),
                world_size,
                8,
                max_context_batch_size=len(cp_request_lengths),
                chunking_enabled=False,
            )
            assert capacity == tokens_per_rank[rank] > 4096
        torch.manual_seed(20263001)
        n_shared_experts = 2
        global_weights = _make_global_weights(
            device, include_shared=True, n_shared_experts=n_shared_experts
        )
        local_start = rank * (_EXPERTS // world_size)
        local_end = local_start + (_EXPERTS // world_size)
        mega_se_weights = {
            W.moe_gate: global_weights[W.moe_gate],
            W.moe_gate_bias: global_weights[W.moe_gate_bias],
            W.ffn_w13: global_weights[W.ffn_w13].clone(),
            W.ffn_s13: global_weights[W.ffn_s13].clone(),
            W.ffn_w2: global_weights[W.ffn_w2].clone(),
            W.ffn_s2: global_weights[W.ffn_s2].clone(),
        }
        for key in (
            W.moe_w1,
            W.moe_s1,
            W.moe_w2,
            W.moe_s2,
        ):
            mega_se_weights[key] = global_weights[key][local_start:local_end].clone()

        # The production default must consume the shared weights in the
        # fused MegaMoESE strategy. The independent EP=1 reference executes
        # the same shared expert through the standalone path.
        mega_se = _make_moe(
            mega_se_weights,
            ep_size=world_size,
            ep_rank=rank,
            strategy=None,
            n_shared_experts=n_shared_experts,
            max_tokens_per_rank=capacity,
            chunking_enabled=not cp_request_lengths,
        )
        reference = _make_moe(
            dict(global_weights),
            ep_size=1,
            ep_rank=0,
            strategy="local_loop",
            n_shared_experts=n_shared_experts,
            max_tokens_per_rank=capacity,
            chunking_enabled=not cp_request_lengths,
        )
        # DeepGEMM 2.8 removed the fused shared_recipe API: when the wheel
        # still provides it, auto selects MegaMoE-SE; otherwise the strategy
        # degrades to ordinary MegaMoE plus the standalone shared expert
        # (same weights and quantization), which the numerical comparison
        # below covers either way.
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            _mega_moe_se_available,
        )

        if _mega_moe_se_available():
            assert isinstance(mega_se.fused_moe.fused_experts, MegaMoeSEExecutor)
            assert mega_se.fused_moe.includes_shared_expert
            assert mega_se._moe.shared_experts is None
            assert mega_se._moe._shared_executor is None
        else:
            assert isinstance(mega_se.fused_moe.fused_experts, MegaMoeExecutor)
            assert not mega_se.fused_moe.includes_shared_expert
            assert mega_se._moe.shared_experts is not None
            assert mega_se._moe._shared_executor is not None
        assert isinstance(reference.fused_moe.fused_experts, LocalLoopExecutor)
        assert reference.shared_experts is not None
        assert reference._moe._shared_executor is not None

        tokens = tokens_per_rank[rank]
        torch.manual_seed(20263100 + rank)
        x = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
        input_ids = torch.arange(tokens, dtype=torch.long, device=device)
        _check_eager_layer(mega_se, reference, x, input_ids, label=f"SE/rank={rank}")

        if not run_graph:
            return
        torch.manual_seed(20263200 + rank)
        graph_x = torch.randn(2, _DIM, dtype=torch.bfloat16, device=device)
        graph_ids = torch.arange(2, dtype=torch.long, device=device)
        _check_graph_replays(
            mega_se, reference, graph_x, graph_ids, device, seed=20263300 + rank
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class RoutedOnlyMegaMoEExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated target")
        if torch.cuda.device_count() < _WORLD_SIZE:
            raise AssertionError(
                f"{_WORLD_SIZE} GPUs are required by this dedicated target"
            )

    def _spawn(self, worker, *args) -> None:
        test_tmpdir = os.environ.get("TEST_TMPDIR")
        with tempfile.TemporaryDirectory(dir=test_tmpdir) as tmpdir:
            rendezvous_path = os.path.join(tmpdir, "torch_dist_rendezvous")
            mp.spawn(
                worker,
                args=(_WORLD_SIZE, rendezvous_path, *args),
                nprocs=_WORLD_SIZE,
                join=True,
            )

    def test_two_rank_default_handles_empty_rank(self):
        self._spawn(_worker, (8, 0), False)

    def test_cp_multi_request_unchunked_mega_moe(self):
        # Three 4097-token requests each contribute 2050 padded tokens/rank.
        self._spawn(_worker, (6150, 6150), False, False, False, 1, (4097,) * 3)

    def test_cp_multi_request_unchunked_mega_moe_se(self):
        self._spawn(_se_worker, (6150, 6150), False, (4097,) * 3)

    def test_two_rank_chunking_keeps_collectives_symmetric(self):
        self._spawn(_worker, (65, 1), False)

    def test_two_rank_layer_stack_reuses_one_synchronized_chunk_plan(self):
        self._spawn(_worker, (65, 1), False, False, False, 2)

    def test_two_rank_default_matches_local_reference_and_cuda_graph(self):
        self._spawn(_worker, (3, 2), True)

    def test_two_rank_hash_routing_matches_local_reference_and_cuda_graph(self):
        self._spawn(_worker, (3, 2), True, False, True)

    def test_two_rank_mega_moe_standalone_shared_expert_fallback(self):
        self._spawn(_worker, (3, 2), True, True)

    def test_two_rank_multi_se_handles_empty_rank(self):
        self._spawn(_se_worker, (8, 0), False)

    def test_two_rank_multi_se_chunking_keeps_collectives_symmetric(self):
        self._spawn(_se_worker, (65, 1), False)

    def test_two_rank_multi_se_matches_standalone_and_cuda_graph(self):
        self._spawn(_se_worker, (3, 2), True)


if __name__ == "__main__":
    unittest.main()
