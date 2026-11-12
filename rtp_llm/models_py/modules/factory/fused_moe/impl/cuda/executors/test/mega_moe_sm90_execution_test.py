"""Two-rank SM90 integration: numerical reference, empty ranks, chunks and graphs."""

import os
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test import (
    mega_moe_execution_test_util as helpers,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.utils.model_weight import W


def _fp4(weight, scale):
    raw = weight.view(torch.uint8)
    bits = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    return lut[bits] * scale.float().repeat_interleave(32, dim=-1)


def _quant_dequant(x):
    q, s = sgl_per_token_group_quant_fp8(x.contiguous(), 128, eps=1e-4)
    return q.float() * s.repeat_interleave(128, dim=-1)


def _reference(x, route_weights, route_ids, weights, shared):
    """Independent PyTorch GEMMs, including the documented per-route rounding."""
    if not x.numel():
        return torch.empty_like(x)
    quantized = _quant_dequant(x)
    w13 = _fp4(weights[W.moe_w1], weights[W.moe_s1])
    w2 = _fp4(weights[W.moe_w2], weights[W.moe_s2])
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w13.size(0)):
        rows, slots = torch.where(route_ids == expert)
        if not rows.numel():
            continue
        gate, up = (quantized[rows] @ w13[expert].T).chunk(2, dim=-1)
        hidden = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        hidden *= route_weights[rows, slots, None]
        result = (_quant_dequant(hidden) @ w2[expert].T).bfloat16().float()
        output.index_add_(0, rows, result)
    if shared:

        def dequant(wk, sk):
            return weights[wk].float() * weights[sk].float().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)

        gate, up = (quantized @ dequant(W.ffn_w13, W.ffn_s13).T).chunk(2, -1)
        hidden = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        output += (
            (_quant_dequant(hidden) @ dequant(W.ffn_w2, W.ffn_s2).T).bfloat16().float()
        )
    return output.bfloat16()


def _layer_reference(layer, weights, shared, x, ids):
    route_weights, route_ids = layer.gate(x, ids)
    return _reference(x, route_weights, route_ids, weights, shared)


def _worker(rank, rendezvous, name, shared, hash_routing, scenario):
    os.environ["MOE_SHARED_EXPERT_MODE"] = "auto"
    device = helpers._init_process_group(rank, 2, rendezvous, timeout_seconds=180)
    try:
        capacity = helpers._MAX_TOKENS
        if scenario == "unchunked_cp":
            # Match SM100: three 4097-token requests, padded across two ranks.
            request_lengths = (4097,) * 3
            capacity = resolve_moe_max_tokens_per_rank(
                max(request_lengths),
                sum(request_lengths),
                2,
                8,
                max_context_batch_size=len(request_lengths),
                chunking_enabled=False,
            )
            assert capacity == 6150
        torch.manual_seed(312)
        global_weights = helpers._make_global_weights(
            device,
            include_shared=True,
            n_shared_experts=max(shared, 1),
            hash_routing=hash_routing,
            dim=1024,
            inter_dim=256,
        )
        # Vary scales across experts, rows and K groups without triggering
        # the transform's wide-exponent-range requantization.
        for key, low in (
            (W.moe_s1, 120),
            (W.moe_s2, 120),
            (W.ffn_s13, 125),
            (W.ffn_s2, 125),
        ):
            global_weights[key].view(torch.uint8).random_(low, low + 4)
        local = {
            key: (
                value[rank * 8 : (rank + 1) * 8].clone()
                if key in (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2)
                else value.clone()
            )
            for key, value in global_weights.items()
        }
        layers = []
        for index, layout in enumerate(("gate_up", "up_gate")):
            layer_weights = {key: value.clone() for key, value in local.items()}
            if layout == "up_gate":
                for key in (W.moe_w1, W.moe_s1):
                    gate, up = layer_weights[key].view(torch.uint8).chunk(2, dim=1)
                    layer_weights[key] = torch.cat((up, gate), dim=1).view(
                        local[key].dtype
                    )
            layer = helpers._make_moe(
                layer_weights,
                ep_size=2,
                ep_rank=rank,
                strategy=name,
                n_shared_experts=shared,
                hash_routing=hash_routing,
                max_tokens_per_rank=capacity,
                chunking_enabled=scenario != "unchunked_cp",
                layer_id=index,
                dim=1024,
                inter_dim=256,
                moe_w1_layout=layout,
            )
            fused_shared = name in ("mega_moe_se", "auto") and shared == 1
            assert layer.fused_moe.includes_shared_expert == fused_shared
            assert (layer._moe._shared_executor is not None) == (
                shared > 0 and not fused_shared
            )
            layers.append(layer)
        references = [
            partial(_layer_reference, layer, global_weights, shared) for layer in layers
        ]
        label = f"{name}/shared={shared}/hash={hash_routing}/rank={rank}"
        if scenario == "stack":
            torch.manual_seed(400 + rank)
            x = torch.randn((67, 9)[rank], 1024, dtype=torch.bfloat16, device=device)
            ids = torch.arange(x.size(0), device=device) % 64
            # The entire stack must share one planning collective. Backend
            # communication uses symmetric memory, not dist.all_reduce.
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
                helpers._check_eager_layers(layers, references, x, ids, label=label)
            assert reduce.call_count == 1, reduce.call_count
            return

        counts_cases = (
            ((capacity, capacity),)
            if scenario == "unchunked_cp"
            else ((7, 3), (0, 5), (0, 0), (67, 9))
        )
        for layout, layer, reference in zip(("gate_up", "up_gate"), layers, references):
            for counts in counts_cases:
                torch.manual_seed(400 + rank + counts[rank])
                x = torch.randn(counts[rank], 1024, dtype=torch.bfloat16, device=device)
                ids = torch.arange(x.size(0), device=device) % 64
                helpers._check_eager_layer(
                    layer,
                    reference,
                    x,
                    ids,
                    label=f"{label}/{layout}/counts={counts}",
                )
            if scenario == "unchunked_cp":
                continue
            x = torch.randn(2, 1024, dtype=torch.bfloat16, device=device)
            ids = torch.arange(2, device=device)
            helpers._check_graph_replays(
                layer, reference, x, ids, device, seed=500 + rank
            )
    finally:
        dist.destroy_process_group()


class MegaMoeSm90ExecutionTest(unittest.TestCase):
    def _run(self, strategy, shared, *, hash_routing=False, scenario="execution"):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] != 9
            or torch.cuda.device_count() < 2
        ):
            self.skipTest("requires two SM90 GPUs")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _worker,
                args=(
                    os.path.join(directory, "rendezvous"),
                    strategy,
                    shared,
                    hash_routing,
                    scenario,
                ),
                nprocs=2,
                join=True,
            )

    def test_routed_topk_matches_reference_and_cuda_graph(self):
        self._run("mega_moe", 0)

    def test_routed_hash_matches_reference_and_cuda_graph(self):
        self._run("mega_moe", 0, hash_routing=True)

    def test_standalone_shared_matches_reference_and_cuda_graph(self):
        self._run("mega_moe", 1)

    def test_native_shared_matches_reference_and_cuda_graph(self):
        self._run("mega_moe_se", 1)

    def test_wider_shared_fallback_matches_reference_and_cuda_graph(self):
        self._run("mega_moe_se", 2)

    def test_auto_selects_native_shared(self):
        self._run("auto", 1)

    def test_auto_selects_wider_shared_fallback(self):
        self._run("auto", 2)

    def test_layer_stack_reuses_one_synchronized_chunk_plan(self):
        for strategy, shared in (("mega_moe", 0), ("mega_moe_se", 1)):
            with self.subTest(strategy=strategy):
                self._run(strategy, shared, scenario="stack")

    def test_cp_multi_request_unchunked_prefill(self):
        for strategy, shared in (("mega_moe", 0), ("mega_moe_se", 1)):
            with self.subTest(strategy=strategy):
                self._run(strategy, shared, scenario="unchunked_cp")


if __name__ == "__main__":
    unittest.main()
