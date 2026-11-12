"""Shared fixtures and CUDA graph checks for MegaMoE execution tests."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer as MoE,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    synchronized_moe_chunk_plan,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

_WORLD_SIZE = 2
_EXPERTS = 16
_TOPK = 4
_DIM = 4096
_INTER_DIM = 2048
_MAX_TOKENS = 32


def _sync_ranks_for_mega(device: torch.device) -> None:
    """Finish NCCL work before DeepGEMM occupies every SM for grid sync."""

    dist.barrier()
    torch.cuda.synchronize(device)


def _assert_outputs_do_not_alias(expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Catch result-cache self-comparisons, including zero-length tensors."""

    assert expected is not actual
    assert not expected.is_set_to(actual)
    if expected.numel() and actual.numel():
        assert expected.data_ptr() != actual.data_ptr()


def _ue8m0_ones(*shape: int, device: torch.device) -> torch.Tensor:
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )


def _make_global_weights(
    device: torch.device,
    *,
    include_shared: bool = False,
    n_shared_experts: int = 1,
    hash_routing: bool = False,
    dim: int = _DIM,
    inter_dim: int = _INTER_DIM,
) -> dict:
    def packed_fp4(out_dim: int, in_dim: int) -> torch.Tensor:
        shape = (_EXPERTS, out_dim, in_dim // 2)
        # Build each packed byte from finite E2M1 nibbles. Arbitrary int8
        # bytes can encode reserved/non-finite FP4 patterns and make a
        # numerical regression fail for reasons unrelated to the kernel.
        low = torch.randint(0, 7, shape, dtype=torch.uint8, device=device)
        high = torch.randint(0, 7, shape, dtype=torch.uint8, device=device)
        low |= torch.randint(0, 2, shape, dtype=torch.uint8, device=device) << 3
        high |= torch.randint(0, 2, shape, dtype=torch.uint8, device=device) << 3
        return (low | (high << 4)).view(torch.int8)

    def fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
        return _ue8m0_ones(
            _EXPERTS,
            out_dim,
            in_dim // 32,
            device=device,
        )

    weights = {
        W.moe_gate: torch.randn(_EXPERTS, dim, dtype=torch.bfloat16, device=device),
        W.moe_gate_bias: torch.zeros(_EXPERTS, dtype=torch.float32, device=device),
        W.moe_w1: packed_fp4(inter_dim, dim),
        W.moe_s1: fp4_scale(inter_dim, dim),
        W.moe_w2: packed_fp4(dim, inter_dim),
        W.moe_s2: fp4_scale(dim, inter_dim),
        "routed_up": packed_fp4(inter_dim, dim),
        "routed_up_scale": fp4_scale(inter_dim, dim),
    }
    if hash_routing:
        weights[W.moe_gate_tid2eid] = torch.stack(
            [torch.randperm(_EXPERTS, device=device)[:_TOPK] for _ in range(64)]
        ).contiguous()
    if include_shared:
        shared_inter = inter_dim * n_shared_experts
        weights.update(
            {
                W.ffn_w13: (
                    torch.randn(
                        2 * shared_inter,
                        dim,
                        dtype=torch.float32,
                        device=device,
                    )
                    * 0.05
                ).to(torch.float8_e4m3fn),
                W.ffn_s13: _ue8m0_ones(
                    2 * shared_inter // 128,
                    dim // 128,
                    device=device,
                ),
                W.ffn_w2: (
                    torch.randn(
                        dim,
                        shared_inter,
                        dtype=torch.float32,
                        device=device,
                    )
                    * 0.05
                ).to(torch.float8_e4m3fn),
                W.ffn_s2: _ue8m0_ones(
                    dim // 128,
                    shared_inter // 128,
                    device=device,
                ),
            }
        )
    # Preserve the original split-tensor random draws before canonical packing.
    weights[W.moe_w1] = torch.cat((weights[W.moe_w1], weights.pop("routed_up")), dim=1)
    weights[W.moe_s1] = torch.cat(
        (
            weights[W.moe_s1].view(torch.uint8),
            weights.pop("routed_up_scale").view(torch.uint8),
        ),
        dim=1,
    ).view(torch.float8_e8m0fnu)
    return weights


def _make_moe(
    layer_weights: dict,
    *,
    ep_size: int,
    ep_rank: int,
    strategy,
    n_shared_experts: int = 0,
    hash_routing: bool = False,
    layer_id: int = 7,
    max_tokens_per_rank: int = _MAX_TOKENS,
    chunking_enabled: bool = True,
    moe_w1_layout: str = "gate_up",
    dim: int = _DIM,
    inter_dim: int = _INTER_DIM,
) -> MoE:
    model_config = ModelConfig()
    model_config.hidden_size = dim
    model_config.moe_inter_size = inter_dim
    model_config.expert_num = _EXPERTS
    model_config.moe_k = _TOPK
    model_config.n_shared_experts = n_shared_experts
    model_config.routed_scaling_factor = 1.5
    model_config.swiglu_limit = 10.0
    model_config.moe_w1_layout = moe_w1_layout
    parallelism = ParallelismConfig()
    parallelism.ep_size = ep_size
    parallelism.ep_rank = ep_rank
    parallelism.world_size = ep_size
    parallelism.world_rank = ep_rank
    parallelism.dp_size = ep_size
    parallelism.dp_rank = ep_rank
    moe_config = MoeConfig()
    moe_config.moe_strategy = strategy
    config = MoEConfigAdapter(model_config, parallelism, moe_config).for_layer(
        layer_id=layer_id,
        moe_inter_dim=inter_dim,
        moe_w1_layout=moe_w1_layout,
        max_tokens_per_rank=max_tokens_per_rank,
        moe_quant_method="FP8_FP4",
        warmup_include_capacity=chunking_enabled,
    )
    return MoE(
        config=config,
        score_func="sqrtsoftplus",
        n_hash_layers=8 if hash_routing else 0,
        vocab_size=64,
        layer_weights=layer_weights,
        chunking_enabled=chunking_enabled,
    )


def _check_graph_replays(layer, reference, x, ids, device, *, seed, replays=3):
    """Check changed-input replay against a caller-provided numerical reference."""
    with torch.inference_mode():
        reference(x, ids)
        _sync_ranks_for_mega(device)
        layer(x, ids)
        _sync_ranks_for_mega(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer(x, ids)
        torch.manual_seed(seed)
        for replay in range(replays):
            x.copy_(torch.randn_like(x))
            ids.copy_(torch.arange(ids.numel() - 1, -1, -1, device=device).roll(replay))
            expected = reference(x, ids).clone()
            _assert_outputs_do_not_alias(expected, actual)
            actual.fill_(float("nan"))
            _sync_ranks_for_mega(device)
            graph.replay()
            torch.cuda.synchronize(device)
            dist.barrier()
            assert torch.isfinite(actual).all()
            relative = (
                actual.float() - expected.float()
            ).abs().mean() / expected.float().abs().mean().clamp_min(1e-6)
            assert relative < 0.05, (replay, relative.item())


def _init_process_group(rank, world_size, rendezvous, *, timeout_seconds=120):
    os.environ["MODEL_WARM_UP"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return device


def _check_eager_layer(layer, reference, x, ids, *, label):
    return _check_eager_layers([layer], [reference], x, ids, label=label)


def _check_eager_layers(layers, references, x, ids, *, label):
    """Validate every layer using one synchronized plan and actual stack inputs."""
    assert len(layers) == len(references) > 0
    with torch.inference_mode(), synchronized_moe_chunk_plan(
        layers, x.size(0), x.device
    ):
        actual = x
        for index, (layer, reference) in enumerate(zip(layers, references)):
            # Use the same input for both paths: small numerical differences
            # must not select different top-k experts in the following layer.
            layer_input = actual
            expected = reference(layer_input, ids).clone()
            _sync_ranks_for_mega(x.device)
            actual = layer(layer_input, ids).clone()
            torch.cuda.synchronize(x.device)
            dist.barrier()
            _assert_outputs_do_not_alias(expected, actual)
            assert actual.shape == expected.shape
            assert torch.isfinite(actual).all(), (label, index)
            if actual.numel():
                relative = (
                    actual.float() - expected.float()
                ).abs().mean() / expected.float().abs().mean().clamp_min(1e-6)
                assert relative < 0.05, (label, index, relative.item())
                print(
                    f"{label}/layer={index}: relative MAE={relative.item():.6f}",
                    flush=True,
                )
        return actual
