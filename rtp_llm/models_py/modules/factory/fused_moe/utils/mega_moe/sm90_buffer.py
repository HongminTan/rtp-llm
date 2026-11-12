"""SM90 MegaMoE capture-stable staging buffers."""

import torch

_BUFFER_CACHE: dict = {}
_OUTPUT_CACHE: dict = {}
_SE_OUTPUT_CACHE: dict = {}


def get_sm90_buffer(config, device, *, shared: bool):
    import deep_gemm
    import torch.distributed as dist

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
        _get_validated_world_ep_group,
    )

    group = _get_validated_world_ep_group(config, dist)
    key = (
        group,
        device,
        config.n_routed_experts,
        config.max_tokens_per_rank,
        config.n_activated_experts,
        config.dim,
        config.moe_inter_dim,
        shared,
    )
    if key not in _BUFFER_CACHE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SM90 MegaMoE buffers must be allocated before capture")
        with torch.cuda.device(device):
            _BUFFER_CACHE[key] = deep_gemm.get_symm_buffer_for_mega_moe_sm90(
                group,
                config.n_routed_experts,
                config.max_tokens_per_rank,
                config.n_activated_experts,
                config.dim,
                config.moe_inter_dim,
                activation="swiglu",
                num_shared_experts=int(shared),
            )
    return _BUFFER_CACHE[key]


def get_sm90_output(capacity, hidden, dtype, device, *, shared: bool):
    """Reuse local output storage across sequential layers of the same path."""
    cache = _SE_OUTPUT_CACHE if shared else _OUTPUT_CACHE
    key = (device, hidden, dtype)
    cached = cache.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM90 MegaMoE output must be allocated before capture")
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    cache[key] = cached
    return cached
