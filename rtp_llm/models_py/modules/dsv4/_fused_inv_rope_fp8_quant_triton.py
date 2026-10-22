"""Fused inverse RoPE and block-128 FP8 quantization for DSv4 wo_a.

Outputs grouped or flattened FP8 data with FP32 or packed UE8M0 scales.
Both formats preserve power-of-two scale rounding.
Rotations are supplied as per-request or per-token complex64 values.

Adapted from vLLM's deepseek_v4_ops/fused_inv_rope_fp8_quant.py.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)


def _is_blackwell_device(device: torch.device | int | None = None) -> bool:
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 10


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k"])
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,  # [M, H, D] bf16
    cos_ptr,  # [B, RD_HALF] fp32 (from freqs_cis.real)
    sin_ptr,  # [B, RD_HALF] fp32 (from freqs_cis.imag)
    fp8_ptr,  # [G, M_pad, d] fp8 e4m3 (d = heads_per_group * D)
    scale_ptr,  # TMA-aligned MN-major scales
    num_tokens,
    q_len_per_b: tl.constexpr,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cos_stride_b,
    sin_stride_b,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,  # for TMA-aligned layout this is ``tma_aligned_M``
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,  # 128
    CHUNKS_PER_HEAD: tl.constexpr,  # head_dim // 128 (= 4 for dsv4 head_dim=512)
    ROPE_START: tl.constexpr,  # nope_dim % QUANT_GROUP_SIZE
    SCALE_UE8M0: tl.constexpr,
    HALF_ROPE: tl.constexpr,  # rope_head_dim // 2
):
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh  # flat head index in original o layout [M, H, D]

    # Padding rows in the TMA-aligned scale buffer: fill with zero, skip quant.
    if pid_token >= num_tokens:
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token  # M offset (innermost, stride=1)
            + head_in_group * scale_stride_k
        )
        if SCALE_UE8M0:
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    # --- inverse RoPE on last RD columns ---------------------------------
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    # Partner load: interleaved (real, imag, real, imag, …) — XOR last bit
    # swaps real↔imag.  Masked to zero outside rope region so non-rope
    # positions pass through unchanged via the ``tl.where`` below.
    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    # cs_idx = floor(rope_local / 2).  ``tl.maximum(.., 0)`` protects
    # non-rope lanes (they'd get negative indices and load OOB even
    # behind a mask, which Triton dislikes).
    cs_idx = tl.maximum(rope_local >> 1, 0)
    b_idx = pid_token // q_len_per_b
    cos_v = tl.load(cos_ptr + b_idx * cos_stride_b + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(sin_ptr + b_idx * sin_stride_b + cs_idx, mask=is_rope, other=0.0)
    # Inverse RoPE == multiply by conj(freqs_cis):
    #   (a + bi) · conj(c + di) = (ac + bd) + (bc − ad) i
    # For our interleaved layout, even offsets hold real (a), odd hold
    # imag (b).  At an even lane we output (a·cos + partner·sin); at an
    # odd lane (b·cos − partner·sin).  Matches ``apply_rotary_emb_batched(…,
    # inverse=True)`` in rope.py.
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    # --- per-128-column FP8 quant ----------------------------------------
    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))  # UE8M0-round

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + head_in_group * HEAD_DIM
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    if SCALE_UE8M0:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + (head_in_group * CHUNKS_PER_HEAD + block_offsets) * scale_stride_k
        )
        tl.store(scale_addr, scales)


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k"])
def _fused_inv_rope_fp8_quant_group_heads(
    o_ptr,  # [M, H, D] bf16
    freqs_ri_ptr,  # torch.view_as_real(freqs), float32 [B, RD_HALF, 2]
    fp8_ptr,  # [G, M_pad, d] fp8 e4m3
    scale_ptr,  # TMA-aligned MN-major scales
    num_tokens,
    q_len_per_b: tl.constexpr,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    freqs_stride_b,
    freqs_stride_k,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1).to(tl.int64)
    head_tile = tl.program_id(2).to(tl.int64)
    head_base = head_tile * HEADS_PER_CTA

    for head_step in tl.static_range(HEADS_PER_CTA):
        head_in_group = head_base + head_step
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        if pid_token >= num_tokens:
            if SCALE_UE8M0:
                tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            global_head = g * heads_per_group + head_in_group
            input_base = (
                o_ptr + pid_token * o_stride_token + global_head * o_stride_head
            )

            HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
            offsets = tl.arange(0, HEAD_DIM)
            x = tl.load(input_base + offsets).to(tl.float32)

            rope_abs_start: tl.constexpr = (
                CHUNKS_PER_HEAD - 1
            ) * QUANT_GROUP_SIZE + ROPE_START
            is_rope = offsets >= rope_abs_start
            rope_local = offsets - rope_abs_start
            x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
                tl.float32
            )

            cs_idx = tl.maximum(rope_local >> 1, 0)
            b_idx = pid_token // q_len_per_b
            freq_base = freqs_ri_ptr + b_idx * freqs_stride_b + cs_idx * freqs_stride_k
            cos_v = tl.load(freq_base, mask=is_rope, other=1.0)
            sin_v = tl.load(freq_base + 1, mask=is_rope, other=0.0)

            x_add = x * cos_v + x_partner * sin_v
            x_sub = x * cos_v - x_partner * sin_v
            is_even = (rope_local & 1) == 0
            x = tl.where(is_rope, tl.where(is_even, x_add, x_sub), x)

            x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
            block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
            scale_raw = block_absmax * (1.0 / fp8_max)
            scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
            exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            scale_bits = exp << 23
            scales = scale_bits.to(tl.float32, bitcast=True)

            scales_exp = tl.reshape(
                tl.broadcast_to(
                    tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
                    (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
                ),
                (HEAD_DIM,),
            )
            x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

            fp8_base = (
                fp8_ptr
                + g * fp8_stride_group
                + pid_token * fp8_stride_token
                + head_in_group * HEAD_DIM
            )
            tl.store(fp8_base + offsets, x_quant)

            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            if not SCALE_UE8M0:
                chunk_addr = (
                    scale_ptr
                    + g * scale_stride_group
                    + pid_token
                    + (head_in_group * CHUNKS_PER_HEAD + block_offsets) * scale_stride_k
                )
                tl.store(chunk_addr, scales)
            else:
                packed_val = tl.sum(exp << (block_offsets * 8))
                tl.store(scale_addr, packed_val)


def _alloc_output_buffers(
    M: int,
    n_groups: int,
    d_per_group: int,
    scale_ue8m0: bool,
    flatten_groups: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_work = torch.empty(
        (n_groups, M, d_per_group), dtype=torch.float8_e4m3fn, device=device
    )
    shape = (
        (n_groups * M, d_per_group) if flatten_groups else (n_groups, M, d_per_group)
    )
    scale_work = create_per_token_group_quant_fp8_output_scale(
        shape,
        device,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if flatten_groups:
        scale_work = scale_work.view(n_groups, M, -1)
    return fp8_work, scale_work


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,  # [B, S, H, head_dim] or [M, H, head_dim] bf16
    freqs_cis_per_b: torch.Tensor,  # [B or M, rope_head_dim // 2] complex64
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    impl: str | None = None,
    heads_per_cta: int | None = None,
    scale_ue8m0: bool = True,
    flatten_groups: bool = False,
):
    """Fused inverse-RoPE + block-scaled FP8 quant for wo_a input.

    Args:
        o: Attention output ``[B, S, H, head_dim]`` or flattened
           ``[M, H, head_dim]`` bf16.  ``H == n_groups * heads_per_group``.
           ``head_dim == nope_dim + rope_head_dim`` and must be a multiple
           of ``quant_group_size``.
        freqs_cis_per_b: Per-request ``[B, rope_head_dim // 2]`` or
           per-token ``[M, rope_head_dim // 2]`` complex64 rotations.
        n_groups, heads_per_group: wo_a grouping.
        nope_dim, rope_head_dim: per-head NoPE / RoPE split.
        quant_group_size: FP8 quant block size along K (fixed at 128 for V4).
        eps: Numerical epsilon used inside absmax-scale computation.
        heads_per_cta: Optimized Triton path head grouping.  Valid values are
           1, 2, 4, and 8; ``None`` reads ``DSV4_INV_ROPE_HEADS_PER_CTA``.
           ``DSV4_INV_ROPE_NUM_WARPS`` is an experimental tuning override for
           the optimized Triton launch; default is 2.
        scale_ue8m0: Pack four power-of-two scales into each int32; otherwise
           store the same scale values as FP32. Both use TMA-aligned MN-major storage.
        flatten_groups: Return [G*M, K] instead of [M, G, K].

    Returns:
        FP8 activations and scales with matching leading dimensions. Scale K
        is K/512 for packed UE8M0 and K/128 for FP32. Flattened row g*M+m
        represents token m of group g.
    """
    assert o.is_cuda and o.dtype == torch.bfloat16
    assert freqs_cis_per_b.is_cuda and freqs_cis_per_b.dtype == torch.complex64
    assert o.is_contiguous()
    assert freqs_cis_per_b.is_contiguous()

    # Normalize input to [M, H, D]
    if o.dim() == 4:
        B, S, H, D = o.shape
        M = B * S
        o_flat = o.view(M, H, D)
        freq_rows = int(freqs_cis_per_b.shape[0])
        if freq_rows == M:
            q_len_per_b = 1
        elif freq_rows == B:
            q_len_per_b = S
        else:
            raise AssertionError(f"freq rows must be B={B} or M={M}; got {freq_rows}")
    else:
        assert o.dim() == 3
        M, H, D = o.shape
        B = freqs_cis_per_b.shape[0]
        assert M % B == 0, f"M={M} not divisible by B={B}"
        q_len_per_b = M // B
        o_flat = o

    assert H == n_groups * heads_per_group
    assert D == nope_dim + rope_head_dim
    assert D % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_head_dim), (
        f"RoPE must end the last {quant_group_size}-column quant block: "
        f"nope_dim={nope_dim}, rope_head_dim={rope_head_dim}, "
        f"quant_group_size={quant_group_size}"
    )
    assert rope_head_dim % 2 == 0
    assert freqs_cis_per_b.shape[-1] == rope_head_dim // 2

    chunks_per_head = D // quant_group_size
    assert quant_group_size == 128
    if scale_ue8m0:
        assert chunks_per_head == 4, "packed scales require four blocks per head"

    d_per_group = heads_per_group * D
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_work, scale_work = _alloc_output_buffers(
        M,
        n_groups,
        d_per_group,
        scale_ue8m0,
        flatten_groups,
        o.device,
    )
    tma_M = M if flatten_groups else ((M + 3) // 4) * 4

    selected_impl = (
        impl
        if impl is not None
        else os.environ.get("DSV4_INV_ROPE_FP8_QUANT_IMPL", "optimized")
    ).lower()
    if selected_impl == "legacy":
        # complex64 .real / .imag are stride-2 views; .contiguous() is a real
        # layout copy. Keep this path only as a correctness/perf baseline.
        cos = freqs_cis_per_b.real.contiguous()
        sin = freqs_cis_per_b.imag.contiguous()
        grid = (tma_M, n_groups * heads_per_group)
        _fused_inv_rope_fp8_quant_per_head[grid](
            o_flat,
            cos,
            sin,
            fp8_work,
            scale_work,
            M,
            q_len_per_b=q_len_per_b,
            heads_per_group=heads_per_group,
            o_stride_token=o_flat.stride(0),
            o_stride_head=o_flat.stride(1),
            cos_stride_b=cos.stride(0),
            sin_stride_b=sin.stride(0),
            fp8_stride_group=fp8_work.stride(0),
            fp8_stride_token=fp8_work.stride(1),
            scale_stride_group=scale_work.stride(0),
            scale_stride_k=scale_work.stride(2),
            fp8_max=fp8_max,
            eps=eps,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=nope_dim % quant_group_size,
            SCALE_UE8M0=scale_ue8m0,
            HALF_ROPE=rope_head_dim // 2,
            num_warps=1,
            num_stages=1,
        )
    elif selected_impl == "optimized":
        env_heads_per_cta = os.environ.get("DSV4_INV_ROPE_HEADS_PER_CTA")
        if heads_per_cta is not None:
            selected_heads_per_cta = heads_per_cta
        elif env_heads_per_cta is not None:
            selected_heads_per_cta = int(env_heads_per_cta)
        else:
            selected_heads_per_cta = 8 if _is_blackwell_device(o.device) else 2
        if selected_heads_per_cta not in (1, 2, 4, 8):
            raise ValueError(
                f"invalid DSV4_INV_ROPE_HEADS_PER_CTA={selected_heads_per_cta}; "
                "expected 1, 2, 4, or 8"
            )
        if heads_per_group % selected_heads_per_cta != 0:
            selected_heads_per_cta = 1
        selected_num_warps = int(os.environ.get("DSV4_INV_ROPE_NUM_WARPS", "2"))
        if selected_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                f"invalid DSV4_INV_ROPE_NUM_WARPS={selected_num_warps}; "
                "expected 1, 2, 4, or 8"
            )
        freqs_ri = torch.view_as_real(freqs_cis_per_b)
        grid = (tma_M, n_groups, heads_per_group // selected_heads_per_cta)
        _fused_inv_rope_fp8_quant_group_heads[grid](
            o_flat,
            freqs_ri,
            fp8_work,
            scale_work,
            M,
            q_len_per_b=q_len_per_b,
            heads_per_group=heads_per_group,
            o_stride_token=o_flat.stride(0),
            o_stride_head=o_flat.stride(1),
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            fp8_stride_group=fp8_work.stride(0),
            fp8_stride_token=fp8_work.stride(1),
            scale_stride_group=scale_work.stride(0),
            scale_stride_k=scale_work.stride(2),
            fp8_max=fp8_max,
            eps=eps,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=nope_dim % quant_group_size,
            HEADS_PER_CTA=selected_heads_per_cta,
            SCALE_UE8M0=scale_ue8m0,
            num_warps=selected_num_warps,
            num_stages=1,
        )
    else:
        raise ValueError(
            f"invalid DSV4_INV_ROPE_FP8_QUANT_IMPL={selected_impl!r}; "
            "expected legacy|optimized"
        )
    if flatten_groups:
        return (
            fp8_work.view(n_groups * M, d_per_group),
            scale_work.view(n_groups * M, -1),
        )
    return fp8_work.transpose(0, 1), scale_work.transpose(0, 1)


def fused_inv_rope_fp8_quant_legacy(*args, **kwargs):
    kwargs["impl"] = "legacy"
    return fused_inv_rope_fp8_quant(*args, **kwargs)


def fused_inv_rope_fp8_quant_optimized(*args, **kwargs):
    kwargs["impl"] = "optimized"
    return fused_inv_rope_fp8_quant(*args, **kwargs)
