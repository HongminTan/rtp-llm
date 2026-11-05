# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# E2M1 bit decoding adapted from vLLM's nvfp4_emulation_utils.py (5df63a7).
"""E2M1/group-32 weights to RTP's E4M3/128x128 block representation."""

from functools import cache

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_e2m1(code):
    # E2M1's normal magnitudes map directly to FP32 exponent/mantissa bits.
    magnitude = code & 7
    bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
    value = bits.to(tl.float32, bitcast=True)
    value = tl.where(magnitude == 0, 0.0, value)
    value = tl.where(magnitude == 1, 0.5, value)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _decode_e8m0(code):
    bits = code.to(tl.uint32) << 23
    # E8M0 code zero is 2^-127, not zero; 255 represents NaN.
    value = tl.where(code == 0, 0x00400000, bits).to(tl.float32, bitcast=True)
    return tl.where(code == 255, float("nan"), value)


@triton.jit
def _fp4_to_fp8_scales_kernel(W, S4, S8, N: tl.constexpr, K: tl.constexpr):
    expert = tl.program_id(0).to(tl.int64)
    bn = tl.program_id(1)
    bk = tl.program_id(2)
    n = bn * 128 + tl.arange(0, 128)
    k = bk * 128 + tl.arange(0, 128)
    packed = tl.load(
        W + expert * N * (K // 2) + n[:, None] * (K // 2) + k[None, :] // 2
    ).to(tl.uint8)
    code = (packed >> ((k[None, :] % 2) * 4)) & 15
    scale = tl.load(
        S4 + expert * N * (K // 32) + n[:, None] * (K // 32) + k[None, :] // 32
    )
    value = _decode_e2m1(code) * _decode_e8m0(scale)
    amax = tl.max(tl.max(tl.abs(value), 1), 0)
    # Match PyTorch's scalar division (multiplication by FP32 reciprocal).
    s8 = tl.maximum(amax * (1.0 / 448.0), 1.0e-12)
    tl.store(S8 + expert * (N // 128) * (K // 128) + bn * (K // 128) + bk, s8)


@triton.jit
def _mag_byte(q1, q15, M3: tl.constexpr):
    # E2M1 magnitudes are {1, 1.5} times powers of two. Scale two rounded
    # quotients to build the FP8 table without eight separate divisions.
    if M3 == 0:
        return tl.zeros_like(q1).to(tl.uint8)
    if M3 == 1:
        v = q1 * 0.5
    elif M3 == 3:
        v = q15
    elif M3 == 5:
        v = q15 * 2.0
    elif M3 == 7:
        v = q15 * 4.0
    elif M3 == 2:
        v = q1
    elif M3 == 4:
        v = q1 * 2.0
    else:
        v = q1 * 4.0
    v = tl.minimum(v, 448.0)
    return v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _prmt(lo32, hi32, sel):
    # One-byte gather from an 8-byte register table per element.
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [lo32, hi32, sel],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _convert_tile(
    W, S4, S8, OUT, expert, tile, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    expert = expert.to(tl.int64)
    groups = tile.to(tl.int64) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    mask = groups < N * K // 32
    n, k_group = groups // (K // 32), groups % (K // 32)
    s4 = tl.load(S4 + expert * N * (K // 32) + groups, mask, other=127)
    s8 = tl.load(
        S8 + expert * (N // 128) * (K // 128) + (n // 128) * (K // 128) + k_group // 4,
        mask,
        other=1,
    )
    e8 = _decode_e8m0(s4)
    q1 = tl.div_rn(e8, s8)
    q15 = tl.div_rn(e8 * 1.5, s8)
    # Eight-entry FP8 byte LUT per group, packed into two u32 registers.
    b0 = _mag_byte(q1, q15, 0).to(tl.int32)
    b1 = _mag_byte(q1, q15, 1).to(tl.int32)
    b2 = _mag_byte(q1, q15, 2).to(tl.int32)
    b3 = _mag_byte(q1, q15, 3).to(tl.int32)
    b4 = _mag_byte(q1, q15, 4).to(tl.int32)
    b5 = _mag_byte(q1, q15, 5).to(tl.int32)
    b6 = _mag_byte(q1, q15, 6).to(tl.int32)
    b7 = _mag_byte(q1, q15, 7).to(tl.int32)
    lo32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    hi32 = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)
    lo32b = tl.broadcast_to(lo32[:, None], (BLOCK // 32, 16))
    hi32b = tl.broadcast_to(hi32[:, None], (BLOCK // 32, 16))
    offsets = groups[:, None] * 16 + tl.arange(0, 16)[None, :]
    packed = tl.load(W + expert * N * K // 2 + offsets, mask[:, None], other=0).to(
        tl.uint8
    )
    lo = packed & 15
    hi = packed >> 4
    q_lo = _prmt(lo32b, hi32b, (lo & 7).to(tl.int32)) | ((lo & 8) << 4).to(tl.int32)
    q_hi = _prmt(lo32b, hi32b, (hi & 7).to(tl.int32)) | ((hi & 8) << 4).to(tl.int32)
    # Byte pairs are packed into u16 stores; no cross-lane data movement.
    q16 = (q_lo | (q_hi << 8)).to(tl.uint16)
    out16 = OUT.to(tl.pointer_type(tl.uint16), bitcast=True)
    out_offsets = groups[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(out16 + expert * (N * K // 2) + out_offsets, q16, mask[:, None])


@triton.jit
def _fp4_to_fp8_kernel(
    W,
    S4,
    S8,
    OUT,
    COUNTS,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Rebuild the active-expert schedule on device for each graph replay.
    tiles: tl.constexpr = N * K // BLOCK
    ids = tl.arange(0, triton.next_power_of_2(E))
    if HAS_COUNTS:
        active = tl.load(COUNTS + ids, ids < E, other=0) > 0
    else:
        active = ids < E
    ends = tl.cumsum(active.to(tl.int32))
    total = tl.sum(active.to(tl.int32)) * tiles
    for work in range(tl.program_id(0), total, tl.num_programs(0)):
        wi = work // tiles
        tile = work % tiles
        expert = tl.sum(((ends <= wi) & (ids < E)).to(tl.int32))
        _convert_tile(W, S4, S8, OUT, expert, tile, N, K, BLOCK)


def _validate_source(weight, scale):
    if weight.ndim != 3 or weight.dtype != torch.int8 or not weight.is_cuda:
        raise ValueError("FP4 weights must be CUDA int8 [experts, N, K/2]")
    e, n, packed_k = weight.shape
    k = packed_k * 2
    if e <= 0 or n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("FP4 conversion requires positive E and N/K multiples of 128")
    if (
        scale.dtype != torch.float8_e8m0fnu
        or scale.shape != (e, n, k // 32)
        or scale.device != weight.device
    ):
        raise ValueError(
            "FP4 scales must be E8M0 [experts, N, K/32] on the weight device"
        )
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("FP4 conversion requires contiguous weights and scales")
    return e, n, k


def compute_fp4_to_fp8_scales(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    e, n, k = _validate_source(weight, scale)
    result = torch.empty(
        (e, n // 128, k // 128), dtype=torch.float32, device=weight.device
    )
    _fp4_to_fp8_scales_kernel[(e, n // 128, k // 128)](
        weight,
        scale.view(torch.uint8),
        result,
        n,
        k,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result


_BLOCK_SIZE = 8192
_CTAS_PER_SM = 16


@cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def convert_fp4_to_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    scale_fp8: torch.Tensor,
    out: torch.Tensor,
    expert_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    k = packed_k * 2
    grid = (min(e * (n * k // _BLOCK_SIZE), _num_sms(weight.device) * _CTAS_PER_SM),)
    _fp4_to_fp8_kernel[grid](
        weight,
        scale.view(torch.uint8),
        scale_fp8,
        out,
        expert_counts,
        e,
        n,
        k,
        expert_counts is not None,
        _BLOCK_SIZE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
