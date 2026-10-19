"""M-R1 fused static-EP packing parity gate (R473.2 / r258 指令一①(b)(d)).

Compares ``ep_static_pack_send`` (three Triton kernels) against the aten
chain it replaces in ``grouped_fp4_sm90.py`` ``_forward_ep_static`` on
identical inputs: bitwise send_buf equality (payload, eid sentinel column,
eid/assign metadata columns, bf16 casts included) plus a launch-count
fingerprint -- the fused path must be exactly three kernels.

Grid per r258 (d): N in {1, 8, 128, 4096} with invalid-id mixing.
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_static_pack import ep_static_pack_send

WORLD, E_LOCAL, K = 4, 16, 6


def _make_inputs(num_tokens: int, invalid_frac: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, WORLD * E_LOCAL, (num_tokens, K), generator=gen).cuda()
    bad = (torch.rand(num_tokens, K, generator=gen) < invalid_frac).cuda()
    ids = torch.where(bad, torch.tensor(-1, device="cuda"), ids)
    ids = torch.where(
        bad & (torch.rand(num_tokens, K, generator=gen).cuda() < 0.5),
        torch.tensor(WORLD * E_LOCAL, device="cuda"),
        ids,
    )
    x = torch.randn(num_tokens, 128, generator=gen).to(torch.bfloat16).cuda()
    return ids, x


def _aten_chain(ids: torch.Tensor, x: torch.Tensor):
    """Source form of grouped_fp4_sm90.py _forward_ep_static :1041-1113."""
    N, D = x.shape
    flat_ids = ids.reshape(-1).to(torch.int64)
    valid = (flat_ids >= 0) & (flat_ids < WORLD * E_LOCAL)
    flat_ids_v = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
    dest = flat_ids_v // E_LOCAL
    order = torch.argsort(dest, stable=True)
    counts = torch.zeros(WORLD, dtype=torch.int64, device=x.device).scatter_add_(
        0, dest, torch.ones_like(dest)
    )
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(N * K, device=x.device, dtype=torch.int64)
    dest_sorted = flat_ids_v.index_select(0, order) // E_LOCAL
    within = pos - starts.index_select(0, dest_sorted)
    x_rows = x.repeat_interleave(K, dim=0).index_select(0, order).to(torch.bfloat16)
    cap = N * K
    send_buf = torch.zeros(WORLD, cap, D + 2, dtype=torch.bfloat16, device=x.device)
    send_buf[..., D] = -1.0
    send_buf[dest_sorted, within, :D] = x_rows
    send_buf[dest_sorted, within, D] = flat_ids_v.index_select(0, order).to(
        torch.bfloat16
    )
    send_buf[dest_sorted, within, D + 1] = order.to(torch.bfloat16)
    return send_buf


class EPStaticPackParityTest(unittest.TestCase):
    def _check_case(self, num_tokens, invalid_frac, seed):
        ids, x = _make_inputs(num_tokens, invalid_frac, seed)
        send_ref = _aten_chain(ids, x)
        # Drive the engine call form exactly (mr1-trace-01 lesson): the
        # executor passes its already-flattened 1-D flat_ids into the
        # wrapper, so the parity gate must exercise that shape.
        send_fused, _within = ep_static_pack_send(
            ids.reshape(-1), x, WORLD, E_LOCAL, num_tokens * K
        )
        self.assertTrue(
            torch.equal(send_ref, send_fused),
            "send_buf mismatch: "
            f"max|d|={(send_fused.float() - send_ref.float()).abs().max().item():.3e}",
        )

    def test_parity_grid(self):
        for num_tokens in (1, 8, 128, 4096):
            self._check_case(num_tokens, 0.15, 1000 + num_tokens)

    def test_parity_all_invalid(self):
        # Degenerate: every id invalid -> all rows ship to rank 0 with
        # eid 0 (the zero-weight padding semantics the combine relies on).
        self._check_case(8, 1.0, 2001)

    def test_launch_count_fingerprint(self):
        from torch.profiler import ProfilerActivity, profile

        ids, x = _make_inputs(8, 0.2, 3001)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ep_static_pack_send(ids.reshape(-1), x, WORLD, E_LOCAL, 8 * K)
        kernel_names = [e.name for e in prof.events() if "CUDA" in str(e.device_type)]
        fused_kernels = [
            n
            for n in kernel_names
            if n.startswith(("_rank_scan", "_send_fill", "_send_pack"))
        ]
        print(f"[fingerprint] fused-path kernels: {sorted(set(fused_kernels))}")
        self.assertEqual(
            len(fused_kernels),
            3,
            f"fused path must be exactly 3 kernel launches, got "
            f"{len(fused_kernels)} of {len(kernel_names)} CUDA events",
        )


if __name__ == "__main__":
    unittest.main()
