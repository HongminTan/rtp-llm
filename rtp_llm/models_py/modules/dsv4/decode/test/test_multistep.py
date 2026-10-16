"""Channel-B increment 1 tests: multi-step metadata chain + greedy head.

Covers, on CPU:

1. ``advance_decode_metadata`` chains: N successive in-place
   ``start_pos + 1`` updates produce field-by-field the same tensors as
   fresh ``build_decode_metadata`` at ``base + i`` — the read-modify-write
   chain semantics the captured graph depends on.
2. Buffer stability across the chain (``data_ptr`` unchanged — the
   ``forbid_realloc`` contract the graph capture bakes in).
3. ``greedy_next_token`` head math on a synthetic [vocab, dim] weight.

No CUDA here (dev-box CPU tier); graph integration lives in the smoke arm.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    allocate_decode_metadata,
    build_decode_metadata,
    update_decode_metadata_in_place,
)
from rtp_llm.models_py.modules.dsv4.decode.multistep import (
    advance_decode_metadata,
    greedy_next_token,
)

_DEVICE = torch.device("cpu")


def _alloc(max_bs: int):
    return allocate_decode_metadata(
        max_batch_size=max_bs,
        q_len=1,
        window_size=8,
        head_dim=32,
        max_seq_len=64,
        compress_ratios=[4, 128],
        index_topk=4,
        device=_DEVICE,
    )


def _build(start_pos: torch.Tensor):
    return build_decode_metadata(
        start_pos=start_pos,
        q_len=1,
        window_size=8,
        head_dim=32,
        max_seq_len=64,
        compress_ratios=[4, 128],
        index_topk=4,
        device=_DEVICE,
    )


def _prefix_snapshot(meta: "object", bs: int) -> dict:
    """Copy the active prefix of every comparable field."""
    snap = {
        "start_pos": meta.start_pos[:bs].clone(),
        "slot_mapping_swa": meta.slot_mapping_swa[:bs].clone(),
        "topk_window_idxs": meta.topk_window_idxs[:bs].clone(),
    }
    for r, t in meta.slot_mapping_compressed.items():
        snap[f"slot_compressed[{r}]"] = t[:bs].clone()
    for r, t in meta.compressed_lens.items():
        snap[f"compressed_lens[{r}]"] = t[:bs].clone()
    for r, t in meta.topk_total_by_ratio.items():
        snap[f"topk_total[{r}]"] = t[:bs].clone()
    return snap


class TestAdvanceChain(unittest.TestCase):

    def test_chain_matches_fresh_build(self):
        base = torch.tensor([3, 7, 11], dtype=torch.int32)
        bs = base.shape[0]
        meta = _alloc(max_bs=8)
        update_decode_metadata_in_place(meta, base)
        chained = {}
        for i in range(1, 4):
            advance_decode_metadata(meta, None, None)
            chained[i] = _prefix_snapshot(meta, bs)
        for i in range(1, 4):
            fresh = _build(base + i)
            ref = {
                "start_pos": fresh.start_pos[:bs],
                "slot_mapping_swa": fresh.slot_mapping_swa[:bs],
                "topk_window_idxs": fresh.topk_window_idxs[:bs],
            }
            for r, t in fresh.slot_mapping_compressed.items():
                ref[f"slot_compressed[{r}]"] = t[:bs]
            for r, t in fresh.compressed_lens.items():
                ref[f"compressed_lens[{r}]"] = t[:bs]
            for r, t in fresh.topk_total_by_ratio.items():
                ref[f"topk_total[{r}]"] = t[:bs]
            for key, got in chained[i].items():
                self.assertTrue(
                    torch.equal(got, ref[key]),
                    f"step {i}: {key} mismatch\n got={got.flatten()[:12].tolist()}"
                    f"\n ref={ref[key].flatten()[:12].tolist()}",
                )

    def test_chain_preserves_buffer_storage(self):
        meta = _alloc(max_bs=8)
        update_decode_metadata_in_place(meta, torch.tensor([1, 2], dtype=torch.int32))
        ptrs = {
            "start_pos": meta.start_pos.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
        }
        for r, t in meta.slot_mapping_compressed.items():
            ptrs[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            ptrs[f"topk_total[{r}]"] = t.data_ptr()
        for _ in range(4):
            advance_decode_metadata(meta, None, None)
        for key, ptr in ptrs.items():
            # Re-derive the live pointer for the same field.
            live = {
                "start_pos": meta.start_pos.data_ptr(),
                "slot_swa": meta.slot_mapping_swa.data_ptr(),
                "topk_window": meta.topk_window_idxs.data_ptr(),
            }
            for r, t in meta.slot_mapping_compressed.items():
                live[f"slot_compressed[{r}]"] = t.data_ptr()
            for r, t in meta.topk_total_by_ratio.items():
                live[f"topk_total[{r}]"] = t.data_ptr()
            self.assertEqual(ptr, live[key], f"storage moved for {key}")


class TestGreedyHead(unittest.TestCase):

    def test_argmax_matches_reference(self):
        torch.manual_seed(0)
        B, dim, vocab = 3, 16, 32
        hidden = torch.randn(B, dim, dtype=torch.bfloat16)
        head = torch.randn(vocab, dim, dtype=torch.bfloat16)
        got = greedy_next_token(hidden, head)
        ref = torch.argmax(torch.mm(hidden.to(head.dtype), head.t()).float(), dim=-1)
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(got.dtype, torch.int64)
        self.assertEqual(got.shape, (B,))


if __name__ == "__main__":
    unittest.main()
