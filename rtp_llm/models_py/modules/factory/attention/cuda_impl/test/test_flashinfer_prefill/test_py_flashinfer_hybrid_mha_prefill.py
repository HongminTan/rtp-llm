import math
import unittest
from typing import List

import torch
from flashinfer.prefill import single_prefill_with_kv_cache

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferHybridPrefillAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance


class TestPyFlashinferHybridPrefillAttnOp(BaseAttentionTest):

    def _create_chunked_prefill_attention_inputs(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        input_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.is_cuda_graph = False
        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        cu_seqlens = [0]
        for input_len in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + input_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=torch.float16))
        return attn_inputs

    def _create_prefix_kv_cache(
        self,
        k_prefix: List[torch.Tensor],
        v_prefix: List[torch.Tensor],
        prefix_lengths: List[int],
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        block_table: torch.Tensor,
    ) -> LayerKVCache:
        total_pages = sum(
            math.ceil(seq_len / page_size) for seq_len in sequence_lengths
        )
        paged_kv_cache = torch.zeros(
            total_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=k_prefix[0].dtype,
            device=self.device,
        )
        for batch_idx, prefix_len in enumerate(prefix_lengths):
            num_pages = math.ceil(prefix_len / page_size)
            for page_offset in range(num_pages):
                page_id = int(block_table[batch_idx, page_offset].item())
                start = page_offset * page_size
                end = min(start + page_size, prefix_len)
                paged_kv_cache[page_id, 0, :, : end - start, :] = k_prefix[batch_idx][
                    start:end
                ].transpose(0, 1)
                paged_kv_cache[page_id, 1, :, : end - start, :] = v_prefix[batch_idx][
                    start:end
                ].transpose(0, 1)
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = paged_kv_cache
        return kv_cache

    def _reference_chunked_prefill(
        self,
        q_new: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        prefix_len: int,
        input_len: int,
    ) -> torch.Tensor:
        q_full = torch.zeros(
            prefix_len + input_len,
            q_new.shape[1],
            q_new.shape[2],
            dtype=q_new.dtype,
            device=q_new.device,
        )
        q_full[prefix_len:] = q_new
        return single_prefill_with_kv_cache(
            q_full, k_full, v_full, causal=True, kv_layout="NHD"
        )[prefix_len:]

    def _test_hybrid_prefill_correctness(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        page_size: int,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
    ):
        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )
        config.attn_configs.kv_cache_dtype = kv_cache_dtype

        is_fp8 = kv_cache_dtype == KvCacheDataType.FP8
        compute_dtype = torch.float16

        attn_inputs = self._create_chunked_prefill_attention_inputs(
            batch_size, prefix_lengths, input_lengths, page_size
        )
        attn_op = PyFlashinferHybridPrefillAttnOp(config.attn_configs, attn_inputs)
        assert attn_op.support(attn_inputs)
        attn_op.prepare(attn_inputs)

        q_chunks = []
        k_new_chunks = []
        v_new_chunks = []
        k_prefix_chunks = []
        v_prefix_chunks = []
        ref_chunks = []
        for prefix_len, input_len in zip(prefix_lengths, input_lengths):
            if is_fp8:
                q_new = (
                    torch.rand(
                        input_len,
                        head_num,
                        size_per_head,
                        dtype=compute_dtype,
                        device=self.device,
                    )
                    * 2
                    - 1
                )
                k_prefix = (
                    torch.rand(
                        prefix_len,
                        head_num_kv,
                        size_per_head,
                        dtype=compute_dtype,
                        device=self.device,
                    )
                    * 2
                    - 1
                )
                v_prefix = (
                    torch.rand(
                        prefix_len,
                        head_num_kv,
                        size_per_head,
                        dtype=compute_dtype,
                        device=self.device,
                    )
                    * 2
                    - 1
                )
                k_new = (
                    torch.rand(
                        input_len,
                        head_num_kv,
                        size_per_head,
                        dtype=compute_dtype,
                        device=self.device,
                    )
                    * 2
                    - 1
                )
                v_new = (
                    torch.rand(
                        input_len,
                        head_num_kv,
                        size_per_head,
                        dtype=compute_dtype,
                        device=self.device,
                    )
                    * 2
                    - 1
                )
            else:
                q_new = torch.randn(
                    input_len,
                    head_num,
                    size_per_head,
                    dtype=compute_dtype,
                    device=self.device,
                )
                k_prefix = torch.randn(
                    prefix_len,
                    head_num_kv,
                    size_per_head,
                    dtype=compute_dtype,
                    device=self.device,
                )
                v_prefix = torch.randn_like(k_prefix)
                k_new = torch.randn(
                    input_len,
                    head_num_kv,
                    size_per_head,
                    dtype=compute_dtype,
                    device=self.device,
                )
                v_new = torch.randn_like(k_new)

            k_full = torch.cat([k_prefix, k_new], dim=0)
            v_full = torch.cat([v_prefix, v_new], dim=0)

            ref_chunks.append(
                self._reference_chunked_prefill(
                    q_new,
                    k_full,
                    v_full,
                    prefix_len,
                    input_len,
                )
            )
            q_chunks.append(q_new)
            k_prefix_chunks.append(k_prefix)
            v_prefix_chunks.append(v_prefix)
            k_new_chunks.append(k_new)
            v_new_chunks.append(v_new)

        q = torch.cat(q_chunks, dim=0)
        k_new = torch.cat(k_new_chunks, dim=0)
        v_new = torch.cat(v_new_chunks, dim=0)
        ref_output = torch.cat(ref_chunks, dim=0)

        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        kv_cache = self._create_prefix_kv_cache(
            k_prefix_chunks,
            v_prefix_chunks,
            prefix_lengths,
            sequence_lengths,
            page_size,
            head_num_kv,
            size_per_head,
            attn_inputs.kv_cache_kernel_block_id_host,
        )
        if is_fp8:
            kv_cache.kv_cache_base = kv_cache.kv_cache_base.to(torch.float8_e4m3fn)

        output = attn_op.forward(q, k_new, v_new, kv_cache)
        if is_fp8:
            assert_close_with_mismatch_tolerance(
                output.float(),
                ref_output.float(),
                atol=0.04,
                rtol=0.04,
                max_mismatched_elements=int(1e-5 * ref_output.numel()),
            )
        else:
            compare_tensors(
                output,
                ref_output,
                rtol=1e-2,
                atol=1e-2,
                name=f"Hybrid prefill output ({kv_cache_dtype.name})",
            )

    def test_chunked_prefill_single_batch(self):
        self._test_hybrid_prefill_correctness(
            batch_size=1,
            prefix_lengths=[4884],
            input_lengths=[5],
            head_num=40,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_multi_batch_varied(self):
        self._test_hybrid_prefill_correctness(
            batch_size=3,
            prefix_lengths=[32, 96, 160],
            input_lengths=[8, 16, 24],
            head_num=16,
            head_num_kv=4,
            size_per_head=64,
            page_size=16,
        )

    def test_chunked_prefill_multi_batch_uniform(self):
        self._test_hybrid_prefill_correctness(
            batch_size=4,
            prefix_lengths=[64, 64, 64, 64],
            input_lengths=[16, 16, 16, 16],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_small_page_size(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=32,
        )

    def test_chunked_prefill_large_page_size(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=128,
        )

    def test_chunked_prefill_many_heads(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[64, 128],
            input_lengths=[16, 32],
            head_num=64,
            head_num_kv=16,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_gqa(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[64, 128],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )


class TestPyFlashinferHybridPrefillAttnOpFP8(TestPyFlashinferHybridPrefillAttnOp):

    def _test_hybrid_prefill_correctness(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        page_size: int,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.FP8,
    ):
        super()._test_hybrid_prefill_correctness(
            batch_size=batch_size,
            prefix_lengths=prefix_lengths,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            page_size=page_size,
            kv_cache_dtype=kv_cache_dtype,
        )


if __name__ == "__main__":
    unittest.main()
