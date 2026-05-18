"""Tests for TRTLLMFMHAv2PagedPrefillOp (non-padded mode)

Tests paged prefill with prefix cache via trtllm_fmha_v2_prefill Q_PAGED_KV_HND layout.
This mode is used when there's existing KV cache (prefix/prompt caching).
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    TRTLLMFMHAv2PagedPrefillOp,
)


class TestTRTLLMFMHAv2PagedPrefillOp(TRTAttnTestBase):
    """Test suite for TRTLLMFMHAv2PagedPrefillOp in non-padded mode

    TRTLLMFMHAv2PagedPrefillOp:
    - Paged prefill with prefix cache (prompt caching)
    - Uses Q_PAGED_KV_HND layout via FlashInfer trtllm_fmha_v2_prefill
    - prefix_lengths must be > 0 (already cached KV)
    - Processes new input_lengths tokens with existing prefix_lengths cache
    - Total KV length = prefix_lengths + input_lengths
    """

    def setUp(self):
        super().setUp()
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:
            self.skipTest(f"Requires SM90+, got SM{cap[0]}{cap[1]}")

    def test_basic(self):
        """Test basic paged prefill with single sequence and prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: Basic ===", flush=True)

        batch_size = 1
        input_lengths = [128]
        prefix_lengths = [64]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_batch(self):
        """Test paged prefill with multiple sequences and different prefix lengths"""
        print("\n=== Test FMHAv2PagedPrefill: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [32, 64, 128, 256]
        prefix_lengths = [32, 64, 128, 256]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_gqa(self):
        """Test paged prefill with grouped query attention and prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: GQA ===", flush=True)

        batch_size = 2
        input_lengths = [128, 256]
        prefix_lengths = [128, 256]
        head_num = 32
        head_num_kv = 4
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_long_sequence(self):
        """Test paged prefill with long sequences and long prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [512, 1024]
        prefix_lengths = [512, 1024]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )


if __name__ == "__main__":
    unittest.main()
