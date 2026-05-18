"""Tests for TRTLLMFMHAv2PrefillOp (non-padded mode)

Tests standard prefill without prefix cache via trtllm_fmha_v2_prefill CONTIGUOUS_Q_KV layout.
This mode is used for dynamic batch processing.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    TRTLLMFMHAv2PrefillOp,
)


class TestTRTLLMFMHAv2PrefillOp(TRTAttnTestBase):
    """Test suite for TRTLLMFMHAv2PrefillOp in non-padded mode

    TRTLLMFMHAv2PrefillOp:
    - Standard prefill without prefix cache
    - Uses CONTIGUOUS_Q_KV layout via FlashInfer trtllm_fmha_v2_prefill
    - prefix_lengths must be 0 or None
    - Only processes new input_lengths tokens
    - Non-padded mode: variable sequence lengths (no padding)
    """

    def setUp(self):
        super().setUp()
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:
            self.skipTest(f"Requires SM90+, got SM{cap[0]}{cap[1]}")

    def test_basic(self):
        """Test basic prefill with single sequence"""
        print("\n=== Test FMHAv2Prefill: Basic ===", flush=True)

        batch_size = 1
        input_lengths = [128]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_batch(self):
        """Test prefill with multiple sequences of variable lengths"""
        print("\n=== Test FMHAv2Prefill: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [64, 128, 256, 512]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_gqa(self):
        """Test prefill with grouped query attention"""
        print("\n=== Test FMHAv2Prefill: GQA ===", flush=True)

        batch_size = 2
        input_lengths = [256, 512]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_long_sequence(self):
        """Test prefill with long sequences"""
        print("\n=== Test FMHAv2Prefill: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [1024, 2048]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )


if __name__ == "__main__":
    unittest.main()
