from typing import NamedTuple, Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import (
    is_sm_90,
    is_sm_120,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
)


class TRTMHAImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # CUDA graph copy logic specific to TRTMHAImpl
        if self.prefill_cuda_graph_copy_params:
            # Infer qkv_dim from fmha_input tensor shape
            qkv_dim = fmha_input.shape[1]
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            aligned_attn_buf = torch.zeros(
                (total_len, qkv_dim),
                dtype=fmha_input.dtype,
                device=fmha_input.device,
            )

            cuda_graph_copy_small2large(
                fmha_input,
                aligned_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                qkv_dim,
                self.cu_seq_lens,
            )
            fmha_input = aligned_attn_buf

        # Execute FMHA forward
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        if self.prefill_cuda_graph_copy_params:
            # Infer hidden_size from res tensor shape
            hidden_size = res.shape[1]
            compact_attn_buf = torch.zeros(
                (qkv.shape[0], hidden_size), dtype=res.dtype, device=res.device
            )
            cuda_graph_copy_large2small(
                res,
                compact_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            res = compact_attn_buf
        return res

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        pass


class TRTPagedMHAImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTPagedAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def support_cuda_graph(self) -> bool:
        return False

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTPagedAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if not attn_inputs.is_prefill and (
            attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
        ):
            attn_inputs.prefix_lengths = torch.zeros_like(
                attn_inputs.input_lengths, device=attn_inputs.input_lengths.device
            )
        common.update_trt_params(
            self.fmha_impl,
            self.rope_kvcache_impl,
            self.fmha_params,
            self.rope_params,
            attn_inputs,
        )


# ---------------------------------------------------------------------------
# FlashInfer trtllm_fmha_v2_prefill based implementations (SM90/SM120)
# ---------------------------------------------------------------------------

# flashinfer JIT only generates flash_attention=True kernels;
# fmha_v2_run.cu determine_launch_params requires s >= 16 for flash_attention.
# Pad max_q_len / max_kv_len to this minimum so the dispatch selects flash kernels.
_TRTLLM_FMHA_V2_MIN_SEQ_LEN = 16

_TRTLLM_FMHA_V2_WORKSPACE_SIZE_MB = 512
_g_trtllm_fmha_v2_workspace_pool: list[torch.Tensor] = []
_g_trtllm_fmha_v2_pool_lock = __import__("threading").Lock()


def _get_trtllm_fmha_v2_workspace(device: str = "cuda") -> torch.Tensor:
    with _g_trtllm_fmha_v2_pool_lock:
        if _g_trtllm_fmha_v2_workspace_pool:
            return _g_trtllm_fmha_v2_workspace_pool.pop()
        return torch.zeros(
            _TRTLLM_FMHA_V2_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )


def _release_trtllm_fmha_v2_workspace(buf: torch.Tensor) -> None:
    with _g_trtllm_fmha_v2_pool_lock:
        _g_trtllm_fmha_v2_workspace_pool.append(buf)


class TRTLLMFMHAv2Params(NamedTuple):
    batch_size: int
    max_q_len: int
    max_kv_len: int
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: Optional[torch.Tensor] = None  # paged only
    block_tables: Optional[torch.Tensor] = None  # paged only


class TRTLLMFMHAv2PagedPrefillOp:
    """Paged prefill op via trtllm_fmha_v2_prefill Q_PAGED_KV_HND layout."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.scaling = self.head_dim**-0.5
        self.workspace_buffer = _get_trtllm_fmha_v2_workspace()

    def __del__(self) -> None:
        _release_trtllm_fmha_v2_workspace(self.workspace_buffer)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            (is_sm_90() or is_sm_120())
            and attn_inputs.is_prefill
            and attn_inputs.kv_cache_kernel_block_id_device is not None
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTLLMFMHAv2Params:
        prefix_lengths = torch.zeros_like(attn_inputs.input_lengths, device="cuda")
        input_lengths = torch.zeros_like(attn_inputs.input_lengths, device="cuda")
        prefix_lengths.copy_(attn_inputs.prefix_lengths, non_blocking=True)
        input_lengths.copy_(attn_inputs.input_lengths, non_blocking=True)
        seq_lens = input_lengths + prefix_lengths

        cu_kv_seqlens = torch.zeros(
            attn_inputs.input_lengths.shape[0] + 1, device="cuda", dtype=torch.int32
        )
        cu_kv_seqlens[1:] = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        cu_seqlens = attn_inputs.cu_seqlens
        if not cu_seqlens.is_cuda:
            cu_seqlens = cu_seqlens.to("cuda", non_blocking=True)

        block_tables = attn_inputs.kv_cache_kernel_block_id_device
        if not block_tables.is_cuda:
            block_tables = block_tables.to("cuda", non_blocking=True)

        max_kv = max(
            (attn_inputs.prefix_lengths + attn_inputs.input_lengths).max().item(),
            _TRTLLM_FMHA_V2_MIN_SEQ_LEN,
        )
        return TRTLLMFMHAv2Params(
            batch_size=attn_inputs.input_lengths.size(0),
            max_q_len=max(
                attn_inputs.input_lengths.max().item(), _TRTLLM_FMHA_V2_MIN_SEQ_LEN
            ),
            max_kv_len=max_kv,
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            block_tables=block_tables,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: LayerKVCache,
        params: TRTLLMFMHAv2Params,
    ) -> torch.Tensor:
        import flashinfer.prefill

        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype).contiguous().view(-1, self.head_num, self.head_dim)
        kv_cache_5d = kv_cache.kv_cache_base.view(
            kv_cache.kv_cache_base.shape[0],
            2,
            self.kv_head_num,
            self.seq_size_per_block,
            self.head_dim,
        )
        o = flashinfer.prefill.trtllm_fmha_v2_prefill(
            qkv=(q, kv_cache_5d),
            input_layout="Q_PAGED_KV_HND",
            workspace_buffer=self.workspace_buffer,
            seq_lens=params.seq_lens,
            max_q_len=params.max_q_len,
            max_kv_len=params.max_kv_len,
            bmm1_scale=self.scaling,
            bmm2_scale=1.0,
            batch_size=params.batch_size,
            cum_seq_lens_q=params.cu_seqlens,
            cum_seq_lens_kv=params.cu_kv_seqlens,
            block_tables=params.block_tables,
            out_dtype=q_type,
            mask_mode="causal",
        )
        return o.view(-1, self.head_num * self.head_dim).to(q_type)


class TRTLLMFMHAv2PrefillOp:
    """Non-paged prefill op via trtllm_fmha_v2_prefill CONTIGUOUS_Q_KV layout."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.workspace_buffer = _get_trtllm_fmha_v2_workspace()

    def __del__(self) -> None:
        _release_trtllm_fmha_v2_workspace(self.workspace_buffer)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.any().item()
        )
        return (is_sm_90() or is_sm_120()) and attn_inputs.is_prefill and not has_prefix

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTLLMFMHAv2Params:
        input_lengths = torch.zeros_like(attn_inputs.input_lengths, device="cuda")
        input_lengths.copy_(attn_inputs.input_lengths, non_blocking=True)
        cu_seqlens = attn_inputs.cu_seqlens
        if not cu_seqlens.is_cuda:
            cu_seqlens = cu_seqlens.to("cuda", non_blocking=True)
        max_len = max(
            attn_inputs.input_lengths.max().item(), _TRTLLM_FMHA_V2_MIN_SEQ_LEN
        )
        return TRTLLMFMHAv2Params(
            batch_size=attn_inputs.input_lengths.size(0),
            max_q_len=max_len,
            max_kv_len=max_len,
            seq_lens=input_lengths,
            cu_seqlens=cu_seqlens,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: TRTLLMFMHAv2Params,
    ) -> torch.Tensor:
        import flashinfer.prefill

        q_type = qkv.dtype
        compute_dtype = (
            torch.float8_e4m3fn
            if self.kv_cache_dtype == KvCacheDataType.FP8
            else q_type
        )
        q = (
            qkv[:, : self.q_size]
            .to(compute_dtype)
            .contiguous()
            .view(-1, self.head_num, self.head_dim)
        )
        kv = (
            qkv[:, self.q_size :]
            .to(compute_dtype)
            .contiguous()
            .view(-1, 2, self.kv_head_num, self.head_dim)
        )
        res = flashinfer.prefill.trtllm_fmha_v2_prefill(
            qkv=(q, kv),
            input_layout="CONTIGUOUS_Q_KV",
            workspace_buffer=self.workspace_buffer,
            seq_lens=params.seq_lens,
            max_q_len=params.max_q_len,
            max_kv_len=params.max_kv_len,
            bmm1_scale=self.scaling,
            bmm2_scale=1.0,
            batch_size=params.batch_size,
            cum_seq_lens_q=params.cu_seqlens,
            cum_seq_lens_kv=params.cu_seqlens,
            out_dtype=q_type,
            mask_mode="causal",
        )
        return res.view(-1, self.head_num * self.head_dim)


class FlashInferTRTLLMFMHAv2PagedPrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTLLMFMHAv2PagedPrefillOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return TRTLLMFMHAv2PagedPrefillOp.support(attn_configs, attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        fmha_input = (
            self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if self.need_rope_kv_cache
            else qkv
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class FlashInferTRTLLMFMHAv2PrefillImpl(FMHAImplBase):
    """Non-paged prefill via trtllm_fmha_v2_prefill PACKED_QKV layout."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTLLMFMHAv2PrefillOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return TRTLLMFMHAv2PrefillOp.support(attn_configs, attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        fmha_input = (
            self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if self.need_rope_kv_cache
            else qkv
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
