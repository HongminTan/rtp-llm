from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class RecordingWrapper(FMHAImplBase):
    def __init__(
        self,
        inner_impl: FMHAImplBase,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        active_sequence_ids: Sequence[int],
        record_qkv: bool,
        golden_layer_records: Optional[Dict[int, AttentionLayerRecord]] = None,
    ) -> None:
        self.inner = inner_impl
        self.attn_configs = attn_configs
        self.record_qkv = record_qkv
        self.golden_layer_records = golden_layer_records
        self.layer_records: Dict[int, AttentionLayerRecord] = {}
        self.head_num = int(attn_configs.head_num)
        self.kv_head_num = int(attn_configs.kv_head_num)
        self.head_dim = int(attn_configs.size_per_head)
        self.dtype = getattr(attn_configs, "dtype", None)
        self.is_prefill = bool(attn_inputs.is_prefill)
        self.sequence_lengths = self._snapshot_tensor(
            getattr(attn_inputs, "sequence_lengths", None)
        )
        self.input_lengths = self._snapshot_tensor(
            getattr(attn_inputs, "input_lengths", None)
        )
        self.prefix_lengths = self._snapshot_tensor(
            getattr(attn_inputs, "prefix_lengths", None)
        )
        self.cu_seqlens = self._snapshot_tensor(
            getattr(attn_inputs, "cu_seqlens", None)
        )
        self.active_sequence_ids = tuple(int(i) for i in active_sequence_ids)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if layer_idx in self.layer_records:
            raise RuntimeError(f"duplicate layer record: layer_idx={layer_idx}")
        q = k = v = None
        if self.record_qkv:
            q, k, v = self._split(qkv)
            q = q.detach().clone()
            k = k.detach().clone()
            v = v.detach().clone()
        out = self.inner.forward(qkv, kv_cache, layer_idx)
        # impls return either [T, Hq, D] (paged) or [T, Hq*D] (FMHAv2/XQA)
        # the record copy is always flattened to packed [T, Hq*D]
        # But the model still gets inner's raw out
        out_rec = out.reshape(out.shape[0], -1)
        self.layer_records[layer_idx] = AttentionLayerRecord(
            layer_idx=layer_idx,
            q=q,
            k=k,
            v=v,
            output=out_rec.detach().clone(),
        )
        if self.golden_layer_records is not None:
            reference = self.golden_layer_records.get(layer_idx)
            if reference is None:
                raise RuntimeError(f"golden output is missing layer {layer_idx}")
            if reference.output.numel() != out.numel():
                raise RuntimeError(
                    "golden output shape differs from candidate output at "
                    f"layer {layer_idx}: golden={tuple(reference.output.shape)} "
                    f"candidate={tuple(out.shape)}"
                )
            if (
                reference.output.dtype != out.dtype
                or reference.output.device != out.device
            ):
                raise RuntimeError(
                    "golden output dtype/device differs from candidate output at "
                    f"layer {layer_idx}"
                )
            return reference.output.reshape_as(out).clone()
        return out

    @staticmethod
    def _snapshot_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().clone()

    def _split(
        self, qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.attn_configs.head_num
        kv_h = self.attn_configs.kv_head_num
        d = self.attn_configs.size_per_head
        q, k, v = torch.split(qkv, [h * d, kv_h * d, kv_h * d], dim=-1)
        return q.reshape(-1, h, d), k.reshape(-1, kv_h, d), v.reshape(-1, kv_h, d)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def support_cuda_graph(self) -> bool:
        return False

    @property
    def fmha_params(self) -> Any:
        return self.inner.fmha_params
