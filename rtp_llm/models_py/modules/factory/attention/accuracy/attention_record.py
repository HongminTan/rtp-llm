from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class AttentionLayerRecord:
    layer_idx: int
    output: torch.Tensor  # packed [T, head_num * head_dim]
    q: Optional[torch.Tensor] = None  # packed [T, head_num, head_dim]
    k: Optional[torch.Tensor] = None  # packed [T, kv_head_num, head_dim]
    v: Optional[torch.Tensor] = None  # packed [T, kv_head_num, head_dim]

    def release(self) -> None:
        self.q = None
        self.k = None
        self.v = None
        self.output = None  # type: ignore[assignment]


@dataclass
class AttentionForwardRecord:
    impl_name: str
    phase: str  # "plain" | "prefix" | "decode_{st}"
    layer_records: Dict[int, AttentionLayerRecord]
    head_num: int = 0
    kv_head_num: int = 0
    head_dim: int = 0
    dtype: Optional[torch.dtype] = None
    is_prefill: bool = False
    sequence_lengths: Optional[torch.Tensor] = None
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    active_sequence_ids: Tuple[int, ...] = ()

    def release(self) -> None:
        for r in self.layer_records.values():
            r.release()
        self.layer_records.clear()
        self.sequence_lengths = None
        self.input_lengths = None
        self.prefix_lengths = None
        self.cu_seqlens = None
