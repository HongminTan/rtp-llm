#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on seq_len % seq_size_per_block.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   int64_t                 initial_hash = 0);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks up to
// max_seq_len (defaults to complete_token_ids->seqLength() when negative).
// Also updates BatchKVCacheResource::last_block_aligned.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     int64_t                 initial_hash = 0,
                     int                     max_seq_len  = -1);

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
