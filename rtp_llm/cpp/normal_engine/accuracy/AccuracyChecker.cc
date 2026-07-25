#include "rtp_llm/cpp/normal_engine/accuracy/AccuracyChecker.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <pybind11/stl.h>

#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/TimeUtility.h"

using namespace std;

namespace rtp_llm {

namespace {

constexpr uint64_t kSplitMixIncrement = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kSequenceSeedSalt  = 0xD1B54A32D192ED03ULL;

uint64_t splitMix64(uint64_t value) {
    value += kSplitMixIncrement;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

}  // namespace

// RAII for release Stream resources (KV block...)
struct StreamReleaseGuard {
    GenerateStreamPtr stream_;
    StreamReleaseGuard() = default;
    explicit StreamReleaseGuard(GenerateStreamPtr stream): stream_(std::move(stream)) {}
    StreamReleaseGuard(StreamReleaseGuard&& guard) noexcept: stream_(std::move(guard.stream_)) {}
    StreamReleaseGuard& operator=(StreamReleaseGuard&& guard) noexcept {
        if (this != &guard) {
            release();
            stream_ = std::move(guard.stream_);
        }
        return *this;
    }
    StreamReleaseGuard(const StreamReleaseGuard&)            = delete;
    StreamReleaseGuard& operator=(const StreamReleaseGuard&) = delete;
    void                reset(GenerateStreamPtr stream) {
        release();
        stream_ = std::move(stream);
    }
    void release() {
        if (stream_) {
            stream_->setNeedReleaseResource(true);
            stream_->releaseResource();
            stream_.reset();
        }
    }
    ~StreamReleaseGuard() {
        release();
    }
};

namespace {

// RAII for closing the Python recorder and clearing accuracy_recording_.
struct RecordingSession {
    PyWrappedModel* py_model_;
    py::object      recorder_;
    RecordingSession(PyWrappedModel* py_model, py::object recorder):
        py_model_(py_model), recorder_(std::move(recorder)) {}
    RecordingSession(const RecordingSession&)            = delete;
    RecordingSession& operator=(const RecordingSession&) = delete;
    ~RecordingSession() {
        {
            py::gil_scoped_acquire gil;
            try {
                if (recorder_.ptr()) {
                    recorder_.attr("close")();
                }
            } catch (...) {
                RTP_LLM_LOG_ERROR("accuracy check: recorder close() failed");
            }
            recorder_ = py::object();
        }
        py_model_->setAccuracyRecording(false);
    }
};

// Build a stream and its release guard for each active sequence
// rank0 only; rank>0 builds nothing and forwards via tpSync
bool buildBatch(bool                                                            root,
                const std::vector<size_t>&                                      active_seq_ids,
                const std::function<absl::StatusOr<GenerateStreamPtr>(size_t)>& make_stream,
                std::vector<GenerateStreamPtr>&                                 streams,
                std::vector<StreamReleaseGuard>&                                guards,
                const std::string&                                              impl_name,
                const std::string&                                              phase) {
    streams.clear();
    guards.clear();
    if (!root) {
        return true;
    }
    for (size_t seq_id : active_seq_ids) {
        try {
            auto stream_or = make_stream(seq_id);
            if (!stream_or.ok()) {
                RTP_LLM_LOG_ERROR("accuracy check build stream failed (impl=%s, phase=%s, seq=%zu): %s",
                                  impl_name.c_str(),
                                  phase.c_str(),
                                  seq_id,
                                  stream_or.status().ToString().c_str());
                return false;
            }
            guards.emplace_back(stream_or.value());
            streams.push_back(stream_or.value());
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("accuracy check build stream threw (impl=%s, phase=%s, seq=%zu): %s",
                              impl_name.c_str(),
                              phase.c_str(),
                              seq_id,
                              e.what());
            return false;
        } catch (...) {
            RTP_LLM_LOG_ERROR("accuracy check build stream threw (impl=%s, phase=%s, seq=%zu): unknown",
                              impl_name.c_str(),
                              phase.c_str(),
                              seq_id);
            return false;
        }
    }
    return true;
}

}  // namespace

static const std::vector<AccuracyScenario> kDefaultAccuracyScenarios = {
    {{{0, 128, 0}}, true, 0xA11CE00000000001ULL},    // N=1 plain (ragged)
    {{{0, 64, 4}}, true, 0xA11CE00000000002ULL},     // N=1 decode for small context/token limits
    {{{512, 512, 0}}, true, 0xA11CE00000000003ULL},  // N=1 prefix (paged/hybrid)
    {{{512, 512, 5}}, true, 0xA11CE00000000004ULL},  // N=1 prefix + decode
    {{{0, 128, 0}, {0, 777, 0}, {0, 2048, 0}}, false, 0xA11CE00000000005ULL},          // N=3 variable-length plain
    {{{512, 512, 0}, {256, 1024, 0}, {2048, 2048, 0}}, false, 0xA11CE00000000006ULL},  // N=3 variable-length prefix
    {{{512, 512, 1}, {512, 512, 3}, {512, 512, 7}}, false, 0xA11CE00000000007ULL},     // N=3 shrinking decode
    {{{0, 64, 16}, {0, 64, 16}, {0, 64, 16}, {0, 64, 16}, {0, 64, 16}, {0, 64, 16}, {0, 64, 16}, {0, 64, 16}},
     false,
     0xA11CE00000000008ULL},  // N=8 large-batch decode
};

const std::vector<AccuracyScenario>& AccuracyChecker::defaultScenarios() {
    return kDefaultAccuracyScenarios;
}

std::vector<AccuracyScenario> AccuracyChecker::scenariosForMoeConfig(const MoeConfig& moe_config) {
    if (!moe_config.use_deepep_moe || !moe_config.use_deepep_low_latency) {
        return kDefaultAccuracyScenarios;
    }
    constexpr size_t kMinLowLatencyTokenBudget = 8;
    if (moe_config.ll_num_max_token < static_cast<int>(kMinLowLatencyTokenBudget)) {
        return {};
    }

    const size_t token_budget = static_cast<size_t>(moe_config.ll_num_max_token);

    // Preserve the exact qualifying decode contexts. Long prefills are executed
    // through chunked golden bootstrap; shortening them would weaken a global allowlist.
    std::vector<AccuracyScenario> scenarios = {kDefaultAccuracyScenarios[1], kDefaultAccuracyScenarios[3]};

    const size_t b3_len0 = std::max<size_t>(1, token_budget / 8);
    const size_t b3_len1 = std::max<size_t>(1, token_budget / 4);
    const size_t b3_len2 = token_budget - b3_len0 - b3_len1;
    scenarios.push_back({{{b3_len0 / 2, b3_len0 - b3_len0 / 2, 1},
                          {b3_len1 / 2, b3_len1 - b3_len1 / 2, 3},
                          {b3_len2 / 2, b3_len2 - b3_len2 / 2, 7}},
                         false,
                         0xA11CE00000000007ULL});

    std::vector<AccuracySeq> b8_seqs;
    b8_seqs.reserve(8);
    const size_t b8_base      = token_budget / 8;
    const size_t b8_remainder = token_budget % 8;
    for (size_t seq_id = 0; seq_id < 8; ++seq_id) {
        b8_seqs.push_back({0, b8_base + (seq_id < b8_remainder ? 1 : 0), 16});
    }
    scenarios.push_back({std::move(b8_seqs), false, 0xA11CE00000000008ULL});
    return scenarios;
}

AccuracyChecker::AccuracyChecker(const ModelConfig&       model_config,
                                 const RuntimeConfig&     runtime_config,
                                 const ParallelismConfig& parallelism_config):
    model_config_(model_config), runtime_config_(runtime_config), parallelism_config_(parallelism_config) {}

torch::Tensor
AccuracyChecker::makeDeterministicPrompt(size_t len, int64_t token_size, uint64_t prompt_seed, size_t sequence_id) {
    RTP_LLM_CHECK_WITH_INFO(token_size > 0 && token_size <= std::numeric_limits<int32_t>::max(),
                            "accuracy check token size must fit positive int32, got %ld",
                            token_size);
    RTP_LLM_CHECK_WITH_INFO(len <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "accuracy check prompt length is too large: %zu",
                            len);

    auto prompt = torch::empty({static_cast<int64_t>(len)}, torch::TensorOptions().dtype(torch::kInt32));
    auto tokens = prompt.data_ptr<int32_t>();

    // Derive an independent stream per sequence, then address each token by index.
    const uint64_t sequence_seed =
        splitMix64(prompt_seed ^ (kSequenceSeedSalt * (static_cast<uint64_t>(sequence_id) + 1)));
    for (size_t token_idx = 0; token_idx < len; ++token_idx) {
        const uint64_t sample = splitMix64(sequence_seed + kSplitMixIncrement * static_cast<uint64_t>(token_idx));
        tokens[token_idx]     = static_cast<int32_t>(sample % static_cast<uint64_t>(token_size));
    }
    return prompt;
}

torch::Tensor AccuracyChecker::makeCheckPrompt(size_t len, uint64_t prompt_seed, size_t sequence_id) {
    const int64_t token_size = model_config_.embedding_size ?
                                   std::min(model_config_.embedding_size, model_config_.vocab_size) :
                                   model_config_.vocab_size;
    return makeDeterministicPrompt(len, token_size, prompt_seed, sequence_id);
}

std::shared_ptr<GenerateInput> AccuracyChecker::wrapAccuracyInput(torch::Tensor input_ids, bool need_release) {
    auto input                   = std::make_shared<GenerateInput>();
    input->generate_config       = std::make_shared<GenerateConfig>();
    input->input_ids             = input_ids;
    input->begin_time_us         = autil::TimeUtility::currentTimeInMicroSeconds();
    auto& config                 = *input->generate_config;
    config.top_k                 = 1;
    config.do_sample             = false;
    config.ignore_eos            = true;
    config.reuse_cache           = false;
    config.enable_device_cache   = false;
    config.enable_memory_cache   = false;
    config.enable_remote_cache   = false;
    config.can_use_pd_separation = false;
    config.pd_separation         = false;
    input->need_release_resource = need_release;
    return input;
}

absl::StatusOr<GenerateStreamPtr> AccuracyChecker::buildAccuracyStream(torch::Tensor prompt, bool need_release) {
    auto input  = wrapAccuracyInput(prompt, need_release);
    auto stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config_, *resource_context_, nullptr, 0, false);
    RETURN_IF_STATUS_ERROR(stream->initKVBlock());
    return stream;
}

// Build an independent stream that reuses golden's KV cache for `history_len` tokens
// Prefix: `init_tokens` contains history + chunk; reuse history and execute the chunk
// Decode: `init_tokens` contains history; append and execute `cur_token`
// Full history blocks are referenced, while an unaligned boundary block is copied
absl::StatusOr<GenerateStreamPtr> AccuracyChecker::makeForkStream(
    torch::Tensor init_tokens, size_t history_len, bool is_decode, int cur_token, const GenerateStreamPtr& golden) {
    const size_t block_size  = model_config_.attn_config.tokens_per_block;
    const size_t full_blocks = history_len / block_size;
    const size_t boundary    = history_len % block_size;

    RTP_LLM_CHECK_WITH_INFO(!golden->hasError(), "accuracy check: golden in error state at fork");
    RTP_LLM_CHECK_WITH_INFO(golden->kvCache().groupNums() == 1,
                            "accuracy check: fork requires single kv-cache group, got %d",
                            golden->kvCache().groupNums());
    if (is_decode) {
        // golden must be a finished decode stream and its history is set.
        RTP_LLM_CHECK_WITH_INFO(!golden->isContextStream(), "accuracy check: golden has not entered decode");
        RTP_LLM_CHECK_WITH_INFO(golden->seqLength() >= (int64_t)history_len,
                                "accuracy check: golden seqLength %ld < history_len %zu",
                                (long)golden->seqLength(),
                                history_len);
    }
    const auto&  golden_blocks = golden->kvCache().blocks(0, 0);
    const size_t total_blocks  = full_blocks + (boundary ? 1 : 0);
    // check golden kv cache has been generated completely
    RTP_LLM_CHECK_WITH_INFO(
        golden_blocks.size() >= total_blocks,
        "accuracy check: golden KV cache is incomplete for history_len=%zu: expected at least %zu blocks, got %zu",
        golden_blocks.size(),
        total_blocks,
        history_len);

    auto input  = wrapAccuracyInput(init_tokens, true);
    auto stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config_, *resource_context_, nullptr, 0, false);
    stream->setReserveStep(0);

    RTP_LLM_CHECK_WITH_INFO(stream->currentBatchSize() == 1 && stream->maxBatchSize() == 1,
                            "accuracy check: each fork must contain exactly one sequence");
    RTP_LLM_CHECK_WITH_INFO(!stream->streamCacheResource().reuseCache(),
                            "accuracy check: fork must have reuse_cache=false");

    // Reference golden's complete history blocks
    BlockIndicesType history_blocks(golden_blocks.begin(), golden_blocks.begin() + full_blocks);
    stream->referenceRequestBlocks(history_blocks);

    if (is_decode) {
        // Decode: Append golden's token and set stream on decode phase
        auto new_tokens                   = torch::zeros({1, 1}, torch::kInt32);
        new_tokens.data_ptr<int32_t>()[0] = cur_token;
        StreamUpdateInfo update_info{/*new_tokens=*/new_tokens,
                                     /*num_new_tokens=*/1,
                                     /*hidden_states=*/torch::Tensor(),
                                     /*logits=*/torch::Tensor(),
                                     /*softmax_probs=*/torch::Tensor(),
                                     /*cum_log_probs=*/torch::Tensor(),
                                     /*all_probs=*/torch::Tensor(),
                                     /*loss=*/torch::Tensor(),
                                     /*src_batch_indices=*/torch::Tensor(),
                                     /*all_hidden_states=*/torch::Tensor(),
                                     /*update_remote_generate=*/false};
        stream->update(update_info);
        RTP_LLM_CHECK_WITH_INFO(!stream->hasError(), "accuracy check: decode fork update(cur_token) failed");
        RTP_LLM_CHECK_WITH_INFO(!stream->isContextStream(), "accuracy check: decode fork not in decode mode");
        RTP_LLM_CHECK_WITH_INFO(stream->seqLength() == stream->inputLength() + 1,
                                "accuracy check: decode fork seqLength %ld != inputLength+1",
                                (long)stream->seqLength());
        auto next_execute_token = stream->currentExecuteTokens(0);
        RTP_LLM_CHECK_WITH_INFO(next_execute_token.size() == 1 && next_execute_token[0] == cur_token,
                                "accuracy check: decode fork current token != cur_token");
    } else {
        // Prefix: Stay a context stream, reuse golden's [0:history_len], compute only the chunk.
        stream->setReuseLength(history_len);
        RTP_LLM_CHECK_WITH_INFO(stream->isContextStream(), "accuracy check: prefix fork must stay context");
    }

    // update() clears any stale mapping; prefix never set one. Both paths must be clean before we inject.
    RTP_LLM_CHECK_WITH_INFO(stream->streamCacheResource().getKVBlockUpdateMapping().empty(),
                            "accuracy check: stale block-update mapping before boundary copy");

    // Allocate own blocks for the remainder (decode: one tail; prefix: the chunk blocks)
    RETURN_IF_STATUS_ERROR(stream->incrKVBlock());
    const size_t need_blocks = (stream->seqLength() + block_size - 1) / block_size;
    RTP_LLM_CHECK_WITH_INFO(stream->kvCache().blocks(0, 0).size() == need_blocks,
                            "accuracy check: fork expected %zu blocks, got %zu",
                            need_blocks,
                            stream->kvCache().blocks(0, 0).size());
    if (!is_decode) {
        // Verify the configured reuse length.
        RTP_LLM_CHECK_WITH_INFO(stream->reuseLength() == (int)history_len,
                                "accuracy check: prefix fork reuse_length %d != %zu",
                                stream->reuseLength(),
                                history_len);
    }

    // Copy unaligned block
    if (boundary != 0) {
        const int                      dst_block = stream->kvCache().blocks(0, 0)[full_blocks];
        const std::string&             tag       = golden->kvCache().cacheResource().soleGroupTagForLayer(0);
        std::vector<TaggedBlockIdPair> mapping{TaggedBlockIdPair{tag, golden_blocks[full_blocks], dst_block}};
        stream->setKVBlockUpdateMapping(mapping);
    }
    return stream;
}

// World (DP x TP x EP) reduction: every rank must agree
bool AccuracyChecker::worldAll(bool local_flag) {
    if (parallelism_config_.world_size <= 1) {
        return local_flag;
    }
    auto flag = torch::tensor({local_flag ? 1 : 0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    flag      = execAllReduce({flag, ReduceOp::Min, false, ParallelMode::DP_AND_TP}).buffer;
    return flag.item<int32_t>() != 0;
}

absl::Status AccuracyChecker::abortIfFailed(bool step_flag, const std::string& where) {
    if (worldAll(step_flag)) {
        return absl::OkStatus();
    }
    return absl::InternalError("accuracy check aborted at " + where + " (some rank failed)");
}

absl::StatusOr<std::vector<int32_t>> AccuracyChecker::sumWorldMask(const std::vector<int32_t>& local_mask,
                                                                   const std::string&          where) {
    if (parallelism_config_.world_size <= 1) {
        return local_mask;
    }
    try {
        auto mask = torch::tensor(local_mask, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        mask      = execAllReduce({mask, ReduceOp::Sum, false, ParallelMode::DP_AND_TP}).buffer;
        cudaSyncAndCheck();
        auto                 cpu = mask.to(torch::kCPU).contiguous();
        std::vector<int32_t> out(static_cast<size_t>(cpu.numel()));
        const auto*          data = cpu.data_ptr<int32_t>();
        std::copy(data, data + cpu.numel(), out.begin());
        return out;
    } catch (const std::exception& e) {
        return absl::InternalError("accuracy check mask reduction failed at " + where + ": " + e.what());
    }
}

// Check the scheduler and sequence-length limits
bool AccuracyChecker::runCheck(const AccuracyScenario& scenario, const std::string& scenario_base_name) {
    size_t max_seq_len   = 0;
    size_t max_plain_len = 0;
    size_t plain_tokens  = 0;
    for (const auto& seq : scenario.seqs) {
        const size_t plain_len = seq.prefix_len + seq.chunk_len;
        plain_tokens += plain_len;
        max_plain_len = std::max(max_plain_len, plain_len);
        max_seq_len   = std::max(max_seq_len, plain_len + seq.decode_steps);
    }
    const bool batch_size_check_flag =
        scenario.seqs.size() <= static_cast<size_t>(runtime_config_.max_generate_batch_size)
        && (max_forward_tokens_ == 0 || scenario.seqs.size() <= max_forward_tokens_);
    const size_t max_batch_tokens = static_cast<size_t>(runtime_config_.fifo_scheduler_config.max_batch_tokens_size);
    const size_t batch_tokens =
        max_forward_tokens_ == 0 ? max_plain_len * scenario.seqs.size() : std::min(plain_tokens, max_forward_tokens_);
    const bool batch_tokens_check_flag = (max_batch_tokens == 0) || (batch_tokens < max_batch_tokens);
    // PyTorch SDPA may produce incorrect results for sequences longer than 65536 tokens
    // The fix is expected in torch 2.14
    const bool sequence_length_check_flag =
        (max_seq_len <= static_cast<size_t>(model_config_.max_seq_len)) && (max_seq_len <= 65536);
    if (!worldAll(batch_size_check_flag && batch_tokens_check_flag && sequence_length_check_flag)) {
        RTP_LLM_LOG_WARNING("accuracy check: skip scenario %s (batch_size=%zu limit=%d, batch_tokens=%zu limit=%zu, "
                            "sequence_length=%zu model_limit=%ld sdpa_limit=65536)",
                            scenario_base_name.c_str(),
                            scenario.seqs.size(),
                            runtime_config_.max_generate_batch_size,
                            batch_tokens,
                            max_batch_tokens,
                            max_seq_len,
                            (long)model_config_.max_seq_len);
        return false;
    }
    return true;
}

absl::Status AccuracyChecker::forwardGoldenBootstrap(const py::object&                     recorder,
                                                     const std::string&                    scenario_base_name,
                                                     const std::string&                    phase,
                                                     const std::vector<GenerateStreamPtr>& active_streams,
                                                     const std::vector<size_t>&            active_seq_ids,
                                                     bool                                  build_streams_flag) {
    const bool        root           = (parallelism_config_.tp_rank == 0);
    const std::string bootstrap_name = scenario_base_name + "::golden_bootstrap::" + phase;
    bool              setup_flag     = build_streams_flag;
    if (setup_flag) {
        try {
            py::gil_scoped_acquire gil;
            recorder.attr("start_golden_bootstrap")(scenario_base_name, phase, active_seq_ids);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("accuracy check bootstrap setup failed at %s: %s", bootstrap_name.c_str(), e.what());
            setup_flag = false;
        } catch (...) {
            RTP_LLM_LOG_ERROR("accuracy check bootstrap setup failed at %s: unknown", bootstrap_name.c_str());
            setup_flag = false;
        }
    }
    RETURN_IF_STATUS_ERROR(abortIfFailed(setup_flag, "setup " + bootstrap_name));

    bool forward_flag = true;
    try {
        std::list<GenerateStreamPtr> streams;
        if (root) {
            for (const auto& stream : active_streams) {
                streams.push_back(stream);
            }
        }
        forward_flag = executor_->process(streams).ok();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check bootstrap failed at %s: %s", bootstrap_name.c_str(), e.what());
        forward_flag = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check bootstrap failed at %s: unknown", bootstrap_name.c_str());
        forward_flag = false;
    }
    try {
        py::gil_scoped_acquire gil;
        recorder.attr("stop_golden_bootstrap")(bootstrap_name, forward_flag);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check bootstrap cleanup failed at %s: %s", bootstrap_name.c_str(), e.what());
        forward_flag = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check bootstrap cleanup failed at %s: unknown", bootstrap_name.c_str());
        forward_flag = false;
    }
    return abortIfFailed(forward_flag, bootstrap_name);
}

absl::Status AccuracyChecker::bootstrapGoldenHistory(const py::object&                 recorder,
                                                     const std::string&                scenario_base_name,
                                                     const AccuracyScenario&           scenario,
                                                     const std::vector<torch::Tensor>& prompts,
                                                     std::vector<GenerateStreamPtr>&   golden_streams,
                                                     std::vector<StreamReleaseGuard>&  golden_guards) {
    RTP_LLM_CHECK_WITH_INFO(max_forward_tokens_ > 0, "accuracy check bootstrap requires a positive token budget");
    const bool root = (parallelism_config_.tp_rank == 0);
    golden_streams.resize(scenario.seqs.size());
    golden_guards.resize(scenario.seqs.size());

    for (size_t seq_id = 0; seq_id < scenario.seqs.size(); ++seq_id) {
        const size_t       total_len   = scenario.seqs[seq_id].prefix_len + scenario.seqs[seq_id].chunk_len;
        size_t             history_len = 0;
        size_t             chunk_idx   = 0;
        GenerateStreamPtr  current_stream;
        StreamReleaseGuard current_guard;
        while (history_len < total_len) {
            const size_t                    chunk_end = std::min(total_len, history_len + max_forward_tokens_);
            std::vector<GenerateStreamPtr>  chunk_streams;
            std::vector<StreamReleaseGuard> chunk_guards;
            const std::vector<size_t>       active_seq_ids{seq_id};
            const std::string phase = "bootstrap_s" + std::to_string(seq_id) + "_chunk_" + std::to_string(chunk_idx);
            const bool        build_flag = buildBatch(
                root,
                active_seq_ids,
                [&](size_t) -> absl::StatusOr<GenerateStreamPtr> {
                    auto chunk_prompt = prompts[seq_id].narrow(0, 0, static_cast<int64_t>(chunk_end));
                    if (history_len == 0) {
                        return buildAccuracyStream(chunk_prompt, /*need_release=*/false);
                    }
                    return makeForkStream(chunk_prompt,
                                          history_len,
                                          /*is_decode=*/false,
                                          /*cur_token=*/0,
                                          current_stream);
                },
                chunk_streams,
                chunk_guards,
                "golden_bootstrap",
                phase);
            RETURN_IF_STATUS_ERROR(
                forwardGoldenBootstrap(recorder, scenario_base_name, phase, chunk_streams, active_seq_ids, build_flag));
            if (root) {
                RTP_LLM_CHECK_WITH_INFO(chunk_streams.size() == 1 && chunk_guards.size() == 1,
                                        "accuracy check bootstrap expected one stream for seq %zu",
                                        seq_id);
                current_stream = chunk_streams.front();
                current_guard  = std::move(chunk_guards.front());
            }
            history_len = chunk_end;
            ++chunk_idx;
        }
        if (root) {
            golden_streams[seq_id] = current_stream;
            golden_guards[seq_id]  = std::move(current_guard);
        }
    }
    return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> AccuracyChecker::listCandidates(const py::object&  recorder,
                                                                         const std::string& phase) {
    std::vector<std::string> registry;
    std::vector<int32_t>     local_mask;
    bool                     snapshot_ok = true;
    try {
        py::gil_scoped_acquire gil;
        py::dict               snapshot = recorder.attr("candidate_snapshot")(phase).cast<py::dict>();
        registry                        = snapshot["registry"].cast<std::vector<std::string>>();
        local_mask                      = snapshot["viable_mask"].cast<std::vector<int32_t>>();
        snapshot_ok =
            !registry.empty() && local_mask.size() == registry.size()
            && std::all_of(registry.begin(), registry.end(), [](const std::string& name) { return !name.empty(); })
            && std::unordered_set<std::string>(registry.begin(), registry.end()).size() == registry.size()
            && std::all_of(
                local_mask.begin(), local_mask.end(), [](int32_t value) { return value == 0 || value == 1; });
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check candidate_snapshot(%s) failed: %s", phase.c_str(), e.what());
        snapshot_ok = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check candidate_snapshot(%s) failed: unknown", phase.c_str());
        snapshot_ok = false;
    }

    // Dry construction may have launched device work. Do not enter a collective
    // with a poisoned device; the process supervisor must stop the worker group.
    cudaSyncAndCheck();
    if (!worldAll(snapshot_ok)) {
        return absl::FailedPreconditionError("attention gate candidate snapshot invalid at " + phase);
    }

    auto sums_or = sumWorldMask(local_mask, "candidate " + phase);
    RETURN_IF_STATUS_OR_ERROR(sums_or);
    auto                     sums = std::move(sums_or).value();
    std::vector<std::string> candidates;
    const int32_t            world_size = static_cast<int32_t>(parallelism_config_.world_size);
    for (size_t i = 0; i < registry.size(); ++i) {
        if (sums[i] == world_size) {
            candidates.push_back(registry[i]);
        } else if (sums[i] > 0) {
            RTP_LLM_LOG_WARNING("accuracy check candidate asymmetric phase=%s backend=%s viable_ranks=%d world=%d",
                                phase.c_str(),
                                registry[i].c_str(),
                                sums[i],
                                world_size);
        }
    }

    bool manifest_ok = true;
    try {
        py::gil_scoped_acquire gil;
        recorder.attr("set_expected_candidates")(phase, candidates);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check set_expected_candidates(%s) failed: %s", phase.c_str(), e.what());
        manifest_ok = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check set_expected_candidates(%s) failed: unknown", phase.c_str());
        manifest_ok = false;
    }
    RETURN_IF_STATUS_ERROR(abortIfFailed(manifest_ok, "set_expected_candidates " + phase));
    return candidates;
}

// Run one batched forward for the active sequences with world-consistent setup and execution checks.
absl::Status AccuracyChecker::forward(const py::object&                     recorder,
                                      const std::string&                    scenario_base_name,
                                      const std::string&                    impl_name,
                                      const std::string&                    phase,
                                      const std::vector<GenerateStreamPtr>& active_streams,
                                      const std::vector<size_t>&            active_seq_ids,
                                      bool                                  build_streams_flag,
                                      bool                                  gate_qualifying) {
    const bool        root          = (parallelism_config_.tp_rank == 0);
    const std::string scenario_name = scenario_base_name + "::" + impl_name + "::" + phase;
    bool              setup_flag    = build_streams_flag;
    if (setup_flag) {
        try {
            py::gil_scoped_acquire gil;
            recorder.attr("start_run")(scenario_base_name, impl_name, phase, active_seq_ids, gate_qualifying);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("accuracy check start_run failed at %s: %s", scenario_name.c_str(), e.what());
            setup_flag = false;
        } catch (...) {
            RTP_LLM_LOG_ERROR("accuracy check start_run failed at %s: unknown", scenario_name.c_str());
            setup_flag = false;
        }
    } else {
        RTP_LLM_LOG_ERROR("accuracy check build streams failed at %s", scenario_name.c_str());
    }
    RETURN_IF_STATUS_ERROR(abortIfFailed(setup_flag, "setup " + scenario_name));
    bool forward_flag = true;
    try {
        std::list<GenerateStreamPtr> streams;
        if (root) {
            for (const auto& stream : active_streams) {
                streams.push_back(stream);
            }
        }
        RTP_LLM_CHECK_WITH_INFO(!root || streams.size() == active_seq_ids.size(),
                                "accuracy check: stream count %zu != active sequence count %zu at %s",
                                streams.size(),
                                active_seq_ids.size(),
                                scenario_name.c_str());
        forward_flag = executor_->process(streams).ok();
        if (forward_flag) {
            py::gil_scoped_acquire gil;
            recorder.attr("stop_run")(scenario_name);
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check failed at %s: %s", scenario_name.c_str(), e.what());
        forward_flag = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check failed at %s: unknown", scenario_name.c_str());
        forward_flag = false;
    }
    return abortIfFailed(forward_flag, scenario_name);
}

absl::Status AccuracyChecker::runScenario(const py::object& recorder, const AccuracyScenario& scenario) {
    const bool   root       = (parallelism_config_.tp_rank == 0);
    const size_t batch_size = scenario.seqs.size();
    if (batch_size == 0) {
        return absl::OkStatus();
    }
    // scenario_base_name fully encodes the sequence group: "b{N}__s0_p{P0}c{C0}d{D0}__s1_..."
    std::string scenario_base_name = "b" + std::to_string(batch_size);
    size_t      max_decode         = 0;
    for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
        scenario_base_name += "__s" + std::to_string(seq_id) + "_p" + std::to_string(scenario.seqs[seq_id].prefix_len)
                              + "c" + std::to_string(scenario.seqs[seq_id].chunk_len) + "d"
                              + std::to_string(scenario.seqs[seq_id].decode_steps);
        max_decode = std::max(max_decode, scenario.seqs[seq_id].decode_steps);
    }

    if (!runCheck(scenario, scenario_base_name)) {
        return absl::OkStatus();
    }

    // rank0 builds one prompt per sequence (prefix + chunk); rank>0 keeps prompts empty
    std::vector<torch::Tensor> prompts;
    if (root) {
        prompts.reserve(batch_size);
        for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
            prompts.push_back(makeCheckPrompt(
                scenario.seqs[seq_id].prefix_len + scenario.seqs[seq_id].chunk_len, scenario.prompt_seed, seq_id));
        }
    }
    std::vector<size_t> all_seq_ids(batch_size);
    for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
        all_seq_ids[seq_id] = seq_id;
    }

    size_t total_plain_tokens = 0;
    for (const auto& seq : scenario.seqs) {
        total_plain_tokens += seq.prefix_len + seq.chunk_len;
    }
    const bool chunked_bootstrap = max_forward_tokens_ > 0 && total_plain_tokens > max_forward_tokens_;

    // phase plain: batch golden streams plain prefill, one batched forward; golden kept alive for the whole scenario
    std::vector<GenerateStreamPtr>  golden_streams;
    std::vector<StreamReleaseGuard> golden_guards;
    if (chunked_bootstrap) {
        RTP_LLM_LOG_INFO("accuracy check: chunked golden bootstrap scenario=%s tokens=%zu budget=%zu",
                         scenario_base_name.c_str(),
                         total_plain_tokens,
                         max_forward_tokens_);
        RETURN_IF_STATUS_ERROR(
            bootstrapGoldenHistory(recorder, scenario_base_name, scenario, prompts, golden_streams, golden_guards));
    } else {
        bool build_golden_flag = buildBatch(
            root,
            all_seq_ids,
            [&](size_t seq_id) { return buildAccuracyStream(prompts[seq_id], /*need_release=*/false); },
            golden_streams,
            golden_guards,
            "golden",
            "plain");
        RETURN_IF_STATUS_ERROR(forward(recorder,
                                       scenario_base_name,
                                       "golden",
                                       "plain",
                                       golden_streams,
                                       all_seq_ids,
                                       build_golden_flag,
                                       scenario.gate_qualifying));
        auto plain_candidates_or = listCandidates(recorder, "plain");
        RETURN_IF_STATUS_OR_ERROR(plain_candidates_or);
        auto plain_candidates = std::move(plain_candidates_or).value();
        for (auto& impl_name : plain_candidates) {
            std::vector<GenerateStreamPtr>  plain_streams;
            std::vector<StreamReleaseGuard> plain_guards;
            bool                            build_plain_flag = buildBatch(
                root,
                all_seq_ids,
                [&](size_t seq_id) { return buildAccuracyStream(prompts[seq_id], /*need_release=*/true); },
                plain_streams,
                plain_guards,
                impl_name,
                "plain");
            RETURN_IF_STATUS_ERROR(forward(recorder,
                                           scenario_base_name,
                                           impl_name,
                                           "plain",
                                           plain_streams,
                                           all_seq_ids,
                                           build_plain_flag,
                                           scenario.gate_qualifying));
        }

        // phase prefix: sequences with a prefix, each fork referencing its golden history kv blocks and run paged
        // attention
        std::vector<size_t> prefix_seq_ids;
        for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
            if (scenario.seqs[seq_id].prefix_len > 0) {
                prefix_seq_ids.push_back(seq_id);
            }
        }
        if (!prefix_seq_ids.empty()) {
            {  // scope guard for release golden prefix streams before running candidates
                std::vector<GenerateStreamPtr>  golden_prefix_streams;
                std::vector<StreamReleaseGuard> golden_prefix_guards;
                bool                            build_golden_prefix_flag = buildBatch(
                    root,
                    prefix_seq_ids,
                    [&](size_t seq_id) {
                        return makeForkStream(prompts[seq_id],
                                              scenario.seqs[seq_id].prefix_len,
                                              /*is_decode=*/false,
                                              /*cur_token=*/0,
                                              golden_streams[seq_id]);
                    },
                    golden_prefix_streams,
                    golden_prefix_guards,
                    "golden",
                    "prefix");
                RETURN_IF_STATUS_ERROR(forward(recorder,
                                               scenario_base_name,
                                               "golden",
                                               "prefix",
                                               golden_prefix_streams,
                                               prefix_seq_ids,
                                               build_golden_prefix_flag,
                                               scenario.gate_qualifying));
            }

            auto prefix_candidates_or = listCandidates(recorder, "prefix");
            RETURN_IF_STATUS_OR_ERROR(prefix_candidates_or);
            auto prefix_candidates = std::move(prefix_candidates_or).value();
            for (auto& impl_name : prefix_candidates) {
                std::vector<GenerateStreamPtr>  prefix_streams;
                std::vector<StreamReleaseGuard> prefix_guards;
                bool                            build_prefix_flag = buildBatch(
                    root,
                    prefix_seq_ids,
                    [&](size_t seq_id) {
                        return makeForkStream(prompts[seq_id],
                                              scenario.seqs[seq_id].prefix_len,
                                              /*is_decode=*/false,
                                              /*cur_token=*/0,
                                              golden_streams[seq_id]);
                    },
                    prefix_streams,
                    prefix_guards,
                    impl_name,
                    "prefix");
                RETURN_IF_STATUS_ERROR(forward(recorder,
                                               scenario_base_name,
                                               impl_name,
                                               "prefix",
                                               prefix_streams,
                                               prefix_seq_ids,
                                               build_prefix_flag,
                                               scenario.gate_qualifying));
            }
        }
    }

    // decode: teacher forcing over a shrinking active set, one batched forward per step
    if (max_decode > 0) {
        // Capture t0 (each active seq's first decode token, from the plain prefill) before golden decodes
        std::vector<std::vector<int>> golden_decode_tokens(batch_size);
        bool                          t0_capture_flag = true;
        if (root) {
            for (size_t seq_id = 0; seq_id < batch_size && t0_capture_flag; ++seq_id) {
                if (scenario.seqs[seq_id].decode_steps == 0) {
                    continue;
                }
                try {
                    auto tokens = golden_streams[seq_id]->currentExecuteTokens(0);
                    if (tokens.empty()) {
                        RTP_LLM_LOG_ERROR("accuracy check: golden seq %zu produced no token for t0", seq_id);
                        t0_capture_flag = false;
                    } else {
                        golden_decode_tokens[seq_id].reserve(scenario.seqs[seq_id].decode_steps);
                        golden_decode_tokens[seq_id].push_back(tokens[0]);
                    }
                } catch (const std::exception& e) {
                    RTP_LLM_LOG_ERROR("accuracy check: t0 capture threw: %s", e.what());
                    t0_capture_flag = false;
                } catch (...) {
                    RTP_LLM_LOG_ERROR("accuracy check: t0 capture threw: unknown");
                    t0_capture_flag = false;
                }
            }
        }
        RETURN_IF_STATUS_ERROR(abortIfFailed(t0_capture_flag, "t0-capture"));

        // golden decodes its trajectory over the (maybe shrinking) active set and collects per-sequence tokens
        for (size_t step = 0; step < max_decode; ++step) {
            std::vector<size_t> active_seq_ids;
            for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
                if (step < scenario.seqs[seq_id].decode_steps) {
                    active_seq_ids.push_back(seq_id);
                }
            }
            bool golden_block_flag = true;
            if (root) {
                for (size_t seq_id : active_seq_ids) {
                    if (!golden_streams[seq_id]->incrKVBlock().ok()) {
                        RTP_LLM_LOG_ERROR(
                            "accuracy check: golden seq %zu incrKVBlock failed at decode step %zu", seq_id, step);
                        golden_block_flag = false;
                    }
                }
            }
            std::vector<GenerateStreamPtr> golden_decode_streams;
            if (root) {
                for (size_t seq_id : active_seq_ids) {
                    golden_decode_streams.push_back(golden_streams[seq_id]);
                }
            }
            RETURN_IF_STATUS_ERROR(forward(recorder,
                                           scenario_base_name,
                                           "golden",
                                           "decode_" + std::to_string(step),
                                           golden_decode_streams,
                                           active_seq_ids,
                                           golden_block_flag,
                                           scenario.gate_qualifying));
            // append active_seqs golden tokens on rank0
            if (root) {
                for (size_t seq_id : active_seq_ids) {
                    if (golden_decode_tokens[seq_id].size() < scenario.seqs[seq_id].decode_steps) {
                        auto tokens = golden_streams[seq_id]->currentExecuteTokens(0);
                        if (!tokens.empty()) {
                            golden_decode_tokens[seq_id].push_back(tokens[0]);
                        }
                    }
                }
            }
        }
        // check all golden decode completely
        bool golden_decode_flag = true;
        if (root) {
            for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
                if (golden_decode_tokens[seq_id].size() != scenario.seqs[seq_id].decode_steps) {
                    golden_decode_flag = false;
                }
            }
        }
        RETURN_IF_STATUS_ERROR(abortIfFailed(golden_decode_flag, "golden-decode"));

        // decode candidates: per step, re-fork the active set from golden's state (teacher forcing)
        for (size_t step = 0; step < max_decode; ++step) {
            std::vector<size_t> active_seq_ids;
            for (size_t seq_id = 0; seq_id < batch_size; ++seq_id) {
                if (step < scenario.seqs[seq_id].decode_steps) {
                    active_seq_ids.push_back(seq_id);
                }
            }
            const std::string decode_phase         = "decode_" + std::to_string(step);
            auto              decode_candidates_or = listCandidates(recorder, decode_phase);
            RETURN_IF_STATUS_OR_ERROR(decode_candidates_or);
            auto decode_candidates = std::move(decode_candidates_or).value();
            for (auto& impl_name : decode_candidates) {
                std::vector<GenerateStreamPtr>  decode_streams;
                std::vector<StreamReleaseGuard> decode_guards;
                bool                            build_decode_flag = buildBatch(
                    root,
                    active_seq_ids,
                    [&](size_t seq_id) -> absl::StatusOr<GenerateStreamPtr> {
                        const size_t history_len =
                            scenario.seqs[seq_id].prefix_len + scenario.seqs[seq_id].chunk_len + step;
                        torch::Tensor history;
                        if (step == 0) {
                            history = prompts[seq_id];
                        } else {
                            auto extra =
                                torch::from_blob(golden_decode_tokens[seq_id].data(), {(int64_t)step}, torch::kInt32);
                            history = torch::cat({prompts[seq_id], extra}, 0);
                        }
                        return makeForkStream(history,
                                              history_len,
                                              /*is_decode=*/true,
                                              golden_decode_tokens[seq_id][step],
                                              golden_streams[seq_id]);
                    },
                    decode_streams,
                    decode_guards,
                    impl_name,
                    decode_phase);
                RETURN_IF_STATUS_ERROR(forward(recorder,
                                               scenario_base_name,
                                               impl_name,
                                               decode_phase,
                                               decode_streams,
                                               active_seq_ids,
                                               build_decode_flag,
                                               scenario.gate_qualifying));
            }
        }
    }

    return absl::OkStatus();
}

size_t AccuracyChecker::computeTemporaryBlockNum(const ModelConfig&                   model_config,
                                                 const KVCacheConfig&                 kv_cache_config,
                                                 const std::vector<AccuracyScenario>& scenarios) {
    const size_t block_size      = model_config.attn_config.tokens_per_block;
    size_t       max_peak_blocks = 0;
    for (const auto& scenario : scenarios) {
        size_t plain_peak_blocks    = 0;
        size_t golden_decode_blocks = 0;
        size_t decode_fork_blocks   = 0;
        for (const auto& sequence : scenario.seqs) {
            const size_t plain_len = sequence.prefix_len + sequence.chunk_len;
            // golden + one full candidate
            plain_peak_blocks += 2 * ((plain_len + block_size - 1) / block_size);
            golden_decode_blocks += (plain_len + sequence.decode_steps + block_size - 1) / block_size;
            if (sequence.decode_steps > 0) {
                decode_fork_blocks += 1;
            }
        }
        max_peak_blocks = std::max({max_peak_blocks, plain_peak_blocks, golden_decode_blocks + decode_fork_blocks});
    }
    const int64_t reserve_block_ratio = kv_cache_config.reserve_block_ratio;
    const size_t  min_margin_blocks   = 16;
    // 1/10 => a 10% relative margin
    const size_t margin_ratio = 10;
    const size_t margin_blocks =
        std::max<size_t>(min_margin_blocks, (max_peak_blocks + margin_ratio - 1) / margin_ratio);
    const size_t base_blocks     = max_peak_blocks + margin_blocks;
    const size_t required_blocks = (reserve_block_ratio > 0 && reserve_block_ratio < 100) ?
                                       (base_blocks * 100) / (100 - static_cast<size_t>(reserve_block_ratio)) + 1 :
                                       base_blocks;
    return required_blocks + 1;  // + reserved block 0
}

absl::StatusOr<std::vector<std::string>> AccuracyChecker::finalizeGate(const py::object&  recorder,
                                                                       const std::string& domain) {
    const std::string        finalize_method = "finalize_" + domain + "_gate";
    std::vector<std::string> registry;
    std::vector<int32_t>     applicable_mask;
    std::vector<int32_t>     passed_mask;
    bool                     finalize_ok = true;
    try {
        py::gil_scoped_acquire gil;
        py::dict               local_gate =
            recorder.attr(finalize_method.c_str())(model_config_.attn_config.kv_cache_dtype).cast<py::dict>();
        finalize_ok     = local_gate["valid"].cast<bool>();
        registry        = local_gate["registry"].cast<std::vector<std::string>>();
        applicable_mask = local_gate["applicable_mask"].cast<std::vector<int32_t>>();
        passed_mask     = local_gate["passed_mask"].cast<std::vector<int32_t>>();
        finalize_ok =
            finalize_ok && !registry.empty() && applicable_mask.size() == registry.size()
            && passed_mask.size() == registry.size()
            && std::all_of(registry.begin(), registry.end(), [](const std::string& name) { return !name.empty(); })
            && std::unordered_set<std::string>(registry.begin(), registry.end()).size() == registry.size();
        for (size_t i = 0; finalize_ok && i < registry.size(); ++i) {
            const bool masks_are_binary =
                (applicable_mask[i] == 0 || applicable_mask[i] == 1) && (passed_mask[i] == 0 || passed_mask[i] == 1);
            finalize_ok = masks_are_binary && passed_mask[i] <= applicable_mask[i];
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("accuracy check %s failed: %s", finalize_method.c_str(), e.what());
        finalize_ok = false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("accuracy check %s failed: unknown", finalize_method.c_str());
        finalize_ok = false;
    }

    cudaSyncAndCheck();
    if (!worldAll(finalize_ok)) {
        RTP_LLM_LOG_ERROR("dynamic_%s_gate_failed reason=invalid_local_gate_result", domain.c_str());
        return absl::FailedPreconditionError(domain + " precision gate local result is invalid on at least one rank");
    }

    std::vector<int32_t> stacked;
    stacked.reserve(registry.size() * 2);
    stacked.insert(stacked.end(), applicable_mask.begin(), applicable_mask.end());
    stacked.insert(stacked.end(), passed_mask.begin(), passed_mask.end());
    auto sums_or = sumWorldMask(stacked, domain + " gate masks");
    RETURN_IF_STATUS_OR_ERROR(sums_or);
    auto sums = std::move(sums_or).value();

    std::vector<std::string> passed;
    const int32_t            world_size    = static_cast<int32_t>(parallelism_config_.world_size);
    const size_t             registry_size = registry.size();
    for (size_t i = 0; i < registry_size; ++i) {
        const int32_t applicable_sum = sums[i];
        const int32_t passed_sum     = sums[registry_size + i];
        if (applicable_sum == world_size && passed_sum == world_size) {
            passed.push_back(registry[i]);
        }
        if ((applicable_sum > 0 && applicable_sum < world_size) || (passed_sum > 0 && passed_sum < world_size)) {
            RTP_LLM_LOG_WARNING("dynamic_%s_gate_asymmetric backend=%s applicable=%d passed=%d world=%d",
                                domain.c_str(),
                                registry[i].c_str(),
                                applicable_sum,
                                passed_sum,
                                world_size);
        }
    }
    if (passed.empty()) {
        RTP_LLM_LOG_ERROR(
            "dynamic_%s_gate_failed reason=no_backend_passed registry_size=%zu", domain.c_str(), registry_size);
        return absl::FailedPreconditionError(domain + " precision gate has no globally passed backend");
    }
    RTP_LLM_LOG_INFO(
        "dynamic_%s_gate_passed passed=%zu registry_size=%zu", domain.c_str(), passed.size(), registry_size);
    return passed;
}

absl::StatusOr<AttentionGateResult> AccuracyChecker::runAll(Executor*                            executor,
                                                            ResourceContext&                     resource_context,
                                                            const std::vector<AccuracyScenario>& scenarios,
                                                            size_t                               max_forward_tokens,
                                                            bool finalize_prefill_gate) {
    executor_           = executor;
    resource_context_   = &resource_context;
    max_forward_tokens_ = max_forward_tokens;

    auto* normal_executor = dynamic_cast<NormalExecutor*>(executor_);
    RTP_LLM_CHECK_WITH_INFO(normal_executor, "accuracy check requires NormalExecutor");
    auto* py_model = dynamic_cast<PyWrappedModel*>(normal_executor->getModel());
    RTP_LLM_CHECK_WITH_INFO(py_model, "accuracy check requires PyWrappedModel");

    RecordingSession session{py_model, py::object()};
    {
        py::gil_scoped_acquire gil;
        auto                   recorder_module =
            py::module::import("rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder");
        session.recorder_ = recorder_module.attr("TensorRecorder")(
            py_model->getPyModel(), /*record_qkv=*/runtime_config_.enable_accuracy_check);
    }
    py_model->setAccuracyRecording(true);

    const int64_t record_begin_us = autil::TimeUtility::currentTimeInMicroSeconds();
    try {
        for (const auto& scenario : scenarios) {
            auto scenario_status = runScenario(session.recorder_, scenario);
            if (absl::IsFailedPrecondition(scenario_status)) {
                RTP_LLM_LOG_ERROR("dynamic_decode_gate_failed reason=candidate_schema_invalid detail=%s",
                                  scenario_status.ToString().c_str());
            }
            RETURN_IF_STATUS_ERROR(scenario_status);
        }
    } catch (const std::exception& e) {
        return absl::InternalError(std::string("accuracy check scenario threw: ") + e.what());
    } catch (...) {
        return absl::InternalError("accuracy check scenario threw: unknown");
    }
    const int64_t record_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - record_begin_us;
    RTP_LLM_LOG_INFO("accuracy check: recording completed in %.3f seconds", record_time_us / 1000000.0);

    // The recorder stays valid across both domains; every rank finalizes the
    // domains in the same order so the WORLD collectives stay aligned.
    AttentionGateResult gate_result;
    auto                decode_or = finalizeGate(session.recorder_, "decode");
    RETURN_IF_STATUS_OR_ERROR(decode_or);
    gate_result.decode_passed = std::move(decode_or).value();
    if (finalize_prefill_gate) {
        auto prefill_or = finalizeGate(session.recorder_, "prefill");
        RETURN_IF_STATUS_OR_ERROR(prefill_or);
        gate_result.prefill_passed = std::move(prefill_or).value();
    }
    return gate_result;
}

}  // namespace rtp_llm
