#pragma once

#include <chrono>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

inline constexpr char kMixedForceBatchGroupError[] =
    "force batch group must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kMixedExecutionModeBatchError[] =
    "model execution batch must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kDecodeRolePrefillOnlyError[] = "DECODE role does not support max_new_tokens == 0";

// The engine-loop step cadence for idle ranks (#66 v3 step lock / v3.1.1 skip
// pacing, perf-log R258/R298): the BatchDecodeScheduler's bounded wait and the
// v3.1 global-idle skip's sleep share ONE constant so the two pacing sites can
// never drift apart (pitfall #88: the 20 ms cadence used to be provided
// implicitly by the fake-stream forward; the skip removed that forward, so the
// pacing became explicit and must stay single-sourced).
inline constexpr std::chrono::milliseconds kStepLockInterval{20};

inline bool isPrefillOnly(const GenerateStreamPtr& stream) {
    return stream->generateConfig()->isPrefillOnly();
}

template<typename StreamRange>
bool hasMixedExecutionModes(const StreamRange& streams) {
    bool has_execution_mode = false;
    bool prefill_only       = false;
    for (const auto& stream : streams) {
        const bool stream_prefill_only = isPrefillOnly(stream);
        if (!has_execution_mode) {
            has_execution_mode = true;
            prefill_only       = stream_prefill_only;
        } else if (prefill_only != stream_prefill_only) {
            return true;
        }
    }
    return false;
}

}  // namespace rtp_llm
