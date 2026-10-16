#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

int CudaGraphRunner::multiStepN() const {
    // Env is read once per process (same read-once discipline as the
    // executor flag readers). Values outside [1,8] fall back to single-step.
    static const int n = [] {
        const char* env = std::getenv("RTP_LLM_MULTI_STEP");
        int         v   = env ? std::atoi(env) : 1;
        if (v < 1 || v > 8) {
            RTP_LLM_LOG_WARNING("[multi-step] RTP_LLM_MULTI_STEP=%d out of range [1,8], falling back to 1", v);
            return 1;
        }
        return v;
    }();
    return n;
}

torch::Tensor CudaGraphRunner::replayDecodeMultistep(int bs, int n) {
    const int key = multistepGraphKey(bs, n);
    RTP_LLM_CHECK_WITH_INFO(
        graph_instances_.count(key) > 0, "multi-step graph key %d (bs=%d n=%d) was not captured", key, bs, n);
    replayGraph(key);
    return graph_instances_[key].mem_hold_.multistep_tokens_;
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add key batch sizes up to 32
    for (int i : {1, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    // Add range from 48 to max_generate_batch_size, stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Channel-B multi-step capture (R205.1/R216): enabled when the model
    // exposes the multi-step body and env selects N > 1. Per bs we capture
    // the (bs, N) instance IN ADDITION to the single-step (bs) graph —
    // residual steps fall back to single-step replays (R209.1).
    const int  ms_n = multiStepN();
    const bool ms_enabled =
        ms_n > 1 && !py_multistep_method_.is_none() && num_tokens_per_bs_ == 1 && !is_prefill_cuda_graph_mode_;
    if (ms_enabled) {
        RTP_LLM_LOG_INFO("[multi-step] capturing %d-step variants for %zu batch sizes (RTP_LLM_MULTI_STEP=%d)",
                         ms_n,
                         capture_range_.size(),
                         ms_n);
    }
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs, enable_cuda_graph_debug_mode_);
        if (ms_enabled) {
            graph_instances_.try_emplace(multistepGraphKey(bs, ms_n), enable_cuda_graph_debug_mode_);
        }
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
        // capture-specific metadata above was written after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            prepareFmhaImpl(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(bs, "batch size");
        // Retain graph-owned buffers until the first replay is known complete.
        // A launch/synchronization failure must take the same fail-closed path
        // as a partially captured graph instead of deleting in-flight storage.
        capture_session_may_be_dirty_.store(false, std::memory_order_release);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);

        if (ms_enabled) {
            // The multi-step instance gets its own capture inputs + fmha impl:
            // its metadata buffers are mutated in-graph by the multi-step
            // body, so it must not share the single-step instance's impl (the
            // per-replay prep rewrites whichever impl the replayed graph owns).
            PyModelInputs ms_inputs;
            prepareCaptureInputs(ms_inputs, bs, bs * num_tokens_per_bs_);
            ms_inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
            refreshTaggedAttentionInputs(ms_inputs);
            const int ms_key                   = multistepGraphKey(bs, ms_n);
            graph_instances_[ms_key].mem_hold_ = createCaptureMemoryHold(ms_inputs, bs * num_tokens_per_bs_);
            graph_instances_[ms_key].mem_hold_.attn_pyobj_ =
                prepareFmhaImpl(graph_instances_[ms_key].mem_hold_.py_model_inputs_, true);
            graph_instances_[ms_key].mem_hold_.multistep_tokens_ =
                torch::zeros({ms_n, static_cast<int64_t>(bs)},
                             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
            captureDecodeMultistepOneBatchSize(bs, ms_n);
            cuda_graph::finish_capture_session();
            replayAndSyncCheck(ms_key, "multi-step");
            capture_session_may_be_dirty_.store(false, std::memory_order_release);
            RTP_LLM_LOG_INFO("capture success for multi-step (bs %d n %d)", bs, ms_n);
        }
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
