#include "rtp_llm/cpp/normal_engine/DeviceInputMode.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/HiddenStateCapturePolicy.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

using namespace std;

namespace rtp_llm {

namespace {

// M1-lite diagnostic counters (R201): throttled log lines distinguish
// "gate never passed" from "launched but zero effect" without any behavior
// change. First hit and every 100th hit log one line.
std::atomic<uint64_t> g_m1_lite_launched{0};
std::atomic<uint64_t> g_m1_lite_consumed{0};
std::atomic<uint64_t> g_m1_lite_discarded{0};
std::atomic<uint64_t> g_m1_gate_skip_near_finish{0};
std::atomic<uint64_t> g_m1_gate_skip_kv_boundary{0};
std::atomic<uint64_t> g_m1_gate_skip_device_state{0};

bool readEnvFlagOnce(const char* env_name, const char* log_tag, const char* label) {
    const char* env = std::getenv(env_name);
    const bool  on  = (env != nullptr && std::string(env) == "1");
    RTP_LLM_LOG_INFO("[%s] %s=%s -> %s=%d", log_tag, env_name, env ? env : "(unset)", label, static_cast<int>(on));
    return on;
}

void holdSamplerInputHostBuffers(TensorHolder& holder, const SamplerInputs& inputs) {
    holder.hold_host(inputs.token_ids);
    holder.hold_host(inputs.input_lengths);
    holder.hold_host(inputs.sequence_lengths);
    holder.hold_host(inputs.num_beams_in);
    holder.hold_host(inputs.num_beams_out);
    holder.hold_host(inputs.top_k);
    holder.hold_host(inputs.top_p);
    holder.hold_host(inputs.temperature);
    holder.hold_host(inputs.repetition_penalty);
    holder.hold_host(inputs.presence_penalty);
    holder.hold_host(inputs.frequency_penalty);
    holder.hold_host(inputs.no_repeat_ngram_size);
    holder.hold_host(inputs.do_sample);
    holder.hold_host(inputs.finished_mask);
    holder.hold_host(inputs.cum_log_probs);
}

}  // namespace

NormalExecutor::ModelFactory NormalExecutor::test_model_factory = nullptr;

NormalExecutor::~NormalExecutor() {
    cudaProfilerEnd();
}

NormalExecutor::NormalExecutor(const EngineInitParams&                params,
                               const std::shared_ptr<KVCacheManager>& cache_manager,
                               bool                                   warm_up,
                               bool                                   is_propose,
                               int                                    propose_model_index,
                               MlaOpsType                             mla_ops_type,
                               std::function<void()>                  profile_step_start,
                               std::function<void()>                  profile_step_finish):
    Executor(),
    cache_manager_(cache_manager),
    warm_up_(warm_up),
    use_all_gather_(params.moe_config.use_all_gather && !params.moe_config.use_deepep_low_latency),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    is_propose_(is_propose),
    propose_model_index_(propose_model_index),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    dispatch_runner_(cuda_graph::graphGetStreamFromPool(true)) {
    has_engram_         = !params.model_config_.attn_config.v41_kv_source_layer_ids.empty();
    enable_detail_log_  = params.profiling_debug_logging_config.enable_detail_log;
    tp_rank_            = params.parallelism_config.tp_rank;
    parallelism_config_ = params.parallelism_config;
    role_type_          = params.pd_sep_config.role_type;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, tp_rank_);
    if (params.profiling_debug_logging_config.enable_model_inputs_log) {
        model_inputs_logger_ =
            std::make_shared<ModelInputsLogger>(params.parallelism_config.world_rank,
                                                params.profiling_debug_logging_config.log_file_backup_count,
                                                metrics_reporter_);
    }

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int         first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        auto        moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        bool        is_gated_activation = params.model_config_.isGatedActivation();
        auto        moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

        expert_balancer_ = make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                                       params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                                       params.model_config_.num_layers,
                                                       moe_inter_size,
                                                       params.model_config_.hidden_size,
                                                       params.parallelism_config.ep_rank,
                                                       params.parallelism_config.ep_size,
                                                       params.parallelism_config.world_size,
                                                       params.py_eplb,
                                                       moe_weight_type,
                                                       params.model_config_.quant_algo,
                                                       metrics_reporter_,
                                                       params.eplb_config);
    }

    // Scheduler limits the number of streams, but a stream may fan out into more sampler rows
    // through num_return_sequences or variable beams. Preallocate the common case and grow safely.
    const auto initial_sampler_batch_size =
        static_cast<size_t>(std::max<int64_t>(1, params.runtime_config.max_generate_batch_size));
    sampler_.reset(new Sampler(SamplerInitParams{initial_sampler_batch_size, false}));

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ?
             std::make_optional(is_propose_ ? cache_manager->getMTPModuleGroupedCacheLayerLayout(propose_model_index_) :
                                              cache_manager->getMainModelGroupedCacheLayerLayout()) :
             std::nullopt,
         params.model_id,
         params.parallelism_config,
         params.hw_kernel_config,
         params.profiling_debug_logging_config,
         params.runtime_config,
         params.concurrency_config,
         params.sp_config,
         params.device_resource_config,
         mla_ops_type,
         params.model_config_.max_seq_len,
         params.model_config_.hidden_size,
         cache_manager,
         is_propose_ ? std::make_optional(propose_model_index_) : std::nullopt,
         params.model_config_.hc_mult});
    model_init_params.metrics_reporter = metrics_reporter_;
    if (!is_propose_ && !warm_up_) {
        model_init_params.eplb_monitor_config = params.eplb_config;
    }
    model_init_params.hidden_state_capture_layer_ids = selectHiddenStateCaptureLayerIds(
        is_propose_ ? HiddenStateCaptureModelRole::DRAFT : HiddenStateCaptureModelRole::TARGET,
        role_type_,
        warm_up_,
        params.model_config_.hidden_state_capture_layer_ids);
    model_init_params.hidden_state_capture_dtype = params.model_config_.hidden_state_capture_dtype;
#if USING_CUDA || USING_ROCM
    if (params.hw_kernel_config.enable_cuda_graph && model_init_params.kv_cache_layer_layout.has_value()) {
        const auto& model_cache_config =
            is_propose_ ? cache_manager->getMTPModuleCacheConfig(propose_model_index_) : cache_manager->cacheConfig();
        const size_t runtime_tokens_per_block = model_cache_config.seq_size_per_block;
        const size_t max_reserved_step        = params.sp_config.speculativeReserveStep();
        const auto&  topology                 = model_init_params.kv_cache_layer_layout->topology();
        RTP_LLM_CHECK_WITH_INFO(params.model_config_.max_seq_len > 0,
                                "CUDA graph max sequence length must be positive");
        const size_t max_seq_len = static_cast<size_t>(params.model_config_.max_seq_len);
        const size_t real_width =
            CudaGraphRunner::captureKernelBlockTableWidth(topology, max_seq_len, max_reserved_step);

        // Fake callers use one uniform physical count for every group: decode
        // warmup reserves seq+reserve with model pages, generation-prefill
        // warmup reserves the same fake input length with cache pages, and idle batches use one.
        const size_t warmup_span = params.model_config_.attn_config.tokens_per_block;
        RTP_LLM_CHECK_WITH_INFO(warmup_span > 0 && runtime_tokens_per_block > 0,
                                "CUDA graph fake block spans must be positive");
        const size_t warmup_len           = warmUpInputLength(max_seq_len, max_reserved_step);
        const size_t decode_tokens        = warmup_len + max_reserved_step;
        const size_t decode_warmup_blocks = decode_tokens / warmup_span + (decode_tokens % warmup_span != 0);
        const size_t prefill_warmup_blocks =
            warmup_len / runtime_tokens_per_block + (warmup_len % runtime_tokens_per_block != 0);
        const size_t fake_count = std::max<size_t>({1, decode_warmup_blocks, prefill_warmup_blocks});
        const size_t fake_width = CudaGraphRunner::captureKernelBlockTableWidth(topology, fake_count);
        RTP_LLM_CHECK_WITH_INFO(std::max(real_width, fake_width) <= std::numeric_limits<int64_t>::max(),
                                "CUDA graph kernel block table width exceeds int64 range");
        model_init_params.kernel_block_table_width = std::max(real_width, fake_width);
    }
#endif

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ?
                                   (is_propose_ ? cache_manager->getMTPModuleCacheConfig(propose_model_index_) :
                                                  cache_manager->cacheConfig()) :
                                   CacheConfig();

    const int async_worker_count =
        tp_rank_ > 0 || warm_up_ || is_propose_ ? 0 : params.runtime_config.output_dispatcher_worker_count;
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(params.model_config_,
                                                                 params.pd_sep_config,
                                                                 params.profiling_debug_logging_config,
                                                                 cache_config,
                                                                 warm_up_,
                                                                 async_worker_count));
    LogitsProcessorFactory::init(params.model_config_,
                                 params.grammar_config,
                                 params.sp_config.tree_decode_config,
                                 params.parallelism_config.tp_rank == 0 && !warm_up_ ? metrics_reporter_ : nullptr);
    cudaProfilerBegin();
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    const int64_t process_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    if (schedule_time_us <= 0) {
        schedule_time_us = process_start_time_us;
    }
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    auto                           tps_active_guard =
        tps_reporter_.makeActiveGuard(metrics_reporter_ && tp_rank_ == 0 && !warm_up_ && !streams.empty());
    auto wall_tps_active_guard =
        wall_tps_reporter_.makeActiveGuard(metrics_reporter_ && tp_rank_ == 0 && !warm_up_ && !streams.empty());
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    RTP_LLM_PROFILE_FUNCTION();
    // Cap outstanding stream-async bookkeeping to one step unless DROP_BROAD_SYNC is on.
    // Still sync when gatherModelInput lacks NormalAsyncDeviceState; host
    // token/seq_len fallbacks race the previous worker.
    bool worker_synced = false;
    if (useStreamAsync() && !useDropBroadSync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch(stream_count=%zu)", streams.size());
        dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
        worker_synced = true;
    }

    StreamGroups stream_groups(streams);
    prepareGrpcNormalDeviceState(stream_groups);

    if (useStreamAsync() && useDropBroadSync() && !gatherCanUseDeviceState(stream_groups)) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch(stream_count=%zu)", streams.size());
        dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
        worker_synced = true;
        // Rebuild StreamGroups after waiting: cached maxSeqLen/batch sizes can
        // be stale while the previous worker mutates GenerateStream host state.
        stream_groups = StreamGroups(streams);
        prepareGrpcNormalDeviceState(stream_groups);
    }

    bool m1_consumed = false;
    if (m1_pre_launched_) {
        // M1-lite: the previous process() already gathered, synced, and
        // launched this step's forward (launchNextStepEarly) before its
        // dispatch tail. Consume the pending pair and skip straight to the
        // sampler — but only when the live stream set is exactly the set the
        // forward was launched for (composition-safety guard (a)); otherwise
        // discard the orphaned pair and fall through to the ordinary path.
        RTP_LLM_PROFILE_SCOPE("executor.m1_lite.consume_pre_launched");
        const bool composition_match =
            m1_pre_launched_streams_.size() == streams.size()
            && std::equal(m1_pre_launched_streams_.begin(), m1_pre_launched_streams_.end(), streams.begin());
        if (composition_match) {
            model_input             = std::move(m1_pending_input_);
            model_output            = std::move(m1_pending_output_);
            m1_consumed             = true;
            const uint64_t consumed = g_m1_lite_consumed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (consumed == 1 || consumed % 100 == 0) {
                RTP_LLM_LOG_INFO("[normal-m1-lite] consumed=%lu launched=%lu discarded=%lu",
                                 (unsigned long)consumed,
                                 (unsigned long)g_m1_lite_launched.load(std::memory_order_relaxed),
                                 (unsigned long)g_m1_lite_discarded.load(std::memory_order_relaxed));
            }
            if (profile_step_start_) {
                profile_step_start_();
            }
            buffer_holder_.release();
        } else {
            RTP_LLM_LOG_WARNING("[normal-m1-lite] pre-launched pair discarded: stream set changed");
            g_m1_lite_discarded.fetch_add(1, std::memory_order_relaxed);
            m1_pending_input_  = GptModelInputs();
            m1_pending_output_ = GptModelOutputs();
        }
        m1_pre_launched_streams_.clear();
        m1_pre_launched_ = false;
    }
    if (!m1_consumed) {
        {
            RTP_LLM_PROFILE_SCOPE("executor.gather_model_input");
            int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
            auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
            RETURN_IF_STATUS_OR_ERROR(model_input_status);
            model_input                              = std::move(model_input_status.value());
            executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }
        {
            RTP_LLM_PROFILE_SCOPE("executor.tp_sync_input");
            int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;

            // Move metadata to CUDA before tpSyncModelInputs so broadcasts use the
            // GPU packed-buffer path instead of CPU execBroadcastCpu/unpack loops.
            ensureModelInputsOnCuda(model_input, "process.before_tp_sync");

            tpSyncModelInputs(model_input, parallelism_config_);
            if (model_input.skip_run) {
                return absl::OkStatus();
            }

            // Post-sync guard for fields not covered by the root device bitmap
            // (notably sequence_lengths_plus_1) before model_->forward.
            ensureModelInputsOnCuda(model_input, "process.after_tp_sync");

            executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }

        if (profile_step_start_) {
            profile_step_start_();
        }

        // make sure last model input is released before forward
        // TensorHolder release point (NormalExecutor): after current TP sync has
        // consumed model-input H2D staging, advance the one-extra-round hold window
        // for model-input sources and previous sampler-input host tensors.
        buffer_holder_.release();
        // PyWrappedModel TensorHolder release point for the previous model forward.
        model_->releaseBuffers();

        {
            // update kv cache
            if (model_input.kv_cache_update_mapping.defined() && model_input.kv_cache_update_mapping.numel() > 0) {
                RTP_LLM_PROFILE_SCOPE("executor.kv_cache_update");
                RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(cache_manager_),
                                        "cache update mapping requires a cache manager");
                cache_manager_->blockBatchCopyForForward(model_input.kv_cache_update_mapping,
                                                         model_input.kv_cache_group_tags);
            }
        }
        {
            bool force = tp_rank_ == 0 && enable_detail_log_;
            if (force) {
                RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
            } else {
                RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
            }
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.model_forward(ctx_batch=%zu,gen_batch=%zu,tokens=%zu,max_seq=%zu)",
                                          stream_groups.totalContextBatchSize(),
                                          stream_groups.totalDecodeBatchSize(),
                                          stream_groups.modelExecuteTokenSize(),
                                          stream_groups.maxSeqLen());
            int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            if (model_inputs_logger_) {
                const auto role = is_propose_ ? ModelInputsModelRole::DRAFT : ModelInputsModelRole::NORMAL;
                model_inputs_logger_->log(model_input, role, model_->model_id_);
            }
            model_output                        = std::move(model_->forward(model_input));
            executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (auto capture_error = model_->takeDeferredHiddenStateCaptureError(); capture_error.has_value()) {
        cudaSyncAndCheck();
        model_->releaseBuffers();
        for (const auto& stream : streams) {
            stream->reportError(ErrorCode::EXECUTION_EXCEPTION, *capture_error);
        }
        if (profile_step_finish_) {
            profile_step_finish_();
        }
        return absl::OkStatus();
    }

    if (tp_rank_ > 0 || warm_up_ || streams.size() == 0) {
        cudaSyncAndCheck();
        model_->releaseBuffers();
        if (profile_step_finish_) {
            profile_step_finish_();
        }
        return absl::OkStatus();
    }

    // Gather selects the execution mode from non-errored streams in model order.
    // Use that same decision: streams.front() may already have failed or be reordered.
    const bool zero_token_batch = model_input.skip_lm_head;
    if (!zero_token_batch) {
        RTP_LLM_PROFILE_SCOPE("executor.sampler_forward");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        // Sampler input still reads CPU stream state mutated by the previous
        // worker. DROP_BROAD_SYNC therefore needs this narrow sync unless an
        // earlier host-fallback gather already waited.
        if (useStreamAsync() && useDropBroadSync() && !worker_synced) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch_pre_sampler(stream_count=%zu)", streams.size());
            dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
            worker_synced = true;
            // Rebuild after waiting so sampler buffers use the current seqLength
            // instead of appending one column behind.
            stream_groups = StreamGroups(streams);
        }

        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // Stream-async dispatch is opt-in and only useful for decode-only batches:
    // the next iteration can overlap forward prep with worker D2H/update work.
    // Prefill gets no overlap benefit and would only add worker startup cost.
    const bool is_decode_only = stream_groups.totalContextBatchSize() == 0 && stream_groups.totalDecodeBatchSize() > 0;

    // M1-lite launch-early (R192 M1, bs1-decode staging; RTP_LLM_M1_LITE
    // default off): with the sampler's GPU work enqueued, publish the
    // device-side state from this step's sampler output and launch the next
    // step's forward BEFORE this step's dispatch host tail. Launch timing
    // moves from "each rank's host-chain end" to the sampler completion
    // point — the graphs' lockstep window — absorbing the inter-rank launch
    // skew the first in-graph collective otherwise pays (R190.1). The
    // dispatch's later D2H blocks on the launched graph's completion, which
    // is harmless: nothing host-side gates until the next sampler's
    // stream-state updates land in this dispatch.
    //
    // Composition-safety (R200 correction 3): the pre-launched forward is
    // bound to THIS step's stream set. Two guards keep a composition change
    // from corrupting the next step:
    //   (b) near-finish valve — skip the launch when any stream is within 2
    //       tokens of its length limit, so the batch the launched forward
    //       assumes cannot dissolve before it is consumed. The EOS floor
    //       (min_new_tokens) makes this sufficient for fixed-length decode
    //       (perf targets set min=max); general serving additionally needs
    //       the phase-3 release gating.
    //   (a) consume-side set match — the next process() discards the pending
    //       pair unless the live stream set is exactly the launched set.
    if (useM1Lite() && is_decode_only && !zero_token_batch && !profile_step_start_) {
        publishNormalDeviceState(stream_groups, sampler_output);
        bool can_launch = gatherCanUseDeviceState(stream_groups);
        if (!can_launch) {
            g_m1_gate_skip_device_state.fetch_add(1, std::memory_order_relaxed);
        }
        if (can_launch) {
            for (const auto& stream : stream_groups.decodeStreams()) {
                const auto*   cfg = stream->generateInput() ? stream->generateInput()->generate_config.get() : nullptr;
                const int64_t max_new = cfg ? cfg->max_new_tokens : 0;
                if (max_new > 0 && static_cast<int64_t>(stream->outputTokenLen()) + 2 >= max_new) {
                    can_launch = false;
                    g_m1_gate_skip_near_finish.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        }
        if (can_launch && cache_manager_ != nullptr) {
            // KV-boundary conservative valve (R192 safety): the next gather
            // runs before this step's dispatch writes the stream state, so
            // its kv-table copy would read the pre-allocation vector. Skip
            // the pre-launch around block boundaries (the ordinary path
            // handles them); the cost is ~2 steps per block period.
            const size_t block = cache_manager_->cacheConfig().seq_size_per_block;
            if (block > 0) {
                for (const auto& stream : stream_groups.decodeStreams()) {
                    const int next_seq = stream->getNormalAsyncDeviceState().next_real_seq_len;
                    if (next_seq > 0 && static_cast<size_t>(next_seq) % block <= 1) {
                        can_launch = false;
                        g_m1_gate_skip_kv_boundary.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }
        if (can_launch) {
            auto launch_status = launchNextStepEarly(stream_groups);
            if (launch_status.ok()) {
                m1_pre_launched_streams_.assign(streams.begin(), streams.end());
                m1_pre_launched_        = true;
                const uint64_t launched = g_m1_lite_launched.fetch_add(1, std::memory_order_relaxed) + 1;
                if (launched == 1 || launched % 100 == 0) {
                    RTP_LLM_LOG_INFO("[normal-m1-lite] launched=%lu (gate skips: near_finish=%lu kv_boundary=%lu "
                                     "device_state=%lu)",
                                     (unsigned long)launched,
                                     (unsigned long)g_m1_gate_skip_near_finish.load(std::memory_order_relaxed),
                                     (unsigned long)g_m1_gate_skip_kv_boundary.load(std::memory_order_relaxed),
                                     (unsigned long)g_m1_gate_skip_device_state.load(std::memory_order_relaxed));
                }
            } else {
                // Conservative fallback: the ordinary next-step path runs.
                RTP_LLM_LOG_WARNING("[normal-m1-lite] early launch fell back: %s", launch_status.ToString().c_str());
            }
        }
    }

    StreamGroups::TokenCountsByPriority token_counts_by_priority;
    if (metrics_reporter_ && tp_rank_ == 0) {
        token_counts_by_priority = stream_groups.tokenCountsByPriority();
    }
    if (useStreamAsync() && is_decode_only && !zero_token_batch) {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output(stream_async)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        // Record as soon as sampler outputs are valid so the worker waits with
        // cudaStreamWaitEvent before pinned D2H staging.
        auto sampler_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        sampler_event->record(cuda_graph::graphGetCurrentStream());

        // Metrics and KV release stay on the main thread; dispatch_output_us
        // now measures launch cost, while worker time is in async_runner.thread.
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        int64_t tps_execute_time_us           = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - process_start_time_us;
        }
        reportMetrics(stream_groups, executor_collector, tps_collector, tps_execute_time_us, token_counts_by_priority);

        return dispatchOutputAsync(stream_groups,
                                   std::move(model_output),
                                   std::move(sampler_output),
                                   std::move(sampler_event),
                                   profile_step_finish_);
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output");
        int64_t      start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        absl::Status result;
        if (zero_token_batch) {
            cudaSyncAndCheck();
            result = batch_stream_processor_->dispatchPrefillOnly(stream_groups,
                                                                  model_output.generation_prefill_cuda_graph_status);
        } else {
            MergedOutput merge_outputs{std::move(model_output), std::move(sampler_output)};
            if (useDeviceInput()) {
                publishNormalDeviceState(stream_groups, merge_outputs.sampler_output);
            }
            result = batch_stream_processor_->dispatch(stream_groups, merge_outputs);
        }
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        int64_t tps_execute_time_us           = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - process_start_time_us;
        }
        reportMetrics(stream_groups, executor_collector, tps_collector, tps_execute_time_us, token_counts_by_priority);

        if (profile_step_finish_) {
            profile_step_finish_();
        }
        return result;
    }
}

bool NormalExecutor::useM1Lite() const {
    static const bool enabled = []() { return readEnvFlagOnce("RTP_LLM_M1_LITE", "normal-m1-lite", "useM1Lite"); }();
    // The device-state gather path is a hard prerequisite: without it the
    // next-step gather would need the D2H'd stream state, which the reorder
    // has deliberately not waited for.
    return enabled && useDeviceInput();
}

absl::Status NormalExecutor::launchNextStepEarly(const StreamGroups& stream_groups) {
    // Same-composition pre-commit (R192 M1): the next step runs these exact
    // streams. Composition changes surface at the next schedule() and take
    // effect one step later via the m1 gate re-evaluation — the fallback is
    // the ordinary path, never a mixed batch.
    RTP_LLM_PROFILE_SCOPE("executor.m1_lite.launch_next_step_early");
    auto gather_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(gather_status);
    GptModelInputs next_input = std::move(gather_status.value());
    next_input.skip_run       = false;

    // Move metadata to CUDA before the replay so broadcasts use the GPU path.
    ensureModelInputsOnCuda(next_input, "process.m1_lite_next");
    tpSyncModelInputs(next_input, parallelism_config_);

    // KV update must precede the forward, mirroring the ordinary path.
    if (next_input.kv_cache_update_mapping.defined() && next_input.kv_cache_update_mapping.numel() > 0) {
        RTP_LLM_PROFILE_SCOPE("executor.m1_lite.kv_cache_update");
        RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(cache_manager_), "cache update mapping requires a cache manager");
        cache_manager_->blockBatchCopyForForward(next_input.kv_cache_update_mapping, next_input.kv_cache_group_tags);
    }

    model_->releaseBuffers();
    m1_pending_output_ = std::move(model_->forward(next_input));
    m1_pending_input_  = std::move(next_input);
    return absl::OkStatus();
}

void NormalExecutor::reportMetrics(const StreamGroups&                        stream_groups,
                                   RtpLLMExecutorMetricsCollector&            executor_collector,
                                   RtpLLMTokenPSMetricsCollector&             tps_collector,
                                   int64_t                                    tps_execute_time_us,
                                   const StreamGroups::TokenCountsByPriority& token_counts_by_priority) {
    if (tp_rank_ > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context  = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context  = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context         = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                   stream_groups.contextExecuteTokenSizeWithCache(),
                                   stream_groups.totalDecodeBatchSize(),
                                   stream_groups.modelExecuteTokenSize(),
                                   tps_execute_time_us);
        tps_collector.addTokenSizeByPriority(token_counts_by_priority, tps_execute_time_us);
        tps_reporter_.report(&tps_collector);
        wall_tps_reporter_.report(&tps_collector);
    }
}

bool NormalExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

bool NormalExecutor::useStreamAsync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_STREAM_ASYNC", "normal-stream-async", "useStreamAsync");
    }();
    return enabled;
}

bool NormalExecutor::useDropBroadSync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DROP_BROAD_SYNC", "normal-drop-broad-sync", "enabled");
    }();
    return enabled;
}

bool NormalExecutor::useDeviceInput() const {
    static const bool enabled = []() {
        const bool  on           = deviceInputEnabled();
        const char* stream_async = std::getenv("RTP_LLM_STREAM_ASYNC");
        const char* device_input = std::getenv("RTP_LLM_DEVICE_INPUT");
        RTP_LLM_LOG_INFO("[normal-device-input] RTP_LLM_STREAM_ASYNC=%s RTP_LLM_DEVICE_INPUT=%s -> enabled=%d",
                         stream_async ? stream_async : "(unset)",
                         device_input ? device_input : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

bool NormalExecutor::checkDeviceInput() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DEVICE_INPUT_CHECK", "normal-device-input", "enabled");
    }();
    return enabled;
}

void NormalExecutor::ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag) {
    if (!useDeviceInput()) {
        return;
    }

    auto to_cuda = [this, tag](torch::Tensor& tensor, const char* name) {
        if (!tensor.defined() || tensor.is_cuda()) {
            return;
        }
        if (tensor.numel() == 0) {
            tensor = torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
            return;
        }
        if (!tensor.is_pinned()) {
            // Non-pinned CPU H2D forces a synchronous copy, defeating the
            // whole point of pre-staging tensors to CUDA. Fall back rather
            // than abort, but warn loudly so we can fix the producer.
            RTP_LLM_LOG_WARNING(
                "[normal-device-input] %s.%s is CPU but not pinned; H2D falls back to blocking copy", tag, name);
            tensor = tensor.to(torch::kCUDA);
            return;
        }
        // non_blocking=true requires the source tensor to outlive the copy;
        // the holder keeps a reference until the next process() iteration
        // releases it (after the broadcast has consumed the tensor).
        buffer_holder_.hold_host(tensor);
        tensor = tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };

    to_cuda(model_input.combo_tokens, "combo_tokens");
    to_cuda(model_input.input_lengths, "input_lengths");
    to_cuda(model_input.sequence_lengths, "sequence_lengths");
    to_cuda(model_input.prefix_lengths, "prefix_lengths");
    to_cuda(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    to_cuda(model_input.lm_output_indexes, "lm_output_indexes");
    to_cuda(model_input.custom_output_indexes, "custom_output_indexes");
    checkModelInputsOnCuda(model_input, tag);
}

void NormalExecutor::checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const {
    if (!checkDeviceInput()) {
        return;
    }
    auto check = [tag](const torch::Tensor& tensor, const char* name) {
        if (!tensor.defined()) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(),
                                "[normal-device-input] %s.%s expected CUDA tensor, got device=%s numel=%ld",
                                tag,
                                name,
                                tensor.device().str().c_str(),
                                tensor.numel());
    };
    check(model_input.combo_tokens, "combo_tokens");
    check(model_input.input_lengths, "input_lengths");
    check(model_input.sequence_lengths, "sequence_lengths");
    check(model_input.prefix_lengths, "prefix_lengths");
    check(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    check(model_input.lm_output_indexes, "lm_output_indexes");
    check(model_input.custom_output_indexes, "custom_output_indexes");
}

bool NormalExecutor::gatherCanUseDeviceState(const StreamGroups& stream_groups) const {
    // Decode-only batch is the only shape the device-state path supports.
    if (stream_groups.totalContextBatchSize() != 0 || stream_groups.totalDecodeBatchSize() == 0) {
        return false;
    }
    for (const auto& stream : stream_groups.decodeStreams()) {
        // Use config-only proxies for the batch-1 check. currentBatchSize() reads
        // outputTokenLen() → seqLength() which itself races with the worker we're
        // trying to decide whether to skip syncing.
        if (stream->hasNumBeams() || stream->numReturnSequences() > 1) {
            return false;
        }
        bool has_blocking_stateful_processor = false;
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor != nullptr && processor->isStateful() && !processor->supportsNormalAsyncDeviceState()) {
                has_blocking_stateful_processor = true;
                break;
            }
        }
        // NormalAsyncDeviceState mirrors token ids and sequence length. Stateful
        // processors that cannot publish their own device-side next-step state
        // still need the old wait before the next sampler consumes logits.
        if (has_blocking_stateful_processor && stream->hasPendingAsyncBookkeeping()) {
            return false;
        }
        const auto& state = stream->getNormalAsyncDeviceState();
        if (!state.last_sample_token_gpu.defined() || !state.last_sample_token_gpu.is_cuda()
            || !state.next_seq_len_gpu.defined() || !state.next_seq_len_gpu.is_cuda()) {
            return false;
        }
    }
    return true;
}

void NormalExecutor::prepareGrpcNormalDeviceState(const StreamGroups& stream_groups) {
    // publishNormalDeviceState (the per-step refresher) is gated on
    // useDeviceInput(); without the same gate here the published snapshot goes
    // stale and GenerateStateMachine overrides incrKVBlock with an old seq_len,
    // skipping block-boundary allocation on PD decode.
    if (!useDeviceInput()) {
        return;
    }
    if (role_type_ != RoleType::DECODE || stream_groups.totalContextBatchSize() != 0
        || stream_groups.totalDecodeBatchSize() == 0) {
        return;
    }

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (const auto& stream : stream_groups.decodeStreams()) {
        if (!stream->consumeGrpcNormalDeviceStatePending()) {
            continue;
        }
        const auto& state = stream->getNormalAsyncDeviceState();
        if (state.last_sample_token_gpu.defined() && state.last_sample_token_gpu.is_cuda()
            && state.next_seq_len_gpu.defined() && state.next_seq_len_gpu.is_cuda()) {
            continue;
        }
        if (stream->outputTokenLen() != 1) {
            continue;
        }
        const auto current_batch_size = stream->currentBatchSize();
        if (current_batch_size != 1) {
            RTP_LLM_LOG_WARNING("[normal-device-state] skip grpc publish: stream=%ld cur_bs=%d",
                                stream->streamId(),
                                current_batch_size);
            continue;
        }

        auto current_tokens = stream->currentExecuteTokens(0);
        if (current_tokens.size() != 1) {
            RTP_LLM_LOG_WARNING("[normal-device-state] skip grpc publish: stream=%ld token_count=%zu",
                                stream->streamId(),
                                current_tokens.size());
            continue;
        }

        const auto seq_length = stream->seqLength();
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .epoch                 = 0,
            .last_sample_token_gpu = torch::full({1}, static_cast<int64_t>(current_tokens[0]), cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(seq_length), cuda_i32),
            .last_real_seq_len     = seq_length,
            .next_real_seq_len     = seq_length,
        });
    }
}

void NormalExecutor::publishNormalDeviceState(const StreamGroups& stream_groups, const SamplerOutput& sampler_output) {
    RTP_LLM_PROFILE_SCOPE("executor.publish_normal_device_state");

    auto all_streams = stream_groups.allStreams();
    if (all_streams.empty()) {
        return;
    }

    auto clear_states = [&all_streams]() {
        for (auto& stream : all_streams) {
            stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{});
        }
    };

    const auto& token_ids = sampler_output.token_ids;
    if (!token_ids.defined() || token_ids.numel() == 0 || token_ids.dim() < 1) {
        RTP_LLM_LOG_WARNING("[normal-device-state] skip publish: token_ids undefined/empty");
        clear_states();
        return;
    }

    for (const auto& stream : all_streams) {
        if (stream->currentBatchSize() != 1 || stream->nextBatchSize() != 1) {
            RTP_LLM_LOG_WARNING(
                "[normal-device-state] skip publish: only batch-1 decode is supported, stream=%ld cur_bs=%d next_bs=%d",
                stream->streamId(),
                stream->currentBatchSize(),
                stream->nextBatchSize());
            clear_states();
            return;
        }
    }

    const auto    cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor token_ids_gpu =
        (token_ids.is_cuda() && token_ids.scalar_type() == torch::kInt32) ? token_ids : token_ids.to(cuda_i32);
    const int64_t token_rows = token_ids_gpu.size(0);
    if (token_rows < static_cast<int64_t>(all_streams.size())) {
        RTP_LLM_LOG_WARNING(
            "[normal-device-state] skip publish: token_ids rows=%ld stream_count=%zu", token_rows, all_streams.size());
        clear_states();
        return;
    }

    int64_t batch_idx_out = 0;
    for (auto& stream : all_streams) {
        torch::Tensor last_sample_token_gpu;
        if (token_ids_gpu.dim() == 1) {
            last_sample_token_gpu = token_ids_gpu.narrow(0, batch_idx_out, 1).to(torch::kInt32);
        } else {
            const int64_t last_col = token_ids_gpu.size(-1) - 1;
            last_sample_token_gpu =
                token_ids_gpu.narrow(0, batch_idx_out, 1).select(-1, last_col).reshape({1}).to(torch::kInt32);
        }

        // Mirror next_seq_len_gpu on host for the next iter's scheduler.
        // Fall back to live seqLength only on first publish (no prior worker).
        const auto& prev_state = stream->getNormalAsyncDeviceState();
        const int   cur_real_seq_len =
            prev_state.next_real_seq_len > 0 ? prev_state.next_real_seq_len : stream->seqLength();

        torch::Tensor cur_seq_len_gpu;
        const auto&   prev_next_seq_len = prev_state.next_seq_len_gpu;
        if (prev_next_seq_len.defined() && prev_next_seq_len.is_cuda()) {
            cur_seq_len_gpu = prev_next_seq_len;
        } else {
            cur_seq_len_gpu = torch::full({1}, static_cast<int64_t>(cur_real_seq_len), cuda_i32);
        }

        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor != nullptr && processor->supportsNormalAsyncDeviceState()) {
                processor->prepareNormalAsyncUpdate(last_sample_token_gpu, 1);
            }
        }

        GenerateStream::NormalAsyncDeviceState state;
        state.last_sample_token_gpu = std::move(last_sample_token_gpu);
        state.next_seq_len_gpu      = (cur_seq_len_gpu + 1).to(torch::kInt32);
        if (has_engram_) {
            torch::Tensor previous = prev_state.engram_token_window_gpu;
            if (!previous.defined()) {
                // The first publication (including PD admission) owns stable
                // host history. Later rounds use the immutable GPU tail.
                auto                 history = stream->completeTokenIdsVec(0);
                std::vector<int32_t> tail(3, -1);
                for (int lag = 0; lag < 3 && lag < static_cast<int>(history.size()); ++lag) {
                    tail[lag] = history[history.size() - 1 - lag];
                }
                previous = torch::tensor(tail, cuda_i32).reshape({1, 3});
            }
            state.engram_token_window_gpu =
                torch::cat({state.last_sample_token_gpu.reshape({1, 1}), previous.narrow(1, 0, 3)}, 1);
        }
        state.last_real_seq_len = cur_real_seq_len;
        state.next_real_seq_len = cur_real_seq_len + 1;
        stream->setNormalAsyncDeviceState(std::move(state));
        batch_idx_out += 1;
    }
}

absl::Status NormalExecutor::dispatchOutputAsync(const StreamGroups&           stream_groups,
                                                 GptModelOutputs               model_output,
                                                 SamplerOutput                 sampler_output,
                                                 std::shared_ptr<torch::Event> sampler_event,
                                                 std::function<void()>         profile_step_finish) {
    RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_async");

    if (useDeviceInput()) {
        publishNormalDeviceState(stream_groups, sampler_output);
    }

    auto* processor           = batch_stream_processor_.get();
    auto  stream_groups_copy  = stream_groups;
    auto  model_output_copy   = std::move(model_output);
    auto  sampler_output_copy = std::move(sampler_output);

    // Claim each stream's KV resource before handing work to the runner so
    // releaseResource defers while the worker reads/updates those blocks.
    auto streams_for_inc = stream_groups_copy.allStreams();
    for (auto& s : streams_for_inc) {
        s->incPendingAsyncBookkeeping();
    }

    dispatch_runner_.launch([processor,
                             stream_groups_copy  = std::move(stream_groups_copy),
                             model_output_copy   = std::move(model_output_copy),
                             sampler_output_copy = std::move(sampler_output_copy),
                             sampler_event]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_worker");

        auto worker_streams = stream_groups_copy.allStreams();

        // RAII: every captured stream is decremented exactly once, and the
        // value capture keeps GenerateStreamPtr alive until after dec runs.
        auto dec_guard = std::shared_ptr<void>(nullptr, [worker_streams](void*) {
            for (auto& s : worker_streams) {
                s->decPendingAsyncBookkeepingAndMaybeRelease();
            }
        });

        // Queue a stream wait, then do pinned D2H on the worker stream while
        // the main thread continues to the next gather.
        if (sampler_event) {
            sampler_event->block(cuda_graph::graphGetCurrentStream());
        }

        auto status =
            processor->dispatch(stream_groups_copy, {std::move(model_output_copy), std::move(sampler_output_copy)});
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[normal-stream-async] dispatch (worker) failed: %s", status.ToString().c_str());
        }
        // dec_guard destructs here, dec'ing each stream's pending count.
    });

    // Kineto requires profiler enable/disable on the same thread. Keep finish
    // on the engine loop thread even when output dispatch runs asynchronously.
    if (profile_step_finish) {
        profile_step_finish();
    }

    return absl::OkStatus();
}

}  // namespace rtp_llm
