#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams> propose_params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
         enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;

    KVCacheInfo  getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status step();
    absl::Status startLoop();
    int64_t      getLastScheduleTime() override;
    void         reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool updateEplbConfig(const EPLBConfig& config) override;
    void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) override;

private:
    void                            initScheduler();
    std::shared_ptr<GenerateStream> createStream(const std::shared_ptr<GenerateInput>& input);
    bool                            validateStreamForScheduling(const GenerateStreamPtr& stream) const;
    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens);
    WarmUpResult                    warmUp(const EngineInitParams& params);
    WarmUpResult                    prefillWarmUp(const EngineInitParams& params);
    WarmUpResult                    decodeWarmUp(const EngineInitParams& params);
    void                            initLoadBalance();
    absl::Status                    trySaveStepError() const;
    void                            loop();
    void                            normalizeSystemPromptCacheConfig();
    void                            initCacheManager(std::optional<WarmUpResult> warm_up_result);
    static void                     initializeAndPublishCacheManager(ResourceContext&                            resource_context,
                                                                     int&                                        kv_cache_group_num,
                                                                     RoleType                                    role_type,
                                                                     std::shared_ptr<KVCacheManager>             cache_manager,
                                                                     const std::function<bool(KVCacheManager&)>& initializer);
    absl::Status                    initSystemPrompt();
    std::shared_ptr<GenerateInput>  makeFakeInput(size_t seq_len);
    size_t                          getWarmUpInputLength() const;
    static size_t                   reserveStepForStream(size_t configured_reserve_step, bool is_prefill_only);
    static size_t warmUpReservedBlockCount(size_t seq_len, size_t reserve_tokens, size_t tokens_per_block);
    static bool   shouldEarlyReturnEmptyBatch(bool streams_empty, int64_t tp_size, bool enable_ffn_disaggregate);
    static bool   shouldAddMtpFakePrefill(bool has_prefill, bool use_batch_decode_scheduler, int64_t dp_size);
    void          mayAddFakeStream(std::list<GenerateStreamPtr>& streams);

    // #66 fix (R240/r122): unconditional per-step shape-consensus anchor.
    // Called at the top of every engine step (window iteration) with
    // dp_size > 1. Every rank contributes its local batch size to a
    // Max all-reduce over the DP group and the host blocks until the
    // k-th all-reduce completes on ALL ranks — the per-index matching
    // clamps cross-rank host drift to ~one step, so a batch-composition
    // transition can no longer desynchronize the engines' collective
    // sequences (the H-A' wedge: ranks diverging between finishing and
    // admitting, mismatching in-graph collective order forever).
    // The call itself is unconditional per step (r122 hard revision):
    // no rank-local "batch changed" gate — a conditional trigger would
    // re-create the same divergence on the consensus collective.
    // Returns the consensus (max) batch size; ranks below it pad with
    // fake streams at the window start so every rank replays the same
    // graph shape. Steady-state cost: one 1-element all-reduce + the
    // completion wait (~tens of us); the fill only runs when the local
    // batch size changes.
    // Returns (consensus_phase << 32) | consensus_bs (R251: phase = 1 when
    // any rank's batch contains a context/prefill stream this step).
    int64_t shapeConsensusAnchor(int64_t local_bs, bool local_has_prefill);
    void    padToConsensusBatch(std::list<GenerateStreamPtr>& streams, int64_t consensus_bs);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;
    bool isDSpark() override;

private:
    autil::ThreadPtr                              loop_thread_;
    std::atomic<bool>                             running_{false};
    std::unique_ptr<Executor>                     executor_;
    ModelConfig                                   model_config_;
    py::object                                    custom_output_selector_;
    ParallelismConfig                             parallelism_config;
    RuntimeConfig                                 runtime_config;
    EPLBConfig                                    eplb_config;
    PDSepConfig                                   pd_sep_config;
    ProfilingDebugLoggingConfig                   profiling_debug_logging_config;
    KVCacheConfig                                 kv_cache_config;
    CacheStoreConfig                              cache_store_config;
    FfnDisAggregateConfig                         ffn_disaggregate_config;
    ModelSpecificConfig                           model_specific_config;
    SpeculativeExecutionConfig                    sp_config;
    kmonitor::MetricsReporterPtr                  metrics_reporter_;
    std::unique_ptr<ProposeModelEngineInitParams> propose_params_;
    StepWindowProfiler                            step_profiler_;
    int                                           reserve_step_ = 0;
    // Multi-step decode window (env RTP_LLM_MULTI_STEP_MAX, 0 = off): the
    // schedule/mayAddFake/gather host chain runs once per window instead of
    // once per decode step. Decode-only, non-spec batches only; the window
    // breaks early when any stream finishes or errors so the state machine
    // re-runs on the next full schedule (the batch shape never changes
    // mid-window, so the graph replay stays stable).
    int multi_step_max_ = 0;
    // shapeConsensusAnchor state (#66 fix): 1-element Max-all-reduce buffer
    // (lazy CUDA int64) + the last locally-published bs (fills are rare —
    // only on batch-composition changes) + telemetry counters.
    torch::Tensor         shape_consensus_buf_;
    int64_t               shape_consensus_last_bs_    = -1;
    int64_t               shape_consensus_last_phase_ = -1;
    std::atomic<uint64_t> shape_consensus_calls_{0};
    std::atomic<uint64_t> shape_consensus_pads_{0};
    std::atomic<uint64_t> shape_consensus_phase_pads_{0};
};

}  // namespace rtp_llm
