#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

namespace py = pybind11;

namespace rtp_llm {

struct StreamReleaseGuard;

struct AccuracySeq {
    size_t prefix_len;
    size_t chunk_len;
    size_t decode_steps;
};

struct AccuracyScenario {
    std::vector<AccuracySeq> seqs;
    bool                     gate_qualifying = true;
    uint64_t                 prompt_seed     = 0;
};

enum class DecodeBackendGateStatus {
    NOT_REQUESTED,
    READY,
    UNAVAILABLE,
};

struct DecodeBackendGate {
    DecodeBackendGateStatus  status = DecodeBackendGateStatus::NOT_REQUESTED;
    std::string              reason;
    int64_t                  protocol_version     = 0;
    int64_t                  registry_fingerprint = 0;
    int64_t                  manifest_fingerprint = 0;
    std::vector<std::string> applicable;
    std::vector<std::string> verified;
    std::vector<std::string> passed;

    static DecodeBackendGate notRequested(std::string reason = "not_requested") {
        return {DecodeBackendGateStatus::NOT_REQUESTED, std::move(reason)};
    }

    static DecodeBackendGate unavailable(std::string reason) {
        return {DecodeBackendGateStatus::UNAVAILABLE, std::move(reason)};
    }

    const char* statusString() const {
        switch (status) {
            case DecodeBackendGateStatus::NOT_REQUESTED:
                return "NOT_REQUESTED";
            case DecodeBackendGateStatus::READY:
                return "READY";
            case DecodeBackendGateStatus::UNAVAILABLE:
                return "UNAVAILABLE";
        }
        return "UNAVAILABLE";
    }
};

class AccuracyChecker {
public:
    AccuracyChecker(const ModelConfig&       model_config,
                    const RuntimeConfig&     runtime_config,
                    const ParallelismConfig& parallelism_config);

    static const std::vector<AccuracyScenario>& defaultScenarios();

    static std::vector<AccuracyScenario> scenariosForMoeConfig(const MoeConfig& moe_config);

    static torch::Tensor
    makeDeterministicPrompt(size_t len, int64_t token_size, uint64_t prompt_seed, size_t sequence_id);

    static bool
    acceptsWorldBackend(int32_t passed_sum, int32_t verified_sum, int32_t soft_outlier_sum, int32_t world_size);

    // Entry point for accuracy checks
    absl::StatusOr<DecodeBackendGate> runAll(Executor*                            executor,
                                             ResourceContext&                     resource_context,
                                             const std::vector<AccuracyScenario>& scenarios = defaultScenarios(),
                                             size_t                               max_forward_tokens = 0);

    // Sizes the temporary KV pool for peak block usage across all scenarios, plus a safety margin
    // The scenarios must match those passed to runAll()
    static size_t computeTemporaryBlockNum(const ModelConfig&                   model_config,
                                           const KVCacheConfig&                 kv_cache_config,
                                           const std::vector<AccuracyScenario>& scenarios = defaultScenarios());

private:
    torch::Tensor                     makeCheckPrompt(size_t len, uint64_t prompt_seed, size_t sequence_id);
    std::shared_ptr<GenerateInput>    wrapAccuracyInput(torch::Tensor input_ids, bool need_release);
    absl::StatusOr<GenerateStreamPtr> buildAccuracyStream(torch::Tensor prompt, bool need_release);
    absl::StatusOr<GenerateStreamPtr> makeForkStream(
        torch::Tensor init_tokens, size_t history_len, bool is_decode, int cur_token, const GenerateStreamPtr& golden);
    absl::Status runScenario(const py::object& recorder, const AccuracyScenario& scenario);
    absl::Status bootstrapGoldenHistory(const py::object&                 recorder,
                                        const std::string&                scenario_base_name,
                                        const AccuracyScenario&           scenario,
                                        const std::vector<torch::Tensor>& prompts,
                                        std::vector<GenerateStreamPtr>&   golden_streams,
                                        std::vector<StreamReleaseGuard>&  golden_guards);

    bool         worldAll(bool local_flag);
    absl::Status abortIfFailed(bool step_flag, const std::string& where);
    absl::Status validateWorldMetadata(const std::vector<int64_t>& local_metadata, const std::string& where);
    absl::StatusOr<std::vector<int32_t>> sumWorldMask(const std::vector<int32_t>& local_mask, const std::string& where);
    absl::StatusOr<DecodeBackendGate>    finalizeDecodeGate(const py::object& recorder);
    bool runCheck(const AccuracyScenario& scenario, const std::string& scenario_base_name);
    absl::StatusOr<std::vector<std::string>> listCandidates(const py::object& recorder, const std::string& phase);
    absl::Status                             forward(const py::object&                     recorder,
                                                     const std::string&                    scenario_base_name,
                                                     const std::string&                    impl_name,
                                                     const std::string&                    phase,
                                                     const std::vector<GenerateStreamPtr>& active_streams,
                                                     const std::vector<size_t>&            active_seq_ids,
                                                     bool                                  build_streams_flag,
                                                     bool                                  gate_qualifying);
    absl::Status                             forwardGoldenBootstrap(const py::object&                     recorder,
                                                                    const std::string&                    scenario_base_name,
                                                                    const std::string&                    phase,
                                                                    const std::vector<GenerateStreamPtr>& active_streams,
                                                                    const std::vector<size_t>&            active_seq_ids,
                                                                    bool                                  build_streams_flag);

    const ModelConfig&       model_config_;
    const RuntimeConfig&     runtime_config_;
    const ParallelismConfig& parallelism_config_;

    // Borrowed only during runAll()
    Executor*        executor_           = nullptr;
    ResourceContext* resource_context_   = nullptr;
    size_t           max_forward_tokens_ = 0;
};

}  // namespace rtp_llm
