#include "ATen/CPUGeneratorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <cstdlib>

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/accuracy/AccuracyChecker.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

TEST_F(NormalEngineTest, testDynamicDecodeBackendRoleMatrix) {
    HWKernelConfig hw_kernel_config;
    hw_kernel_config.enable_dynamic_decode_backend = true;

    EXPECT_FALSE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/false, SP_TYPE_NONE));
    EXPECT_FALSE(dynamic_decode_detail::shouldDeferCapture(
        /*warm_up=*/false, hw_kernel_config, SP_TYPE_NONE));

    hw_kernel_config.enable_cuda_graph = true;
    EXPECT_TRUE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/false, SP_TYPE_NONE));
    EXPECT_TRUE(dynamic_decode_detail::shouldDeferCapture(
        /*warm_up=*/false, hw_kernel_config, SP_TYPE_NONE));
    EXPECT_FALSE(dynamic_decode_detail::shouldDeferCapture(
        /*warm_up=*/true, hw_kernel_config, SP_TYPE_NONE));
    // Target verify decode.
    EXPECT_FALSE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/false, SP_TYPE_MTP));
    // Draft decode.
    EXPECT_FALSE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/false, SP_TYPE_EAGLE));
    // Draft prefill.
    EXPECT_FALSE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/true, SP_TYPE_MTP));

    hw_kernel_config.enable_dynamic_decode_backend = false;
    EXPECT_FALSE(
        dynamic_decode_detail::enabledForGraph(hw_kernel_config, /*is_prefill_cuda_graph_mode=*/false, SP_TYPE_NONE));
}

TEST_F(NormalEngineTest, testAccuracyCheckPromptIsDeterministic) {
    constexpr size_t   len        = 256;
    constexpr int64_t  token_size = 32000;
    constexpr uint64_t seed       = 0x123456789ABCDEF0ULL;

    const auto rng_state_before = at::detail::getDefaultCPUGenerator().get_state().clone();
    const auto first            = AccuracyChecker::makeDeterministicPrompt(len, token_size, seed, 0);
    const auto repeated         = AccuracyChecker::makeDeterministicPrompt(len, token_size, seed, 0);
    const auto different_seed   = AccuracyChecker::makeDeterministicPrompt(len, token_size, seed + 1, 0);
    const auto different_seq    = AccuracyChecker::makeDeterministicPrompt(len, token_size, seed, 1);
    const auto rng_state_after  = at::detail::getDefaultCPUGenerator().get_state();

    EXPECT_TRUE(torch::equal(first, repeated));
    EXPECT_FALSE(torch::equal(first, different_seed));
    EXPECT_FALSE(torch::equal(first, different_seq));
    EXPECT_EQ(first.scalar_type(), torch::kInt32);
    EXPECT_EQ(first.device(), torch::Device(torch::kCPU));
    EXPECT_GE(first.min().item<int32_t>(), 0);
    EXPECT_LT(first.max().item<int32_t>(), token_size);
    EXPECT_TRUE(torch::equal(rng_state_before, rng_state_after));
}

TEST_F(NormalEngineTest, testDefaultAccuracyScenarioQualification) {
    const auto& scenarios = AccuracyChecker::defaultScenarios();
    ASSERT_EQ(scenarios.size(), 8);

    const std::vector<uint64_t> expected_seeds = {
        0xA11CE00000000001ULL,
        0xA11CE00000000002ULL,
        0xA11CE00000000003ULL,
        0xA11CE00000000004ULL,
        0xA11CE00000000005ULL,
        0xA11CE00000000006ULL,
        0xA11CE00000000007ULL,
        0xA11CE00000000008ULL,
    };
    for (size_t scenario_idx = 0; scenario_idx < scenarios.size(); ++scenario_idx) {
        const auto& scenario = scenarios[scenario_idx];
        EXPECT_EQ(scenario.gate_qualifying, scenario.seqs.size() == 1) << "scenario " << scenario_idx;
        EXPECT_EQ(scenario.prompt_seed, expected_seeds[scenario_idx]) << "scenario " << scenario_idx;
    }

    MoeConfig regular_moe;
    regular_moe.use_deepep_moe         = true;
    regular_moe.use_deepep_low_latency = false;
    const auto regular_scenarios       = AccuracyChecker::scenariosForMoeConfig(regular_moe);
    ASSERT_EQ(regular_scenarios.size(), scenarios.size());
    for (size_t scenario_idx = 0; scenario_idx < scenarios.size(); ++scenario_idx) {
        ASSERT_EQ(regular_scenarios[scenario_idx].seqs.size(), scenarios[scenario_idx].seqs.size());
        EXPECT_EQ(regular_scenarios[scenario_idx].gate_qualifying, scenarios[scenario_idx].gate_qualifying);
        EXPECT_EQ(regular_scenarios[scenario_idx].prompt_seed, scenarios[scenario_idx].prompt_seed);
        for (size_t seq_idx = 0; seq_idx < scenarios[scenario_idx].seqs.size(); ++seq_idx) {
            EXPECT_EQ(regular_scenarios[scenario_idx].seqs[seq_idx].prefix_len,
                      scenarios[scenario_idx].seqs[seq_idx].prefix_len);
            EXPECT_EQ(regular_scenarios[scenario_idx].seqs[seq_idx].chunk_len,
                      scenarios[scenario_idx].seqs[seq_idx].chunk_len);
            EXPECT_EQ(regular_scenarios[scenario_idx].seqs[seq_idx].decode_steps,
                      scenarios[scenario_idx].seqs[seq_idx].decode_steps);
        }
    }
}

TEST_F(NormalEngineTest, testDeepEPLowLatencyAccuracyScenariosPreserveDecodeCoverage) {
    MoeConfig moe_config;
    moe_config.use_deepep_moe         = true;
    moe_config.use_deepep_low_latency = true;

    for (const int token_budget : {8, 32, 64, 160}) {
        moe_config.ll_num_max_token = token_budget;
        const auto scenarios        = AccuracyChecker::scenariosForMoeConfig(moe_config);
        ASSERT_FALSE(scenarios.empty()) << "token budget " << token_budget;

        ASSERT_EQ(scenarios.size(), 4) << "token budget " << token_budget;
        bool has_kv64_qualifier   = false;
        bool has_kv1024_qualifier = false;
        for (const auto& scenario : scenarios) {
            size_t plain_tokens = 0;
            for (const auto& seq : scenario.seqs) {
                plain_tokens += seq.prefix_len + seq.chunk_len;
                has_kv64_qualifier |=
                    scenario.gate_qualifying && seq.prefix_len == 0 && seq.chunk_len == 64 && seq.decode_steps == 4;
                has_kv1024_qualifier |=
                    scenario.gate_qualifying && seq.prefix_len == 512 && seq.chunk_len == 512 && seq.decode_steps == 5;
            }
            EXPECT_LE(scenario.seqs.size(), static_cast<size_t>(token_budget));
            if (!scenario.gate_qualifying) {
                EXPECT_LE(plain_tokens, static_cast<size_t>(token_budget));
            }
        }
        EXPECT_TRUE(has_kv64_qualifier);
        EXPECT_TRUE(has_kv1024_qualifier);
    }

    moe_config.ll_num_max_token = 32;
    const auto budget_32        = AccuracyChecker::scenariosForMoeConfig(moe_config);
    ASSERT_EQ(budget_32.size(), 4);
    EXPECT_EQ(budget_32[0].prompt_seed, 0xA11CE00000000002ULL);
    EXPECT_EQ(budget_32[1].prompt_seed, 0xA11CE00000000004ULL);
    ASSERT_EQ(budget_32[2].seqs.size(), 3);
    EXPECT_EQ(budget_32[2].seqs[0].prefix_len + budget_32[2].seqs[0].chunk_len, 4);
    EXPECT_EQ(budget_32[2].seqs[1].prefix_len + budget_32[2].seqs[1].chunk_len, 8);
    EXPECT_EQ(budget_32[2].seqs[2].prefix_len + budget_32[2].seqs[2].chunk_len, 20);
    EXPECT_EQ(budget_32[2].prompt_seed, 0xA11CE00000000007ULL);
    ASSERT_EQ(budget_32[3].seqs.size(), 8);
    for (const auto& seq : budget_32[3].seqs) {
        EXPECT_EQ(seq.prefix_len, 0);
        EXPECT_EQ(seq.chunk_len, 4);
        EXPECT_EQ(seq.decode_steps, 16);
    }
    EXPECT_EQ(budget_32[3].prompt_seed, 0xA11CE00000000008ULL);

    moe_config.ll_num_max_token = 7;
    EXPECT_TRUE(AccuracyChecker::scenariosForMoeConfig(moe_config).empty());
}

TEST_F(NormalEngineTest, testDecodeGateSoftOutlierBudgetIsWorldScoped) {
    EXPECT_TRUE(AccuracyChecker::acceptsWorldBackend(1, 1, 1, 1));
    EXPECT_TRUE(AccuracyChecker::acceptsWorldBackend(2, 2, 1, 2));
    EXPECT_FALSE(AccuracyChecker::acceptsWorldBackend(2, 2, 2, 2));
    EXPECT_FALSE(AccuracyChecker::acceptsWorldBackend(1, 2, 1, 2));
    EXPECT_FALSE(AccuracyChecker::acceptsWorldBackend(2, 1, 1, 2));
    EXPECT_FALSE(AccuracyChecker::acceptsWorldBackend(2, 2, -1, 2));
    EXPECT_FALSE(AccuracyChecker::acceptsWorldBackend(0, 0, 0, 0));
}

TEST_F(NormalEngineTest, testNonServingExecutorsKeepFixedEagerCaptureLifecycle) {
    HWKernelConfig hw_kernel_config;
    hw_kernel_config.enable_cuda_graph             = true;
    hw_kernel_config.enable_dynamic_decode_backend = true;

    const auto serving_config = dynamic_decode_detail::executorHWKernelConfig(
        hw_kernel_config, /*warm_up=*/false, /*accuracy_check_executor=*/false);
    EXPECT_TRUE(serving_config.enable_dynamic_decode_backend);
    EXPECT_TRUE(dynamic_decode_detail::shouldDeferCapture(/*warm_up=*/false, serving_config, SP_TYPE_NONE));

    const auto warmup_config = dynamic_decode_detail::executorHWKernelConfig(
        hw_kernel_config, /*warm_up=*/true, /*accuracy_check_executor=*/false);
    EXPECT_TRUE(warmup_config.enable_cuda_graph);
    EXPECT_FALSE(warmup_config.enable_dynamic_decode_backend);
    EXPECT_FALSE(dynamic_decode_detail::shouldDeferCapture(/*warm_up=*/true, warmup_config, SP_TYPE_NONE));

    const auto accuracy_config = dynamic_decode_detail::executorHWKernelConfig(
        hw_kernel_config, /*warm_up=*/false, /*accuracy_check_executor=*/true);
    EXPECT_TRUE(accuracy_config.enable_cuda_graph);
    EXPECT_FALSE(accuracy_config.enable_dynamic_decode_backend);
    EXPECT_FALSE(dynamic_decode_detail::shouldDeferCapture(/*warm_up=*/false, accuracy_config, SP_TYPE_NONE));
}

TEST_F(NormalEngineTest, testDynamicDecodeCaptureTriggerLifecycle) {
    struct TestCase {
        const char*     name;
        bool            enable_cuda_graph;
        bool            enable_dynamic_decode_backend;
        SpeculativeType sp_type;
        size_t          expected_trigger_count;
    };
    const std::vector<TestCase> cases = {
        {"default", false, false, SP_TYPE_NONE, 0},
        {"dynamic_without_graph", false, true, SP_TYPE_NONE, 0},
        {"ordinary_decode", true, true, SP_TYPE_NONE, 1},
        {"speculative_decode", true, true, SP_TYPE_MTP, 0},
    };

    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        CustomConfig config;
        config.enable_cuda_graph             = test_case.enable_cuda_graph;
        config.enable_dynamic_decode_backend = test_case.enable_dynamic_decode_backend;
        config.sp_type                       = test_case.sp_type;
        auto lifecycle                       = std::make_shared<MockModelLifecycle>();
        auto engine                          = createMockEngine(config, lifecycle);

        EXPECT_EQ(lifecycle->trigger_init_capture_count, test_case.expected_trigger_count);
    }
}

TEST_F(NormalEngineTest, testFp8KVCache) {
    CustomConfig config;
    config.kv_cache_data_type = DataType::TYPE_FP8_E4M3;
    auto engine               = createMockEngine(config);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    shared_ptr<GenerateStream> stream = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    auto output2 = stream->nextOutput();
    ASSERT_TRUE(!output2.ok());
}

TEST_F(NormalEngineTest, testSimple) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // test streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 3;
        query->generate_config->is_streaming   = true;
        query->generate_config->gen_timeline   = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.iter_count, 1);

        auto output2 = stream->nextOutput();
        ASSERT_TRUE(output2.ok());
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.output_len, 2);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.iter_count, 2);

        auto output3 = stream->nextOutput();
        ASSERT_TRUE(output3.ok());
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.output_len, 3);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.iter_count, 3);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output4 = stream->nextOutput();
        ASSERT_TRUE(!output4.ok());
    }

    // test non-streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 5;
        query->generate_config->is_streaming   = false;

        shared_ptr<GenerateStream> stream = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testSystemPrompt) {
    CustomConfig config;
    vector<int>  prompt_1           = {1, 2, 3};
    vector<int>  prompt_2           = {4, 5, 6, 7, 8, 9};
    config.multi_task_prompt_tokens = {{"1", prompt_1}, {"2", prompt_2}};
    auto engine                     = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_TRUE(engine->resourceContext().system_prompt);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 2);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->task_id        = "2";
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testReuseCacheOption) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    config.reuse_cache = false;
    auto engine2       = createMockEngine(config);
    ASSERT_FALSE(engine2->resourceContext().reuse_cache);
}

TEST_F(NormalEngineTest, testReuseCache) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOn) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    // First query with reuse_cache = true
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Second query with reuse_cache = false (should not reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Third query with reuse_cache = true (should reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);  // Should be 4 because reuse_cache = true
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOff) {
    // Test with engine-level reuse_cache = false (master switch off)
    CustomConfig config;
    config.reuse_cache = false;
    auto engine        = createMockEngine(config);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // Query with reuse_cache = true, but should be ignored because engine-level is false
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;  // This should be ignored
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Query with reuse_cache = false, should also result in no cache reuse
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

}  // namespace rtp_llm
