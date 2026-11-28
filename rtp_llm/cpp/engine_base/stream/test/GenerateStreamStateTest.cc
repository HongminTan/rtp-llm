
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/test/MockAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

using namespace std;

namespace rtp_llm {

namespace {

thread_local const AsyncContext* tracked_abort_context = nullptr;
thread_local size_t              tracked_abort_count   = 0;

void expectQps(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags& tags, double expected) {
    ASSERT_NE(metric, nullptr);
    auto* series = metric->DeclareMetric(&tags);
    ASSERT_NE(series, nullptr);
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    series->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(series));
    ASSERT_EQ(record.Values().size(), 1u);
    EXPECT_DOUBLE_EQ(std::stod(record.Values().front()->Value()), expected);
}

}  // namespace

// KVCacheManager::abortPendingLoad is intentionally non-virtual. This target-local
// GNU ld --wrap seam counts only the AsyncContext selected by the current test
// and forwards every call to the real implementation. The Itanium ABI symbol is
// _ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE.
extern "C" bool
realKVCacheManagerAbortPendingLoad(KVCacheManager* manager, const std::shared_ptr<AsyncContext>& context) asm(
    "__real__ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE");

extern "C" bool
wrappedKVCacheManagerAbortPendingLoad(KVCacheManager* manager, const std::shared_ptr<AsyncContext>& context) asm(
    "__wrap__ZN7rtp_llm14KVCacheManager16abortPendingLoadERKSt10shared_ptrINS_12AsyncContextEE");

extern "C" bool wrappedKVCacheManagerAbortPendingLoad(KVCacheManager*                      manager,
                                                      const std::shared_ptr<AsyncContext>& context) {
    if (context.get() == tracked_abort_context) {
        ++tracked_abort_count;
    }
    return realKVCacheManagerAbortPendingLoad(manager, context);
}

class GenerateStreamStateTest: public DeviceTestBase {
protected:
    GenerateStreamStateTest(): perf_scope("PERF_TEST", "1") {
        // 降低显存配置以避免 OOM
        max_seq_len_ = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens = {1, 2, 3, 4, 5, 6},
                                   bool                    reuse_cache  = false,
                                   RoleType                role_type    = RoleType::PDFUSION) {
        cache_manager_ =
            std::make_shared<KVCacheManager>(init_config(), /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(cache_manager_->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;
        resource_context.role_type     = role_type;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 1;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }

    std::shared_ptr<testing::NiceMock<MockCoordinatorCacheManager>> installRetryableInitMalloc() {
        auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
        ON_CALL(*allocator, totalBlocksNum()).WillByDefault(testing::Return(64));
        ON_CALL(*allocator, getNeedBlocks(testing::_)).WillByDefault(testing::Return(1));
        ON_CALL(*allocator, initMallocForCommonLen(testing::_))
            .WillByDefault(testing::Return(MallocResult{false, 0, 0, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED}));
        cache_manager_->coordinator_manager_ = allocator;
        return allocator;
    }

protected:
    autil::EnvGuard                 perf_scope;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ============================================================================
// 1. Initial state and basic state checks
// ============================================================================

TEST_F(GenerateStreamStateTest, testInitialStateIsWaiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->isFinished());
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->getStatus() == StreamState::LOADING_CACHE);
}

TEST_F(GenerateStreamStateTest, testMallocRetryMetricsIndependentOfTerminalOutcome) {
    const struct {
        ErrorCode error;
        int       retries;
    } cases[] = {{ErrorCode::NONE_ERROR, 3},
                 {ErrorCode::MALLOC_FAILED, 3},
                 {ErrorCode::DECODE_MALLOC_FAILED, 3},
                 {ErrorCode::CANCELLED, 1},
                 {ErrorCode::GENERATE_TIMEOUT, 3},
                 {ErrorCode::MALLOC_FAILED, 0},
                 {ErrorCode::DECODE_MALLOC_FAILED, 0}};
    for (const auto& test : cases) {
        SCOPED_TRACE(testing::Message() << "error=" << static_cast<int>(test.error) << " retries=" << test.retries);
        kmonitor::MetricsTags tags;
        auto                  reporter = std::make_shared<kmonitor::MetricsReporter>("", "", tags);
        auto                  stream   = createStream({1, 2, 3}, /*reuse_cache=*/false, RoleType::PREFILL);
        stream->setMetricsReporter(reporter);
        auto  allocator  = installRetryableInitMalloc();
        auto* operations = reporter->getMetricsGroup<RtpLLMCacheOperationMetrics>();
        stream->reportEvent(StreamEvents::CanRun);
        for (int i = 0; i < test.retries; ++i) {
            EXPECT_EQ(stream->moveToNext(), StreamState::WAITING);
        }
        // Retry decisions are visible while waiting, even if cancelled before the next attempt.
        expectQps(operations->malloc_retry_qps_metric, tags, test.retries);
        if (test.error == ErrorCode::NONE_ERROR) {
            ON_CALL(*allocator, initMallocForCommonLen(testing::_))
                .WillByDefault(testing::Return(MallocResult{true, 0}));
            ON_CALL(*allocator, incrMalloc(testing::_)).WillByDefault(testing::Return(MallocResult{true, 0}));
            EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
            stream->reportEvent(StreamEvents::GenerateDone);
        } else {
            stream->reportError(test.error, "terminal error");
        }
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        stream->reportMetricOnce();
        stream->reportMetricOnce();
        stream.reset();
        expectQps(reporter->getMetricsGroup<RtpLLMStreamMetrics>()->kv_cache_malloc_failed_qps_metric,
                  tags,
                  test.error == ErrorCode::MALLOC_FAILED || test.error == ErrorCode::DECODE_MALLOC_FAILED);
        expectQps(operations->malloc_retry_qps_metric, tags, 0);
    }
}

TEST_F(GenerateStreamStateTest, testDirectStateManipulation) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE via direct manipulation
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    ASSERT_FALSE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_FALSE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToWaitingTransition) {
    auto stream = createStream();
    // Set to LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // Transition back to WAITING
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    ASSERT_FALSE(stream->getStatus() == StreamState::LOADING_CACHE);
}

// ============================================================================
// 2. WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED paths
// ============================================================================

TEST_F(GenerateStreamStateTest, testWaitingToLoadingCacheToWaitingPath) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // LOADING_CACHE -> WAITING (load done, back to waiting for scheduling)
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    // WAITING -> RUNNING (normal scheduling)
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToFinishedViaError) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_TRUE(stream->getStatus() == StreamState::LOADING_CACHE);
    // Report error and set FINISHED during LOADING_CACHE
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testLoadingCacheToFinishedViaTimeout) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    // Error during LOADING_CACHE transitions to FINISHED
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "timeout during loading");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::GENERATE_TIMEOUT);
}

// ============================================================================
// 3. RUNNING -> FINISHED transitions
// ============================================================================

TEST_F(GenerateStreamStateTest, testRunningToFinishedNormal) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->generate_status_->status = StreamState::FINISHED;
    stream->fillSubGenerateStatus(StreamState::FINISHED);
    ASSERT_TRUE(stream->isFinished());
    ASSERT_FALSE(stream->getStatus() == StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testRunningToFinishedViaError) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
}

// ============================================================================
// 4. reportEvent(Error) / hasError / statusInfo
// ============================================================================

TEST_F(GenerateStreamStateTest, testReportErrorDoesNotChangeState) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    // reportEvent(Error) only stores error, does NOT directly change state
    stream->reportError(ErrorCode::CANCELLED, "cancel from RPC");
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);  // state unchanged
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);
}

TEST_F(GenerateStreamStateTest, testReportErrorFirstWins) {
    auto stream = createStream();
    stream->reportError(ErrorCode::CANCELLED, "first error");
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "second error");
    auto err = stream->statusInfo();
    // First error should win
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished_Waiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->getStatus() == StreamState::WAITING);
    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

TEST_F(GenerateStreamStateTest, testReportErrorAndReleaseResource) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    auto& resource                   = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_GT(resource.curBlocksNum(), 0);

    stream->reportError(ErrorCode::MALLOC_FAILED, "OOM");
    stream->generate_status_->status = StreamState::FINISHED;
    stream->releaseResource();
    ASSERT_TRUE(stream->isFinished());
    ASSERT_TRUE(stream->hasError());
}

// ============================================================================
// 5. reportEvent(Error) + state set replaces cancelIfNotRunning
// ============================================================================

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished_LoadingCache) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testReportError_Running_StateUnchanged) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    // reportEvent(Error) should NOT change state
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    ASSERT_TRUE(stream->hasError());
}

TEST_F(GenerateStreamStateTest, testReportErrorAndSetFinished) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    stream->generate_status_->status = StreamState::FINISHED;
    ASSERT_TRUE(stream->isFinished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 6. Use reportEvent(Error) to cancel (external error reporting)
// ============================================================================

TEST_F(GenerateStreamStateTest, testCancelUsesReportError) {
    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::RUNNING;
    stream->reportError(ErrorCode::CANCELLED, "cancel stream");
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);
    auto err = stream->statusInfo();
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 7. Complete state machine cycle: WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED
// ============================================================================

TEST_F(GenerateStreamStateTest, testFullStateMachineCycle) {
    auto stream = createStream();
    // Initial: WAITING
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> LOADING_CACHE
    stream->generate_status_->status = StreamState::LOADING_CACHE;
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);

    // LOADING_CACHE -> WAITING (load done)
    stream->generate_status_->status = StreamState::WAITING;
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> RUNNING
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_EQ(stream->generate_status_->status, StreamState::RUNNING);

    // RUNNING -> FINISHED
    stream->generate_status_->status = StreamState::FINISHED;
    stream->fillSubGenerateStatus(StreamState::FINISHED);
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

// ============================================================================
// 8. StreamStateToString
// ============================================================================

TEST_F(GenerateStreamStateTest, testStreamStateToString) {
    ASSERT_EQ(StreamStateToString(StreamState::WAITING), "WAITING");
    ASSERT_EQ(StreamStateToString(StreamState::LOADING_CACHE), "LOADING_CACHE");
    ASSERT_EQ(StreamStateToString(StreamState::RUNNING), "RUNNING");
    ASSERT_EQ(StreamStateToString(StreamState::FINISHED), "FINISHED");
}

// ============================================================================
// 9. LoadInitiated event: Verify Decode mode cache load fix
// ============================================================================

TEST_F(GenerateStreamStateTest, testLoadInitiatedPreventsDuplicateInitKVBlock) {
    auto stream = createStream();
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Simulate DecodeRpcServer: call initKVBlock directly and set LoadInitiated
    auto& resource = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    stream->reportEvent(StreamEvents::LoadInitiated);

    // FIFOScheduler calls moveToNext, should skip initKVBlock and asyncLoadCache
    auto new_state = stream->moveToNext();
    // Should stay in WAITING because CanRun is not set yet
    ASSERT_EQ(new_state, StreamState::WAITING);

    // Now simulate FIFOScheduler setting CanRun
    stream->reportEvent(StreamEvents::CanRun);
    new_state = stream->moveToNext();
    ASSERT_EQ(new_state, StreamState::RUNNING);
}

TEST_F(GenerateStreamStateTest, testLoadInitiatedSkipsAsyncLoadCache) {
    auto stream = createStream();
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Simulate DecodeRpcServer: only initKVBlock, no asyncLoadCache
    auto& resource = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    stream->reportEvent(StreamEvents::LoadInitiated);

    // No allocator-owned load is in flight.
    ASSERT_FALSE(resource.allocator_load_context_);

    // moveToNext should not trigger asyncLoadCache because LoadInitiated is set
    stream->reportEvent(StreamEvents::CanRun);
    auto new_state = stream->moveToNext();
    ASSERT_EQ(new_state, StreamState::RUNNING);

    // Still no allocator load context.
    ASSERT_FALSE(resource.allocator_load_context_);
}

TEST_F(GenerateStreamStateTest, testPrefillFallbackDecodeGrowsBlocksAfterContext) {
    auto stream = createStream({1, 2}, /*reuse_cache=*/false, RoleType::PREFILL);

    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_EQ(stream->curBlocksNum(), 1u);

    // PD fallback can continue decoding in the PREFILL role. The next decode
    // token crosses the two-token test block boundary and needs another column.
    stream->setIsContextStream(false);
    stream->setSeqLength(3);

    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_EQ(stream->curBlocksNum(), 2u);
}

TEST_F(GenerateStreamStateTest, testIncrementalAsyncAllocationTerminatesBeforeModelAndReleasesOnce) {
    using ::testing::_;
    using ::testing::Return;

    auto stream = createStream({1, 2}, /*reuse_cache=*/false, RoleType::PREFILL);
    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    ASSERT_EQ(stream->curBlocksNum(), 1u);

    const auto real_allocator = cache_manager_->coordinator_manager_;
    ASSERT_EQ(real_allocator->groupBlockPools().size(), 1u);
    const auto device_pool = real_allocator->groupBlockPools().front();
    ASSERT_NE(device_pool, nullptr);
    const auto& request_blocks = stream->streamCacheResource().kvCache().blocks(/*batch_id=*/0, "default");
    ASSERT_EQ(request_blocks.size(), 1u);
    const BlockIdxType request_block = request_blocks.front();
    ASSERT_EQ(device_pool->refCount(request_block), 1u);
    const size_t free_after_request_alloc = device_pool->freeBlocksNum();

    auto   async_context  = std::make_shared<MockAsyncContext>();
    auto   mock_allocator = std::make_shared<MockCoordinatorCacheManager>(init_config());
    size_t free_count     = 0;
    EXPECT_CALL(*mock_allocator, incrMalloc(_))
        .WillOnce(Return(MallocResult{/*success=*/true,
                                      /*reuse_len=*/0,
                                      /*match_cost_time_us=*/0,
                                      async_context}));
    EXPECT_CALL(*mock_allocator, free(_)).WillOnce([&](const FreeInfo& free_info) {
        ++free_count;
        real_allocator->free(free_info);
    });
    cache_manager_->coordinator_manager_ = mock_allocator;

    stream->setIsContextStream(false);
    stream->setSeqLength(3);
    tracked_abort_context = async_context.get();
    tracked_abort_count   = 0;

    // A model executor may only consume RUNNING streams. The state-machine
    // caller must make the unexpected incremental async result terminal first.
    size_t     model_entry_count     = 0;
    const auto run_model_if_eligible = [&]() {
        if (stream->getStatus() == StreamState::RUNNING) {
            ++model_entry_count;
        }
    };

    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    run_model_if_eligible();
    EXPECT_TRUE(stream->isFinished());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_NE(stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(model_entry_count, 0u);
    EXPECT_EQ(tracked_abort_count, 1u);
    EXPECT_EQ(free_count, 1u);
    EXPECT_EQ(stream->curBlocksNum(), 0u);
    EXPECT_FALSE(device_pool->isAllocated(request_block));
    EXPECT_EQ(device_pool->freeBlocksNum(), free_after_request_alloc + 1);
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());

    // Re-entering the terminal state and an explicit stream release are both
    // idempotent: neither the observer nor request blocks are released twice.
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
    stream->releaseResource();
    EXPECT_EQ(tracked_abort_count, 1u);
    EXPECT_EQ(free_count, 1u);
    EXPECT_FALSE(device_pool->isAllocated(request_block));
    EXPECT_EQ(device_pool->freeBlocksNum(), free_after_request_alloc + 1);

    tracked_abort_context = nullptr;
    tracked_abort_count   = 0;
}

TEST_F(GenerateStreamStateTest, testNormalPathTriggersAsyncLoadCache) {
    // Create stream with reuse_cache enabled to trigger asyncLoadCache
    auto stream = createStream({1, 2, 3, 4, 5, 6}, /*reuse_cache=*/true);
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    // Normal path: moveToNext should trigger initKVBlock + asyncLoadCache
    auto new_state = stream->moveToNext();

    // Should transition to LOADING_CACHE if asyncLoadCache was initiated
    // or stay in WAITING if no connectors are available
    ASSERT_TRUE(new_state == StreamState::LOADING_CACHE || new_state == StreamState::WAITING);
}

TEST_F(GenerateStreamStateTest, testNonLinearBoundaryGrowthAvoidsJoinAndPreservesInFlightSwaBlock) {
    // 1. FULL + SWA topology with sufficient retained SWA tail width
    //    (tokens_per_block=16, active_tail_blocks=2, sliding_window=16,
    //    reserve_step=4, max_mtp_advance=4 => (6-1)*16 = 80 >= 16+8+4+2 = 30)
    //    grows across a block boundary without joining the pending worker.
    CacheConfig config;
    config.dtype               = DataType::TYPE_INT8;
    config.layer_num           = 2;
    config.seq_size_per_block  = 16;
    auto             full_spec = test::makeMhaSpec("full", /*tokens_per_block=*/16, DataType::TYPE_INT8, 1, 1);
    auto             swa_spec  = test::makeMhaSpec("swa", /*tokens_per_block=*/16, DataType::TYPE_INT8, 1, 1);
    CacheGroupPolicy full_policy;
    full_policy.group_type = CacheGroupType::FULL;
    CacheGroupPolicy swa_policy;
    swa_policy.group_type          = CacheGroupType::SWA;
    swa_policy.active_tail_blocks  = 2;
    swa_policy.sliding_window_size = 16;
    swa_policy.enable_prefix_reuse = false;
    config.fromGroupedSpecs({full_spec, swa_spec},
                            {{0}, {1}},
                            {CacheGroupType::FULL, CacheGroupType::SWA},
                            {"full", "swa"},
                            {full_policy, swa_policy});
    config.finalizeBlockNums(/*block_num=*/16, RuntimeConfig{});

    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = kv_cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PDFUSION;

    auto generate_input                             = std::make_shared<GenerateInput>();
    generate_input->input_ids                       = torch::arange(1, 13, torch::kInt32);  // seq_len = 12
    generate_input->generate_config                 = std::make_shared<GenerateConfig>();
    generate_input->generate_config->max_new_tokens = 32;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 2048;
    model_config.attn_config.tokens_per_block = 16;
    auto stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->streamCacheResource().hasLinearCacheGroup());
    stream->setReserveStep(4);  // 12 + 4 = 16 => 1 block initially.

    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);
    ASSERT_EQ(stream->curBlocksNum(), 1u);

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    // Without allow_async_cache_reservation, non-LINEAR joins at the boundary.
    {
        GenerateStream::MtpAsyncDeviceState unready;
        unready.async_decode_state_ready          = true;
        unready.previous_seq_len_upper_bound      = 12;
        unready.next_seq_len_upper_bound          = 14;  // 14 + 4 = 18 > 16 => boundary
        unready.next_kv_cache_block_id_gpu        = torch::zeros({2, 1, 1}, cuda_i32);
        unready.next_kv_cache_kernel_block_id_gpu = torch::zeros({2, 1, 1}, cuda_i32);
        stream->setMtpAsyncDeviceState(std::move(unready));
        stream->incPendingAsyncBookkeeping();
        EXPECT_TRUE(stream->needsAsyncBookkeepingJoinForKvReservation());
        stream->decPendingAsyncBookkeepingAndMaybeRelease();
    }

    // With allow_async_cache_reservation and sufficient SWA tail width:
    // moveToNext() grows host blocks without joining, clears the stale GPU
    // snapshot, keeps async_decode_state_ready, and retains the in-flight SWA
    // tail block.
    GenerateStream::MtpAsyncDeviceState state;
    state.async_decode_state_ready          = true;
    state.allow_async_cache_reservation     = true;
    state.max_mtp_advance                   = 4;
    state.previous_seq_len_upper_bound      = 12;
    state.next_seq_len_upper_bound          = 14;
    state.next_kv_cache_block_id_gpu        = torch::zeros({2, 1, 1}, cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu = torch::zeros({2, 1, 1}, cuda_i32);
    stream->setMtpAsyncDeviceState(std::move(state));
    stream->incPendingAsyncBookkeeping();

    EXPECT_FALSE(stream->needsAsyncBookkeepingJoinForKvReservation());
    // Pending async KV estimate accounts for the in-flight upper-bound slack across both groups.
    EXPECT_EQ(stream->estimateInitialNeedBlocks(), 2);
    // Would deadlock if moveToNext() called waitPendingAsyncBookkeeping() while pending=1.
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    EXPECT_EQ(stream->curBlocksNum(), 2u);
    EXPECT_EQ(stream->estimateInitialNeedBlocks(), 0);
    EXPECT_FALSE(stream->hasMtpCacheSnapshot());
    EXPECT_TRUE(stream->hasMtpAsyncDecodeState());

    {
        const auto& swa_blocks = stream->streamCacheResource().kvCache().blocks(0, "swa");
        ASSERT_EQ(swa_blocks.size(), 2u);
        EXPECT_FALSE(isNullBlockIdx(swa_blocks[0])) << "in-flight Step N SWA tail block must stay allocated";
        EXPECT_FALSE(isNullBlockIdx(swa_blocks[1]));
    }

    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    stream->releaseResource();
    const auto pools = kv_cache_manager->coordinator_manager_->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    EXPECT_EQ(pools[0]->freeBlocksNum(), pools[0]->totalBlocksNum());
    EXPECT_EQ(pools[1]->freeBlocksNum(), pools[1]->totalBlocksNum());

    // 2. Narrow SWA tail width (tokens_per_block=1, active_tail_blocks=1,
    //    reserve_step=6, max_mtp_advance=3 => (7-1)*1 = 6 < 1+6+6+2 = 15)
    //    avoids join on steady steps but falls back to worker join at a block
    //    boundary so pruning cannot reclaim in-flight pages.
    {
        CacheConfig narrow_cfg;
        narrow_cfg.dtype              = DataType::TYPE_INT8;
        narrow_cfg.layer_num          = 1;
        narrow_cfg.seq_size_per_block = 1;
        auto             narrow_spec  = test::makeMhaSpec("swa", /*tokens_per_block=*/1, DataType::TYPE_INT8, 1, 1);
        CacheGroupPolicy narrow_policy;
        narrow_policy.group_type          = CacheGroupType::SWA;
        narrow_policy.active_tail_blocks  = 1;
        narrow_policy.enable_prefix_reuse = false;
        narrow_cfg.fromGroupedSpecs({narrow_spec}, {{0}}, {CacheGroupType::SWA}, {"swa"}, {narrow_policy});
        narrow_cfg.finalizeBlockNums(/*block_num=*/32, RuntimeConfig{});

        auto narrow_mgr = std::make_shared<KVCacheManager>(narrow_cfg, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(narrow_mgr->init());
        ResourceContext narrow_ctx;
        narrow_ctx.cache_manager = narrow_mgr;
        narrow_ctx.role_type     = RoleType::PDFUSION;

        ModelConfig narrow_model_cfg                  = model_config;
        narrow_model_cfg.attn_config.tokens_per_block = 1;
        auto narrow_input                             = std::make_shared<GenerateInput>();
        narrow_input->input_ids                       = torch::arange(1, 11, torch::kInt32);  // seq_len = 10
        narrow_input->generate_config                 = std::make_shared<GenerateConfig>();
        narrow_input->generate_config->max_new_tokens = 16;
        auto narrow_stream =
            std::make_shared<NormalGenerateStream>(narrow_input, narrow_model_cfg, runtime_config, narrow_ctx, nullptr);
        narrow_stream->setReserveStep(6);  // 16 slots allocated initially.
        narrow_stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(narrow_stream->moveToNext(), StreamState::RUNNING);
        narrow_stream->setIsContextStream(false);

        GenerateStream::MtpAsyncDeviceState narrow_state;
        narrow_state.async_decode_state_ready          = true;
        narrow_state.allow_async_cache_reservation     = true;
        narrow_state.max_mtp_advance                   = 3;
        narrow_state.previous_seq_len_upper_bound      = 10;
        narrow_state.next_seq_len_upper_bound          = 10;  // steady: 10 + 6 <= 16
        narrow_state.next_kv_cache_block_id_gpu        = torch::zeros({1, 1, 16}, cuda_i32);
        narrow_state.next_kv_cache_kernel_block_id_gpu = torch::zeros({1, 1, 16}, cuda_i32);
        narrow_stream->setMtpAsyncDeviceState(narrow_state);
        narrow_stream->incPendingAsyncBookkeeping();
        EXPECT_FALSE(narrow_stream->canReserveMtpCacheWithoutBookkeepingJoin(narrow_state));
        EXPECT_FALSE(narrow_stream->needsAsyncBookkeepingJoinForKvReservation());

        narrow_state.next_seq_len_upper_bound = 11;  // boundary: 11 + 6 = 17 > 16
        narrow_stream->setMtpAsyncDeviceState(narrow_state);
        EXPECT_TRUE(narrow_stream->needsAsyncBookkeepingJoinForKvReservation());

        narrow_stream->decPendingAsyncBookkeepingAndMaybeRelease();
        narrow_stream->releaseResource();
    }

    // 3. LINEAR hybrid topology always joins at a block boundary even when
    //    allow_async_cache_reservation is set.
    CacheConfig linear_config;
    linear_config.dtype              = DataType::TYPE_INT8;
    linear_config.layer_num          = 2;
    linear_config.seq_size_per_block = 16;
    auto lin_full_spec               = test::makeMhaSpec("full", 16, DataType::TYPE_INT8, 1, 1);
    auto lin_spec                    = test::makeLinearSpec("linear", 16, DataType::TYPE_INT8, 1, 1);
    linear_config.fromGroupedSpecs(
        {lin_full_spec, lin_spec}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::LINEAR}, {"full", "linear"});
    linear_config.finalizeBlockNums(/*block_num=*/16, RuntimeConfig{});
    auto lin_mgr = std::make_shared<KVCacheManager>(linear_config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(lin_mgr->init());

    ResourceContext lin_ctx;
    lin_ctx.cache_manager = lin_mgr;
    lin_ctx.role_type     = RoleType::PDFUSION;
    auto lin_stream =
        std::make_shared<NormalGenerateStream>(generate_input, model_config, runtime_config, lin_ctx, nullptr);
    ASSERT_TRUE(lin_stream->streamCacheResource().hasLinearCacheGroup());
    lin_stream->setReserveStep(4);
    lin_stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(lin_stream->moveToNext(), StreamState::RUNNING);
    lin_stream->setIsContextStream(false);

    GenerateStream::MtpAsyncDeviceState lin_state;
    lin_state.async_decode_state_ready          = true;
    lin_state.allow_async_cache_reservation     = true;
    lin_state.max_mtp_advance                   = 4;
    lin_state.previous_seq_len_upper_bound      = 12;
    lin_state.next_seq_len_upper_bound          = 14;
    lin_state.next_kv_cache_block_id_gpu        = torch::zeros({2, 1, 1}, cuda_i32);
    lin_state.next_kv_cache_kernel_block_id_gpu = torch::zeros({2, 1, 1}, cuda_i32);
    lin_stream->setMtpAsyncDeviceState(std::move(lin_state));
    lin_stream->incPendingAsyncBookkeeping();
    EXPECT_TRUE(lin_stream->needsAsyncBookkeepingJoinForKvReservation());
    lin_stream->decPendingAsyncBookkeepingAndMaybeRelease();
    lin_stream->releaseResource();
}

// An overlapped async specUpdate arriving after GenerateDone is published (and
// before moveToNext() transitions status to FINISHED) must neither append tokens
// nor overwrite the queued terminal output with a late error_info.
TEST_F(GenerateStreamStateTest, testOverlappedSpecUpdateIgnoredAfterGenerateDoneBeforeMoveToNext) {
    auto stream = createStream({1, 2}, /*reuse_cache=*/false, RoleType::PDFUSION);
    stream->setPerfTest(false);
    stream->vocab_size_                                      = 100;
    stream->generate_input_->generate_config->max_new_tokens = 10;
    stream->special_tokens_.eos_token_id                     = 7;

    stream->reportEvent(StreamEvents::CanRun);
    ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
    stream->setIsContextStream(false);
    auto sp_buf    = std::make_shared<SpeculativeExecutorStreamOutput>();
    sp_buf->tokens = torch::zeros({1, 2}, torch::kInt32);
    stream->setSPOutputBuffer(sp_buf);

    // Worker k hits EOS token 7 -> publishes GenerateDone, status stays RUNNING.
    StreamSpecUpdateInfo eos_update{
        torch::tensor({{7}}, torch::kInt32), 1, /*draft_token=*/3, torch::Tensor(), torch::Tensor()};
    stream->specUpdate(eos_update);
    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    auto normal_stream = std::dynamic_pointer_cast<NormalGenerateStream>(stream);
    ASSERT_NE(normal_stream, nullptr);
    const int    seq_after_eos     = stream->seqLength();
    const size_t outputs_after_eos = normal_stream->generate_outputs_.size();
    ASSERT_EQ(seq_after_eos, 3);
    ASSERT_EQ(outputs_after_eos, 1u);

    // Late normal update from overlapped worker k+1 is ignored.
    StreamSpecUpdateInfo late_update{
        torch::tensor({{4}}, torch::kInt32), 1, /*draft_token=*/5, torch::Tensor(), torch::Tensor()};
    stream->specUpdate(late_update);
    EXPECT_EQ(stream->seqLength(), seq_after_eos);
    EXPECT_EQ(normal_stream->generate_outputs_.size(), outputs_after_eos);
    EXPECT_FALSE(stream->hasError());

    // Late error_info from overlapped worker k+1 must not override the queued EOS output.
    StreamSpecUpdateInfo late_error_update{
        torch::tensor({{5}}, torch::kInt32), 1, /*draft_token=*/6, torch::Tensor(), torch::Tensor()};
    late_error_update.error_info = ErrorInfo(ErrorCode::INVALID_PARAMS, "late speculative verify error");
    stream->specUpdate(late_error_update);
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->seqLength(), seq_after_eos);
    EXPECT_EQ(normal_stream->generate_outputs_.size(), outputs_after_eos);

    auto out = normal_stream->nextOutput(100);
    ASSERT_TRUE(out.ok()) << out.status().ToString();
    ASSERT_EQ(out.value().generate_outputs.size(), 1u);
    EXPECT_TRUE(out.value().generate_outputs[0].finished);

    // Next moveToNext() consumes GenerateDone cleanly.
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
}

// Committed output metadata uses the commit-order iteration counter so an
// overlapped next-round step() on the main thread does not shift aux_info.iter_count
// or suppress first-round return_all_hidden_states while iterCount() keeps tracking
// started rounds for runtime monitoring.
TEST_F(GenerateStreamStateTest, testCommittedIterCountIsIsolatedFromOverlappedNextStep) {
    auto config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/4, DataType::TYPE_INT8);
    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = kv_cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PDFUSION;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.vocab_size                   = 128;
    model_config.special_tokens.eos_token_id  = 7;
    model_config.attn_config.tokens_per_block = 4;

    // 1. Normal update(): Round 2's step() runs before Round 1 commits.
    {
        auto input                                       = std::make_shared<GenerateInput>();
        input->input_ids                                 = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                           = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens           = 4;
        input->generate_config->is_streaming             = true;
        input->generate_config->aux_info                 = true;
        input->generate_config->return_all_hidden_states = true;

        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);

        stream->step();  // Round 1 started
        stream->step();  // Round 2 started before Round 1 worker commits
        EXPECT_EQ(stream->iterCount(), 2u);

        auto round1_hidden = torch::ones({2, 4}, torch::kFloat32);
        stream->update(StreamUpdateInfo{.new_tokens        = torch::tensor({{44}}, torch::kInt32),
                                        .num_new_tokens    = 1,
                                        .all_hidden_states = round1_hidden});
        auto out1 = stream->nextOutput(100);
        ASSERT_TRUE(out1.ok());
        ASSERT_EQ(out1.value().generate_outputs.size(), 1u);
        EXPECT_EQ(out1.value().generate_outputs[0].aux_info.iter_count, 1);
        ASSERT_TRUE(out1.value().generate_outputs[0].all_hidden_states.has_value());
        EXPECT_TRUE(out1.value().generate_outputs[0].all_hidden_states->defined());

        stream->update(StreamUpdateInfo{.new_tokens        = torch::tensor({{45}}, torch::kInt32),
                                        .num_new_tokens    = 1,
                                        .all_hidden_states = round1_hidden});
        auto out2 = stream->nextOutput(100);
        ASSERT_TRUE(out2.ok()) << out2.status().ToString();
        ASSERT_EQ(out2.value().generate_outputs.size(), 1u);
        EXPECT_EQ(out2.value().generate_outputs[0].aux_info.iter_count, 2);
        EXPECT_FALSE(out2.value().generate_outputs[0].all_hidden_states.has_value());

        stream->releaseResource();
    }

    // 2. Speculative specUpdate(): Round 2's step() runs before Round 1 commits.
    {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 4;
        input->generate_config->is_streaming   = true;
        input->generate_config->aux_info       = true;

        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->setPerfTest(false);
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);

        auto sp_buf    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_buf->tokens = torch::zeros({1, 2}, torch::kInt32);
        stream->setSPOutputBuffer(sp_buf);

        stream->step();  // Round 1 started
        stream->step();  // Round 2 started before Round 1 worker commits
        EXPECT_EQ(stream->iterCount(), 2u);

        stream->specUpdate(StreamSpecUpdateInfo{
            torch::tensor({{44}}, torch::kInt32), 1, /*draft_token=*/3, torch::Tensor(), torch::Tensor()});
        auto out1 = stream->nextOutput(100);
        ASSERT_TRUE(out1.ok()) << out1.status().ToString();
        ASSERT_EQ(out1.value().generate_outputs.size(), 1u);
        EXPECT_EQ(out1.value().generate_outputs[0].aux_info.iter_count, 1);
        EXPECT_EQ(out1.value().generate_outputs[0].output_ids.item<int32_t>(), 44);

        stream->specUpdate(StreamSpecUpdateInfo{
            torch::tensor({{45}}, torch::kInt32), 1, /*draft_token=*/4, torch::Tensor(), torch::Tensor()});
        auto out2 = stream->nextOutput(100);
        ASSERT_TRUE(out2.ok()) << out2.status().ToString();
        ASSERT_EQ(out2.value().generate_outputs.size(), 1u);
        EXPECT_EQ(out2.value().generate_outputs[0].aux_info.iter_count, 2);
        EXPECT_EQ(out2.value().generate_outputs[0].output_ids.item<int32_t>(), 45);

        stream->releaseResource();
    }

    // 3. Chunked prefill: middle chunks advance committed_iter_count_ without
    // publishing output, so the final chunk reports the full chunk count.
    {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 2;
        input->generate_config->is_streaming   = true;
        input->generate_config->aux_info       = true;

        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        stream->setChunkSize(2);
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);

        for (int chunk = 0; chunk < 2; ++chunk) {
            ASSERT_TRUE(stream->isMiddleChunk());
            stream->step();
            stream->update(StreamUpdateInfo{.new_tokens = torch::tensor({{10}}, torch::kInt32), .num_new_tokens = 1});
            EXPECT_FALSE(stream->hasOutput());
        }
        ASSERT_FALSE(stream->isMiddleChunk());
        stream->step();
        stream->update(StreamUpdateInfo{.new_tokens = torch::tensor({{44}}, torch::kInt32), .num_new_tokens = 1});
        auto out = stream->nextOutput(100);
        ASSERT_TRUE(out.ok());
        ASSERT_EQ(out.value().generate_outputs.size(), 1u);
        EXPECT_EQ(out.value().generate_outputs[0].aux_info.iter_count, 3);
        EXPECT_EQ(stream->iterCount(), 3u);

        stream->releaseResource();
    }

    // 4. Copying a stream preserves independent value semantics for iter_count_.
    {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 4;

        auto parent =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        parent->step();
        parent->step();
        ASSERT_EQ(parent->iterCount(), 2u);

        NormalGenerateStream child(*parent);
        EXPECT_EQ(child.iterCount(), 2u);

        child.step();
        EXPECT_EQ(child.iterCount(), 3u);
        EXPECT_EQ(parent->iterCount(), 2u);

        parent->step();
        EXPECT_EQ(parent->iterCount(), 3u);
        EXPECT_EQ(child.iterCount(), 3u);
    }
}

// Releasing a finished stream publishes full-block cache keys only up to the
// materialized KV prefix [0, seqLength - 1): an un-forwarded final sampled token
// must not publish its block, while a forwarded full block skipped by the steady
// fast path is synchronized and published.
TEST_F(GenerateStreamStateTest, testReleaseSyncsOnlyMaterializedFullBlockCacheKeys) {
    auto config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/4, DataType::TYPE_INT8);
    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager       = kv_cache_manager;
    resource_context.reuse_cache         = true;
    resource_context.enable_device_cache = true;
    resource_context.role_type           = RoleType::PDFUSION;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 2048;
    model_config.attn_config.tokens_per_block = 4;

    auto make_stream = [&](std::vector<int32_t> tokens, int max_new_tokens) {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor(tokens, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = max_new_tokens;
        return std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
    };

    // Case 1: Prompt [11, 22, 33] samples 44 and finishes immediately (seqLength=4,
    // materialized KV prefix=3). Block 0 has no KV for token 44 and must not be published.
    {
        auto stream1 = make_stream({11, 22, 33}, 1);
        stream1->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream1->moveToNext(), StreamState::RUNNING);
        stream1->setIsContextStream(false);
        int err_tok = 0;
        ASSERT_TRUE(stream1->complete_token_ids_->update(
            torch::tensor({{44}}, torch::kInt32), 0, 1, 3, 16, 1024, false, stream1->streamId(), err_tok));
        ASSERT_EQ(stream1->seqLength(), 4);
        stream1->reportEvent(StreamEvents::GenerateDone);
        ASSERT_EQ(stream1->moveToNext(), StreamState::FINISHED);

        auto probe = make_stream({11, 22, 33, 44, 55}, 4);
        probe->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(probe->moveToNext(), StreamState::RUNNING);
        EXPECT_EQ(probe->reuseLength(), 0);
        probe->releaseResource();
    }

    // Case 2: Prompt [11, 22, 33] samples 44, then forwards 44 on a steady fast-path
    // round (skipping incrKVBlock) and samples 55 before finishing (seqLength=5,
    // materialized KV prefix=4). Full block [11, 22, 33, 44] is published.
    {
        auto stream2 = make_stream({11, 22, 33}, 16);
        stream2->setReserveStep(2);  // 3 + 2 = 5 => 2 blocks allocated initially.
        stream2->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream2->moveToNext(), StreamState::RUNNING);
        stream2->setIsContextStream(false);
        ASSERT_EQ(stream2->curBlocksNum(), 2u);
        ASSERT_FALSE(stream2->streamCacheResource().kvCache().lastBlockAligned());

        const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        GenerateStream::MtpAsyncDeviceState state;
        state.async_decode_state_ready          = true;
        state.previous_seq_len_upper_bound      = 3;
        state.next_seq_len_upper_bound          = 5;
        state.next_kv_cache_block_id_gpu        = torch::zeros({1, 1, 2}, cuda_i32);
        state.next_kv_cache_kernel_block_id_gpu = torch::zeros({1, 1, 2}, cuda_i32);
        stream2->setMtpAsyncDeviceState(std::move(state));
        stream2->incPendingAsyncBookkeeping();
        ASSERT_EQ(stream2->moveToNext(), StreamState::RUNNING);

        int err_tok = 0;
        ASSERT_TRUE(stream2->complete_token_ids_->update(
            torch::tensor({{44, 55}}, torch::kInt32), 0, 2, 3, 16, 1024, false, stream2->streamId(), err_tok));
        ASSERT_EQ(stream2->seqLength(), 5);
        stream2->decPendingAsyncBookkeepingAndMaybeRelease();
        stream2->reportEvent(StreamEvents::GenerateDone);
        ASSERT_EQ(stream2->moveToNext(), StreamState::FINISHED);

        auto probe = make_stream({11, 22, 33, 44, 55}, 4);
        probe->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(probe->moveToNext(), StreamState::RUNNING);
        EXPECT_EQ(probe->reuseLength(), 4);
        probe->releaseResource();
    }
}

// Terminal EOS and cancellation transitions coordinate atomically with the
// pending async bookkeeping worker so KV pages are never stranded until stream
// destruction regardless of thread interleaving.
TEST_F(GenerateStreamStateTest, testTerminalReleaseCoordinatesAtomicallyWithPendingWorker) {
    auto config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/4, DataType::TYPE_INT8);
    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());
    const auto pools = kv_cache_manager->coordinator_manager_->groupBlockPools();
    ASSERT_EQ(pools.size(), 1u);
    const size_t total_blocks = pools[0]->totalBlocksNum();

    ResourceContext resource_context;
    resource_context.cache_manager = kv_cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PDFUSION;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.attn_config.tokens_per_block = 4;

    auto make_running_stream = [&]() {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        auto s = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        s->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(s->moveToNext(), StreamState::RUNNING);
        s->setIsContextStream(false);
        return s;
    };

    for (bool test_cancel : {false, true}) {
        // Deterministic deferred path: moveToNext while pending defers release
        // until the worker finishes.
        {
            auto stream = make_running_stream();
            ASSERT_LT(pools[0]->freeBlocksNum(), total_blocks);
            stream->incPendingAsyncBookkeeping();
            if (test_cancel) {
                stream->reportEvent(StreamEvents::Error, ErrorCode::CANCELLED, "cancelled");
            } else {
                stream->reportEvent(StreamEvents::GenerateDone);
            }
            EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
            EXPECT_TRUE(stream->isDeferredReleasePending());
            EXPECT_LT(pools[0]->freeBlocksNum(), total_blocks);

            stream->decPendingAsyncBookkeepingAndMaybeRelease();
            EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_FALSE(stream->isDeferredReleasePending());
            EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
        }

        // Concurrent race: terminal moveToNext vs last worker exit must always
        // reclaim all blocks before the stream object is destroyed.
        for (int iter = 0; iter < 150; ++iter) {
            auto stream = make_running_stream();
            ASSERT_LT(pools[0]->freeBlocksNum(), total_blocks);
            stream->incPendingAsyncBookkeeping();
            if (test_cancel) {
                stream->reportEvent(StreamEvents::Error, ErrorCode::CANCELLED, "cancelled");
            } else {
                stream->reportEvent(StreamEvents::GenerateDone);
            }

            std::atomic<bool> start{false};
            std::thread       worker([&]() {
                while (!start.load(std::memory_order_acquire)) {}
                stream->decPendingAsyncBookkeepingAndMaybeRelease();
            });
            start.store(true, std::memory_order_release);
            EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
            worker.join();

            EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_FALSE(stream->isDeferredReleasePending());
            EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
        }
    }
}

// AsyncBookkeepingHold coordinates round-scoped GPU event synchronization and
// KV release across normal worker exits, host exceptions, GPU sync errors,
// runner.launch() unwinds, stream reuse, and deferred-release callback errors.
TEST_F(GenerateStreamStateTest, testAsyncBookkeepingHoldProtectsGpuAccessAndHandlesFailures) {
    struct ManualGpuFence {
        struct State {
            std::mutex              mu;
            std::condition_variable cv;
            std::atomic<bool>       done{false};
            std::atomic<bool>       sync_waiting{false};
            std::atomic<int>        query_count{0};
            std::atomic<int>        sync_count{0};
        };
        std::shared_ptr<State> state = std::make_shared<State>();

        std::function<bool()> queryFn() const {
            auto s = state;
            return [s]() {
                s->query_count.fetch_add(1, std::memory_order_acq_rel);
                return s->done.load(std::memory_order_acquire);
            };
        }
        std::function<void()> syncFn() const {
            auto s = state;
            return [s]() {
                s->sync_count.fetch_add(1, std::memory_order_acq_rel);
                if (s->done.load(std::memory_order_acquire)) {
                    return;
                }
                std::unique_lock<std::mutex> lk(s->mu);
                s->sync_waiting.store(true, std::memory_order_release);
                s->cv.notify_all();
                s->cv.wait(lk, [&]() { return s->done.load(std::memory_order_acquire); });
            };
        }
        void waitUntilSyncWaiting() const {
            std::unique_lock<std::mutex> lk(state->mu);
            state->cv.wait(lk, [&]() { return state->sync_waiting.load(std::memory_order_acquire); });
        }
        void complete() const {
            {
                std::lock_guard<std::mutex> lk(state->mu);
                state->done.store(true, std::memory_order_release);
            }
            state->cv.notify_all();
        }
    };

    auto make_env = [](uint32_t block_num) {
        CacheConfig config;
        config.dtype              = DataType::TYPE_INT8;
        config.layer_num          = 1;
        config.seq_size_per_block = 1;
        auto             swa_spec = test::makeMhaSpec("swa", /*tokens_per_block=*/1, DataType::TYPE_INT8, 1, 1);
        CacheGroupPolicy swa_policy;
        swa_policy.group_type          = CacheGroupType::SWA;
        swa_policy.active_tail_blocks  = 2;
        swa_policy.enable_prefix_reuse = false;
        swa_policy.explicit_block_num  = block_num;
        config.fromGroupedSpecs({swa_spec}, {{0}}, {CacheGroupType::SWA}, {"swa"}, {swa_policy});
        config.finalizeBlockNums(block_num, RuntimeConfig{});
        auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(mgr->init());
        return mgr;
    };

    auto       kv_cache_manager = make_env(/*block_num=*/32);
    const auto pools            = kv_cache_manager->coordinator_manager_->groupBlockPools();
    ASSERT_EQ(pools.size(), 1u);
    const size_t total_blocks = pools[0]->totalBlocksNum();

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.attn_config.tokens_per_block = 1;

    auto make_swa_stream = [&](const std::shared_ptr<KVCacheManager>& mgr, int initial_len, int reserve_step) {
        ResourceContext ctx;
        ctx.cache_manager = mgr;
        ctx.reuse_cache   = false;
        ctx.role_type     = RoleType::PDFUSION;

        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::arange(1, initial_len + 1, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 16;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
        stream->setReserveStep(reserve_step);
        stream->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
        stream->setIsContextStream(false);
        return stream;
    };

    // 1. Host exception before GPU completion still synchronizes the healthy
    //    GPU event so active block-table pages cannot be freed or re-allocated
    //    while GPU work is running; once completed, all pages return cleanly.
    {
        auto         stream    = make_swa_stream(kv_cache_manager, /*initial_len=*/4, /*reserve_step=*/2);
        const size_t held_free = pools[0]->freeBlocksNum();
        ASSERT_LT(held_free, total_blocks);
        ManualGpuFence fence;
        auto           hold = std::make_shared<AsyncBookkeepingHold>(
            std::list<GenerateStreamPtr>{stream}, fence.queryFn(), fence.syncFn());

        std::exception_ptr caught_err;
        std::thread        worker([&]() {
            std::exception_ptr worker_err;
            try {
                throw std::runtime_error("injected host dispatch exception");
            } catch (...) {
                worker_err = std::current_exception();
            }
            hold->finish(&worker_err);
            caught_err = worker_err;
        });

        fence.waitUntilSyncWaiting();
        // Terminal EOS while blocked on the unfinished GPU event must defer release
        // and keep every active block-table page allocated.
        stream->reportEvent(StreamEvents::GenerateDone);
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        EXPECT_TRUE(stream->isDeferredReleasePending());
        EXPECT_EQ(pools[0]->freeBlocksNum(), held_free);

        fence.complete();
        worker.join();
        ASSERT_NE(caught_err, nullptr);
        EXPECT_THROW(std::rethrow_exception(caught_err), std::runtime_error);
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(stream->hasUnconfirmedAsyncGpuAccess());
        EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
    }

    // 2. GPU event sync failure propagates cleanly out of normal control flow
    //    (never std::terminate), marks the stream unconfirmed + Error, and
    //    protects all active block-table pages across explicit releaseResource(),
    //    terminal GenerateDone, stream destruction, and incrKVBlock() pruning.
    {
        auto       isolated_mgr = make_env(/*block_num=*/32);
        const auto iso_pool     = isolated_mgr->coordinator_manager_->groupBlockPools()[0];

        // 2a. Explicit releaseResource()
        {
            auto         s           = make_swa_stream(isolated_mgr, /*initial_len=*/4, /*reserve_step=*/2);
            const size_t free_before = iso_pool->freeBlocksNum();
            auto         hold        = std::make_shared<AsyncBookkeepingHold>(
                std::list<GenerateStreamPtr>{s},
                []() { return false; },
                []() { throw std::runtime_error("injected gpu sync error"); });
            std::exception_ptr err;
            hold->finish(&err);
            ASSERT_NE(err, nullptr);
            EXPECT_THROW(std::rethrow_exception(err), std::runtime_error);
            EXPECT_FALSE(s->hasPendingAsyncBookkeeping());
            EXPECT_TRUE(s->hasUnconfirmedAsyncGpuAccess());
            EXPECT_TRUE(s->hasError());

            s->releaseResource();
            EXPECT_EQ(iso_pool->freeBlocksNum(), free_before);
        }

        // 2b. Terminal GenerateDone transition
        {
            auto         s           = make_swa_stream(isolated_mgr, /*initial_len=*/4, /*reserve_step=*/2);
            const size_t free_before = iso_pool->freeBlocksNum();
            auto         hold        = std::make_shared<AsyncBookkeepingHold>(
                std::list<GenerateStreamPtr>{s},
                []() { return false; },
                []() { throw std::runtime_error("injected gpu sync error"); });
            std::exception_ptr err;
            hold->finish(&err);
            ASSERT_NE(err, nullptr);

            s->reportEvent(StreamEvents::GenerateDone);
            EXPECT_EQ(s->moveToNext(), StreamState::FINISHED);
            EXPECT_EQ(iso_pool->freeBlocksNum(), free_before);
        }

        // 2c. Stream destructor (~GenerateStream) and incrKVBlock() pruning guard
        {
            auto               s        = make_swa_stream(isolated_mgr, /*initial_len=*/4, /*reserve_step=*/2);
            const BlockIdxType old_slot = s->streamCacheResource().kvCache().blocks(0, "swa")[2];
            ASSERT_FALSE(isNullBlockIdx(old_slot));
            auto hold = std::make_shared<AsyncBookkeepingHold>(
                std::list<GenerateStreamPtr>{s},
                []() { return false; },
                []() { throw std::runtime_error("injected gpu sync error"); });
            std::exception_ptr err;
            hold->finish(&err);
            ASSERT_NE(err, nullptr);

            // Growing an unconfirmed stream must not prune old SWA blocks.
            EXPECT_TRUE(s->streamCacheResource().incrKVBlock(/*seq_len_override=*/6).ok());
            EXPECT_TRUE(iso_pool->isAllocated(old_slot));

            const size_t free_before_dtor = iso_pool->freeBlocksNum();
            s.reset();
            EXPECT_EQ(iso_pool->freeBlocksNum(), free_before_dtor);
        }
    }

    // 3. Reusing a stream for Round 2 after Round 1 failed still queries and
    //    synchronizes Round 2's own GPU event, while keeping Round 1's
    //    unconfirmed protection sticky on terminal release.
    {
        auto         isolated_mgr = make_env(/*block_num=*/16);
        const auto   iso_pool     = isolated_mgr->coordinator_manager_->groupBlockPools()[0];
        auto         stream       = make_swa_stream(isolated_mgr, /*initial_len=*/4, /*reserve_step=*/2);
        const size_t free_before  = iso_pool->freeBlocksNum();

        // Round 1 fails GPU sync.
        {
            auto hold1 = std::make_shared<AsyncBookkeepingHold>(
                std::list<GenerateStreamPtr>{stream},
                []() { return false; },
                []() { throw std::runtime_error("round 1 gpu error"); });
            std::exception_ptr err1;
            hold1->finish(&err1);
            ASSERT_NE(err1, nullptr);
            EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_TRUE(stream->hasUnconfirmedAsyncGpuAccess());
        }

        // Round 2 has its own hold and fence: it must query and synchronize event2.
        ManualGpuFence fence2;
        auto           hold2 = std::make_shared<AsyncBookkeepingHold>(
            std::list<GenerateStreamPtr>{stream}, fence2.queryFn(), fence2.syncFn());
        std::thread worker2([&]() {
            std::exception_ptr err2;
            hold2->finish(&err2);
            EXPECT_EQ(err2, nullptr);
        });
        fence2.waitUntilSyncWaiting();
        EXPECT_TRUE(stream->hasPendingAsyncBookkeeping());
        fence2.complete();
        worker2.join();

        EXPECT_GE(fence2.state->query_count.load(std::memory_order_acquire), 1);
        EXPECT_EQ(fence2.state->sync_count.load(std::memory_order_acquire), 1);
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_TRUE(stream->hasUnconfirmedAsyncGpuAccess());

        stream->reportEvent(StreamEvents::GenerateDone);
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        EXPECT_EQ(iso_pool->freeBlocksNum(), free_before);
    }

    // 4. Pre-worker unwind (e.g. runner.launch() throwing a prior round's
    //    exception before executing the worker): ~AsyncBookkeepingHold()
    //    synchronizes the healthy GPU event and releases pending claims so
    //    waiters never deadlock.
    {
        auto           stream = make_swa_stream(kv_cache_manager, /*initial_len=*/4, /*reserve_step=*/2);
        ManualGpuFence fence;
        {
            auto hold = std::make_shared<AsyncBookkeepingHold>(
                std::list<GenerateStreamPtr>{stream}, fence.queryFn(), fence.syncFn());
            EXPECT_TRUE(stream->hasPendingAsyncBookkeeping());
            std::thread releaser([&]() {
                fence.waitUntilSyncWaiting();
                fence.complete();
            });
            hold.reset();  // Simulates launch() throwing before taking ownership.
            releaser.join();
        }
        EXPECT_EQ(fence.state->sync_count.load(std::memory_order_acquire), 1);
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(stream->hasUnconfirmedAsyncGpuAccess());
        stream->releaseResource();
        EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
    }

    // 5. Multi-stream deferred release when pool capacity callback throws on
    //    the first stream: both streams return all blocks, both reach
    //    pending == 0, waiters wake immediately, and the exception is captured.
    {
        auto         cb_mgr   = make_env(/*block_num=*/16);
        auto         cb_pool  = cb_mgr->coordinator_manager_->groupBlockPools()[0];
        const size_t cb_total = cb_pool->totalBlocksNum();

        auto s0 = make_swa_stream(cb_mgr, /*initial_len=*/4, /*reserve_step=*/0);
        auto s1 = make_swa_stream(cb_mgr, /*initial_len=*/4, /*reserve_step=*/0);
        ASSERT_EQ(cb_pool->freeBlocksNum(), cb_total - 4);

        auto hold = std::make_shared<AsyncBookkeepingHold>(std::list<GenerateStreamPtr>{s0, s1},
                                                           std::shared_ptr<torch::Event>{});
        s0->reportEvent(StreamEvents::GenerateDone);
        s1->reportEvent(StreamEvents::GenerateDone);
        ASSERT_EQ(s0->moveToNext(), StreamState::FINISHED);
        ASSERT_EQ(s1->moveToNext(), StreamState::FINISHED);
        ASSERT_TRUE(s0->isDeferredReleasePending());
        ASSERT_TRUE(s1->isDeferredReleasePending());

        std::atomic<int> cb_calls{0};
        cb_pool->setCapacityChangeCallback([&]() {
            if (cb_calls.fetch_add(1, std::memory_order_acq_rel) == 0) {
                throw std::runtime_error("injected capacity callback error");
            }
        });

        std::exception_ptr err;
        hold->finish(&err);
        ASSERT_NE(err, nullptr);
        EXPECT_THROW(std::rethrow_exception(err), std::runtime_error);
        EXPECT_FALSE(s0->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(s1->hasPendingAsyncBookkeeping());
        EXPECT_EQ(cb_pool->freeBlocksNum(), cb_total);
    }

    // 6. CompleteTokenIds::getLatestTokens checks bounds in full size_t width.
    {
        auto stream = make_swa_stream(kv_cache_manager, /*initial_len=*/4, /*reserve_step=*/2);
        EXPECT_EQ(stream->completeTokenIdsPtr()->getLatestTokens(2).size(), 2u);
        const size_t huge_count = static_cast<size_t>(std::numeric_limits<int>::max()) + 2;
        EXPECT_THROW((void)stream->completeTokenIdsPtr()->getLatestTokens(huge_count), std::exception);
        stream->releaseResource();
        EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
    }

    // 7. Real CUDA torch::Event through AsyncBookkeepingHold (safe under both
    //    default launch and CUDA_LAUNCH_BLOCKING=1).
#if USING_CUDA
    if (torch::cuda::is_available()) {
        auto gpu_stream = c10::cuda::getStreamFromPool();
        auto ev         = std::make_shared<torch::Event>(torch::kCUDA);
        {
            c10::cuda::CUDAStreamGuard guard(gpu_stream);
            (void)torch::ones({16, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).sum();
            ev->record(gpu_stream);
        }
        auto               stream = make_swa_stream(kv_cache_manager, /*initial_len=*/4, /*reserve_step=*/2);
        auto               hold   = std::make_shared<AsyncBookkeepingHold>(std::list<GenerateStreamPtr>{stream}, ev);
        std::exception_ptr err;
        hold->finish(&err);
        EXPECT_EQ(err, nullptr);
        EXPECT_TRUE(ev->query());
        stream->reportEvent(StreamEvents::GenerateDone);
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        EXPECT_EQ(pools[0]->freeBlocksNum(), total_blocks);
    }
#endif
}

// genTimeline() short-circuits when disabled, uses published async sequence
// bounds without racing worker commits, and joins only when an MTP bound
// interval straddles the profile_step threshold.
TEST_F(GenerateStreamStateTest, testGenTimelineResolvesPendingAsyncSequenceLengthWithoutRace) {
    auto config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/4, DataType::TYPE_INT8);
    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = kv_cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PDFUSION;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.attn_config.tokens_per_block = 4;

    auto make_stream = [&](bool gen_timeline, int profile_step) {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);  // input_len = 2
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        input->generate_config->gen_timeline   = gen_timeline;
        input->generate_config->profile_step   = profile_step;
        auto s = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        s->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(s->moveToNext(), StreamState::RUNNING);
        s->setIsContextStream(false);
        return s;
    };

    // 1. Disabled gen_timeline short-circuits immediately without waiting.
    {
        auto s = make_stream(/*gen_timeline=*/false, /*profile_step=*/2);
        s->incPendingAsyncBookkeeping();
        EXPECT_FALSE(s->genTimeline());
        s->decPendingAsyncBookkeepingAndMaybeRelease();
        s->releaseResource();
    }

    // 2. MTP lower > threshold (threshold = 2 + 2 - 1 = 3, lower = 4, upper = 6)
    //    returns false immediately without waiting and matches post-commit state.
    {
        auto s = make_stream(/*gen_timeline=*/true, /*profile_step=*/2);
        s->setSeqLength(3);
        GenerateStream::MtpAsyncDeviceState mtp_state;
        mtp_state.next_seq_len_lower_bound = 4;
        mtp_state.next_seq_len_upper_bound = 6;
        s->setMtpAsyncDeviceState(std::move(mtp_state));
        s->incPendingAsyncBookkeeping();

        EXPECT_FALSE(s->genTimeline());

        s->setSeqLength(4);
        s->decPendingAsyncBookkeepingAndMaybeRelease();
        EXPECT_FALSE(s->genTimeline());
        s->releaseResource();
    }

    // 3. MTP upper <= threshold (threshold = 2 + 5 - 1 = 6, lower = 4, upper = 6)
    //    returns true immediately without waiting.
    {
        auto s = make_stream(/*gen_timeline=*/true, /*profile_step=*/5);
        s->setSeqLength(3);
        GenerateStream::MtpAsyncDeviceState mtp_state;
        mtp_state.next_seq_len_lower_bound = 4;
        mtp_state.next_seq_len_upper_bound = 6;
        s->setMtpAsyncDeviceState(std::move(mtp_state));
        s->incPendingAsyncBookkeeping();

        EXPECT_TRUE(s->genTimeline());

        s->decPendingAsyncBookkeepingAndMaybeRelease();
        s->releaseResource();
    }

    // 4. MTP interval straddles threshold (threshold = 4, lower = 4, upper = 6):
    //    joins the in-flight worker and resolves against the committed seqLength.
    for (int committed_len : {4, 5}) {
        auto s = make_stream(/*gen_timeline=*/true, /*profile_step=*/3);
        s->setSeqLength(3);
        GenerateStream::MtpAsyncDeviceState mtp_state;
        mtp_state.next_seq_len_lower_bound = 4;
        mtp_state.next_seq_len_upper_bound = 6;
        s->setMtpAsyncDeviceState(std::move(mtp_state));
        s->incPendingAsyncBookkeeping();

        std::thread worker([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            {
                std::lock_guard<std::mutex> lk(*s->mutex_);
                s->setSeqLength(committed_len);
            }
            s->decPendingAsyncBookkeepingAndMaybeRelease();
        });

        EXPECT_EQ(s->genTimeline(), committed_len <= 4);
        worker.join();
        EXPECT_FALSE(s->hasPendingAsyncBookkeeping());
        EXPECT_EQ(s->genTimeline(), committed_len <= 4);
        s->releaseResource();
    }

    // 5. Normal async decode uses published next_real_seq_len directly.
    {
        auto s = make_stream(/*gen_timeline=*/true, /*profile_step=*/2);  // threshold = 3
        s->setSeqLength(3);
        GenerateStream::NormalAsyncDeviceState normal_state;
        normal_state.last_real_seq_len = 3;
        normal_state.next_real_seq_len = 4;
        s->setNormalAsyncDeviceState(std::move(normal_state));
        s->incPendingAsyncBookkeeping();

        EXPECT_FALSE(s->genTimeline());

        s->setSeqLength(4);
        s->decPendingAsyncBookkeepingAndMaybeRelease();
        EXPECT_FALSE(s->genTimeline());
        s->releaseResource();
    }
}

// GPU sync failure must publish StreamEvents::Error before pending bookkeeping
// reaches zero, and worker-side deferred release must re-defer without waiting
// if another claim arrives concurrently.
TEST_F(GenerateStreamStateTest, testGpuSyncFailurePublishesErrorBeforeClearingPendingCount) {
    auto config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/4, DataType::TYPE_INT8);
    auto kv_cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
    ASSERT_TRUE(kv_cache_manager->init());
    const auto pools = kv_cache_manager->coordinator_manager_->groupBlockPools();
    ASSERT_EQ(pools.size(), 1u);
    const size_t total_blocks = pools[0]->totalBlocksNum();

    ResourceContext resource_context;
    resource_context.cache_manager = kv_cache_manager;
    resource_context.reuse_cache   = false;
    resource_context.role_type     = RoleType::PDFUSION;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.attn_config.tokens_per_block = 4;

    auto make_running_stream = [&]() {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        auto s = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
        s->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(s->moveToNext(), StreamState::RUNNING);
        s->setIsContextStream(false);
        return s;
    };

    // 1. Hold stream mutex while the last worker hits a GPU sync failure.
    //    Pending count must remain > 0 until Error is published under mutex_,
    //    so any scheduler waking from waitPendingAsyncBookkeeping() immediately
    //    observes Error and transitions to FINISHED (never RUNNING).
    {
        auto         stream      = make_running_stream();
        const size_t free_before = pools[0]->freeBlocksNum();
        ASSERT_LT(free_before, total_blocks);

        std::atomic<bool> sync_invoked{false};
        auto              hold = std::make_shared<AsyncBookkeepingHold>(
            std::list<GenerateStreamPtr>{stream},
            []() { return false; },
            [&]() {
                sync_invoked.store(true, std::memory_order_release);
                throw std::runtime_error("injected gpu sync error");
            });

        std::exception_ptr           worker_err;
        std::unique_lock<std::mutex> stream_lock(*stream->mutex_);
        std::thread                  worker([&]() { hold->finish(&worker_err); });

        while (!sync_invoked.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        // While the scheduler holds stream->mutex_, the worker must not clear
        // pending or publish unconfirmed GPU access ahead of StreamEvents::Error.
        EXPECT_EQ(stream->pendingAsyncBookkeepingCount(), 1);
        EXPECT_FALSE(stream->hasUnconfirmedAsyncGpuAccess());
        EXPECT_FALSE(stream->hasErrorWithoutLock());

        stream_lock.unlock();
        stream->waitPendingAsyncBookkeeping();

        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_TRUE(stream->hasUnconfirmedAsyncGpuAccess());
        EXPECT_TRUE(stream->hasErrorWithoutLock());
        EXPECT_TRUE(stream->hasError());
        EXPECT_TRUE(stream->hasEvent(StreamEvents::Error));
        EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        EXPECT_EQ(pools[0]->freeBlocksNum(), free_before);

        worker.join();
        ASSERT_NE(worker_err, nullptr);
        EXPECT_THROW(std::rethrow_exception(worker_err), std::runtime_error);
        EXPECT_EQ(pools[0]->freeBlocksNum(), free_before);
    }

    // 2. Worker-side deferred release must re-defer without blocking on cv if
    //    another claim is active, then release cleanly once that claim exits.
    {
        const size_t baseline_free = pools[0]->freeBlocksNum();
        auto         stream        = make_running_stream();
        const size_t free_before   = pools[0]->freeBlocksNum();
        ASSERT_LT(free_before, baseline_free);

        stream->incPendingAsyncBookkeeping();
        stream->tryReleaseResourceAfterBookkeeping();
        EXPECT_TRUE(stream->isDeferredReleasePending());
        EXPECT_EQ(pools[0]->freeBlocksNum(), free_before);

        stream->decPendingAsyncBookkeepingAndMaybeRelease();
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(stream->isDeferredReleasePending());
        EXPECT_EQ(pools[0]->freeBlocksNum(), baseline_free);
    }
}

// Multi-group (FULL + SWA) KV release isolates per-group unreference callback
// exceptions so every remaining group still returns its blocks before clearing
// stream block IDs, across deferred worker release, explicit releaseResource(),
// terminal moveToNext(), and stream destruction.
TEST_F(GenerateStreamStateTest, testMultiGroupReleaseExceptionFreesAllGroupsWithoutLeak) {
    auto make_hybrid_env = [](uint32_t block_num) {
        CacheConfig config;
        config.dtype               = DataType::TYPE_INT8;
        config.layer_num           = 2;
        config.seq_size_per_block  = 4;
        auto             full_spec = test::makeMhaSpec("full", /*tokens_per_block=*/4, DataType::TYPE_INT8, 1, 1);
        auto             swa_spec  = test::makeMhaSpec("swa", /*tokens_per_block=*/4, DataType::TYPE_INT8, 1, 1);
        CacheGroupPolicy full_policy;
        full_policy.group_type          = CacheGroupType::FULL;
        full_policy.enable_prefix_reuse = false;
        full_policy.explicit_block_num  = block_num;
        CacheGroupPolicy swa_policy;
        swa_policy.group_type          = CacheGroupType::SWA;
        swa_policy.active_tail_blocks  = 2;
        swa_policy.enable_prefix_reuse = false;
        swa_policy.explicit_block_num  = block_num;
        config.fromGroupedSpecs({full_spec, swa_spec},
                                {{0}, {1}},
                                {CacheGroupType::FULL, CacheGroupType::SWA},
                                {"full", "swa"},
                                {full_policy, swa_policy});
        config.finalizeBlockNums(block_num, RuntimeConfig{});
        auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(mgr->init());
        return mgr;
    };

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len                  = 64;
    model_config.attn_config.tokens_per_block = 4;

    auto make_hybrid_stream = [&](const std::shared_ptr<KVCacheManager>& mgr) {
        ResourceContext ctx;
        ctx.cache_manager = mgr;
        ctx.reuse_cache   = false;
        ctx.role_type     = RoleType::PDFUSION;

        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        auto s = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
        s->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(s->moveToNext(), StreamState::RUNNING);
        s->setIsContextStream(false);
        return s;
    };

    // 1. Deferred worker release via AsyncBookkeepingHold when group 0 callback throws.
    {
        auto       mgr   = make_hybrid_env(/*block_num=*/16);
        const auto pools = mgr->coordinator_manager_->groupBlockPools();
        ASSERT_EQ(pools.size(), 2u);
        const size_t full_total = pools[0]->totalBlocksNum();
        const size_t swa_total  = pools[1]->totalBlocksNum();
        ASSERT_EQ(full_total, 15u);
        ASSERT_EQ(swa_total, 15u);

        auto stream = make_hybrid_stream(mgr);
        ASSERT_LT(pools[0]->freeBlocksNum(), full_total);
        ASSERT_LT(pools[1]->freeBlocksNum(), swa_total);

        auto hold = std::make_shared<AsyncBookkeepingHold>(std::list<GenerateStreamPtr>{stream},
                                                           std::shared_ptr<torch::Event>{});
        stream->reportEvent(StreamEvents::GenerateDone);
        ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
        ASSERT_TRUE(stream->isDeferredReleasePending());

        pools[0]->setCapacityChangeCallback([]() { throw std::runtime_error("pool 0 capacity callback error"); });

        std::exception_ptr err;
        hold->finish(&err);
        ASSERT_NE(err, nullptr);
        EXPECT_THROW(std::rethrow_exception(err), std::runtime_error);
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);

        stream.reset();
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);
    }

    // 2. Explicit releaseResource() when group 0 callback throws.
    {
        auto       mgr   = make_hybrid_env(/*block_num=*/16);
        const auto pools = mgr->coordinator_manager_->groupBlockPools();
        ASSERT_EQ(pools.size(), 2u);
        const size_t full_total = pools[0]->totalBlocksNum();
        const size_t swa_total  = pools[1]->totalBlocksNum();

        auto stream = make_hybrid_stream(mgr);
        pools[0]->setCapacityChangeCallback([]() { throw std::runtime_error("pool 0 explicit release error"); });

        EXPECT_THROW(stream->releaseResource(), std::runtime_error);
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);

        stream.reset();
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);
    }

    // 3. Terminal moveToNext() immediate release when group 0 callback throws
    //    must still wake finishOrCancel() waiters and free all groups.
    {
        auto       mgr   = make_hybrid_env(/*block_num=*/16);
        const auto pools = mgr->coordinator_manager_->groupBlockPools();
        ASSERT_EQ(pools.size(), 2u);
        const size_t full_total = pools[0]->totalBlocksNum();
        const size_t swa_total  = pools[1]->totalBlocksNum();

        auto stream = make_hybrid_stream(mgr);
        pools[0]->setCapacityChangeCallback([]() { throw std::runtime_error("pool 0 moveToNext error"); });

        std::atomic<bool> cancel_finished{false};
        std::thread       canceller([&]() {
            cancel_finished.store(stream->finishOrCancel(/*wait_timeout_ms=*/2000, "test cancel"),
                                  std::memory_order_release);
        });
        while (!stream->hasEvent(StreamEvents::Error)) {
            std::this_thread::yield();
        }
        EXPECT_THROW((void)stream->moveToNext(), std::runtime_error);
        canceller.join();

        EXPECT_TRUE(cancel_finished.load(std::memory_order_acquire));
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);
    }

    // 4. Stream destructor (~GenerateStream) swallows release callback exceptions
    //    without terminating the process and still frees every group.
    {
        auto       mgr   = make_hybrid_env(/*block_num=*/16);
        const auto pools = mgr->coordinator_manager_->groupBlockPools();
        ASSERT_EQ(pools.size(), 2u);
        const size_t full_total = pools[0]->totalBlocksNum();
        const size_t swa_total  = pools[1]->totalBlocksNum();

        auto stream = make_hybrid_stream(mgr);
        pools[0]->setCapacityChangeCallback([]() { throw std::runtime_error("pool 0 destructor error"); });

        stream.reset();
        EXPECT_EQ(pools[0]->freeBlocksNum(), full_total);
        EXPECT_EQ(pools[1]->freeBlocksNum(), swa_total);
    }
}

// Covers normal async decode boundary joining without device-state mirrors,
// narrow SWA pruning protection while async bookkeeping is pending, and
// syncCacheKeys truncation against max_seq_len.
TEST_F(GenerateStreamStateTest, testNormalAsyncBoundarySwaGuardAndCacheKeyTruncate) {
    // 1. Normal async decode without NormalAsyncDeviceState (RTP_LLM_DEVICE_INPUT=0)
    //    must join the pending worker when the pending token crosses a block boundary.
    {
        auto config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/2, DataType::TYPE_INT8);
        auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(mgr->init());

        ResourceContext ctx;
        ctx.cache_manager = mgr;
        ctx.reuse_cache   = false;
        ctx.role_type     = RoleType::PDFUSION;

        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len                  = 64;
        model_config.attn_config.tokens_per_block = 2;

        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 16;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
        stream->setIsContextStream(false);
        ASSERT_EQ(stream->curBlocksNum(), 2u);

        // Non-boundary (3 -> 4 with block_size=2, 2 blocks allocated) does not join.
        stream->incPendingAsyncBookkeeping();
        EXPECT_FALSE(stream->needsAsyncBookkeepingJoinForKvReservation());
        EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
        EXPECT_EQ(stream->curBlocksNum(), 2u);
        stream->setSeqLength(4);
        stream->decPendingAsyncBookkeepingAndMaybeRelease();

        // Boundary (4 -> 5 with block_size=2, 2 blocks allocated) joins worker first
        // so incrKVBlock allocates the 3rd page for token 5.
        stream->incPendingAsyncBookkeeping();
        EXPECT_TRUE(stream->needsAsyncBookkeepingJoinForKvReservation());
        std::thread worker([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            {
                std::lock_guard<std::mutex> lk(*stream->mutex_);
                stream->setSeqLength(5);
            }
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        });
        EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
        worker.join();
        EXPECT_EQ(stream->seqLength(), 5);
        EXPECT_EQ(stream->curBlocksNum(), 3u);
        stream->releaseResource();
    }

    // 2. Narrow SWA (block=16, window=17, active_tail=2) normal async decode
    //    growing 32 -> 33 while pending=1 must not prune page 0 before GPU finishes.
    {
        CacheConfig config;
        config.dtype              = DataType::TYPE_INT8;
        config.layer_num          = 1;
        config.seq_size_per_block = 16;
        auto             swa_spec = test::makeMhaSpec("swa", /*tokens_per_block=*/16, DataType::TYPE_INT8, 1, 1);
        CacheGroupPolicy swa_policy;
        swa_policy.group_type          = CacheGroupType::SWA;
        swa_policy.sliding_window_size = 17;
        swa_policy.active_tail_blocks  = 2;
        swa_policy.enable_prefix_reuse = false;
        swa_policy.explicit_block_num  = 16;
        config.fromGroupedSpecs({swa_spec}, {{0}}, {CacheGroupType::SWA}, {"swa"}, {swa_policy});
        config.finalizeBlockNums(16, RuntimeConfig{});
        auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(mgr->init());
        const auto pool = mgr->coordinator_manager_->groupBlockPools()[0];

        ResourceContext ctx;
        ctx.cache_manager = mgr;
        ctx.reuse_cache   = false;
        ctx.role_type     = RoleType::PDFUSION;

        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len                  = 128;
        model_config.attn_config.tokens_per_block = 16;

        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::arange(1, 33, torch::kInt32);  // len = 32 (2 pages)
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 16;
        // 2a. Direct incrKVBlock while pending=1 must not prune page 0 before GPU finishes.
        {
            auto s = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
            s->reportEvent(StreamEvents::CanRun);
            ASSERT_EQ(s->moveToNext(), StreamState::RUNNING);
            s->setIsContextStream(false);
            const BlockIdxType page0 = s->streamCacheResource().kvCache().blocks(0, "swa")[0];
            ASSERT_FALSE(isNullBlockIdx(page0));
            ASSERT_TRUE(pool->isAllocated(page0));

            GenerateStream::NormalAsyncDeviceState normal_state;
            normal_state.last_real_seq_len = 32;
            normal_state.next_real_seq_len = 33;
            s->setNormalAsyncDeviceState(std::move(normal_state));
            s->incPendingAsyncBookkeeping();

            EXPECT_FALSE(s->canPruneSwaBlocksWhileAsyncPending());
            EXPECT_TRUE(s->needsAsyncBookkeepingJoinForKvReservation());
            ASSERT_TRUE(s->streamCacheResource().incrKVBlock(/*seq_len_override=*/33).ok());
            EXPECT_EQ(s->curBlocksNum(), 3u);
            EXPECT_TRUE(pool->isAllocated(page0));
            EXPECT_EQ(s->streamCacheResource().kvCache().blocks(0, "swa")[0], page0);

            s->decPendingAsyncBookkeepingAndMaybeRelease();
            s->releaseResource();
        }

        // 2b. moveToNext() on a narrow SWA boundary joins the pending worker so GPU
        //     finishes before pruning page 0 (works cleanly without retry).
        {
            auto stream = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
            stream->reportEvent(StreamEvents::CanRun);
            ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
            stream->setIsContextStream(false);
            const BlockIdxType page0 = stream->streamCacheResource().kvCache().blocks(0, "swa")[0];
            ASSERT_FALSE(isNullBlockIdx(page0));
            ASSERT_TRUE(pool->isAllocated(page0));

            GenerateStream::NormalAsyncDeviceState normal_state;
            normal_state.last_real_seq_len = 32;
            normal_state.next_real_seq_len = 33;
            stream->setNormalAsyncDeviceState(std::move(normal_state));
            stream->incPendingAsyncBookkeeping();

            ASSERT_TRUE(stream->needsAsyncBookkeepingJoinForKvReservation());
            std::atomic<bool> worker_done{false};
            std::thread       worker([&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                {
                    std::lock_guard<std::mutex> lk(*stream->mutex_);
                    stream->setSeqLength(33);
                }
                worker_done.store(true, std::memory_order_release);
                stream->decPendingAsyncBookkeepingAndMaybeRelease();
            });
            EXPECT_EQ(stream->moveToNext(), StreamState::RUNNING);
            worker.join();
            EXPECT_TRUE(worker_done.load(std::memory_order_acquire));
            EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
            EXPECT_FALSE(pool->isAllocated(page0));
            stream->releaseResource();
        }
    }

    // 3. syncCacheKeys(max_seq_len) truncates already-computed full keys past max_seq_len.
    {
        auto config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/8, /*tokens_per_block=*/2, DataType::TYPE_INT8);
        auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        ASSERT_TRUE(mgr->init());

        ResourceContext ctx;
        ctx.cache_manager       = mgr;
        ctx.reuse_cache         = true;
        ctx.enable_device_cache = true;
        ctx.role_type           = RoleType::PDFUSION;

        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len                  = 64;
        model_config.attn_config.tokens_per_block = 2;

        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({10, 20, 30, 40}, torch::kInt32);  // 2 full blocks
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 8;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, ctx, nullptr);
        stream->reportEvent(StreamEvents::CanRun);
        ASSERT_EQ(stream->moveToNext(), StreamState::RUNNING);
        ASSERT_EQ(stream->cacheKeys(0).size(), 2u);

        mgr->syncCacheKeys(stream->kvCachePtr(), stream->completeTokenIdsPtr(), /*max_seq_len=*/3);
        EXPECT_EQ(stream->cacheKeys(0).size(), 1u);
        stream->releaseResource();
    }
}

}  // namespace rtp_llm
