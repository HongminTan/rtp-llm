#include "gtest/gtest.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "autil/NetUtil.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

// The real NormalCacheStore executor sends to this transport. Tests choose when
// each peer completes, without relying on NIC availability or callback timing.
class ControlledLoadMessager: public Messager {
public:
    ControlledLoadMessager(): Messager(nullptr, nullptr, nullptr) {}
    bool init(MessagerInitParams) override {
        return true;
    }
    bool generateBlockInfo(::BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return false;
    }
    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        callback = std::move(request->callback);
        calls_[request->ip].push_back({request, std::move(callback), collector});
        changed_.notify_all();
    }
    bool waitCalls(const std::string& peer, size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, 3s, [&] { return calls_[peer].size() >= count; });
    }
    size_t count(const std::string& peer) {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_[peer].size();
    }
    std::shared_ptr<LoadRequest> request(const std::string& peer, size_t attempt) {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_.at(peer).at(attempt).request;
    }
    void complete(const std::string& peer, size_t attempt, CacheStoreErrorCode ec = CacheStoreErrorCode::None) {
        CacheStoreLoadDoneCallback                            callback;
        std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto&                       call = calls_.at(peer).at(attempt);
            callback                         = std::move(call.callback);
            collector                        = call.collector;
        }
        ASSERT_TRUE(callback);
        collector->markEnd(ec == CacheStoreErrorCode::None);
        callback(ec == CacheStoreErrorCode::None, ec);
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        calls_.clear();
    }

private:
    struct Call {
        std::shared_ptr<LoadRequest>                          request;
        CacheStoreLoadDoneCallback                            callback;
        std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector;
    };
    std::mutex                               mutex_;
    std::condition_variable                  changed_;
    std::map<std::string, std::vector<Call>> calls_;
};

struct WorkerGate {
    std::promise<void>       entered;
    std::promise<void>       released;
    std::shared_future<void> release_future = released.get_future().share();
    std::atomic_bool         open{false};
    void                     release() {
        if (!open.exchange(true)) {
            released.set_value();
        }
    }
};

class CacheLoadConnectRetryTest: public testing::Test {
protected:
    void SetUp() override {
        CacheStoreInitParams params;
        params.listen_port   = autil::NetUtil::randomPort();
        params.rdma_mode     = false;
        params.enable_metric = false;
        store_               = NormalCacheStore::createNormalCacheStore(params);
        ASSERT_TRUE(store_);
        // Use a real bounded executor without the unrelated store-readiness task.
        store_->thread_pool_close_ = true;
        store_->thread_pool_->stop();
        replacePool(2, 64);
        store_->device_id_ = -1;
        messager_          = std::make_shared<ControlledLoadMessager>();
        store_->messager_  = messager_;
        buffer_            = std::make_shared<RequestBlockBuffer>("request");
        buffer_->addBlock("block", std::make_shared<char>(0), 1, false, true);
    }
    void TearDown() override {
        for (auto& gate : gates_) {
            gate->release();
        }
        if (store_) {
            store_->thread_pool_->stop();
        }
        if (messager_) {
            messager_->clear();
        }
        store_.reset();
    }
    void replacePool(size_t threads, size_t queue_size) {
        store_->thread_pool_->stop();
        pool_ = std::make_shared<autil::ThreadPool>(threads, queue_size);
        ASSERT_TRUE(pool_->start("PeerRetryTest"));
        store_->thread_pool_ = pool_;
    }
    std::shared_ptr<WorkerGate> blockWorker() {
        auto gate = std::make_shared<WorkerGate>();
        gates_.push_back(gate);
        auto entered = gate->entered.get_future();
        EXPECT_EQ(pool_->pushTask(
                      [gate] {
                          gate->entered.set_value();
                          gate->release_future.wait();
                      },
                      false),
                  autil::ThreadPoolBase::ERROR_NONE);
        EXPECT_EQ(entered.wait_for(3s), std::future_status::ready);
        return gate;
    }
    std::shared_ptr<LoadContext> start(const std::string&                peer       = "p0",
                                       int64_t                           retries    = 2,
                                       int64_t                           timeout_ms = 5000,
                                       bool                              combine    = true,
                                       std::shared_ptr<std::atomic_bool> stopped    = nullptr,
                                       LoadContext::CheckCancelFunc      cancel     = nullptr) {
        auto context = std::make_shared<LoadContext>(store_, combine);
        context->load({buffer_}, peer, 10000, 10001, timeout_ms, std::move(cancel), 2, 1, retries, stopped);
        return context;
    }
    ErrorCode result(const std::shared_ptr<LoadContext>& context) {
        context->waitDone();
        return context->getErrorInfo().code();
    }
    std::shared_ptr<NormalCacheStore>        store_;
    std::shared_ptr<autil::ThreadPool>       pool_;
    std::shared_ptr<ControlledLoadMessager>  messager_;
    std::shared_ptr<RequestBlockBuffer>      buffer_;
    std::vector<std::shared_ptr<WorkerGate>> gates_;
};

TEST_F(CacheLoadConnectRetryTest, LaterPeerRetriesWhileFirstPeerIsStillPending) {
    auto first  = start("p0");
    auto second = start("p1");
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    ASSERT_TRUE(messager_->waitCalls("p1", 1));
    messager_->complete("p1", 0, CacheStoreErrorCode::LoadConnectFailed);
    ASSERT_TRUE(messager_->waitCalls("p1", 2));
    messager_->complete("p1", 1);
    EXPECT_EQ(result(second), ErrorCode::NONE_ERROR);
    // No waiter has advanced past p0, and p0 has not received any completion.
    EXPECT_EQ(first->done_layer_cnt_.load(), 0);
    EXPECT_EQ(messager_->count("p0"), 1);
    messager_->complete("p0", 0);
    EXPECT_EQ(result(first), ErrorCode::NONE_ERROR);
}

TEST_F(CacheLoadConnectRetryTest, BothPeersRetryWithoutWaitingForEachOthersTransfer) {
    auto first  = start("p0");
    auto second = start("p1");
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    ASSERT_TRUE(messager_->waitCalls("p1", 1));
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    messager_->complete("p1", 0, CacheStoreErrorCode::LoadRdmaConnectFailed);
    ASSERT_TRUE(messager_->waitCalls("p0", 2));
    ASSERT_TRUE(messager_->waitCalls("p1", 2));
    EXPECT_EQ(first->done_layer_cnt_.load(), 0);
    EXPECT_EQ(second->done_layer_cnt_.load(), 0);
    messager_->complete("p1", 1);
    EXPECT_EQ(result(second), ErrorCode::NONE_ERROR);
    messager_->complete("p0", 1);
    EXPECT_EQ(result(first), ErrorCode::NONE_ERROR);
}

TEST_F(CacheLoadConnectRetryTest, RecoversTransientErrorsWithoutResettingPayloadOrDeadline) {
    auto context = start();
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    ASSERT_TRUE(messager_->waitCalls("p0", 2));
    messager_->complete("p0", 1, CacheStoreErrorCode::LoadRdmaConnectFailed);
    ASSERT_TRUE(messager_->waitCalls("p0", 3));
    auto first = messager_->request("p0", 0);
    auto last  = messager_->request("p0", 2);
    EXPECT_EQ(last->request_block_buffer, first->request_block_buffer);
    EXPECT_EQ(last->request_block_buffer->getBlock("block"), buffer_->getBlock("block"));
    EXPECT_EQ(last->rdma_port, first->rdma_port);
    EXPECT_EQ(last->partition_count, 2);
    EXPECT_EQ(last->partition_id, 1);
    EXPECT_LE(last->timeout_ms, first->timeout_ms);
    EXPECT_EQ(last->control->deadline, first->control->deadline);
    messager_->complete("p0", 2);
    EXPECT_EQ(result(context), ErrorCode::NONE_ERROR);
    EXPECT_EQ(context->done_layer_cnt_.load(), 1);
}

TEST_F(CacheLoadConnectRetryTest, ExhaustedRetriesReturnLastFailureAndItsBuffers) {
    auto context = start("p0", 1);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    ASSERT_TRUE(messager_->waitCalls("p0", 2));
    messager_->complete("p0", 1, CacheStoreErrorCode::LoadRdmaConnectFailed);
    EXPECT_EQ(result(context), ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED);
    EXPECT_EQ(messager_->count("p0"), 2);
    EXPECT_FALSE(context->failedBlockDebugInfos().empty());
}

TEST_F(CacheLoadConnectRetryTest, DoesNotRetryWriteTimeoutSendOrQueueErrors) {
    const std::vector<CacheStoreErrorCode> errors = {CacheStoreErrorCode::LoadRdmaWriteFailed,
                                                     CacheStoreErrorCode::LoadBufferTimeout,
                                                     CacheStoreErrorCode::CallPrefillTimeout,
                                                     CacheStoreErrorCode::LoadSendRequestFailed,
                                                     CacheStoreErrorCode::PushWorkerItemFailed};
    for (size_t i = 0; i < errors.size(); ++i) {
        const auto peer    = "p" + std::to_string(i);
        auto       context = start(peer);
        ASSERT_TRUE(messager_->waitCalls(peer, 1));
        messager_->complete(peer, 0, errors[i]);
        EXPECT_NE(result(context), ErrorCode::NONE_ERROR);
        EXPECT_EQ(messager_->count(peer), 1);
    }
}

TEST_F(CacheLoadConnectRetryTest, DisabledRetryAndTcpLoadsKeepSingleAttempt) {
    auto disabled = start("p0", 0);
    auto tcp      = start("p1", 2, 5000, false);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    ASSERT_TRUE(messager_->waitCalls("p1", 1));
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    messager_->complete("p1", 0, CacheStoreErrorCode::LoadConnectFailed);
    EXPECT_EQ(result(disabled), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(result(tcp), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(messager_->count("p0"), 1);
    EXPECT_EQ(messager_->count("p1"), 1);
}

TEST_F(CacheLoadConnectRetryTest, FullExecutorQueueFinishesInsteadOfBlockingCompletionThread) {
    replacePool(1, 1);
    auto context = start();
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    auto gate = blockWorker();
    ASSERT_EQ(pool_->pushTask([] {}, false), autil::ThreadPoolBase::ERROR_NONE);
    std::promise<void> completed;
    auto               done = completed.get_future();
    std::thread        completion([&] {
        messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
        completed.set_value();
    });
    auto               status = done.wait_for(1s);
    // Always unblock the executor before joining, including on assertion failure.
    gate->release();
    completion.join();
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_EQ(result(context), ErrorCode::CACHE_STORE_PUSH_ITEM_FAILED);
    EXPECT_EQ(messager_->count("p0"), 1);
}

TEST_F(CacheLoadConnectRetryTest, CancellationDropsQueuedRetryBeforeSending) {
    replacePool(1, 8);
    auto stopped = std::make_shared<std::atomic_bool>(false);
    auto context = start("p0", 2, 5000, true, stopped);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    auto gate = blockWorker();
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    stopped->store(true);
    gate->release();
    pool_->waitFinish();
    EXPECT_EQ(result(context), ErrorCode::CANCELLED);
    EXPECT_EQ(messager_->count("p0"), 1);
}

TEST_F(CacheLoadConnectRetryTest, DeadlineDropsQueuedRetryBeforeSending) {
    replacePool(1, 8);
    auto context = start("p0", 2, 200);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    auto gate = blockWorker();
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    // Exercise expiration in the queued task, not only the waiter's timeout path.
    std::this_thread::sleep_until(context->load_control_->deadline + 5ms);
    gate->release();
    pool_->waitFinish();
    EXPECT_EQ(result(context), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    EXPECT_EQ(messager_->count("p0"), 1);
}

TEST_F(CacheLoadConnectRetryTest, QueueDelayIsDeductedFromRetryTimeout) {
    replacePool(1, 8);
    auto context = start();
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    auto gate = blockWorker();
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    std::this_thread::sleep_for(40ms);
    gate->release();
    ASSERT_TRUE(messager_->waitCalls("p0", 2));
    EXPECT_LE(messager_->request("p0", 1)->timeout_ms, messager_->request("p0", 0)->timeout_ms - 30);
    messager_->complete("p0", 1);
    EXPECT_EQ(result(context), ErrorCode::NONE_ERROR);
}

TEST_F(CacheLoadConnectRetryTest, WaiterCancellationStopsOtherPeersAndLateCallbacksStayFinal) {
    auto stopped = std::make_shared<std::atomic_bool>(false);
    auto first   = start("p0", 2, 5000, true, stopped, [] { return true; });
    auto second  = start("p1", 2, 5000, true, stopped);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    ASSERT_TRUE(messager_->waitCalls("p1", 1));
    EXPECT_EQ(result(first), ErrorCode::CANCELLED);
    messager_->complete("p1", 0, CacheStoreErrorCode::LoadConnectFailed);
    EXPECT_EQ(result(second), ErrorCode::CANCELLED);
    messager_->complete("p0", 0);
    EXPECT_EQ(result(first), ErrorCode::CANCELLED);
    EXPECT_EQ(messager_->count("p1"), 1);
}

TEST_F(CacheLoadConnectRetryTest, LateConnectFailureAfterTimeoutDoesNotRetryOrOverwriteResult) {
    auto context = start("p0", 2, 100);
    ASSERT_TRUE(messager_->waitCalls("p0", 1));
    EXPECT_EQ(result(context), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    messager_->complete("p0", 0, CacheStoreErrorCode::LoadConnectFailed);
    EXPECT_EQ(result(context), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    EXPECT_EQ(messager_->count("p0"), 1);
}

TEST_F(CacheLoadConnectRetryTest, SynchronousValidationFailureFinishesWithoutDeadlock) {
    auto context = start("", 2);
    EXPECT_EQ(result(context), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(messager_->count(""), 0);
}

}  // namespace
}  // namespace rtp_llm
