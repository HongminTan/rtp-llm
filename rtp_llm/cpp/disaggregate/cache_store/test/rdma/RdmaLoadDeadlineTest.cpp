#include "gtest/gtest.h"
#include <chrono>
#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientConnectionCallback.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/MockMemoryUtil.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"

namespace rtp_llm {
namespace {
using namespace testing;
using namespace std::chrono_literals;

class MockRequestMessager: public RdmaMessager {
public:
    using RdmaMessager::RdmaMessager;
    MOCK_METHOD(CacheLoadRequest*,
                makeLoadRequest,
                (const std::shared_ptr<RdmaConnection>&, const std::shared_ptr<LoadRequest>&),
                (override));
};

class RecordingLoadChannel: public arpc::RPCChannelBase {
public:
    bool ChannelBroken() override {
        return false;
    }
    bool ChannelConnected() override {
        return true;
    }
    void CallMethod(const google::protobuf::MethodDescriptor*,
                    google::protobuf::RpcController* controller,
                    const google::protobuf::Message* request,
                    google::protobuf::Message*       response,
                    google::protobuf::Closure*       done) override {
        ++calls;
        rpc_timeout_ms  = static_cast<arpc::ANetRPCController*>(controller)->GetExpireTime();
        peer_timeout_ms = static_cast<const CacheLoadRequest*>(request)->timeout_ms();
        static_cast<CacheLoadResponse*>(response)->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
        done->Run();
    }
    int     calls           = 0;
    int     rpc_timeout_ms  = 0;
    int64_t peer_timeout_ms = 0;
};

class RdmaLoadDeadlineTest: public Test {
protected:
    void SetUp() override {
        auto memory            = std::make_shared<MockMemoryUtil>(nullptr);
        messager_              = std::make_shared<NiceMock<MockRequestMessager>>(memory, nullptr, nullptr);
        messager_->tcp_client_ = std::make_shared<TcpClient>();
        channel_               = std::make_shared<RecordingLoadChannel>();
        messager_->tcp_client_->channel_map_["tcp:peer:10000"] = channel_;
        messager_->rdma_client_                                = std::make_shared<RdmaClient>(memory, 1000, 1);
        transport_                       = std::make_shared<NiceMock<::accl::barex::MockXClientTransport>>();
        messager_->rdma_client_->client_ = transport_;
        auto buffer                      = std::make_shared<RequestBlockBuffer>("request");
        buffer->addBlock("block", std::make_shared<char>(0), 1, false, true);
        request_ = std::make_shared<LoadRequest>(
            "peer",
            10000,
            10001,
            buffer,
            [this](bool success, CacheStoreErrorCode ec) {
                ++callbacks_;
                EXPECT_EQ(success, expect_success_);
                EXPECT_EQ(ec, expect_success_ ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadBufferTimeout);
            },
            5000,
            1,
            0);
    }
    void run() {
        messager_->load(request_, std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1));
        EXPECT_EQ(callbacks_, 1);
        EXPECT_EQ(channel_->calls, expect_success_ ? 1 : 0);
    }
    void connectImmediately() {
        EXPECT_CALL(*transport_, Connect(_, _, _, _)).WillOnce(Invoke([](auto, auto, auto, auto callback) {
            callback->OnConnectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
        }));
    }
    std::shared_ptr<NiceMock<MockRequestMessager>>                 messager_;
    std::shared_ptr<RecordingLoadChannel>                          channel_;
    bool                                                           expect_success_ = false;
    std::shared_ptr<NiceMock<::accl::barex::MockXClientTransport>> transport_;
    std::shared_ptr<LoadRequest>                                   request_;
    int                                                            callbacks_ = 0;
};

TEST_F(RdmaLoadDeadlineTest, InsufficientResponseBudgetDoesNotConnectOrUnderflowTimeout) {
    request_->control = std::make_shared<CacheLoadControl>(8, nullptr);
    EXPECT_CALL(*transport_, Connect(_, _, _, _)).Times(0);
    run();
}

TEST_F(RdmaLoadDeadlineTest, DeadlineConsumedByConnectionAcquisitionRecyclesWithoutSending) {
    request_->control = std::make_shared<CacheLoadControl>(100, nullptr);
    EXPECT_CALL(*transport_, Connect(_, _, _, _)).WillOnce(Invoke([&](auto, auto, auto, auto callback) {
        std::this_thread::sleep_until(request_->control->deadline + 5ms);
        callback->OnConnectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
    }));
    run();
    EXPECT_EQ(messager_->rdma_client_->getAvailableConnectionCount("peer", 10001), 1);
}

TEST_F(RdmaLoadDeadlineTest, CancellationDuringConnectionAcquisitionRecyclesWithoutSending) {
    auto stopped      = std::make_shared<std::atomic_bool>(false);
    request_->control = std::make_shared<CacheLoadControl>(5000, stopped);
    EXPECT_CALL(*transport_, Connect(_, _, _, _)).WillOnce(Invoke([&](auto, auto, auto, auto callback) {
        stopped->store(true);
        callback->OnConnectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
    }));
    run();
    EXPECT_EQ(messager_->rdma_client_->getAvailableConnectionCount("peer", 10001), 1);
}

TEST_F(RdmaLoadDeadlineTest, DeadlineDuringRequestConstructionRecyclesWithoutSending) {
    request_->control = std::make_shared<CacheLoadControl>(100, nullptr);
    connectImmediately();
    EXPECT_CALL(*messager_, makeLoadRequest(_, _)).WillOnce(Invoke([&](const auto&, const auto& request) {
        auto rpc_request = messager_->Messager::makeLoadRequest(request);
        std::this_thread::sleep_until(request->control->deadline + 5ms);
        return rpc_request;
    }));
    run();
    EXPECT_EQ(messager_->rdma_client_->getAvailableConnectionCount("peer", 10001), 1);
}

TEST_F(RdmaLoadDeadlineTest, CancellationDuringRequestConstructionRecyclesWithoutSending) {
    auto stopped      = std::make_shared<std::atomic_bool>(false);
    request_->control = std::make_shared<CacheLoadControl>(5000, stopped);
    connectImmediately();
    EXPECT_CALL(*messager_, makeLoadRequest(_, _)).WillOnce(Invoke([&](const auto&, const auto& request) {
        auto rpc_request = messager_->Messager::makeLoadRequest(request);
        stopped->store(true);
        return rpc_request;
    }));
    run();
    EXPECT_EQ(messager_->rdma_client_->getAvailableConnectionCount("peer", 10001), 1);
}

TEST_F(RdmaLoadDeadlineTest, ConstructionTimeIsDeductedFromBothRpcAndPeerTimeouts) {
    request_->control = std::make_shared<CacheLoadControl>(5000, nullptr);
    expect_success_   = true;
    connectImmediately();
    int64_t budget_before_build = 0;
    EXPECT_CALL(*messager_, makeLoadRequest(_, _)).WillOnce(Invoke([&](const auto&, const auto& request) {
        budget_before_build = request->timeout_ms;
        auto rpc_request    = messager_->Messager::makeLoadRequest(request);
        std::this_thread::sleep_for(40ms);
        return rpc_request;
    }));
    run();
    EXPECT_GT(channel_->rpc_timeout_ms, 10);
    EXPECT_LE(channel_->rpc_timeout_ms, budget_before_build - 30);
    EXPECT_EQ(channel_->peer_timeout_ms, channel_->rpc_timeout_ms - 10);
    EXPECT_EQ(messager_->rdma_client_->getAvailableConnectionCount("peer", 10001), 1);
}

}  // namespace
}  // namespace rtp_llm
