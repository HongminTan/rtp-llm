#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/MockMemoryUtil.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"

#include "autil/NetUtil.h"

using namespace testing;

namespace rtp_llm {

// Pool tests use mock transports and do not initialize CUDA or NICs.
class RdmaClientHostConnectionsTest: public ::testing::Test {
protected:
    std::shared_ptr<MemoryUtil> memory_util_ = std::make_shared<MockMemoryUtil>(nullptr);
};

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections =
        std::make_shared<RdmaClientHostConnections>(memory_util_, autil::NetUtil::getBindIp(), 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // connect failed later
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed_Busy) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _)).Times(0);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // get rdma connection
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(1, host_connections->busy_connections_.size());

    // connect failed after
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed_Available) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _)).Times(0);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // get rdma connection
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(1, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());

    host_connections->recycleConnection(rdma_connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(1, host_connections->available_connections_.size());

    // connect failed after
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_Success) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    ASSERT_EQ(2, connections.size());
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_PartialSuccess) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectFailed(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    ASSERT_EQ(1, connections.size());
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_connectFailed) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectFailed(connection1);
        host_connections->connectFailed(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection == nullptr);

    connect_callback_thread.join();
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_connectTimeout) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection == nullptr);
}

TEST_F(RdmaClientHostConnectionsTest, testReuseLateConnectionsWithoutReconnect) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 1);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    // The first request times out before either callback arrives.
    ASSERT_TRUE(host_connections->getConnection(xclient_transport) == nullptr);

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    host_connections->connectSuccess(connection1);
    host_connections->connectSuccess(connection2);

    // No additional Connect calls are needed for the next request.
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(2, rdma_connection->getConnectionsCount());
    ASSERT_TRUE(host_connections->newly_connections_.empty());
    ASSERT_EQ(2, host_connections->busy_connections_.size());

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testReusePartialLateConnectionsWithoutReconnect) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections       = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 1);
    auto mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    ASSERT_TRUE(host_connections->getConnection(xclient_transport) == nullptr);

    auto                                        mock_connection = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection(mock_connection);
    host_connections->connectSuccess(connection);

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_TRUE(host_connections->newly_connections_.empty());
    ASSERT_EQ(1, host_connections->busy_connections_.size());

    EXPECT_CALL(*mock_connection, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testRecycleConnections) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    host_connections->recycleConnection(rdma_connection);

    auto rdma_connection2 = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection2 != nullptr);
    connections = rdma_connection2->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testDestroyConnections) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
    host_connections->destroyConnection(rdma_connection);

    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());
    auto rdma_connection2 = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection2 == nullptr);
}

TEST_F(RdmaClientHostConnectionsTest, testConcurrentColdRequestsSerializePoolGrowth) {
    auto               pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 2000);
    auto               client = std::make_shared<::accl::barex::MockXClientTransport>();
    std::atomic<int>   connect_calls{0};
    std::promise<void> first_group_started;
    std::mutex         callback_mutex;
    std::vector<std::shared_ptr<::accl::barex::XConnection>> first_group;
    EXPECT_CALL(*client, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .WillRepeatedly(Invoke([&](auto, const auto&, auto, const auto&) {
            std::lock_guard<std::mutex> callback_lock(callback_mutex);
            auto                        connection = std::make_shared<NiceMock<::accl::barex::MockXConnection>>();
            const int                   call       = ++connect_calls;
            if (call <= 2) {
                first_group.push_back(connection);
                if (call == 2) {
                    first_group_started.set_value();
                }
            } else {
                pool->connectSuccess(connection);
            }
        }));
    std::promise<void>             start;
    auto                           start_signal = start.get_future().share();
    std::atomic<int>               entered{0};
    std::vector<std::future<bool>> results;
    for (int i = 0; i < 16; ++i) {
        results.push_back(std::async(std::launch::async, [&] {
            start_signal.wait();
            ++entered;
            auto       connection = pool->getConnection(client);
            const bool valid      = connection && connection->getConnectionsCount() == 2;
            pool->recycleConnection(connection);
            return valid;
        }));
    }
    start.set_value();
    first_group_started.get_future().wait();
    while (entered != 16) {
        std::this_thread::yield();
    }
    // Hold the first handshake open while all callers contend for the cold pool.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(2, connect_calls.load());
    for (const auto& connection : first_group) {
        pool->connectSuccess(connection);
    }
    for (auto& result : results) {
        EXPECT_TRUE(result.get());
    }
    EXPECT_TRUE(pool->busy_connections_.empty());
}

TEST_F(RdmaClientHostConnectionsTest, testQueuedRequestReusesRecycledConnection) {
    auto pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 500);
    auto client = std::make_shared<::accl::barex::MockXClientTransport>();
    EXPECT_CALL(*client, Connect(_, _, _, _)).Times(0);
    auto first  = std::make_shared<NiceMock<::accl::barex::MockXConnection>>();
    auto second = std::make_shared<NiceMock<::accl::barex::MockXConnection>>();
    pool->connectSuccess(first);
    pool->connectSuccess(second);
    auto                               busy = pool->getConnection(client);
    std::unique_lock<std::timed_mutex> expanding(pool->connect_mutex_);
    auto result = std::async(std::launch::async, [&] { return pool->getConnection(client); });
    EXPECT_EQ(std::future_status::timeout, result.wait_for(std::chrono::milliseconds(20)));
    pool->recycleConnection(busy);
    expanding.unlock();
    auto reused = result.get();
    ASSERT_NE(nullptr, reused);
    EXPECT_EQ(2, reused->getConnectionsCount());
    pool->recycleConnection(reused);
}

TEST_F(RdmaClientHostConnectionsTest, testWaitingForPoolGrowthHonorsDeadline) {
    auto pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 40);
    auto client = std::make_shared<::accl::barex::MockXClientTransport>();
    EXPECT_CALL(*client, Connect(_, _, _, _)).Times(0);
    std::unique_lock<std::timed_mutex> expanding(pool->connect_mutex_);
    auto       result = std::async(std::launch::async, [&] { return pool->getConnection(client); });
    const auto status = result.wait_for(std::chrono::milliseconds(500));
    EXPECT_EQ(std::future_status::ready, status);
    expanding.unlock();
    EXPECT_EQ(nullptr, result.get());
}

TEST_F(RdmaClientHostConnectionsTest, testExpiredWaiterCannotStealPendingExpansionQps) {
    auto pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 40);
    auto client = std::make_shared<::accl::barex::MockXClientTransport>();
    EXPECT_CALL(*client, Connect(_, _, _, _)).Times(0);
    std::unique_lock<std::timed_mutex> expanding(pool->connect_mutex_);
    pool->connectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
    pool->connectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
    auto waiter = std::async(std::launch::async, [&] { return pool->getConnection(client); });
    EXPECT_EQ(nullptr, waiter.get());
    EXPECT_EQ(2, pool->newly_connections_.size());
    expanding.unlock();
    auto connection = pool->getConnection(client);
    ASSERT_NE(nullptr, connection);
    EXPECT_EQ(2, connection->getConnectionsCount());
    pool->recycleConnection(connection);
}

TEST_F(RdmaClientHostConnectionsTest, testExplicitFillStillGrowsPool) {
    auto pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);
    auto client = std::make_shared<::accl::barex::MockXClientTransport>();
    EXPECT_CALL(*client, Connect(_, _, _, _)).Times(6).WillRepeatedly(Invoke([&](auto, const auto&, auto, const auto&) {
        pool->connectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
    }));
    EXPECT_TRUE(pool->fillConnectionPool(client, 3));
    EXPECT_EQ(6, pool->getAvailableConnectionCount());
}

TEST_F(RdmaClientHostConnectionsTest, testLateCallerCannotStealQpsFromPoolExpansion) {
    auto               pool   = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 200);
    auto               client = std::make_shared<::accl::barex::MockXClientTransport>();
    std::promise<void> connected;
    std::promise<void> finish_connect;
    auto               finish = finish_connect.get_future().share();
    std::atomic<int>   calls{0};
    EXPECT_CALL(*client, Connect(_, _, _, _)).WillRepeatedly(Invoke([&](auto, const auto&, auto, const auto&) {
        pool->connectSuccess(std::make_shared<NiceMock<::accl::barex::MockXConnection>>());
        if (++calls == 2) {
            connected.set_value();
            finish.wait();
        }
    }));
    auto owner = std::async(std::launch::async, [&] { return pool->getConnection(client); });
    connected.get_future().wait();
    // QPs are ready, but their creator has not finished Connect. A new caller
    // must not steal them and strand the creator holding the expansion lock.
    auto late = std::async(std::launch::async, [&] { return pool->getConnection(client); });
    EXPECT_EQ(std::future_status::timeout, late.wait_for(std::chrono::milliseconds(20)));
    finish_connect.set_value();
    auto first = owner.get();
    EXPECT_NE(nullptr, first);
    pool->recycleConnection(first);
    auto second = late.get();
    EXPECT_NE(nullptr, second);
    pool->recycleConnection(second);
}

}  // namespace rtp_llm
