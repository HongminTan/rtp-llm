#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientConnectionCallback.h"

#include "autil/EnvUtil.h"

#include <chrono>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaClientHostConnections::RdmaClientHostConnections(const std::shared_ptr<MemoryUtil>& memory_util,
                                                     const std::string&                 host,
                                                     uint32_t                           port,
                                                     uint32_t                           qp_count_per_rdma_connection,
                                                     uint32_t                           connect_timeout_ms):
    memory_util_(memory_util),
    host_(host),
    port_(port),
    qp_count_per_rdma_connection_(qp_count_per_rdma_connection),
    connect_timeout_ms_(connect_timeout_ms) {}

RdmaClientHostConnections::~RdmaClientHostConnections() {
    for (auto& connection : newly_connections_) {
        connection->Close();
    }
    newly_connections_.clear();

    for (auto& connection : available_connections_) {
        connection->Close();
    }
    available_connections_.clear();

    if (busy_connections_.size() > 0) {
        RTP_LLM_LOG_WARNING("still has inprocessing rdma connection while rdma client dtor");
    }
    for (auto& connection : busy_connections_) {
        connection->Close();
    }
    busy_connections_.clear();
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::getConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client) {
    auto rdma_connection = getRdmaConnectionFromConnectionPool(available_connections_, true);
    if (rdma_connection != nullptr) {
        return rdma_connection;
    }

    // Fresh QPs belong to the thread currently extending this pool. Looking
    // at newly_connections_ here can steal its completed handshake before it
    // wakes, stranding it (and every queued caller) behind connect_mutex_.
    // Late callbacks are still reused under the expansion lock below.
    rdma_connection = newRdmaConnection(client, true);
    if (rdma_connection == nullptr) {
        RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
    }
    return rdma_connection;
}

std::shared_ptr<RdmaConnection> RdmaClientHostConnections::getRdmaConnectionFromConnectionPool(
    std::set<std::shared_ptr<::accl::barex::XConnection>>& connection_pool, bool all_connected) {
    std::lock_guard<std::mutex> lock(mutex_);
    return takeConnectionLocked(connection_pool, all_connected);
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::takeConnectionLocked(std::set<std::shared_ptr<::accl::barex::XConnection>>& connection_pool,
                                                bool                                                   all_connected) {
    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;
    if (connection_pool.empty()) {
        return nullptr;
    }

    if (all_connected && connection_pool.size() < qp_count_per_rdma_connection_) {
        return nullptr;
    }

    for (int i = 0; i < qp_count_per_rdma_connection_ && !connection_pool.empty(); i++) {
        auto iter       = connection_pool.begin();
        auto connection = *iter;
        connection_pool.erase(connection);
        busy_connections_.insert(connection);
        connections.push_back(connection);
    }
    return std::make_shared<RdmaConnection>(memory_util_, host_, port_, connections);
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::newRdmaConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client,
                                             bool                                                    reuse_available) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(connect_timeout_ms_);
    std::unique_lock<std::timed_mutex> connect_lock(connect_mutex_, std::defer_lock);
    if (!connect_lock.try_lock_until(deadline)) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Another caller still owns the in-flight expansion, even if this
        // waiter's deadline expires first. Only recycled QPs are ours to take.
        return reuse_available ? takeConnectionLocked(available_connections_, false) : nullptr;
    }

    // Cold concurrent requests must recheck the pool after waiting for its
    // previous expansion. Otherwise every waiter starts another QP group even
    // though an earlier transfer has already recycled a usable connection.
    if (reuse_available) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        connection = takeConnectionLocked(available_connections_, true);
        if (!connection) {
            // Preserve partial late-callback reuse after acquiring the expansion lock.
            connection = takeConnectionLocked(newly_connections_, false);
        }
        if (connection) {
            return connection;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return takeConnectionLocked(newly_connections_, false);
        }
    }

    auto callback = std::make_shared<RdmaClientConnectionCallback>(shared_from_this());
    for (uint32_t i = 0; i < qp_count_per_rdma_connection_; ++i) {
        client->Connect(::accl::barex::TCP_V4, host_, port_, callback);
    }

    std::unique_lock<std::mutex>    lock(mutex_);
    std::shared_ptr<RdmaConnection> connection;
    connections_changed_.wait_until(lock, deadline, [&] {
        connection = reuse_available ? takeConnectionLocked(available_connections_, true) : nullptr;
        if (!connection) {
            connection = takeConnectionLocked(newly_connections_, true);
        }
        return connection != nullptr;
    });
    if (!connection && reuse_available) {
        connection = takeConnectionLocked(available_connections_, false);
    }
    return connection ? connection : takeConnectionLocked(newly_connections_, false);
}

void RdmaClientHostConnections::recycleConnection(const std::shared_ptr<RdmaConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    auto connections = connection->getAllConnections();
    doRecycleConnections(connections);
}

void RdmaClientHostConnections::doRecycleConnections(
    const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& connection : connections) {
        if (connection == nullptr) {
            continue;
        }
        // connection may already broken and call failed callback
        if (busy_connections_.find(connection) != busy_connections_.end()) {
            busy_connections_.erase(connection);
            available_connections_.insert(connection);
        }
    }
    connections_changed_.notify_all();
}

void RdmaClientHostConnections::destroyConnection(const std::shared_ptr<RdmaConnection>& rdma_connection) {
    if (rdma_connection == nullptr) {
        return;
    }

    auto connections = rdma_connection->getAllConnections();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& connection : connections) {
            if (busy_connections_.find(connection) != busy_connections_.end()) {
                busy_connections_.erase(connection);
                RTP_LLM_LOG_INFO("rdma connection to %s:%u call disconnect", host_.c_str(), port_);
            }
        }
    }

    // trigger close
    for (auto& connection : connections) {
        connection->Close();
    }
}

void RdmaClientHostConnections::connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u connect success", host_.c_str(), port_);
    std::lock_guard<std::mutex> lock(mutex_);
    newly_connections_.insert(connection);
    connections_changed_.notify_all();
}

void RdmaClientHostConnections::connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto iter = newly_connections_.find(connection);
    if (iter != newly_connections_.end()) {
        newly_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = available_connections_.find(connection);
    if (iter != available_connections_.end()) {
        available_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = busy_connections_.find(connection);
    if (iter != busy_connections_.end()) {
        busy_connections_.erase(iter);
        RTP_LLM_LOG_WARNING("rdma connection to %s:%u disconnect while inuse", host_.c_str(), port_);
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
}

int RdmaClientHostConnections::getAvailableConnectionCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_connections_.size();
}

bool RdmaClientHostConnections::fillConnectionPool(const std::shared_ptr<::accl::barex::XClientTransport>& client,
                                                   int                                                     count) {
    for (int i = 0; i < count; i++) {
        auto rdma_connection = newRdmaConnection(client);
        if (rdma_connection == nullptr) {
            RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
            return false;
        }
        recycleConnection(rdma_connection);
    }
    RTP_LLM_LOG_INFO(
        "rdma client host connections to %s:%u fill connection pool success, count %d", host_.c_str(), port_, count);
    return true;
}

}  // namespace rtp_llm
