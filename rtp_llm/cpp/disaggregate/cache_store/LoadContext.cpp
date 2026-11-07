#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include <algorithm>

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func) {
    if (request_block_buffers.empty()) {
        return;
    }

    auto cache_store = cache_store_.lock();
    if (cache_store == nullptr) {
        error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, ErrorCodeToString(ErrorCode::UNKNOWN_ERROR));
        RTP_LLM_LOG_WARNING("load failed, cache store is nullptr");
        return;
    }

    start_time_ms_     = autil::TimeUtility::currentTimeInMilliSeconds();
    deadline_ms_       = start_time_ms_ + timeout_ms;
    check_cancel_func_ = check_cancel_func;

    if (combine_load_) {  // for rdma only call rpc once
        auto new_buffer = std::make_shared<RequestBlockBuffer>(request_block_buffers[0]->getRequestId());
        for (auto& request_block_buffer : request_block_buffers) {
            auto blocks = request_block_buffer->getBlocks();
            for (auto& [_, block] : blocks) {
                new_buffer->addBlock(block);
            }
        }
        request_block_buffers_ = {new_buffer};
    } else {
        request_block_buffers_ = request_block_buffers;
    }

    expect_layer_cnt_ = request_block_buffers_.size();

    for (auto& request_block_buffer : request_block_buffers_) {
        if (!doCall(request_block_buffer, timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    updateResultLocked(success, ec, request_block_buffer);
}

void SyncContext::finishErrorLocked(ErrorCode code) {
    if (finished_) {
        return;
    }
    error_info_ = ErrorInfo(code, ErrorCodeToString(code));
    timed_out_  = code == ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
    finished_   = true;
    if (load_control_) {
        load_control_->stopped.store(true);
    }
    cond_.notify_all();
}

void SyncContext::updateResultLocked(bool                                       success,
                                     CacheStoreErrorCode                        ec,
                                     const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    if (finished_) {
        return;
    }
    if (!success) {
        auto error_code = transCacheStoreErrorCode(ec);
        error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
        failed_request_block_buffers_.push_back(request_block_buffer);
        RTP_LLM_LOG_WARNING("request %s call finished, state:[%s], error code[%s], cost time %ldms",
                            request_block_buffer->getRequestKey().c_str(),
                            success ? "success" : "failed",
                            CacheStoreErrorCodeToString(ec).c_str(),
                            autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    } else {
        RTP_LLM_LOG_DEBUG("request %s call finished, state:[%s], cost time %ldms",
                          request_block_buffer->getRequestKey().c_str(),
                          success ? "success" : "failed",
                          autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    }

    if (++done_layer_cnt_ == expect_layer_cnt_) {
        finished_ = true;
        if (load_control_) {
            load_control_->stopped.store(true);
        }
        cond_.notify_all();
    }
}

void SyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!finished_ && done_layer_cnt_ != expect_layer_cnt_) {
        // Only the waiting thread invokes check_cancel_func_, which may capture references to the RPC stack.
        if (check_cancel_func_ && check_cancel_func_()) {
            if (load_control_ && load_control_->request_stopped) {
                load_control_->request_stopped->store(true);
            }
            finishErrorLocked(ErrorCode::CANCELLED);
            return;
        }
        if (load_control_ && load_control_->isStopped()) {
            finishErrorLocked(ErrorCode::CANCELLED);
            return;
        }
        if (load_control_ ? load_control_->remainingMs() <= 0 :
                            autil::TimeUtility::currentTimeInMilliSeconds() >= deadline_ms_) {
            finishErrorLocked(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
            return;
        }
        cond_.wait_for(
            lock, std::chrono::milliseconds(30), [this] { return finished_ || done_layer_cnt_ == expect_layer_cnt_; });
    }
}

bool SyncContext::success() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ok();
}

std::string SyncContext::getErrorInfoString() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ToString();
}

ErrorInfo SyncContext::getErrorInfo() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_;
}

std::vector<std::string> SyncContext::failedBlockDebugInfos() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (error_info_.ok() || error_info_.code() == ErrorCode::CANCELLED) {
        return {};
    }

    const auto& buffers =
        (timed_out_ || failed_request_block_buffers_.empty()) ? request_block_buffers_ : failed_request_block_buffers_;
    std::vector<std::string> debug_infos;
    debug_infos.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        debug_infos.push_back(buffer == nullptr ? "null" : buffer->debugInfo());
    }
    return debug_infos;
}

LoadContext::LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    SyncContext(cache_store, combine_load) {}

void LoadContext::load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
                       const std::string&                                      ip,
                       uint32_t                                                port,
                       uint32_t                                                rdma_port,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func,
                       int                                                     partition_count,
                       int                                                     partition_id,
                       int64_t                                                 connect_retries,
                       std::shared_ptr<std::atomic_bool>                       request_stopped) {
    retries_left_ = combine_load_ ? std::max<int64_t>(0, connect_retries) : 0;
    if (combine_load_ && (connect_retries > 0 || request_stopped)) {
        load_control_ = std::make_shared<CacheLoadControl>(timeout_ms, std::move(request_stopped));
    }
    peer_ip_         = ip;
    port_            = port;
    rdma_port_       = rdma_port;
    partition_count_ = partition_count;
    partition_id_    = partition_id;
    call(request_block_buffer, timeout_ms, check_cancel_func);
}

void LoadContext::onLoadDone(const std::shared_ptr<RequestBlockBuffer>& buffer,
                             uint64_t                                   attempt,
                             bool                                       success,
                             CacheStoreErrorCode                        ec) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_ || attempt != attempt_id_ || attempt_finished_) {
            return;
        }
        attempt_finished_ = true;
        if (load_control_->isStopped()) {
            finishErrorLocked(ErrorCode::CANCELLED);
            return;
        }
        if (load_control_->remainingMs() <= 0) {
            finishErrorLocked(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
            return;
        }
        if (success || retries_left_ == 0
            || (ec != CacheStoreErrorCode::LoadConnectFailed && ec != CacheStoreErrorCode::LoadRdmaConnectFailed)) {
            updateResultLocked(success, ec, buffer);
            return;
        }
        --retries_left_;
        RTP_LLM_LOG_INFO("request %s retry load to %s:%u after %s, retries left %ld",
                         buffer->getRequestKey().c_str(),
                         peer_ip_.c_str(),
                         rdma_port_,
                         CacheStoreErrorCodeToString(ec).c_str(),
                         retries_left_);
    }
    // NormalCacheStore queues the attempt without blocking. Never wait for a
    // connection or completion while running an IO/worker completion callback.
    doCall(buffer, load_control_->remainingMs());
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    uint64_t attempt = 0;
    if (load_control_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_) {
            return true;
        }
        if (load_control_->isStopped()) {
            finishErrorLocked(ErrorCode::CANCELLED);
            return true;
        }
        timeout_ms = load_control_->remainingMs();
        if (timeout_ms <= 0) {
            finishErrorLocked(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
            return true;
        }
        attempt           = ++attempt_id_;
        attempt_finished_ = false;
    }
    auto cache_store = cache_store_.lock();
    if (!cache_store) {
        std::lock_guard<std::mutex> lock(mutex_);
        finishErrorLocked(ErrorCode::UNKNOWN_ERROR);
        return true;
    }
    auto self                = std::static_pointer_cast<LoadContext>(shared_from_this());
    auto load_layer_callback = [request_block_buffer, self, attempt](bool success, CacheStoreErrorCode ec) {
        if (self->load_control_) {
            self->onLoadDone(request_block_buffer, attempt, success, ec);
        } else {
            self->updateResult(success, ec, request_block_buffer);
        }
    };
    cache_store->load(request_block_buffer,
                      load_layer_callback,
                      peer_ip_,
                      port_,
                      rdma_port_,
                      timeout_ms,
                      partition_count_,
                      partition_id_,
                      load_control_);
    return true;
}

StoreContext::StoreContext(const std::shared_ptr<CacheStore>& cache_store): SyncContext(cache_store, true) {}

void StoreContext::store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                         int64_t                                                 timeout_ms) {
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
