#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

#include <sstream>
#include <algorithm>
#include <cstring>

namespace rtp_llm {

CompleteTokenIds::CompleteTokenIds(int batch_size, int max_batch_size, int max_seq_len, int seq_size_per_block):
    batch_size_(batch_size),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    seq_size_per_block_(seq_size_per_block) {}

CompleteTokenIds::CompleteTokenIds(const CompleteTokenIds& other, bool share, int shift_token_num):
    batch_size_(other.batch_size_),
    max_batch_size_(other.max_batch_size_),
    max_seq_len_(other.max_seq_len_),
    seq_size_per_block_(other.seq_size_per_block_),
    seq_length_(other.seq_length_.load(std::memory_order_acquire)),
    common_len_(other.common_len_.load(std::memory_order_acquire)),
    start_check_seq_length_(other.start_check_seq_length_),
    first_token_time_us_(other.first_token_time_us_),
    first_token_latency_us_(other.first_token_latency_us_) {
    v41_inputs_       = other.v41_inputs_;
    canonical_offset_ = other.canonical_offset_ + shift_token_num;
    if (share) {
        if (shift_token_num == 0) {
            complete_token_ids_ = other.complete_token_ids_;
        } else {
            RTP_LLM_CHECK(batch_size_ == 1);
            // Create a narrowed view (shares storage with other)
            complete_token_ids_ = other.complete_token_ids_.narrow(
                1, shift_token_num, other.complete_token_ids_.size(1) - shift_token_num);
        }
    } else {
        complete_token_ids_ =
            torch::empty({other.complete_token_ids_.size(0), other.complete_token_ids_.size(1)}, torch::kInt32);
        auto copy_rows = batch_size_;
        auto cols      = complete_token_ids_.size(1);
        std::memcpy(complete_token_ids_.data_ptr<int32_t>(),
                    other.complete_token_ids_.data_ptr<int32_t>(),
                    copy_rows * cols * sizeof(int32_t));
    }
}

void CompleteTokenIds::init(const std::shared_ptr<GenerateInput>& generate_input, size_t extra_reserve_token_num) {
    RTP_LLM_CHECK(generate_input != nullptr);

    v41_inputs_        = generate_input->v41_inputs;
    canonical_offset_  = 0;
    const int init_len = generate_input->inputLength();
    seq_length_.store(init_len, std::memory_order_release);
    RTP_LLM_CHECK_WITH_INFO(
        (init_len <= max_seq_len_), "seq_length[%d] must be less than max_seq_len[%d]", init_len, max_seq_len_);

    common_len_.store(max_batch_size_ == 1 ? init_len : init_len / seq_size_per_block_ * seq_size_per_block_,
                      std::memory_order_release);

    start_check_seq_length_ = init_len;

    size_t max_token_num = max_seq_len_ + extra_reserve_token_num;

    complete_token_ids_ = torch::zeros({(int64_t)max_batch_size_, (int64_t)max_token_num}, torch::kInt32);
    for (int i = 0; i < batch_size_; ++i) {
        memcpy(complete_token_ids_.data_ptr<int32_t>() + i * max_token_num,
               generate_input->input_ids.data_ptr<int32_t>(),
               generate_input->input_ids.nbytes());
    }

    RTP_LLM_LOG_DEBUG("complete tokenids init done, %s", showStatus(0).c_str());
}

std::vector<int32_t> CompleteTokenIds::imageCacheIdentity(int begin, int count) const {
    std::vector<int32_t> result;
    if (!v41_inputs_) {
        return result;
    }
    for (const auto& image : v41_inputs_->images) {
        const int64_t start = image.start - canonical_offset_;
        const int64_t end   = start + image.types.numel();
        if (end <= begin || start >= static_cast<int64_t>(begin) + count) {
            continue;
        }
        // Domain-separate image content from ordinary canonical token words.
        result.insert(result.end(),
                      {-41, static_cast<int32_t>(start), static_cast<int32_t>(end), image.n_vit_h, image.n_vit_w});
        for (const auto* digest : {&image.content_sha256, &image.processor_identity}) {
            // Hash the complete, length-delimited identity without parsing RPC strings.
            result.push_back(static_cast<int32_t>(digest->size()));
            for (unsigned char byte : *digest) {
                result.push_back(byte);
            }
        }
    }
    return result;
}

bool CompleteTokenIds::isValidReuseLength(int reuse_length, int min_fresh_tokens) const {
    if (min_fresh_tokens > 0 && reuse_length > std::max(0, seqLength() - min_fresh_tokens)) {
        return false;
    }
    if (!v41_inputs_) {
        return true;
    }
    for (const auto& image : v41_inputs_->images) {
        const int64_t start = image.start - canonical_offset_;
        const int64_t end   = start + image.types.numel();
        // Without both content and preprocessing identity no later prefix is
        // trustworthy, even when its endpoint lies beyond the image.
        if (reuse_length > start && (image.content_sha256.empty() || image.processor_identity.empty())) {
            return false;
        }
        if (reuse_length > start && reuse_length < end) {
            return false;
        }
    }
    return true;
}

int CompleteTokenIds::maxBatchSize() {
    return max_batch_size_;
}

int CompleteTokenIds::batchSize() {
    return batch_size_;
}

torch::Tensor CompleteTokenIds::completeTokenIds() {
    return complete_token_ids_;
}

std::vector<int> CompleteTokenIds::completeTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK_WITH_INFO(
        batch_idx < batch_size_, "batch_idx is out of bound, expected < %d, found %d", batch_size_, batch_idx);
    auto* ptr = data(batch_idx);
    return std::vector<int>(ptr, ptr + seqLength());
}

std::vector<int> CompleteTokenIds::commonCompleteTokenIdsVec(int batch_idx) {
    RTP_LLM_CHECK_WITH_INFO(
        batch_idx < batch_size_, "batch_idx is out of bound, expected < %d, found %d", batch_size_, batch_idx);
    auto* ptr = data(batch_idx);
    return std::vector<int>(ptr, ptr + commonSeqLength());
}

std::vector<int> CompleteTokenIds::currentExecuteTokens(int batch_idx) {
    RTP_LLM_CHECK_WITH_INFO(
        batch_idx < batch_size_, "batch_idx is out of bound, expected < %d, found %d", batch_size_, batch_idx);
    return {*(data(batch_idx) + seqLength() - 1)};
}

std::vector<int> CompleteTokenIds::contextTokens(int batch_idx, int prefix_length, int context_length) {
    RTP_LLM_CHECK_WITH_INFO(
        batch_idx < batch_size_, "batch_idx is out of bound, expected < %d, found %d", batch_size_, batch_idx);
    auto* ptr = data(batch_idx) + prefix_length;
    return std::vector<int>(ptr, ptr + context_length);
}

std::vector<int> CompleteTokenIds::getLatestTokens(size_t token_num) {
    const int cur_len = seqLength();
    RTP_LLM_CHECK(cur_len >= 0 && static_cast<size_t>(cur_len) >= token_num);
    auto* ptr = data(0) + cur_len - token_num;
    return std::vector<int>(ptr, ptr + token_num);
}

bool CompleteTokenIds::matchEosToken(int batch_id, int token_id) {
    int*         token_ids = data(batch_id);
    const int    raw_len   = seqLength();
    const size_t cur_len   = raw_len > 0 ? static_cast<size_t>(raw_len) : 0;
    for (size_t i = start_check_seq_length_; i <= cur_len; ++i) {
        if (token_id == token_ids[i - 1]) {
            seq_length_.store(static_cast<int>(i), std::memory_order_release);
            return true;
        }
    }
    return false;
}

bool CompleteTokenIds::matchStopWordsList(int batch_id, const std::vector<int>& stop_words) {
    int*         token_ids = data(batch_id);
    const int    raw_len   = seqLength();
    const size_t cur_len   = raw_len > 0 ? static_cast<size_t>(raw_len) : 0;
    for (size_t i = start_check_seq_length_; i <= cur_len; ++i) {
        bool   match_one   = true;
        size_t begin_index = i - stop_words.size();
        for (auto& token : stop_words) {
            if (token != token_ids[begin_index++]) {
                match_one = false;
                break;
            }
        }
        if (match_one) {
            seq_length_.store(static_cast<int>(i), std::memory_order_release);
            return true;
        }
    }
    return false;
}

bool CompleteTokenIds::update(const torch::Tensor& new_tokens,
                              int64_t              begin_time_us,
                              int                  num_new_tokens,
                              int                  input_length,
                              int                  max_token_num,
                              int                  vocab_size,
                              bool                 is_beam_search,
                              int64_t              stream_id,
                              int&                 error_token_id) {
    int new_batch_size = new_tokens.size(0);
    RTP_LLM_CHECK_WITH_INFO(
        new_batch_size <= max_batch_size_, "too many batches, expect < %d, found %d", max_batch_size_, new_batch_size);

    const int cur_len = seqLength();
    if (cur_len + num_new_tokens > max_token_num) {
        num_new_tokens = max_token_num - cur_len;
    }
    const bool commits_first_token = cur_len == input_length && num_new_tokens > 0;

    // # NOTE: new tokens indicate num of newly genearted tokens
    // # typically 1 but can be > 1 under speculative decoding
    // # This differs from new_tokens.shape[-1] under beam search case,
    // # which needs to update all the generated tokens each update.
    RTP_LLM_CHECK(new_tokens.dim() == 2);

    auto       new_tokens_ptr     = new_tokens.data_ptr<int>();  // [batch_size, max_num_new_tokens]
    auto       max_num_new_tokens = new_tokens.size(1);
    const auto get_token_id       = [&](auto batch_idx, auto token_idx) {
        if (is_beam_search) {
            return (new_tokens_ptr + max_num_new_tokens * batch_idx)[cur_len + token_idx];
        } else {
            return (new_tokens_ptr + num_new_tokens * batch_idx)[token_idx];
        }
    };

    for (size_t i = 0; i < new_batch_size; ++i) {
        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = get_token_id(i, j);
            if (!(current_token_id >= 0 && current_token_id < vocab_size)) {  // check tokenid
                error_token_id = current_token_id;
                return false;
            }
        }
        if (is_beam_search) {
            memcpy(data(i), new_tokens_ptr + i * max_num_new_tokens, sizeof(int) * max_num_new_tokens);
        } else {
            if (batch_size_ != new_batch_size && i > 0) {
                memcpy(data(i), data(0), sizeof(int) * cur_len);
            }
            memcpy(data(i) + cur_len, new_tokens_ptr + i * num_new_tokens, sizeof(int) * num_new_tokens);
        }
    }
    batch_size_ = new_batch_size;
    setSeqLength(cur_len + num_new_tokens);
    if (commits_first_token) {
        first_token_time_us_    = autil::TimeUtility::currentTimeInMicroSeconds();
        first_token_latency_us_ = first_token_time_us_ - begin_time_us;
    }

    RTP_LLM_LOG_DEBUG("update token, num_new_tokens: %d, after update is %s", num_new_tokens, showStatus(0).c_str());
    return true;
}

void CompleteTokenIds::setReserveStep(int reserve_step) {
    reserve_step_ = reserve_step;
}

int CompleteTokenIds::getReserveStep() const {
    return reserve_step_;
}

void CompleteTokenIds::setSeqLength(int seq_length) {
    RTP_LLM_CHECK(seq_length <= complete_token_ids_.size(1));
    const int prev_len = seq_length_.load(std::memory_order_relaxed);
    if (seq_length > prev_len) {
        start_check_seq_length_ = prev_len + 1;
    } else {
        start_check_seq_length_ = seq_length;
    }
    seq_length_.store(seq_length, std::memory_order_release);

    if (batch_size_ == 1) {  // reset common len
        common_len_.store(seq_length, std::memory_order_release);
    }
}

int CompleteTokenIds::totalSeqLength() const {
    return seqLength() + (int)reserve_step_;
}

int CompleteTokenIds::commonSeqLength() const {
    return common_len_.load(std::memory_order_acquire);
}

int CompleteTokenIds::seqLength() const {
    return seq_length_.load(std::memory_order_acquire);
}

void CompleteTokenIds::copyTokensTo(int batch_id, void* dst, int offset, size_t token_num) {
    memcpy(dst, data(batch_id) + offset, sizeof(int32_t) * token_num);
}

int64_t CompleteTokenIds::firstTokenTimeUs() const {
    return first_token_time_us_;
}

int64_t CompleteTokenIds::firstTokenLatencyUs() const {
    return first_token_latency_us_;
}

std::string CompleteTokenIds::toString(int batch_id) const {
    auto*              ptr     = const_cast<CompleteTokenIds*>(this)->data(batch_id);
    const int          cur_len = seqLength();
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < cur_len; ++i) {
        if (i > 0)
            oss << ", ";
        oss << ptr[i];
    }
    oss << "]";
    return oss.str();
}

int32_t* CompleteTokenIds::data(int batch_id) {
    return complete_token_ids_.data_ptr<int32_t>() + batch_id * complete_token_ids_.size(1);
}

std::string CompleteTokenIds::showStatus(int batch_id) {
    const int cur_len = seqLength();
    int       start   = cur_len > 10 ? cur_len - 10 : 0;
    int       len     = cur_len - start;
    auto*     ptr     = data(batch_id) + start;

    std::ostringstream debug_oss;
    debug_oss << "[";
    for (int i = 0; i < len; ++i) {
        if (i > 0)
            debug_oss << ", ";
        debug_oss << ptr[i];
    }
    debug_oss << "]";

    std::ostringstream oss;
    oss << "complete tokenids seq length " << cur_len << ", tokenids size " << completeTokenIdsVec(batch_id).size()
        << ", last 10 tokenids is " << debug_oss.str();
    return oss.str();
}

}  // namespace rtp_llm
