#include "asr_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>

namespace cpp_asr {

namespace {
std::string TrimWhitespace(const std::string& input) {
  const auto first =
      std::find_if_not(input.begin(), input.end(), [](unsigned char ch) { return std::isspace(ch); });
  if (first == input.end()) {
    return {};
  }
  const auto last =
      std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
  if (last <= first) {
    return {};
  }
  return std::string(first, last);
}

struct NormalizedText {
  std::string normalized;
  std::vector<size_t> mapping;
};

NormalizedText NormalizeForDedup(const std::string& input) {
  NormalizedText result;
  bool last_was_space = true;
  for (size_t i = 0; i < input.size(); ++i) {
    unsigned char ch = static_cast<unsigned char>(input[i]);
    if (std::isspace(ch)) {
      if (result.normalized.empty() || last_was_space) {
        continue;
      }
      result.normalized.push_back(' ');
      result.mapping.push_back(i);
      last_was_space = true;
    } else {
      result.normalized.push_back(static_cast<char>(std::tolower(ch)));
      result.mapping.push_back(i);
      last_was_space = false;
    }
  }

  while (!result.normalized.empty() && result.normalized.back() == ' ') {
    result.normalized.pop_back();
    result.mapping.pop_back();
  }

  return result;
}
}  // namespace

ASREngine::ASREngine() = default;

ASREngine::~ASREngine() { Shutdown(); }

bool ASREngine::Initialize(const std::string& model_path) {
  if (initialized_.load()) {
    return true;
  }

  ring_buffer_ = std::make_shared<AudioRingBuffer>();
  vad_ = std::make_shared<VADDetector>();
  whisper_ = std::make_unique<WhisperWrapper>();
  heartbeat_logger_ = std::make_unique<HeartbeatLogger>();
  exception_logger_ = std::make_unique<ExceptionLogger>();

  whisper_->LoadModel(model_path);

  snapshot_worker_ =
      std::make_unique<SnapshotWorker>(ring_buffer_, vad_);
  snapshot_worker_->SetSnapshotCallback(
      [this](const std::vector<float>& snapshot) { HandleSnapshot(snapshot); });
  snapshot_worker_->Start();

  initialized_.store(true);
  return true;
}

void ASREngine::Shutdown() {
  if (!initialized_.exchange(false)) {
    return;
  }

  if (snapshot_worker_) {
    snapshot_worker_->Stop();
    snapshot_worker_.reset();
  }

  whisper_.reset();
  vad_.reset();
  ring_buffer_.reset();
  heartbeat_logger_.reset();
  exception_logger_.reset();

  std::lock_guard<std::mutex> lock(transcript_mutex_);
  std::queue<std::string> empty;
  std::swap(transcript_queue_, empty);
}

bool ASREngine::PushAudio(const float* samples, size_t count) {
  if (!initialized_.load() || !ring_buffer_) {
    return false;
  }
  ring_buffer_->PushSamples(samples, count);
  return true;
}

bool ASREngine::PollTranscript(std::string& transcript) {
  std::lock_guard<std::mutex> lock(transcript_mutex_);
  if (transcript_queue_.empty()) {
    transcript.clear();
    return false;
  }
  transcript = transcript_queue_.front();
  transcript_queue_.pop();
  return true;
}

void ASREngine::ResetCall() {
  std::lock_guard<std::mutex> lock(transcript_mutex_);
  tail_text_.clear();
  std::queue<std::string> empty;
  std::swap(transcript_queue_, empty);

  if (ring_buffer_) {
    ring_buffer_->Clear();
  }
  if (vad_) {
    vad_->Reset();
  }
  if (snapshot_worker_) {
    snapshot_worker_->Reset();
  }

  if (heartbeat_logger_) {
    heartbeat_logger_->SetLogPath(heartbeat_log_path_);
  }
  if (exception_logger_) {
    exception_logger_->SetLogPath(exception_log_path_);
  }
}

void ASREngine::SetTranscriptCallback(TranscriptCallback cb) {
  transcript_callback_ = std::move(cb);
}

void ASREngine::ConfigureLogs(const std::string& heartbeat_log_path,
                              const std::string& exception_log_path) {
  if (heartbeat_logger_) {
    heartbeat_logger_->SetLogPath(heartbeat_log_path);
  }
  if (exception_logger_) {
    exception_logger_->SetLogPath(exception_log_path);
  }
  heartbeat_log_path_ = heartbeat_log_path;
  exception_log_path_ = exception_log_path;
}

void ASREngine::HandleSnapshot(const std::vector<float>& snapshot) {
  if (!whisper_ || snapshot.empty()) {
    return;
  }

  const auto start_time = std::chrono::steady_clock::now();
  try {
    const auto result = whisper_->DecodeBlocking(snapshot);
    std::string transcript = TrimWhitespace(result.transcript);
    if (transcript.empty()) {
      return;
    }

    std::string novel_text;
    {
      std::lock_guard<std::mutex> lock(transcript_mutex_);
      novel_text = ExtractNovelTextLocked(transcript);
    }

    if (!novel_text.empty()) {
      EnqueueTranscript(novel_text);
    }

    const auto end_time = std::chrono::steady_clock::now();
    if (heartbeat_logger_) {
      (void)start_time;
      (void)end_time;
      // TODO: Emit heartbeat metrics (latency, queue depth) when wiring is ready.
    }
  } catch (const std::exception& ex) {
    if (exception_logger_) {
      exception_logger_->LogException("ASREngine::HandleSnapshot", ex.what());
    }
  } catch (...) {
    if (exception_logger_) {
      exception_logger_->LogException("ASREngine::HandleSnapshot", "Unknown exception");
    }
  }
}

void ASREngine::EnqueueTranscript(const std::string& text) {
  const std::string trimmed = TrimWhitespace(text);
  if (trimmed.empty()) {
    return;
  }

  TranscriptCallback callback;
  {
    std::lock_guard<std::mutex> lock(transcript_mutex_);
    transcript_queue_.push(trimmed);
    callback = transcript_callback_;
  }

  if (callback) {
    callback(trimmed);
  }
}

std::string ASREngine::ExtractNovelTextLocked(const std::string& transcript) {
  const NormalizedText norm_tail = NormalizeForDedup(tail_text_);
  const NormalizedText norm_cur = NormalizeForDedup(transcript);
  if (norm_cur.normalized.empty()) {
    return {};
  }

  const size_t window =
      std::min(dedup_window_, norm_tail.normalized.size());
  size_t match_len = 0;
  for (size_t k = window; k > 0; --k) {
    const size_t tail_start = norm_tail.normalized.size() - k;
    if (norm_tail.normalized.compare(
            tail_start, k, norm_cur.normalized, 0, k) == 0) {
      match_len = k;
      break;
    }
  }

  size_t raw_start = 0;
  if (match_len == 0) {
    raw_start = 0;
  } else if (match_len >= norm_cur.mapping.size()) {
    raw_start = transcript.size();
  } else {
    raw_start = norm_cur.mapping[match_len];
  }

  if (raw_start >= transcript.size()) {
    return {};
  }

  std::string novel = transcript.substr(raw_start);
  if (TrimWhitespace(novel).empty()) {
    return {};
  }

  tail_text_ += novel;
  if (tail_text_.size() > max_tail_chars_) {
    tail_text_ = tail_text_.substr(tail_text_.size() - max_tail_chars_);
  }
  return novel;
}

}  // namespace cpp_asr
