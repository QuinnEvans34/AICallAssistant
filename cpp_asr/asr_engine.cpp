#include "asr_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cstdlib>
#include <iostream>

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

size_t DurationToSamples(int duration_ms, size_t sample_rate) {
  if (duration_ms <= 0) return 0;
  const double samples = (static_cast<double>(duration_ms) * static_cast<double>(sample_rate)) / 1000.0;
  return samples < 1.0 ? 1 : static_cast<size_t>(samples);
}

}  // namespace

ASREngine::ASREngine() = default;

ASREngine::~ASREngine() { Shutdown(); }

bool ASREngine::Initialize(const std::string& model_path) {
  bool expected = false;
  if (initialized_.load()) {
    return true;
  }

  // Guard initialization so it only runs once
  std::lock_guard<std::mutex> init_lock(transcript_mutex_);
  if (initialized_.load()) {
    return true;
  }

  ring_buffer_ = std::make_shared<AudioRingBuffer>();
  vad_ = std::make_shared<VADDetector>();
  whisper_ = std::make_unique<WhisperWrapper>();
  heartbeat_logger_ = std::make_unique<HeartbeatLogger>();
  exception_logger_ = std::make_unique<ExceptionLogger>();

  // Determine model path: parameter overrides env
  std::string model = model_path;
  if (model.empty()) {
    const char* env_model = std::getenv("ASR_MODEL_PATH");
    if (env_model) model = env_model;
  }

  bool loaded = false;
  try {
    if (!model.empty()) {
      loaded = whisper_->LoadModel(model);
      if (!loaded) {
        exception_logger_->LogException("ASREngine::Initialize", "Failed to load model from path: " + model);
      }
    } else {
      // No model path provided. Check if fake mode allowed.
      const char* allow_fake = std::getenv("ASR_ALLOW_FAKE");
      if (allow_fake && std::string(allow_fake) == "1") {
        // WhisperWrapper.LoadModel may be a stub that supports fake mode.
        loaded = whisper_->LoadModel("");
      } else {
        exception_logger_->LogException("ASREngine::Initialize", "No ASR model path provided and fake mode not enabled.");
      }
    }
  } catch (const std::exception& ex) {
    exception_logger_->LogException("ASREngine::Initialize", ex.what());
    loaded = false;
  } catch (...) {
    exception_logger_->LogException("ASREngine::Initialize", "Unknown exception during model load");
    loaded = false;
  }

  if (!loaded) {
    // Tear down partially-constructed components so callers can retry initialize.
    snapshot_worker_.reset();
    whisper_.reset();
    vad_.reset();
    ring_buffer_.reset();
    heartbeat_logger_.reset();
    exception_logger_.reset();
    return false;
  }

  // Start snapshot worker
  snapshot_worker_ = std::make_unique<SnapshotWorker>(ring_buffer_, vad_);
  snapshot_worker_->SetSnapshotCallback([this](const std::vector<float>& snapshot) {
    // Queue decode work asynchronously.
    {
      std::lock_guard<std::mutex> lock(decode_mutex_);
      decode_queue_.emplace(snapshot, std::chrono::steady_clock::now());
    }
    decode_cv_.notify_one();
  });
  snapshot_worker_->Start();

  // Start decode workers
  StartDecodeWorkers(2);

  initialized_.store(true);
  return loaded || initialized_.load();
}

void ASREngine::StartDecodeWorkers(size_t num_threads) {
  decode_running_.store(true);
  for (size_t i = 0; i < num_threads; ++i) {
    decode_threads_.emplace_back(&ASREngine::DecodeWorkerLoop, this);
  }
}

void ASREngine::StopDecodeWorkers() {
  decode_running_.store(false);
  decode_cv_.notify_all();
  for (auto& t : decode_threads_) {
    if (t.joinable()) t.join();
  }
  decode_threads_.clear();
}

void ASREngine::DecodeWorkerLoop() {
  while (decode_running_.load()) {
    std::pair<std::vector<float>, std::chrono::steady_clock::time_point> work;
    {
      std::unique_lock<std::mutex> lock(decode_mutex_);
      decode_cv_.wait(lock, [this] { return !decode_running_.load() || !decode_queue_.empty(); });
      if (!decode_running_.load() && decode_queue_.empty()) break;
      work = std::move(decode_queue_.front());
      decode_queue_.pop();
    }

    const auto snapshot_time = work.second;
    const auto& snapshot = work.first;
    const auto decode_start = std::chrono::steady_clock::now();

    try {
      const auto result = whisper_->DecodeBlocking(snapshot);
      const auto decode_end = std::chrono::steady_clock::now();
      const double decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();

      snapshots_processed_.fetch_add(1);
      total_decode_ms_.fetch_add(decode_ms);

      // VAD decision: run VAD on snapshot frames and compute ratio
      size_t sample_rate = ring_buffer_ ? ring_buffer_->SampleRate() : 16000;
      size_t frame_samples = DurationToSamples(vad_->config().frame_duration_ms, sample_rate);
      size_t total_frames = 0;
      size_t speech_frames = 0;
      for (size_t offset = 0; offset + frame_samples <= snapshot.size(); offset += frame_samples) {
        ++total_frames;
        if (vad_->IsSpeech(snapshot.data() + offset, frame_samples)) {
          ++speech_frames;
        }
      }
      const float speech_ratio = total_frames ? static_cast<float>(speech_frames) / static_cast<float>(total_frames) : 0.0f;

      // Enqueue transcript if novel
      if (!result.transcript.empty()) {
        std::string novel_text;
        {
          std::lock_guard<std::mutex> lock(transcript_mutex_);
          novel_text = ExtractNovelTextLocked(result.transcript);
          if (!novel_text.empty()) {
            transcript_queue_.push(novel_text);
            std::lock_guard<std::mutex> lt(last_transcript_mutex_);
            last_transcript_ = novel_text;
          }
        }

        // Emit heartbeat entry
        if (heartbeat_logger_) {
          double avg_decode = 0.0;
          size_t count = snapshots_processed_.load();
          if (count > 0) avg_decode = total_decode_ms_.load() / static_cast<double>(count);
          heartbeat_logger_->LogHeartbeat(static_cast<int>(decode_queue_.size()), snapshot.size() / 16.0, decode_ms, speech_ratio > 0.2f);
        }
      }
    } catch (const std::exception& ex) {
      if (exception_logger_) exception_logger_->LogException("ASREngine::DecodeWorkerLoop", ex.what());
    } catch (...) {
      if (exception_logger_) exception_logger_->LogException("ASREngine::DecodeWorkerLoop", "Unknown exception");
    }
  }
}

void ASREngine::Shutdown() {
  if (!initialized_.exchange(false)) {
    return;
  }

  // Stop snapshot worker first so no new work is queued
  if (snapshot_worker_) {
    snapshot_worker_->Stop();
    snapshot_worker_.reset();
  }

  // Stop decode workers and flush queue
  StopDecodeWorkers();
  {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    while (!decode_queue_.empty()) decode_queue_.pop();
  }

  // Clear whisper
  whisper_.reset();

  if (vad_) {
    vad_.reset();
  }

  if (ring_buffer_) {
    ring_buffer_.reset();
  }

  if (heartbeat_logger_) {
    heartbeat_logger_.reset();
  }
  if (exception_logger_) {
    exception_logger_.reset();
  }

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
  // Reset per-call state and clear buffers
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

  // Reset metrics
  snapshots_processed_.store(0);
  total_decode_ms_.store(0.0);
  {
    std::lock_guard<std::mutex> lt(last_transcript_mutex_);
    last_transcript_.clear();
  }
}

void ASREngine::ResetForCall(const std::string& call_id) {
  ResetCall();
  if (!call_id.empty() && heartbeat_logger_) {
    std::string heartbeat_path = "logs/heartbeat/" + call_id + ".jsonl";
    heartbeat_logger_->SetLogPath(heartbeat_path);
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

// Reuse TrimWhitespace/NormalizeForDedup functions from earlier

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
    std::lock_guard<std::mutex> lt(last_transcript_mutex_);
    last_transcript_ = trimmed;
  }

  if (callback) {
    try {
      callback(trimmed);
    } catch (...) {
      if (exception_logger_) {
        exception_logger_->LogException("ASREngine::EnqueueTranscript", "Transcript callback threw an exception");
      }
    }
  }
}

std::string ASREngine::ExtractNovelTextLocked(const std::string& transcript) {
  const NormalizedText norm_tail = NormalizeForDedup(tail_text_);
  const NormalizedText norm_cur = NormalizeForDedup(transcript);
  if (norm_cur.normalized.empty()) {
    return {};
  }

  const size_t window = std::min(dedup_window_, norm_tail.normalized.size());
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
