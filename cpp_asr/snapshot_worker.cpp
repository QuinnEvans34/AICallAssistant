#include "snapshot_worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace cpp_asr {

namespace {
size_t DurationToSamples(int duration_ms, size_t sample_rate) {
  if (duration_ms <= 0) {
    return 0;
  }
  const double samples =
      (static_cast<double>(duration_ms) * static_cast<double>(sample_rate)) /
      1000.0;
  return samples < 1.0 ? 1 : static_cast<size_t>(samples);
}
}  // namespace

SnapshotWorker::SnapshotWorker(std::shared_ptr<AudioRingBuffer> ring_buffer,
                               std::shared_ptr<VADDetector> vad,
                               Config config)
    : ring_buffer_(std::move(ring_buffer)), vad_(std::move(vad)), config_(config) {}

SnapshotWorker::~SnapshotWorker() { Stop(); }

void SnapshotWorker::Start() {
  if (running_.exchange(true)) {
    return;
  }
  worker_thread_ = std::thread(&SnapshotWorker::WorkerLoop, this);
}

void SnapshotWorker::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void SnapshotWorker::Reset() {
  Stop();
  Start();
}

void SnapshotWorker::SetSnapshotCallback(SnapshotCallback cb) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  snapshot_callback_ = std::move(cb);
}

void SnapshotWorker::WorkerLoop() {
  const auto sleep_duration = std::chrono::milliseconds(
      std::max(1, config_.snapshot_period_ms));

  std::vector<float> snapshot;
  while (running_.load()) {
    std::this_thread::sleep_for(sleep_duration);
    if (!running_.load()) {
      break;
    }

    try {
      if (!ring_buffer_ || !vad_) {
        continue;
      }

      const size_t sample_rate = ring_buffer_->SampleRate();
      const size_t snapshot_samples =
          DurationToSamples(config_.snapshot_duration_ms, sample_rate);
      if (snapshot_samples == 0) {
        continue;
      }

      snapshot = ring_buffer_->GetLatestSamples(snapshot_samples);
      if (snapshot.size() < snapshot_samples) {
        continue;
      }

      const VADDetector::Config vad_config = vad_->config();
      size_t frame_samples =
          DurationToSamples(vad_config.frame_duration_ms, sample_rate);
      if (frame_samples == 0) {
        frame_samples = sample_rate / 100;  // ~10 ms fallback.
        frame_samples = std::max<size_t>(1, frame_samples);
      }

      size_t total_frames = 0;
      size_t speech_frames = 0;
      for (size_t offset = 0; offset + frame_samples <= snapshot.size();
           offset += frame_samples) {
        ++total_frames;
        if (vad_->IsSpeech(snapshot.data() + offset, frame_samples)) {
          ++speech_frames;
        }
      }

      if (total_frames == 0) {
        continue;
      }

      const float speech_ratio =
          static_cast<float>(speech_frames) / static_cast<float>(total_frames);
      if (speech_ratio < config_.min_speech_ratio) {
        continue;
      }

      float peak = 0.0f;
      for (float sample : snapshot) {
        peak = std::max(peak, std::fabs(sample));
      }
      if (peak > 0.0f) {
        const float scale = 0.95f / peak;
        for (float& sample : snapshot) {
          sample *= scale;
        }
      }

      SnapshotCallback callback_copy;
      {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback_copy = snapshot_callback_;
      }
      if (callback_copy) {
        callback_copy(snapshot);
      }
    } catch (...) {
      // Swallow all exceptions to keep the worker alive.
    }
  }
}

}  // namespace cpp_asr
