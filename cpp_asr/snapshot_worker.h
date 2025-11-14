#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "audio_ringbuffer.h"
#include "vad_detector.h"

namespace cpp_asr {

/**
 * @brief Background worker that periodically extracts VAD-gated snapshots.
 *
 * The worker owns a dedicated thread that loops at SnapshotPeriodMs and
 * obtains slices from the AudioRingBuffer. When VAD fires, the worker invokes
 * the snapshot callback, allowing ASREngine to submit inference requests.
 */
class SnapshotWorker {
 public:
  using SnapshotCallback = std::function<void(const std::vector<float>&)>;

  struct Config {
    int snapshot_duration_ms = 2000;
    int snapshot_period_ms = 500;
    float min_speech_ratio = 0.2f;
  };

  SnapshotWorker(std::shared_ptr<AudioRingBuffer> ring_buffer,
                 std::shared_ptr<VADDetector> vad,
                 Config config = {});
  ~SnapshotWorker();

  SnapshotWorker(const SnapshotWorker&) = delete;
  SnapshotWorker& operator=(const SnapshotWorker&) = delete;

  void Start();
  void Stop();
  void Reset();
  void SetSnapshotCallback(SnapshotCallback cb);

 private:
  void WorkerLoop();

  std::shared_ptr<AudioRingBuffer> ring_buffer_;
  std::shared_ptr<VADDetector> vad_;
  Config config_;

  SnapshotCallback snapshot_callback_;

  std::thread worker_thread_;
  std::atomic<bool> running_{false};
  mutable std::mutex callback_mutex_;
};
}  // namespace cpp_asr
