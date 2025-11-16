#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "audio_ringbuffer.h"
#include "exception_logger.h"
#include "heartbeat_logger.h"
#include "snapshot_worker.h"
#include "vad_detector.h"
#include "whisper_wrapper.h"

namespace cpp_asr {

/**
 * @brief High-level coordinator that exposes the ASR subsystem API.
 *
 * ASREngine owns the ring buffer, snapshot worker, VAD, and whisper wrapper.
 * It provides a thread-safe interface to push audio, retrieve transcripts,
 * reset state between calls, and shutdown cleanly. Python will interact with
 * this class through C bindings defined in c_api_bindings.cpp.
 */
class ASREngine {
 public:
  using TranscriptCallback = std::function<void(const std::string&)>;

  ASREngine();
  ~ASREngine();

  ASREngine(const ASREngine&) = delete;
  ASREngine& operator=(const ASREngine&) = delete;

  bool Initialize(const std::string& model_path);  // TODO: Wire up components and load Whisper.
  void Shutdown();  // TODO: Stop workers, release buffers, reset metrics.

  bool PushAudio(const float* samples,
                 std::size_t count);  // TODO: Ingest PCM from Python capture thread.
  bool PollTranscript(std::string& transcript);        // TODO: Pop most recent transcript for Python poll.
  void ResetCall();                                    // TODO: Clear per-call state before new session.

  void SetTranscriptCallback(TranscriptCallback cb);
  void ConfigureLogs(const std::string& heartbeat_log_path,
                     const std::string& exception_log_path);

  void ResetForCall(const std::string& call_id);

 private:
  void HandleSnapshot(const std::vector<float>& snapshot);
  void EnqueueTranscript(const std::string& text);
  std::string ExtractNovelTextLocked(const std::string& transcript);

  // Decode worker
  void StartDecodeWorkers(size_t num_threads = 2);
  void StopDecodeWorkers();
  void DecodeWorkerLoop();

  std::mutex transcript_mutex_;
  std::queue<std::string> transcript_queue_;

  TranscriptCallback transcript_callback_;
  std::string tail_text_;
  std::size_t max_tail_chars_ = 200;
  std::size_t dedup_window_ = 40;

  std::shared_ptr<AudioRingBuffer> ring_buffer_;
  std::shared_ptr<VADDetector> vad_;
  std::unique_ptr<WhisperWrapper> whisper_;
  std::unique_ptr<SnapshotWorker> snapshot_worker_;
  std::unique_ptr<HeartbeatLogger> heartbeat_logger_;
  std::unique_ptr<ExceptionLogger> exception_logger_;
  std::string heartbeat_log_path_;
  std::string exception_log_path_;

  std::atomic<bool> initialized_{false};

  // Decode threadpool
  std::vector<std::thread> decode_threads_;
  std::queue<std::pair<std::vector<float>, std::chrono::steady_clock::time_point>> decode_queue_;
  std::mutex decode_mutex_;
  std::condition_variable decode_cv_;
  std::atomic<bool> decode_running_{false};

  // Metrics
  std::atomic<size_t> snapshots_processed_{0};
  std::atomic<double> total_decode_ms_{0.0};
  std::string last_transcript_;
  std::mutex last_transcript_mutex_;
};

}  // namespace cpp_asr
