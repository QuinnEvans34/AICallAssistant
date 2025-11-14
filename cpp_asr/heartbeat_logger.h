#pragma once

#include <mutex>
#include <string>

namespace cpp_asr {

/**
 * @brief Structured heartbeat writer used for observability and backpressure.
 *
 * The logger appends JSONL entries describing ASR state (queue backlog, VAD
 * status, processing latency). This makes it easy for the Python controller
 * to inspect the state without digging into native logs.
 */
class HeartbeatLogger {
 public:
  explicit HeartbeatLogger(std::string log_path = "");
  ~HeartbeatLogger();

  HeartbeatLogger(const HeartbeatLogger&) = delete;
  HeartbeatLogger& operator=(const HeartbeatLogger&) = delete;

  void LogHeartbeat(int queue_depth,
                    double snapshot_latency_ms,
                    double processing_time_ms,
                    bool vad_detected);

  void SetLogPath(const std::string& log_path);

 private:
  std::string log_path_;
  std::mutex mutex_;
};

}  // namespace cpp_asr

