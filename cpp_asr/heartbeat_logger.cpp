#include "heartbeat_logger.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cpp_asr {

namespace {
std::string CurrentTimestamp() {
  using clock = std::chrono::system_clock;
  auto now = clock::now();
  std::time_t now_time = clock::to_time_t(now);
  std::tm tm;
#if defined(_WIN32)
  localtime_s(&tm, &now_time);
#else
  localtime_r(&now_time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}
}  // namespace

HeartbeatLogger::HeartbeatLogger(std::string log_path)
    : log_path_(std::move(log_path)) {}

HeartbeatLogger::~HeartbeatLogger() = default;

void HeartbeatLogger::LogHeartbeat(int queue_depth,
                                   double snapshot_latency_ms,
                                   double processing_time_ms,
                                   bool vad_detected) {
  if (log_path_.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream out(log_path_, std::ios::app);
  if (!out.is_open()) {
    return;
  }
  out << "{\"ts\":\"" << CurrentTimestamp() << "\","
      << "\"queue_depth\":" << queue_depth << ","
      << "\"snapshot_latency_ms\":" << snapshot_latency_ms << ","
      << "\"processing_time_ms\":" << processing_time_ms << ","
      << "\"vad_detected\":" << (vad_detected ? "true" : "false") << "}"
      << std::endl;
}

void HeartbeatLogger::SetLogPath(const std::string& log_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  log_path_ = log_path;
}

}  // namespace cpp_asr

