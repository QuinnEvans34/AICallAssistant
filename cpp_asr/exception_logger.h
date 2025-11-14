#pragma once

#include <mutex>
#include <string>

namespace cpp_asr {

/**
 * @brief Simple structured logger that records exceptions thrown by worker threads.
 *
 * This keeps the native subsystem resilient: asynchronous exceptions are persisted
 * and surfaced to Python without crashing the process. The logger intentionally
 * avoids throwing; failures to write logs are silently ignored.
 */
class ExceptionLogger {
 public:
  explicit ExceptionLogger(std::string log_path = "");
  ~ExceptionLogger();

  ExceptionLogger(const ExceptionLogger&) = delete;
  ExceptionLogger& operator=(const ExceptionLogger&) = delete;

  void LogException(const std::string& component, const std::string& message);
  void SetLogPath(const std::string& log_path);

 private:
  std::string log_path_;
  std::mutex mutex_;
};

}  // namespace cpp_asr

