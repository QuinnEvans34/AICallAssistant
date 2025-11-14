#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cpp_asr {

/**
 * @brief Thin wrapper around whisper.cpp / faster-whisper inference.
 *
 * The wrapper isolates model loading and decoding logic. The blocking decode
 * method is used by the SnapshotWorker for synchronous processing, while
 * DecodeAsync provides a future upgrade path for overlapped inference.
 */
class WhisperWrapper {
 public:
  struct DecodeResult {
    std::string transcript;
    double average_logprob = 0.0;
    bool is_final = false;
  };

  WhisperWrapper();
  ~WhisperWrapper();

  WhisperWrapper(const WhisperWrapper&) = delete;
  WhisperWrapper& operator=(const WhisperWrapper&) = delete;

  bool LoadModel(const std::string& model_path);
  bool IsLoaded() const;

  DecodeResult DecodeBlocking(const std::vector<float>& audio);
  std::future<DecodeResult> DecodeAsync(const std::vector<float>& audio);

  void SetSegmentCallback(std::function<void(const DecodeResult&)> callback);

 private:
  std::string model_path_;
  mutable std::mutex mutex_;
  bool loaded_{false};

  std::function<void(const DecodeResult&)> segment_callback_;

  // Placeholder for future whisper.cpp context pointer.
  void* whisper_context_{nullptr};
};

}  // namespace cpp_asr

