#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace cpp_asr {

/**
 * @brief Lightweight wrapper around WebRTC VAD (or equivalent).
 *
 * The detector exposes a synchronous IsSpeech() method that inspects a PCM frame
 * and returns whether speech is present. The implementation can be swapped out
 * for WebRTC, Silero, or energy-based heuristics without affecting callers.
 */
class VADDetector {
 public:
  struct Config {
    int mode = 2;               // WebRTC aggressiveness 0-3.
    float energy_threshold = 0;  // Optional fallback when VAD unavailable.
    int frame_duration_ms = 30;
    int sample_rate_hz = 16000;
  };

  explicit VADDetector(Config config = {});
  ~VADDetector();

  VADDetector(const VADDetector&) = delete;
  VADDetector& operator=(const VADDetector&) = delete;

  /**
   * @brief Returns true when speech is detected in the provided frame.
   *
   * Thread-safe and alloc-free. In the initial scaffolding the implementation
   * simply returns true when the energy is above the configured threshold,
   * allowing higher-level components to be validated before wiring WebRTC.
   */
  bool IsSpeech(const float* audio, size_t num_samples);

  Config config() const { return config_; }
  void Reset();

 private:
  Config config_;
  std::mutex vad_mutex_;

  // Placeholder pointer for the actual WebRTC handle.
  void* webrtc_handle_{nullptr};
};

}  // namespace cpp_asr
