#include "vad_detector.h"

#include <cmath>

namespace cpp_asr {

VADDetector::VADDetector(Config config) : config_(config) {
  webrtc_handle_ = nullptr;
}

VADDetector::~VADDetector() {
  webrtc_handle_ = nullptr;
}

bool VADDetector::IsSpeech(const float* audio, size_t num_samples) {
  if (!audio || num_samples == 0) {
    return false;
  }

  double sum_sq = 0.0;
  for (size_t i = 0; i < num_samples; ++i) {
    const double sample = static_cast<double>(audio[i]);
    sum_sq += sample * sample;
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(num_samples));

  const float effective_threshold =
      (config_.energy_threshold > 0.0f) ? config_.energy_threshold : 0.015f;
  return rms >= static_cast<double>(effective_threshold);
}

void VADDetector::Reset() {
  // TODO: Reset adaptive thresholds or WebRTC state when implemented.
}

}  // namespace cpp_asr
