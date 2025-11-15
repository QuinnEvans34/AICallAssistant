#include "vad_detector.h"

#include <cmath>
#include <cstring>
#include <filesystem>

#if __has_include("whisper.h")
#define HAVE_WHISPER 1
#include "whisper.h"
#endif

namespace cpp_asr {

VADDetector::VADDetector(Config config) : config_(config) {
  webrtc_handle_ = nullptr;
#if defined(HAVE_WHISPER)
  // whisper VAD uses a separate model; leave null until needed
  webrtc_handle_ = nullptr;
#endif
}

VADDetector::~VADDetector() {
#if defined(HAVE_WHISPER)
  if (webrtc_handle_) {
    try {
      whisper_vad_free(reinterpret_cast<struct whisper_vad_context*>(webrtc_handle_));
    } catch (...) {
    }
    webrtc_handle_ = nullptr;
  }
#else
  (void)webrtc_handle_;
#endif
}

bool VADDetector::IsSpeech(const float* audio, size_t num_samples) {
  if (!audio || num_samples == 0) {
    return false;
  }

#if defined(HAVE_WHISPER)
  std::lock_guard<std::mutex> lock(vad_mutex_);
  struct whisper_vad_context* vctx = reinterpret_cast<struct whisper_vad_context*>(webrtc_handle_);
  if (!vctx) {
    // Lazily initialize a whisper VAD context with default params
    try {
      struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
      vparams.n_threads = 1;
      vctx = whisper_vad_init_from_file_with_params(nullptr, vparams);
      // Note: whisper_vad_init_from_file_with_params expects a file path; passing nullptr may return a default context in some builds.
      webrtc_handle_ = reinterpret_cast<void*>(vctx);
    } catch (...) {
      vctx = nullptr;
      webrtc_handle_ = nullptr;
    }
  }

  if (vctx) {
    // whisper_vad_detect_speech expects float samples
    try {
      bool speech = whisper_vad_detect_speech(vctx, audio, static_cast<int>(num_samples));
      return speech;
    } catch (...) {
      // fallback to energy
    }
  }
#endif

  // Energy-based fallback
  double sum_sq = 0.0;
  for (size_t i = 0; i < num_samples; ++i) {
    const double sample = static_cast<double>(audio[i]);
    sum_sq += sample * sample;
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(num_samples));

  const float effective_threshold = (config_.energy_threshold > 0.0f) ? config_.energy_threshold : 0.015f;
  return rms >= static_cast<double>(effective_threshold);
}

void VADDetector::Reset() {
  std::lock_guard<std::mutex> lock(vad_mutex_);
#if defined(HAVE_WHISPER)
  if (webrtc_handle_) {
    try {
      whisper_vad_free(reinterpret_cast<struct whisper_vad_context*>(webrtc_handle_));
    } catch (...) {
    }
    webrtc_handle_ = nullptr;
  }
#endif
}

}  // namespace cpp_asr
