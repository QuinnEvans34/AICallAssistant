#include "whisper_wrapper.h"

#include <algorithm>
#include <cctype>
#include <future>
#include <utility>
#include <vector>
#include <iostream>
#include <filesystem>

#if defined(_WIN32)
#define WHISPER_DLL_EXPORT
#endif

#if __has_include("whisper.h")
#define HAVE_WHISPER 1
#include "whisper.h"
#endif

namespace cpp_asr {

namespace {
std::string TrimWhitespace(std::string text) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  text.erase(text.begin(),
             std::find_if(text.begin(), text.end(), not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(),
             text.end());
  return text;
}

void DestroyContext(void*& ctx) {
  if (!ctx) {
    return;
  }
#if defined(HAVE_WHISPER)
  try {
    struct whisper_context* c = reinterpret_cast<struct whisper_context*>(ctx);
    whisper_free(c);
  } catch (...) {
    // swallow
  }
#endif
  ctx = nullptr;
}
}  // namespace

WhisperWrapper::WhisperWrapper() = default;
WhisperWrapper::~WhisperWrapper() {
  std::lock_guard<std::mutex> lock(mutex_);
  DestroyContext(whisper_context_);
  loaded_ = false;
}

bool WhisperWrapper::LoadModel(const std::string& model_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (loaded_ && model_path == model_path_) {
    return true;
  }

  DestroyContext(whisper_context_);
  loaded_ = false;

  model_path_ = model_path;

#if defined(HAVE_WHISPER)
  try {
    if (!std::filesystem::exists(model_path_)) {
      return false;
    }
    // Use default context params
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path_.c_str(), cparams);
    if (!ctx) {
      DestroyContext(whisper_context_);
      loaded_ = false;
      return false;
    }
    whisper_context_ = reinterpret_cast<void*>(ctx);
    loaded_ = true;
  } catch (...) {
    DestroyContext(whisper_context_);
    loaded_ = false;
  }
#else
  // Fallback stub behavior if whisper not available at compile time
  try {
    whisper_context_ = reinterpret_cast<void*>(0x1);  // sentinel
    loaded_ = true;
  } catch (...) {
    DestroyContext(whisper_context_);
    loaded_ = false;
  }
#endif
  return loaded_;
}

bool WhisperWrapper::IsLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loaded_;
}

WhisperWrapper::DecodeResult WhisperWrapper::DecodeBlocking(
    const std::vector<float>& audio) {
  DecodeResult result;
  std::function<void(const DecodeResult&)> callback;

#if defined(HAVE_WHISPER)
  struct whisper_context* ctx = nullptr;
#endif
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
      result.transcript.clear();
      return result;
    }
#if defined(HAVE_WHISPER)
    ctx = reinterpret_cast<struct whisper_context*>(whisper_context_);
    if (!ctx) {
      result.transcript.clear();
      return result;
    }
#endif
    callback = segment_callback_;
  }

#if defined(HAVE_WHISPER)
  try {
    // whisper_full expects float PCM samples (32-bit float) at 16kHz
    const float* pcmf32 = audio.data();
    int n_samples = static_cast<int>(audio.size());

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // Configure for low-latency single-pass decoding
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.token_timestamps = false;
    params.translate = false;
    params.language = "en"; // default language
    params.single_segment = true;

    int rv = whisper_full(ctx, params, pcmf32, n_samples);
    if (rv != 0) {
      // decode failed
      result.transcript.clear();
      result.average_logprob = 0.0;
      result.is_final = false;
    } else {
      // iterate segments
      std::string out;
      int n_segments = whisper_full_n_segments(ctx);
      for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text && *text) {
          if (!out.empty()) out.push_back(' ');
          out += text;
        }
      }
      result.transcript = TrimWhitespace(out);
      result.average_logprob = 0.0;
      result.is_final = !result.transcript.empty();
    }
  } catch (...) {
    result.transcript.clear();
    result.average_logprob = 0.0;
    result.is_final = false;
  }
#else
  // Stub behavior when whisper is not compiled in
  try {
    result.transcript = TrimWhitespace("[stub transcript]");
    result.average_logprob = 0.0;
    result.is_final = true;
  } catch (...) {
    result.transcript.clear();
    result.average_logprob = 0.0;
    result.is_final = false;
  }
#endif

  if (callback) {
    try {
      callback(result);
    } catch (...) {
      // swallow callback exceptions
    }
  }
  return result;
}

std::future<WhisperWrapper::DecodeResult> WhisperWrapper::DecodeAsync(
    const std::vector<float>& audio) {
  return std::async(std::launch::async, [this, audio]() { return DecodeBlocking(audio); });
}

void WhisperWrapper::SetSegmentCallback(
    std::function<void(const DecodeResult&)> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  segment_callback_ = std::move(callback);
}

}  // namespace cpp_asr
