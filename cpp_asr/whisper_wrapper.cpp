#include "whisper_wrapper.h"

#include <algorithm>
#include <cctype>
#include <future>
#include <utility>

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
  // TODO: Call whisper_free(ctx) or equivalent once the dependency is available.
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

  // Release any existing model context before loading a new one.
  DestroyContext(whisper_context_);
  loaded_ = false;

  model_path_ = model_path;
  try {
    // TODO: whisper_context_ = whisper_init_from_file(model_path_.c_str());
    whisper_context_ = reinterpret_cast<void*>(0x1);  // Placeholder sentinel.
    loaded_ = (whisper_context_ != nullptr);
  } catch (...) {
    DestroyContext(whisper_context_);
    loaded_ = false;
  }

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

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
      result.transcript.clear();
      return result;
    }

    try {
      // TODO: Configure whisper_full_params for greedy, single-pass decoding.
      // TODO: whisper_full(whisper_context_, params, audio.data(), audio.size());

      result.transcript = TrimWhitespace("[stub transcript]");
      result.average_logprob = 0.0;
      result.is_final = true;
    } catch (...) {
      result.transcript.clear();
      result.average_logprob = 0.0;
      result.is_final = false;
    }

    callback = segment_callback_;
  }

  if (callback) {
    callback(result);
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
