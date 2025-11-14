#include "c_api_bindings.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "asr_engine.h"

#if defined(CPP_ASR_BUILD_PYBIND11)
#include <pybind11/pybind11.h>
#endif

namespace {
std::mutex g_engine_mutex;

cpp_asr::ASREngine& GetEngine() {
  static cpp_asr::ASREngine engine;
  return engine;
}
}  // namespace

extern "C" {

void init_asr_engine(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  if (model_path) {
    GetEngine().Initialize(model_path);
  }
}

void push_audio(const float* samples, int count) {
  if (!samples || count <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  GetEngine().PushAudio(samples, static_cast<size_t>(count));
}

bool poll_transcript(char* buffer, int max_len) {
  if (!buffer || max_len <= 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  std::string transcript;
  if (!GetEngine().PollTranscript(transcript)) {
    buffer[0] = '\0';
    return false;
  }

  const int to_copy =
      static_cast<int>(std::min(static_cast<size_t>(max_len - 1), transcript.size()));
  std::memcpy(buffer, transcript.data(), to_copy);
  buffer[to_copy] = '\0';
  return true;
}

void reset_call() {
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  GetEngine().ResetCall();
}

void shutdown_asr_engine() {
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  GetEngine().Shutdown();
}

}  // extern "C"

#if defined(CPP_ASR_BUILD_PYBIND11)
PYBIND11_MODULE(cpp_asr_native, m) {
  m.doc() = "PyBind11 bridge to the cpp_asr subsystem";

  m.def("init_asr_engine", [](const std::string& path) { init_asr_engine(path.c_str()); });
  m.def("push_audio",
        [](pybind11::buffer samples) {
          pybind11::buffer_info info = samples.request();
          if (info.format != pybind11::format_descriptor<float>::format()) {
            throw std::runtime_error("push_audio expects float32 buffer");
          }
          push_audio(static_cast<const float*>(info.ptr),
                     static_cast<int>(info.size));
        });
  m.def("poll_transcript",
        []() {
          std::array<char, 4096> buffer{};
          const bool has_value = poll_transcript(buffer.data(), static_cast<int>(buffer.size()));
          return has_value ? std::string(buffer.data()) : std::string();
        });
  m.def("reset_call", []() { reset_call(); });
  m.def("shutdown_asr_engine", []() { shutdown_asr_engine(); });
}
#endif
