#define _USE_MATH_DEFINES
#include <cmath>
#include "asr_engine.h"

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
  std::string model_path;
  if (argc > 1) {
    model_path = argv[1];
  } else {
    const char* env_model = std::getenv("ASR_MODEL_PATH");
    if (env_model) {
      model_path = env_model;
    } else {
      std::cerr << "Usage: " << argv[0] << " [model_path]" << std::endl;
      std::cerr << "Or set ASR_MODEL_PATH environment variable." << std::endl;
      return 1;
    }
  }

  cpp_asr::ASREngine engine;
  if (!engine.Initialize(model_path)) {
    std::cerr << "Failed to initialize ASR engine." << std::endl;
    return 1;
  }

  std::cout << "ASR Engine initialized. Pushing test audio..." << std::endl;

  // Generate some dummy audio (sine wave at 440Hz for 1 second at 16kHz)
  const size_t sample_rate = 16000;
  const size_t num_samples = sample_rate;  // 1 second
  std::vector<float> audio(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    audio[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * static_cast<float>(i) / sample_rate);
  }

  // Push audio in chunks
  const size_t chunk_size = 1600;  // 100ms
  for (size_t i = 0; i < num_samples; i += chunk_size) {
    size_t to_push = std::min(chunk_size, num_samples - i);
    engine.PushAudio(audio.data() + i, to_push);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Poll for transcripts
  std::cout << "Polling for transcripts..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::string transcript;
    if (engine.PollTranscript(transcript)) {
      std::cout << "Transcript: " << transcript << std::endl;
    } else {
      std::cout << "No transcript available." << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  engine.Shutdown();
  std::cout << "Test completed." << std::endl;
  return 0;
}