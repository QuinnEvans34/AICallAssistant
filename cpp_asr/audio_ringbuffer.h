#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

namespace cpp_asr {

/**
 * @brief Thread-safe single-producer multi-consumer ring buffer for PCM audio.
 *
 * The buffer stores floating point PCM samples in a circular array. A single writer
 * (the audio ingestion thread) appends samples via PushSamples. Multiple consumers
 * may request snapshots by calling GetLatestSamples or TryReadSlice. Internally the
 * class uses a mutex/condition_variable pair to keep the implementation simple and
 * safe. It can be upgraded to a lock-free structure once the API is validated.
 *
 * The buffer enforces a fixed sample rate (16 kHz) because all downstream DSP
 * components assume that rate. Consumers request slices in samples rather than
 * seconds; helper methods convert between duration and sample count as needed.
 */
class AudioRingBuffer {
 public:
  explicit AudioRingBuffer(size_t max_duration_seconds = 30);
  ~AudioRingBuffer() = default;

  AudioRingBuffer(const AudioRingBuffer&) = delete;
  AudioRingBuffer& operator=(const AudioRingBuffer&) = delete;

  /**
   * @brief Pushes PCM samples into the ring buffer.
   *
   * This method may be called at high frequency from the audio capture thread.
   * When the buffer is full, the oldest samples are overwritten. PushSamples
   * never blocks; consumers are expected to keep up via TryReadSlice.
   */
  void PushSamples(const float* samples, size_t count);

  /**
   * @brief Returns a snapshot containing the most recent num_samples samples.
   *
   * This is typically used for debugging or metric collection where the caller
   * needs to inspect the newest audio regardless of VAD gating.
   */
  std::vector<float> GetLatestSamples(size_t num_samples) const;

  /**
   * @brief Attempts to copy num_samples samples into the provided vector.
   *
   * @param num_samples Number of samples requested.
   * @param out Snapshot buffer. The method resizes it on success.
   * @return true when enough samples are available; false otherwise.
   */
  bool TryReadSlice(size_t num_samples, std::vector<float>& out);

  /**
   * @return Current number of valid samples in the buffer.
   */
  size_t Size() const;

  size_t SampleRate() const { return kSampleRateHz; }
  size_t CapacitySamples() const { return capacity_samples_; }
  void Clear();

  // Expose synchronization primitives for SnapshotWorker optimization
  std::mutex& GetMutexRef() { return mutex_; }
  std::condition_variable& GetDataReadyCvRef() { return data_ready_cv_; }

 private:
  static constexpr size_t kSampleRateHz = 16000;

  size_t capacity_samples_;
  mutable std::mutex mutex_;
  std::condition_variable data_ready_cv_;
  std::vector<float> buffer_;
  size_t write_index_{0};
  size_t available_samples_{0};
};

}  // namespace cpp_asr
