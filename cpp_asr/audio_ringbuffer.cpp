#include "audio_ringbuffer.h"

#include <algorithm>

namespace cpp_asr {

AudioRingBuffer::AudioRingBuffer(size_t max_duration_seconds)
    : capacity_samples_(kSampleRateHz * max_duration_seconds),
      buffer_(capacity_samples_, 0.0f) {}

void AudioRingBuffer::PushSamples(const float* samples, size_t count) {
  if (!samples || count == 0 || capacity_samples_ == 0) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  size_t copy_count = std::min(count, capacity_samples_);
  const float* source = samples + (count - copy_count);

  for (size_t i = 0; i < copy_count; ++i) {
    buffer_[write_index_] = source[i];
    write_index_ = (write_index_ + 1) % capacity_samples_;
  }

  if (copy_count == capacity_samples_) {
    available_samples_ = capacity_samples_;
  } else {
    available_samples_ =
        std::min(capacity_samples_, available_samples_ + copy_count);
  }
  lock.unlock();
  data_ready_cv_.notify_all();
}

std::vector<float> AudioRingBuffer::GetLatestSamples(size_t num_samples) const {
  std::vector<float> snapshot;
  if (num_samples == 0) {
    return snapshot;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (available_samples_ == 0) {
    return snapshot;
  }
  const size_t samples_to_copy = std::min(num_samples, available_samples_);
  snapshot.resize(samples_to_copy);

  const size_t start =
      (write_index_ + capacity_samples_ - samples_to_copy) % capacity_samples_;
  for (size_t i = 0; i < samples_to_copy; ++i) {
    snapshot[i] = buffer_[(start + i) % capacity_samples_];
  }
  return snapshot;
}

bool AudioRingBuffer::TryReadSlice(size_t num_samples,
                                   std::vector<float>& out) {
  if (num_samples == 0) {
    out.clear();
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (available_samples_ < num_samples) {
    return false;
  }

  out.resize(num_samples);
  const size_t start =
      (write_index_ + capacity_samples_ - num_samples) % capacity_samples_;
  for (size_t i = 0; i < num_samples; ++i) {
    out[i] = buffer_[(start + i) % capacity_samples_];
  }
  return true;
}

size_t AudioRingBuffer::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return available_samples_;
}

void AudioRingBuffer::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  write_index_ = 0;
  available_samples_ = 0;
  std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}

}  // namespace cpp_asr
