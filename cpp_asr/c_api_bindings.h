#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the ASR engine with the provided model path.
 */
void init_asr_engine(const char* model_path);

/**
 * Pushes PCM samples captured in Python into the native audio pipeline.
 */
void push_audio(const float* samples, int count);

/**
 * Attempts to retrieve the latest transcript. Returns true when a new result
 * was available and writes a null-terminated UTF-8 string into |buffer|.
 */
bool poll_transcript(char* buffer, int buffer_size);

void reset_call();
void shutdown_asr_engine();

#ifdef __cplusplus
}
#endif
