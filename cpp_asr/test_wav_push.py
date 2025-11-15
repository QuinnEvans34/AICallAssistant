import sys
import wave
import numpy as np
import time

try:
    import cpp_asr_native as native
except Exception as e:
    print('Failed to import cpp_asr_native:', e)
    raise

if len(sys.argv) < 2:
    print('Usage: python test_wav_push.py path/to/file.wav')
    sys.exit(1)

wav_path = sys.argv[1]

# Read WAV
with wave.open(wav_path, 'rb') as wf:
    nch = wf.getnchannels()
    sr = wf.getframerate()
    nframes = wf.getnframes()
    sampwidth = wf.getsampwidth()
    data = wf.readframes(nframes)

# Convert to float32
if sampwidth == 2:
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
elif sampwidth == 4:
    audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
else:
    raise RuntimeError('Unsupported sample width: %d' % sampwidth)

if nch > 1:
    audio = audio.reshape(-1, nch)
    audio = audio.mean(axis=1)

# Resample to 16k if needed (simple linear resample)
if sr != 16000:
    import math
    ratio = 16000.0 / sr
    new_len = int(math.ceil(len(audio) * ratio))
    indices = np.linspace(0, len(audio) - 1, new_len)
    left = np.floor(indices).astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = indices - left
    resampled = (1 - frac) * audio[left] + frac * audio[right]
    audio = resampled.astype(np.float32)

# Initialize engine
native.init_asr_engine("")  # empty -> fake mode if configured

# Push audio in chunks of 0.5s
chunk = 16000 // 2  # 0.5 second
pos = 0
start = time.time()
while pos < len(audio):
    end = min(len(audio), pos + chunk)
    buf = audio[pos:end].astype(np.float32)
    native.push_audio(buf)
    pos = end
    time.sleep(0.05)

# Poll for transcripts
deadline = time.time() + 10.0
while time.time() < deadline:
    t = native.poll_transcript()
    if t:
        print('TRANSCRIPT:', t)
    else:
        time.sleep(0.2)

print('Done')
