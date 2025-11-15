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

# Read WAV and resample if needed (assumes 16kHz mono input for now)
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

# TODO: Resample to 16000 if sr != 16000. For now, require 16k.
if sr != 16000:
    print('Input sample rate is %d, but this test expects 16000.' % sr)
    sys.exit(2)

# Initialize engine
native.init_asr_engine("")  # empty -> fake mode if configured

# Push audio in chunks
chunk = 16000  # 1 second
pos = 0
start = time.time()
while pos < len(audio):
    end = min(len(audio), pos + chunk)
    buf = audio[pos:end].astype(np.float32)
    native.push_audio(buf)
    pos = end
    time.sleep(0.1)

# Poll for transcripts
deadline = time.time() + 10.0
while time.time() < deadline:
    t = native.poll_transcript()
    if t:
        print('TRANSCRIPT:', t)
    else:
        time.sleep(0.2)

print('Done')
