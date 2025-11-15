import os
import sys
import time
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    # Extend sys.path to typical build output dirs
    for p in [repo / "cpp_asr" / "build" / "Release", repo / "build" / "Release", repo]:
        if p.exists():
            sys.path.insert(0, str(p))

    try:
        import cpp_asr_native as native  # type: ignore
    except Exception as e:
        print("FAIL: import cpp_asr_native ->", e)
        return 1

    print("OK: imported cpp_asr_native")

    model = os.getenv("ASR_MODEL_PATH")
    if not model or not Path(model).exists():
        print("WARN: ASR_MODEL_PATH not set or file missing -> skipping engine init test")
        return 0

    try:
        native.init_asr_engine(model)
        native.reset_call()
        # push a small silent buffer
        import numpy as np
        native.push_audio(np.zeros((1600,), dtype=np.float32))
        text = native.poll_transcript()
        print("poll_transcript ->", repr(text))
    except Exception as e:
        print("FAIL: engine interaction ->", e)
        return 1
    finally:
        try:
            native.shutdown_asr_engine()
        except Exception:
            pass

    print("OK: engine init/push/poll/shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
