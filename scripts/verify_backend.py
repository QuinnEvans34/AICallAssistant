import os
import sys
import time
import json
import subprocess
from pathlib import Path

import requests
import asyncio
import websockets


def wait_for_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


async def check_ws(uri: str, name: str) -> None:
    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            # Send a noop to satisfy server receive loop
            try:
                await ws.send("ping")
            except Exception:
                pass
            # Try to read a bit if server sends something (status sends snapshot)
            try:
                await asyncio.wait_for(ws.recv(), timeout=1.0)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"WebSocket {name} failed: {e}")


async def verify_websockets(base_ws: str) -> None:
    tasks = [
        check_ws(f"{base_ws}/ws/transcript", "transcript"),
        check_ws(f"{base_ws}/ws/questions", "questions"),
        check_ws(f"{base_ws}/ws/suggestions", "suggestions"),
        check_ws(f"{base_ws}/ws/status", "status"),
    ]
    await asyncio.gather(*tasks)


def main():
    repo = Path(__file__).resolve().parents[1]

    # Ensure cpp_asr_native is importable by the backend
    py_ext_dirs = [repo / "cpp_asr" / "build" / "Release", repo / "build" / "Release", repo]
    extra_path = os.pathsep.join(str(p) for p in py_ext_dirs if p.exists())
    env = os.environ.copy()
    env["ASR_ALLOW_FAKE"] = env.get("ASR_ALLOW_FAKE", "1")  # enable fake ASR by default for verification
    if extra_path:
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

    # Launch backend
    python = sys.executable
    backend = subprocess.Popen(
        [python, "-u", str(repo / "backend_server" / "main_server.py")],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ok = wait_for_health("http://127.0.0.1:8000/health", timeout=25)
        if not ok:
            # dump some logs
            try:
                out = backend.stdout.read(2000) if backend.stdout else ""
                print("Backend output (partial):\n", out)
            except Exception:
                pass
            return 2

        asyncio.run(verify_websockets("ws://127.0.0.1:8000"))
        print("OK: backend health and websockets verified")
        return 0
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=3)
        except Exception:
            backend.kill()


if __name__ == "__main__":
    raise SystemExit(main())
