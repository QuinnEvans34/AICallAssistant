from __future__ import annotations

import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

# Support running as a package (backend_server.main_server) and as a script (python backend_server/main_server.py)
try:
    from .asr_engine import ASREngine  # type: ignore
    from .heartbeat_manager import HeartbeatManager  # type: ignore
    from .transcript_stream import TranscriptStream  # type: ignore
    from .call_manager import CallManager  # type: ignore
except Exception:  # pragma: no cover
    from asr_engine import ASREngine  # type: ignore
    from heartbeat_manager import HeartbeatManager  # type: ignore
    from transcript_stream import TranscriptStream  # type: ignore
    from call_manager import CallManager  # type: ignore

from call_transcript_recorder import TranscriptRecorder

app = FastAPI(title="Call Assistant Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketChannel:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None

    async def ensure_started(self):
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        if not self.task:
            self.task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self):
        while True:
            kind, payload = await self.queue.get()
            dead = []
            for ws in list(self.connections):
                try:
                    if kind == "text":
                        await ws.send_text(payload)
                    else:
                        await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.connections.discard(ws)

    async def connect(self, websocket: WebSocket):
        await self.ensure_started()
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)

    def send_text(self, message: str):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.queue.put(("text", message)), self.loop)

    def send_json(self, payload: dict):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.queue.put(("json", payload)), self.loop)


transcript_channel = WebSocketChannel()
question_channel = WebSocketChannel()
suggestion_channel = WebSocketChannel()
status_channel = WebSocketChannel()

# Initialize ASR engine and stream components
asr_engine = ASREngine()
transcript_recorder = TranscriptRecorder()
transcript_stream = TranscriptStream(asr_engine, transcript_recorder, lambda _: None, lambda _: None)
heartbeat = HeartbeatManager()

call_manager = CallManager(
    transcript_stream=transcript_stream,
    heartbeat=heartbeat,
    broadcast_transcript=lambda text: transcript_channel.send_text(text),
    broadcast_question=lambda payload: question_channel.send_json(payload),
    broadcast_suggestion=lambda payload: suggestion_channel.send_json(payload),
    broadcast_status=lambda payload: status_channel.send_json(payload),
    transcript_recorder=transcript_recorder,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start_call")
async def start_call():
    call_id = call_manager.start_call()
    return {"status": "ok", "call_id": call_id}


@app.post("/end_call")
async def end_call():
    call_manager.end_call()
    return {"status": "ok"}


@app.get("/current_transcript")
async def current_transcript():
    return {"transcript": call_manager.get_transcript_text()}


@app.websocket("/ws/transcript")
async def ws_transcript(websocket: WebSocket):
    await transcript_channel.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        transcript_channel.disconnect(websocket)


@app.websocket("/ws/questions")
async def ws_questions(websocket: WebSocket):
    await question_channel.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        question_channel.disconnect(websocket)


@app.websocket("/ws/suggestions")
async def ws_suggestions(websocket: WebSocket):
    await suggestion_channel.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        suggestion_channel.disconnect(websocket)


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    await status_channel.connect(websocket)
    try:
        # Send initial snapshot (PascalCase for C# DTOs)
        snapshot = heartbeat.snapshot()
        await websocket.send_json(
            {
                "AsrLatencyMs": snapshot.get("asr_latency_ms", 0.0),
                "QueueDepth": snapshot.get("queue_depth", 0),
                "Errors": snapshot.get("errors", 0),
                "Processing": snapshot.get("processing", False),
            }
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        status_channel.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_server.main_server:app", host="0.0.0.0", port=8000, reload=False)
