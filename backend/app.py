from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from storage import RecordingStorage
from transcription import (
    BYTES_PER_SECOND,
    WhisperTranscriber,
    contains_possible_speech,
    load_legal_context,
    pcm_to_wav,
)
from vad_processor import StreamingVAD


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT_DIR / ".demo"
MODEL_PATH = Path(
    os.getenv("SILERO_MODEL_PATH", RUNTIME_DIR / "models" / "silero_vad.onnx")
)
WHISPER_MODEL = os.getenv("WHISPER_MODEL_NAME", "large-v3-turbo")
WHISPER_MODEL_PATH = Path(
    os.getenv(
        "WHISPER_MODEL_PATH",
        RUNTIME_DIR / "models" / "whisper-large-v3-turbo",
    )
)
SAMPLE_RATE = 16000
RETENTION_DAYS = max(0, int(os.getenv("RECORDING_RETENTION_DAYS", "30")))
MAX_RECORDING_SECONDS = max(
    60, int(os.getenv("MAX_RECORDING_SECONDS", str(30 * 60)))
)
LEGAL_PROMPT_ENABLED = os.getenv("ENABLE_LEGAL_PROMPT", "false").lower() in {
    "1",
    "true",
    "yes",
}
INTERIM_WINDOW_BYTES = 8 * BYTES_PER_SECOND

storage = RecordingStorage(
    RUNTIME_DIR / "easylistener.sqlite3",
    RUNTIME_DIR / "recordings",
)
inference_lock = asyncio.Lock()
configured_legal_prompt, legal_corrections = load_legal_context(ROOT_DIR)
legal_prompt = configured_legal_prompt if LEGAL_PROMPT_ENABLED else ""
transcriber = WhisperTranscriber(
    WHISPER_MODEL_PATH,
    WHISPER_MODEL,
    legal_prompt,
    legal_corrections,
    inference_lock,
)
app = FastAPI(title="EasyListner Local STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SegmentReview(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.on_event("startup")
async def cleanup_expired_recordings() -> None:
    storage.cleanup_older_than(RETENTION_DAYS)


async def whisper_ready() -> bool:
    return await transcriber.ready()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "whisperReady": await whisper_ready(),
        "model": WHISPER_MODEL,
        "engine": "mlx-whisper",
        "sampleRate": SAMPLE_RATE,
    }


@app.get("/api/system/capabilities")
async def capabilities() -> dict[str, Any]:
    vad = StreamingVAD(MODEL_PATH)
    return {
        "stt": await whisper_ready(),
        "vad": vad.engine_name,
        "sqlite": True,
        "localAudio": True,
        "d1": False,
        "r2": False,
        "model": WHISPER_MODEL,
        "engine": "mlx-whisper",
        "recordingCount": storage.count(),
        "retentionDays": RETENTION_DAYS,
        "maxRecordingSeconds": MAX_RECORDING_SECONDS,
        "legalPrompt": bool(legal_prompt),
        "legalCorrections": len(legal_corrections),
    }


@app.get("/api/recordings")
async def list_recordings() -> list[dict[str, Any]]:
    return storage.list()


@app.get("/api/recordings/{recording_id}")
async def get_recording(recording_id: str) -> dict[str, Any]:
    recording = storage.get(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return recording


@app.get("/api/recordings/{recording_id}/audio")
async def get_audio(recording_id: str) -> FileResponse:
    audio_path = storage.audio_path(recording_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="음성 파일을 찾을 수 없습니다.")
    return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)


@app.get("/api/recordings/{recording_id}/transcript")
async def get_transcript(recording_id: str) -> PlainTextResponse:
    recording = storage.get(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    lines = [
        recording["title"],
        f"기록 일시: {recording['createdAt']}",
        f"재생 시간: {recording['duration']:.1f}초",
        "",
    ]
    for segment in recording["segments"]:
        lines.append(f"[{segment['start']:.2f}] {segment['text']}")
    return PlainTextResponse("\n".join(lines), headers={
        "Content-Disposition": f'attachment; filename="{recording_id}_transcript.txt"'
    })


@app.patch("/api/recordings/{recording_id}/segments/{segment_index}")
async def review_segment(
    recording_id: str,
    segment_index: int,
    review: SegmentReview,
) -> dict[str, Any]:
    recording = storage.get(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    if segment_index < 0 or segment_index >= len(recording["segments"]):
        raise HTTPException(status_code=404, detail="자막 구간을 찾을 수 없습니다.")
    segment = recording["segments"][segment_index]
    segment["text"] = review.text.strip()
    segment["reviewRequired"] = False
    segment["reviewReasons"] = []
    segment["reviewedAt"] = datetime.now().isoformat(timespec="seconds")
    updated = storage.update_segments(recording_id, recording["segments"])
    if not updated:
        raise HTTPException(status_code=500, detail="자막을 저장하지 못했습니다.")
    return updated


@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str) -> dict[str, bool]:
    return {"deleted": storage.delete(recording_id)}


@app.delete("/api/recordings")
async def clear_recordings() -> dict[str, int]:
    return {"deleted": storage.clear()}


@app.websocket("/ws/stt")
async def stt_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    pcm_buffer = bytearray()
    speech_start_byte = 0
    segments: list[dict[str, Any]] = []
    waveform: list[float] = []
    vad = StreamingVAD(MODEL_PATH)
    started = False
    speech_observed = False
    maximum_rms = 0.0
    received_frames = 0
    interim_task: asyncio.Task[None] | None = None
    last_interim_byte = 0
    speech_ranges: list[tuple[int, int]] = []

    async def emit_interim(snapshot: bytes) -> None:
        try:
            text, _ = await transcriber.interim(snapshot)
            if text:
                await websocket.send_json({"type": "interim", "text": text})
        except Exception:
            return

    async def finalize_speech(end_byte: int) -> None:
        nonlocal speech_start_byte, interim_task
        segment_pcm = bytes(pcm_buffer[speech_start_byte:end_byte])
        if len(segment_pcm) < SAMPLE_RATE:
            return
        if interim_task and not interim_task.done():
            await interim_task
        interim_task = None
        start_seconds = speech_start_byte / 2 / SAMPLE_RATE
        end_seconds = end_byte / 2 / SAMPLE_RATE
        await websocket.send_json({
            "type": "interim",
            "text": "음성을 분석하고 있습니다…",
        })
        text, latency_ms = await transcriber.interim(segment_pcm)
        if text:
            segment = {
                "text": text,
                "start": start_seconds,
                "end": end_seconds,
            }
            segments.append(segment)
            await websocket.send_json({"type": "final", **segment})
        await websocket.send_json({
            "type": "metrics",
            "latencyMs": round(latency_ms),
            "realtimeFactor": round(
                latency_ms / max(1, (end_seconds - start_seconds) * 1000), 3
            ),
            "model": WHISPER_MODEL,
            "vad": vad.engine_name,
        })

    try:
        while True:
            message = await websocket.receive()
            if message.get("text") is not None:
                payload = json.loads(message["text"])
                message_type = payload.get("type")
                if message_type == "start":
                    if payload.get("format") != "pcm_s16le":
                        await websocket.send_json({
                            "type": "error",
                            "message": "지원하지 않는 오디오 형식입니다. pcm_s16le가 필요합니다.",
                        })
                        continue
                    if int(payload.get("sampleRate", 0)) != SAMPLE_RATE:
                        await websocket.send_json({
                            "type": "error",
                            "message": "지원하지 않는 샘플레이트입니다. 16000Hz가 필요합니다.",
                        })
                        continue
                    session_id = str(payload.get("sessionId") or session_id)
                    started = True
                    await websocket.send_json({
                        "type": "ready",
                        "sessionId": session_id,
                        "model": WHISPER_MODEL,
                        "vad": vad.engine_name,
                        "whisperReady": await whisper_ready(),
                    })
                elif message_type == "stop":
                    waveform = [
                        float(value) for value in payload.get("waveform", [])
                        if isinstance(value, (int, float))
                    ][-180:]
                    if vad.triggered:
                        speech_ranges.append((speech_start_byte, len(pcm_buffer)))
                        await websocket.send_json({"type": "speech_end"})
                    if interim_task and not interim_task.done():
                        await interim_task
                    interim_task = None
                    duration = len(pcm_buffer) / 2 / SAMPLE_RATE
                    await websocket.send_json({
                        "type": "finalizing",
                        "current": 0,
                        "total": 1,
                    })

                    async def report_progress(current: int, total: int) -> None:
                        await websocket.send_json({
                            "type": "finalizing",
                            "current": current,
                            "total": total,
                        })

                    final_segments, diagnostics = (
                        await transcriber.complete_session(
                            bytes(pcm_buffer),
                            speech_ranges=speech_ranges,
                            progress=report_progress,
                        )
                    )
                    segments = final_segments
                    client_samples = int(payload.get("emittedSamples", 0))
                    server_samples = len(pcm_buffer) // 2
                    diagnostics.update({
                        "receivedFrames": received_frames,
                        "serverSamples": server_samples,
                        "clientSamples": client_samples,
                        "missingSamples": max(0, client_samples - server_samples),
                        "workletFlushed": bool(payload.get("workletFlushed")),
                        "vadDetected": speech_observed,
                        "maximumRms": round(maximum_rms, 6),
                    })
                    await websocket.send_json({
                        "type": "final_replace",
                        "segments": segments,
                        "diagnostics": diagnostics,
                    })
                    await websocket.send_json({
                        "type": "metrics",
                        **diagnostics,
                        "vad": vad.engine_name,
                    })
                    recording = {
                        "id": session_id,
                        "title": payload.get("title") or f"음성 명령 기록 {datetime.now():%H%M%S}",
                        "createdAt": datetime.now().isoformat(timespec="seconds"),
                        "duration": duration,
                        "segments": segments or [{
                            "text": "인식된 텍스트가 없습니다.",
                            "start": 0,
                            "end": duration,
                        }],
                        "waveform": waveform,
                        "hasAudio": True,
                        "diagnostics": diagnostics,
                    }
                    storage.save(recording, pcm_to_wav(bytes(pcm_buffer)))
                    await websocket.send_json({"type": "saved", "recording": recording})
                    break
            elif message.get("bytes") is not None and started:
                chunk = message["bytes"]
                if len(chunk) % 2:
                    await websocket.send_json({
                        "type": "error",
                        "message": "손상된 PCM 프레임을 수신했습니다.",
                    })
                    continue
                if len(pcm_buffer) + len(chunk) > MAX_RECORDING_SECONDS * SAMPLE_RATE * 2:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"최대 녹음 시간 {MAX_RECORDING_SECONDS // 60}분을 초과했습니다.",
                    })
                    break
                pcm_buffer.extend(chunk)
                received_frames += 1
                samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                if samples.size:
                    maximum_rms = max(
                        maximum_rms,
                        float(np.sqrt(np.mean(np.square(samples)))),
                    )
                for event_type, probability in vad.accept(samples):
                    if event_type == "speech_start":
                        speech_observed = True
                        speech_start_byte = max(
                            0,
                            len(pcm_buffer) - BYTES_PER_SECOND,
                        )
                    await websocket.send_json({
                        "type": event_type,
                        "probability": round(probability, 4),
                    })
                    if event_type == "speech_end":
                        speech_ranges.append((speech_start_byte, len(pcm_buffer)))
                        await finalize_speech(len(pcm_buffer))
                if (
                    len(pcm_buffer) - last_interim_byte >= BYTES_PER_SECOND * 3
                    and (interim_task is None or interim_task.done())
                ):
                    snapshot_start = (
                        speech_start_byte
                        if vad.triggered
                        else max(0, len(pcm_buffer) - INTERIM_WINDOW_BYTES)
                    )
                    snapshot = bytes(pcm_buffer[snapshot_start:])
                    if vad.triggered or contains_possible_speech(snapshot):
                        last_interim_byte = len(pcm_buffer)
                        interim_task = asyncio.create_task(
                            emit_interim(snapshot)
                        )
    except WebSocketDisconnect:
        if interim_task and not interim_task.done():
            interim_task.cancel()
            with suppress(asyncio.CancelledError):
                await interim_task
        return
    except Exception as error:
        if interim_task and not interim_task.done():
            interim_task.cancel()
            with suppress(asyncio.CancelledError):
                await interim_task
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
        except Exception:
            return
