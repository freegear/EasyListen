from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import math
import os
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np


SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
LOW_CONFIDENCE_LOGPROB = float(os.getenv("LOW_CONFIDENCE_LOGPROB", "-0.65"))
NO_SPEECH_REVIEW_THRESHOLD = float(
    os.getenv("NO_SPEECH_REVIEW_THRESHOLD", "0.6")
)
COMPRESSION_RATIO_REVIEW_THRESHOLD = float(
    os.getenv("COMPRESSION_RATIO_REVIEW_THRESHOLD", "2.4")
)


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def load_legal_context(root_dir: Path) -> tuple[str, dict[str, str]]:
    prompt_path = Path(
        os.getenv("LEGAL_PROMPT_PATH", root_dir / "backend" / "legal_prompt.txt")
    )
    corrections_path = Path(
        os.getenv(
            "LEGAL_CORRECTIONS_PATH",
            root_dir / "backend" / "legal_corrections.json",
        )
    )
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
    corrections = (
        json.loads(corrections_path.read_text(encoding="utf-8"))
        if corrections_path.exists()
        else {}
    )
    return prompt, {
        str(source): str(target)
        for source, target in corrections.items()
        if source and target
    }


def contains_possible_speech(pcm_bytes: bytes) -> bool:
    if not pcm_bytes:
        return False
    samples = pcm_to_float32(pcm_bytes)
    if not samples.size:
        return False
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    return rms >= 0.0015 or peak >= 0.01


def apply_legal_corrections(
    text: str,
    corrections: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    corrected = text
    changes: list[dict[str, str]] = []
    for source, target in sorted(
        corrections.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source not in corrected:
            continue
        corrected = corrected.replace(source, target)
        changes.append({"from": source, "to": target})
    return corrected.strip(), changes


def _normalized_text(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        _normalized_text(left),
        _normalized_text(right),
    ).ratio()


def _is_character_repetition(text: str) -> bool:
    normalized = _normalized_text(text)
    if len(normalized) < 8:
        return False
    most_common = max(normalized.count(character) for character in set(normalized))
    return most_common / len(normalized) >= 0.55


def _append_reason(segment: dict[str, Any], reason: str) -> None:
    reasons = segment.setdefault("reviewReasons", [])
    if reason not in reasons:
        reasons.append(reason)
    segment["reviewRequired"] = True


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _collapse_adjacent_repetitions(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for segment in segments:
        if not collapsed:
            collapsed.append(segment)
            continue
        previous = collapsed[-1]
        same_text = (
            _normalized_text(previous["text"])
            and _normalized_text(previous["text"]) == _normalized_text(segment["text"])
        )
        gap = segment["start"] - previous["end"]
        if same_text and gap <= 1.25:
            previous["end"] = max(previous["end"], segment["end"])
            _append_reason(previous, "repeated_text")
            continue
        collapsed.append(segment)
    return collapsed


def _normalize_speech_ranges(
    speech_ranges: list[tuple[int, int]],
    total_bytes: int,
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start, end in sorted(speech_ranges):
        start = max(0, min(total_bytes, start - (start % 2)))
        end = max(start, min(total_bytes, end - (end % 2)))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return normalized


def _merge_overlapping_segments(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(segments, key=lambda item: (item["start"], item["end"])):
        if not candidate["text"]:
            continue
        if not merged:
            merged.append(candidate)
            continue
        previous = merged[-1]
        overlap = min(previous["end"], candidate["end"]) - max(
            previous["start"], candidate["start"]
        )
        if overlap > 0 and _similarity(previous["text"], candidate["text"]) >= 0.58:
            previous_score = (
                previous["avgLogprob"]
                if previous.get("avgLogprob") is not None
                else -10.0
            )
            candidate_score = (
                candidate["avgLogprob"]
                if candidate.get("avgLogprob") is not None
                else -10.0
            )
            conflicting_raw_text = (
                _normalized_text(previous.get("rawText", ""))
                != _normalized_text(candidate.get("rawText", ""))
            )
            if candidate_score > previous_score:
                candidate["start"] = min(previous["start"], candidate["start"])
                candidate["end"] = max(previous["end"], candidate["end"])
                for reason in previous.get("reviewReasons", []):
                    _append_reason(candidate, reason)
                if conflicting_raw_text:
                    _append_reason(candidate, "overlap_conflict")
                    candidate["alternatives"] = list(dict.fromkeys([
                        previous.get("rawText", ""),
                        candidate.get("rawText", ""),
                    ]))
                merged[-1] = candidate
            else:
                previous["end"] = max(previous["end"], candidate["end"])
                for reason in candidate.get("reviewReasons", []):
                    _append_reason(previous, reason)
                if conflicting_raw_text:
                    _append_reason(previous, "overlap_conflict")
                    previous["alternatives"] = list(dict.fromkeys([
                        previous.get("rawText", ""),
                        candidate.get("rawText", ""),
                    ]))
            continue
        if candidate["start"] < previous["end"]:
            candidate["start"] = previous["end"]
        if candidate["end"] > candidate["start"]:
            merged.append(candidate)
    return merged


class WhisperTranscriber:
    def __init__(
        self,
        model_path: str | Path,
        model_name: str,
        prompt: str,
        corrections: dict[str, str],
        inference_lock: asyncio.Lock,
    ) -> None:
        self.model_path = str(model_path)
        self.model_name = model_name
        self.prompt = prompt
        self.corrections = corrections
        self.inference_lock = inference_lock

    async def ready(self) -> bool:
        model_dir = Path(self.model_path)
        return (
            importlib.util.find_spec("mlx_whisper") is not None
            and (model_dir / "config.json").is_file()
            and (model_dir / "weights.safetensors").is_file()
        )

    async def request(
        self,
        pcm_bytes: bytes,
        verbose: bool = True,
        use_prompt: bool = False,
        condition_on_previous_text: bool = False,
    ) -> tuple[dict[str, Any], float]:
        if not await self.ready():
            raise RuntimeError("MLX Whisper 모델이 준비되지 않았습니다.")

        def transcribe() -> dict[str, Any]:
            import mlx_whisper

            return mlx_whisper.transcribe(
                pcm_to_float32(pcm_bytes),
                path_or_hf_repo=self.model_path,
                language="ko",
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=self.prompt if use_prompt and self.prompt else None,
                compression_ratio_threshold=COMPRESSION_RATIO_REVIEW_THRESHOLD,
                logprob_threshold=-1.0,
                no_speech_threshold=NO_SPEECH_REVIEW_THRESHOLD,
                word_timestamps=False,
                verbose=None,
            )

        started = time.perf_counter()
        async with self.inference_lock:
            worker = asyncio.create_task(asyncio.to_thread(transcribe))
            try:
                payload = await worker
            except asyncio.CancelledError:
                await worker
                raise
        return payload, (time.perf_counter() - started) * 1000

    async def interim(self, pcm_bytes: bytes) -> tuple[str, float]:
        payload, latency_ms = await self.request(
            pcm_bytes,
            verbose=False,
            use_prompt=False,
        )
        return str(payload.get("text", "")).strip(), latency_ms

    async def complete_session(
        self,
        pcm_bytes: bytes,
        speech_ranges: list[tuple[int, int]] | None = None,
        progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        duration = len(pcm_bytes) / BYTES_PER_SECOND
        vad_fallback = not speech_ranges and contains_possible_speech(pcm_bytes)
        if vad_fallback:
            speech_ranges = [(0, len(pcm_bytes))]
        elif speech_ranges is None:
            speech_ranges = []
        normalized_ranges = _normalize_speech_ranges(
            speech_ranges,
            len(pcm_bytes),
        )
        if not normalized_ranges:
            return [], {
                "receivedDuration": round(duration, 3),
                "transcribedDuration": 0.0,
                "speechDuration": 0.0,
                "speechRangeCount": 0,
                "windowCount": 0,
                "deduplicatedSegments": 0,
                "lowConfidenceSegments": 0,
                "model": self.model_name,
                "engine": "mlx-whisper",
                "vadFallback": False,
            }

        speech_duration = sum(
            end - start for start, end in normalized_ranges
        ) / BYTES_PER_SECOND

        candidates: list[dict[str, Any]] = []
        total_latency_ms = 0.0
        transcribed_duration = 0.0
        if progress:
            await progress(1, 1)
        payload, latency_ms = await self.request(
            pcm_bytes,
            verbose=True,
            use_prompt=True,
            condition_on_previous_text=True,
        )
        total_latency_ms += latency_ms
        transcribed_duration = duration
        offset = 0.0
        chunk = pcm_bytes
        if payload:
            response_segments = payload.get("segments") or []
            response_text = str(payload.get("text", "")).strip()
            if not response_segments and response_text:
                response_segments = [{
                    "text": response_text,
                    "start": 0.0,
                    "end": len(chunk) / BYTES_PER_SECOND,
                }]
            for response_segment in response_segments:
                raw_text = str(response_segment.get("text", "")).strip()
                if not raw_text:
                    continue
                corrected_text, changes = apply_legal_corrections(
                    raw_text,
                    self.corrections,
                )
                avg_logprob = _finite_float(response_segment.get("avg_logprob"))
                compression_ratio = _finite_float(
                    response_segment.get("compression_ratio")
                )
                no_speech_probability = _finite_float(
                    response_segment.get("no_speech_prob")
                )
                segment_start = _finite_float(response_segment.get("start"))
                segment_end = _finite_float(response_segment.get("end"))
                start = offset + (segment_start if segment_start is not None else 0.0)
                end = offset + (
                    segment_end
                    if segment_end is not None
                    else len(chunk) / BYTES_PER_SECOND
                )
                candidate = {
                    "text": corrected_text,
                    "rawText": raw_text,
                    "start": round(max(0.0, start), 3),
                    "end": round(min(duration, max(start, end)), 3),
                    "avgLogprob": (
                        round(avg_logprob, 4) if avg_logprob is not None else None
                    ),
                    "compressionRatio": (
                        round(compression_ratio, 4)
                        if compression_ratio is not None
                        else None
                    ),
                    "noSpeechProbability": (
                        round(no_speech_probability, 4)
                        if no_speech_probability is not None
                        else None
                    ),
                    "confidence": None,
                    "reviewRequired": False,
                    "reviewReasons": [],
                    "corrections": changes,
                }
                if changes:
                    _append_reason(candidate, "dictionary_correction")
                if (
                    avg_logprob is not None
                    and avg_logprob < LOW_CONFIDENCE_LOGPROB
                ):
                    _append_reason(candidate, "low_log_probability")
                if (
                    compression_ratio is not None
                    and compression_ratio > COMPRESSION_RATIO_REVIEW_THRESHOLD
                ):
                    _append_reason(candidate, "high_compression_ratio")
                if (
                    no_speech_probability is not None
                    and no_speech_probability > NO_SPEECH_REVIEW_THRESHOLD
                ):
                    _append_reason(candidate, "possible_non_speech")
                if _is_character_repetition(raw_text):
                    _append_reason(candidate, "character_repetition")
                candidates.append(candidate)

        merged = _collapse_adjacent_repetitions(
            _merge_overlapping_segments(candidates)
        )
        low_confidence_count = sum(
            1 for segment in merged if segment.get("reviewRequired")
        )
        return merged, {
            "receivedDuration": round(duration, 3),
            "transcribedDuration": round(transcribed_duration, 3),
            "speechDuration": round(speech_duration, 3),
            "speechRangeCount": len(normalized_ranges),
            "windowCount": 1,
            "deduplicatedSegments": max(0, len(candidates) - len(merged)),
            "lowConfidenceSegments": low_confidence_count,
            "latencyMs": round(total_latency_ms),
            "realtimeFactor": round(
                total_latency_ms / max(1, speech_duration * 1000),
                3,
            ),
            "model": self.model_name,
            "engine": "mlx-whisper",
            "vadFallback": vad_fallback,
            "continuousDecoding": True,
        }
