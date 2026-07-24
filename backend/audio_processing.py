from __future__ import annotations

import math

import numpy as np


CAPTURE_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CAPTURE_BYTES_PER_SECOND = CAPTURE_SAMPLE_RATE * SAMPLE_WIDTH_BYTES
WHISPER_BYTES_PER_SECOND = WHISPER_SAMPLE_RATE * SAMPLE_WIDTH_BYTES


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm(samples: np.ndarray) -> bytes:
    normalized = np.nan_to_num(
        np.asarray(samples, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    clipped = np.clip(normalized, -1.0, 1.0)
    integers = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0)
    return integers.astype("<i2").tobytes()


def resample_float32(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or audio.size == 0:
        return audio.copy()
    try:
        import torch
        from torchaudio.functional import resample

        tensor = torch.from_numpy(audio.copy()).reshape(1, -1)
        output = resample(tensor, source_rate, target_rate)
        return output.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        output_length = max(1, round(audio.size * target_rate / source_rate))
        source_positions = np.arange(audio.size, dtype=np.float64)
        target_positions = np.linspace(
            0,
            max(0, audio.size - 1),
            output_length,
            dtype=np.float64,
        )
        return np.interp(target_positions, source_positions, audio).astype(
            np.float32
        )


def resample_pcm(
    pcm_bytes: bytes,
    source_rate: int,
    target_rate: int,
) -> bytes:
    return float32_to_pcm(
        resample_float32(
            pcm_to_float32(pcm_bytes),
            source_rate,
            target_rate,
        )
    )


class Streaming48kTo16kResampler:
    def __init__(self) -> None:
        self.pending = np.empty(0, dtype=np.int16)
        self.input_samples = 0
        self.output_samples = 0

    def accept(self, pcm_bytes: bytes) -> tuple[bytes, np.ndarray]:
        if len(pcm_bytes) % SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM 데이터 길이는 2바이트 단위여야 합니다.")
        incoming = np.frombuffer(pcm_bytes, dtype="<i2")
        self.input_samples += incoming.size
        if self.pending.size:
            incoming = np.concatenate((self.pending, incoming))
        usable = incoming.size - incoming.size % 3
        if usable <= 0:
            self.pending = incoming.copy()
            return b"", np.empty(0, dtype=np.float32)
        grouped = incoming[:usable].astype(np.float32).reshape(-1, 3)
        samples = grouped.mean(axis=1) / 32768.0
        self.pending = incoming[usable:].copy()
        self.output_samples += samples.size
        return float32_to_pcm(samples), samples

    def flush(self) -> tuple[bytes, np.ndarray]:
        if not self.pending.size:
            return b"", np.empty(0, dtype=np.float32)
        padded = np.pad(
            self.pending.astype(np.float32),
            (0, 3 - self.pending.size),
            mode="edge",
        )
        samples = np.array([padded.mean() / 32768.0], dtype=np.float32)
        self.pending = np.empty(0, dtype=np.int16)
        self.output_samples += 1
        return float32_to_pcm(samples), samples

    @property
    def duration_difference_seconds(self) -> float:
        source_duration = self.input_samples / CAPTURE_SAMPLE_RATE
        output_duration = self.output_samples / WHISPER_SAMPLE_RATE
        difference = output_duration - source_duration
        return difference if math.isfinite(difference) else 0.0
