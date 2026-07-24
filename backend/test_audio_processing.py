from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_processing import (
    CAPTURE_SAMPLE_RATE,
    Streaming48kTo16kResampler,
    WHISPER_SAMPLE_RATE,
    float32_to_pcm,
    pcm_to_float32,
    resample_pcm,
)
from noise_suppression import DeepFilterNoiseSuppressor
from storage import RecordingStorage
from transcription import pcm_to_wav


class FailingNoiseSuppressor(DeepFilterNoiseSuppressor):
    def _enhance_sync(self, source_pcm: bytes):
        raise RuntimeError("test enhancement failure")


class AudioProcessingTests(unittest.IsolatedAsyncioTestCase):
    def test_streaming_resampler_preserves_chunk_boundaries(self) -> None:
        source = np.arange(48003, dtype=np.int16)
        resampler = Streaming48kTo16kResampler()
        output_parts: list[bytes] = []
        for chunk in np.array_split(source, 17):
            output_pcm, _ = resampler.accept(chunk.astype("<i2").tobytes())
            output_parts.append(output_pcm)
        flushed_pcm, _ = resampler.flush()
        output_parts.append(flushed_pcm)
        output = np.frombuffer(b"".join(output_parts), dtype="<i2")
        self.assertEqual(output.size, 16001)
        self.assertAlmostEqual(
            resampler.duration_difference_seconds,
            0.0,
            places=6,
        )

    def test_batch_resampler_returns_expected_duration(self) -> None:
        seconds = 2
        source = np.sin(
            np.linspace(
                0,
                np.pi * 440 * 2 * seconds,
                CAPTURE_SAMPLE_RATE * seconds,
                endpoint=False,
            )
        ).astype(np.float32)
        output = pcm_to_float32(
            resample_pcm(
                float32_to_pcm(source),
                CAPTURE_SAMPLE_RATE,
                WHISPER_SAMPLE_RATE,
            )
        )
        self.assertEqual(output.size, WHISPER_SAMPLE_RATE * seconds)
        self.assertTrue(np.isfinite(output).all())

    async def test_enhancement_failure_falls_back_to_original(self) -> None:
        suppressor = FailingNoiseSuppressor(enabled=True)
        source = np.full(CAPTURE_SAMPLE_RATE, 0.1, dtype=np.float32)
        result = await suppressor.prepare(float32_to_pcm(source))
        self.assertIsNone(result.enhanced_pcm)
        self.assertEqual(
            len(result.whisper_pcm),
            WHISPER_SAMPLE_RATE * 2,
        )
        self.assertTrue(result.diagnostics["noiseSuppressionFallback"])
        self.assertFalse(result.diagnostics["noiseSuppression"])

    def test_storage_preserves_original_and_enhanced_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            storage = RecordingStorage(root / "test.sqlite3", root / "audio")
            source_pcm = np.zeros(CAPTURE_SAMPLE_RATE, dtype="<i2").tobytes()
            recording = {
                "id": "recording-id",
                "title": "테스트",
                "createdAt": "2026-07-25T00:00:00",
                "duration": 1.0,
                "segments": [],
                "waveform": [],
                "diagnostics": {},
            }
            saved = storage.save(
                recording,
                pcm_to_wav(source_pcm, CAPTURE_SAMPLE_RATE),
                pcm_to_wav(source_pcm, CAPTURE_SAMPLE_RATE),
            )
            self.assertTrue(saved["hasEnhancedAudio"])
            self.assertIsNotNone(storage.audio_path("recording-id"))
            self.assertIsNotNone(
                storage.audio_path("recording-id", enhanced=True)
            )
            self.assertTrue(storage.delete("recording-id"))
            self.assertIsNone(storage.audio_path("recording-id"))


if __name__ == "__main__":
    unittest.main()
