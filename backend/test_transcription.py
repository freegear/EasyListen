from __future__ import annotations

import asyncio
import unittest

import numpy as np

from transcription import (
    WhisperTranscriber,
    _collapse_adjacent_repetitions,
    _is_character_repetition,
    apply_legal_corrections,
    contains_possible_speech,
)


class FakeTranscriber(WhisperTranscriber):
    def __init__(self) -> None:
        super().__init__(
            "http://unused",
            "test-model",
            "탄원서, 절절한 심정, 반영되었다",
            {"타난서": "탄원서"},
            asyncio.Lock(),
        )
        self.requests = 0

    async def request(
        self,
        pcm_bytes: bytes,
        verbose: bool = True,
        use_prompt: bool = False,
    ):
        self.requests += 1
        if self.requests == 1:
            return {
                "text": "이 타난서의 내용입니다.",
                "segments": [{
                    "text": "이 타난서의 내용입니다.",
                    "start": 20.0,
                    "end": 25.0,
                    "avg_logprob": -0.8,
                }],
            }, 100.0
        return {
            "text": "이 탄원서의 내용입니다.",
            "segments": [{
                "text": "이 탄원서의 내용입니다.",
                "start": 0.0,
                "end": 3.0,
                "avg_logprob": -0.2,
            }],
        }, 80.0


class TranscriptionTests(unittest.IsolatedAsyncioTestCase):
    def test_silence_detection(self) -> None:
        silence = np.zeros(16000, dtype="<i2").tobytes()
        speech = np.full(16000, 1000, dtype="<i2").tobytes()
        self.assertFalse(contains_possible_speech(silence))
        self.assertTrue(contains_possible_speech(speech))

    def test_legal_correction_tracks_changes(self) -> None:
        corrected, changes = apply_legal_corrections(
            "이 타난서의 내용입니다.",
            {"타난서": "탄원서"},
        )
        self.assertEqual(corrected, "이 탄원서의 내용입니다.")
        self.assertEqual(changes, [{"from": "타난서", "to": "탄원서"}])

    def test_character_repetition_detects_hallucination(self) -> None:
        self.assertTrue(_is_character_repetition("주주주주주주주주주주주주"))
        self.assertFalse(_is_character_repetition("재판부가 판결을 선고했습니다"))

    def test_adjacent_repetition_is_collapsed_and_flagged(self) -> None:
        segments = [
            {
                "text": "박수홍, 김다예.",
                "start": 0.0,
                "end": 1.0,
                "reviewRequired": False,
                "reviewReasons": [],
            },
            {
                "text": "박수홍, 김다예.",
                "start": 1.0,
                "end": 2.0,
                "reviewRequired": False,
                "reviewReasons": [],
            },
        ]
        collapsed = _collapse_adjacent_repetitions(segments)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["end"], 2.0)
        self.assertTrue(collapsed[0]["reviewRequired"])
        self.assertIn("repeated_text", collapsed[0]["reviewReasons"])

    async def test_complete_session_falls_back_when_vad_has_no_ranges(self) -> None:
        transcriber = FakeTranscriber()
        samples = np.full(16000, 1200, dtype="<i2").tobytes()
        segments, diagnostics = await transcriber.complete_session(
            samples,
            speech_ranges=[],
        )
        self.assertEqual(transcriber.requests, 1)
        self.assertEqual(len(segments), 1)
        self.assertEqual(diagnostics["speechRangeCount"], 1)
        self.assertTrue(diagnostics["vadFallback"])

    async def test_complete_session_skips_silence_without_vad_ranges(self) -> None:
        transcriber = FakeTranscriber()
        silence = np.zeros(16000, dtype="<i2").tobytes()
        segments, diagnostics = await transcriber.complete_session(
            silence,
            speech_ranges=[],
        )
        self.assertEqual(transcriber.requests, 0)
        self.assertEqual(segments, [])
        self.assertFalse(diagnostics["vadFallback"])

    async def test_complete_session_merges_overlap(self) -> None:
        transcriber = FakeTranscriber()
        samples = np.full(16000 * 27, 1200, dtype="<i2").tobytes()
        segments, diagnostics = await transcriber.complete_session(samples)
        self.assertEqual(transcriber.requests, 2)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "이 탄원서의 내용입니다.")
        self.assertTrue(segments[0]["reviewRequired"])
        self.assertEqual(diagnostics["deduplicatedSegments"], 1)
        self.assertEqual(diagnostics["windowCount"], 2)


if __name__ == "__main__":
    unittest.main()
