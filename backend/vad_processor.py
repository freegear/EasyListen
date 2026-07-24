from __future__ import annotations

from pathlib import Path

import numpy as np


class SileroModel:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, samples: np.ndarray) -> float:
        audio = np.concatenate((self.context, samples.reshape(1, 512)), axis=1)
        output, self.state = self.session.run(
            None,
            {
                "input": audio.astype(np.float32),
                "state": self.state,
                "sr": np.array(16000, dtype=np.int64),
            },
        )
        self.context = audio[:, -64:]
        return float(output.squeeze())


class StreamingVAD:
    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.4,
        min_silence_ms: int = 800,
        min_speech_ms: int = 150,
    ) -> None:
        self.threshold = threshold
        self.negative_threshold = max(0.01, threshold - 0.15)
        self.min_silence_chunks = max(1, round(min_silence_ms / 32))
        self.min_speech_chunks = max(1, round(min_speech_ms / 32))
        self.model: SileroModel | None = None
        self.engine_name = "energy-fallback"
        try:
            if model_path.exists():
                self.model = SileroModel(model_path)
                self.engine_name = "silero-vad-onnx"
        except Exception:
            self.model = None
        self.reset()

    def reset(self) -> None:
        if self.model:
            self.model.reset()
        self.triggered = False
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.pending = np.empty(0, dtype=np.float32)

    def _probability(self, samples: np.ndarray) -> float:
        if self.model:
            return self.model.probability(samples)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return min(1.0, max(0.0, (rms - 0.004) / 0.026))

    def accept(self, samples: np.ndarray) -> list[tuple[str, float]]:
        self.pending = np.concatenate((self.pending, samples.astype(np.float32)))
        events: list[tuple[str, float]] = []
        while self.pending.size >= 512:
            chunk = self.pending[:512]
            self.pending = self.pending[512:]
            probability = self._probability(chunk)
            if probability >= self.threshold:
                self.speech_chunks += 1
                self.silence_chunks = 0
                if not self.triggered and self.speech_chunks >= self.min_speech_chunks:
                    self.triggered = True
                    events.append(("speech_start", probability))
            elif self.triggered and probability < self.negative_threshold:
                self.silence_chunks += 1
                if self.silence_chunks >= self.min_silence_chunks:
                    self.triggered = False
                    self.speech_chunks = 0
                    self.silence_chunks = 0
                    events.append(("speech_end", probability))
            elif not self.triggered:
                self.speech_chunks = 0
        return events
