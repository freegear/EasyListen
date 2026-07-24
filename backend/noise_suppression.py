from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from audio_processing import (
    CAPTURE_SAMPLE_RATE,
    WHISPER_SAMPLE_RATE,
    float32_to_pcm,
    pcm_to_float32,
    resample_float32,
    resample_pcm,
)


@dataclass
class EnhancementResult:
    whisper_pcm: bytes
    enhanced_pcm: bytes | None
    diagnostics: dict[str, Any]


class DeepFilterNoiseSuppressor:
    def __init__(
        self,
        enabled: bool,
        model_name: str = "DeepFilterNet3",
        attenuation_limit_db: float = 18.0,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.attenuation_limit_db = attenuation_limit_db
        self.model: Any = None
        self.df_state: Any = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    def dependencies_available(self) -> bool:
        return all(
            importlib.util.find_spec(module_name) is not None
            for module_name in ("torch", "torchaudio", "df", "libdf")
        )

    def _load(self) -> None:
        if self.model is not None or not self.enabled:
            return
        if not self.dependencies_available():
            raise RuntimeError("DeepFilterNet3 의존성이 설치되지 않았습니다.")
        from df.enhance import init_df

        loaded = init_df(
            model_base_dir=self.model_name,
            log_level="ERROR",
            log_file=None,
        )
        self.model, self.df_state = loaded[:2]
        if int(self.df_state.sr()) != CAPTURE_SAMPLE_RATE:
            raise RuntimeError("DeepFilterNet3 모델 샘플레이트가 48kHz가 아닙니다.")

    async def initialize(self) -> bool:
        if not self.enabled:
            return False
        try:
            await asyncio.to_thread(self._initialize_sync)
            return True
        except Exception as error:
            self.load_error = str(error)
            return False

    def _initialize_sync(self) -> None:
        with self.lock:
            self._load()

    def _enhance_sync(self, source_pcm: bytes) -> EnhancementResult:
        started = time.perf_counter()
        source_audio = pcm_to_float32(source_pcm)
        if source_audio.size == 0:
            return EnhancementResult(
                whisper_pcm=b"",
                enhanced_pcm=b"",
                diagnostics=self._diagnostics(0.0, 0.0, False),
            )
        with self.lock:
            self._load()
            import torch
            from df.enhance import enhance

            tensor = torch.from_numpy(source_audio.copy()).reshape(1, -1)
            enhanced = enhance(
                self.model,
                self.df_state,
                tensor,
                pad=True,
                atten_lim_db=self.attenuation_limit_db,
            )
            enhanced_audio = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
        if enhanced_audio.size != source_audio.size:
            raise RuntimeError("DeepFilterNet3 출력 길이가 원본과 다릅니다.")
        if not np.isfinite(enhanced_audio).all():
            raise RuntimeError("DeepFilterNet3 출력에 비정상 숫자가 포함됐습니다.")
        source_rms = float(np.sqrt(np.mean(np.square(source_audio))))
        enhanced_rms = float(np.sqrt(np.mean(np.square(enhanced_audio))))
        if source_rms >= 0.001 and enhanced_rms < source_rms * 0.02:
            raise RuntimeError("DeepFilterNet3 출력 음량이 비정상적으로 작습니다.")
        enhanced_pcm = float32_to_pcm(enhanced_audio)
        whisper_audio = resample_float32(
            enhanced_audio,
            CAPTURE_SAMPLE_RATE,
            WHISPER_SAMPLE_RATE,
        )
        elapsed = time.perf_counter() - started
        duration = source_audio.size / CAPTURE_SAMPLE_RATE
        diagnostics = self._diagnostics(elapsed, duration, False)
        diagnostics.update({
            "noiseSuppressionInputRms": round(source_rms, 6),
            "noiseSuppressionOutputRms": round(enhanced_rms, 6),
        })
        return EnhancementResult(
            whisper_pcm=float32_to_pcm(whisper_audio),
            enhanced_pcm=enhanced_pcm,
            diagnostics=diagnostics,
        )

    async def prepare(self, source_pcm: bytes) -> EnhancementResult:
        if not self.enabled:
            return self._fallback(source_pcm, False, None)
        try:
            return await asyncio.to_thread(self._enhance_sync, source_pcm)
        except Exception as error:
            self.load_error = str(error)
            return self._fallback(source_pcm, True, str(error))

    def _fallback(
        self,
        source_pcm: bytes,
        attempted: bool,
        error: str | None,
    ) -> EnhancementResult:
        started = time.perf_counter()
        whisper_pcm = resample_pcm(
            source_pcm,
            CAPTURE_SAMPLE_RATE,
            WHISPER_SAMPLE_RATE,
        )
        elapsed = time.perf_counter() - started
        duration = len(source_pcm) / 2 / CAPTURE_SAMPLE_RATE
        diagnostics = self._diagnostics(elapsed, duration, attempted)
        diagnostics["noiseSuppression"] = False
        if error:
            diagnostics["noiseSuppressionError"] = error[:300]
        return EnhancementResult(
            whisper_pcm=whisper_pcm,
            enhanced_pcm=None,
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        elapsed_seconds: float,
        duration_seconds: float,
        fallback: bool,
    ) -> dict[str, Any]:
        return {
            "noiseSuppression": self.enabled and not fallback,
            "noiseSuppressionModel": self.model_name,
            "noiseSuppressionLatencyMs": round(elapsed_seconds * 1000),
            "noiseSuppressionRealtimeFactor": round(
                elapsed_seconds / max(duration_seconds, 0.001),
                3,
            ),
            "noiseSuppressionFallback": fallback,
            "noiseSuppressionAttenuationLimitDb": self.attenuation_limit_db,
            "sourceSampleRate": CAPTURE_SAMPLE_RATE,
            "whisperSampleRate": WHISPER_SAMPLE_RATE,
        }
