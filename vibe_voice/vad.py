"""End-of-utterance segmentation, plus a Silero frame classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vibe_voice.errors import ModelNotReady


def ms_to_samples(duration_ms: int, sample_rate: int) -> int:
    return max(0, int(sample_rate * duration_ms / 1000))


class UtteranceSegmenter:
    """Turn speech / non-speech frames into one utterance per pause.

    The classifier (Silero, or a test double) decides whether each frame is
    speech. This class owns the end-of-utterance rules: a short pause stays
    inside the phrase, a long enough pause closes it, and a very short burst
    is dropped as noise.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        max_speech_ms: int = 15000,
        speech_pad_ms: int = 200,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        self.sample_rate = sample_rate
        self.min_speech_samples = ms_to_samples(min_speech_ms, sample_rate)
        self.min_silence_samples = ms_to_samples(min_silence_ms, sample_rate)
        self.max_speech_samples = max(1, ms_to_samples(max_speech_ms, sample_rate))
        self.pad_samples = ms_to_samples(speech_pad_ms, sample_rate)
        self._preroll = np.zeros(0, dtype=np.float32)
        self._parts: list[np.ndarray] = []
        self._silence = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._speech_run = 0

    def reset(self) -> None:
        self._preroll = np.zeros(0, dtype=np.float32)
        self._clear_utterance()

    def push(self, samples: np.ndarray, is_speech: bool) -> list[np.ndarray]:
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return []
        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                preroll = self._preroll
                self._preroll = np.zeros(0, dtype=np.float32)
                self._parts = [preroll, frame] if preroll.size else [frame]
                self._speech_run = int(frame.size)
                self._silence = np.zeros(0, dtype=np.float32)
                if self._speech_run >= self.max_speech_samples:
                    return self._emit()
            else:
                self._remember_preroll(frame)
            return []

        if is_speech:
            if self._silence.size:
                self._parts.append(self._silence)
                self._silence = np.zeros(0, dtype=np.float32)
            self._parts.append(frame)
            self._speech_run += int(frame.size)
            if self._speech_run >= self.max_speech_samples:
                return self._emit()
            return []

        self._silence = np.concatenate([self._silence, frame])
        if self._silence.size < self.min_silence_samples:
            return []
        if self._speech_run >= self.min_speech_samples:
            return self._emit()
        tail = self._silence
        self._clear_utterance()
        self._remember_preroll(tail)
        return []

    def flush(self) -> list[np.ndarray]:
        if self._in_speech and self._speech_run >= self.min_speech_samples and self._parts:
            return self._emit()
        self._clear_utterance()
        return []

    def _emit(self) -> list[np.ndarray]:
        parts = list(self._parts)
        if self._silence.size and self.pad_samples:
            parts.append(self._silence[: self.pad_samples])
        self._clear_utterance()
        utterance = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if utterance.size == 0:
            return []
        return [utterance]

    def _clear_utterance(self) -> None:
        self._parts = []
        self._silence = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._speech_run = 0

    def _remember_preroll(self, samples: np.ndarray) -> None:
        if self.pad_samples <= 0 or samples.size == 0:
            return
        combined = np.concatenate([self._preroll, samples])
        if combined.size > self.pad_samples:
            combined = combined[-self.pad_samples :]
        self._preroll = combined


class SileroSpeechClassifier:
    """Speech / non-speech labels from sherpa-onnx's Silero VAD.

    sherpa-onnx also segments internally. Those timeouts are kept very short
    so ``is_speech_detected()`` follows the acoustic decision. The
    :class:`UtteranceSegmenter` applies the pause that ends an utterance.
    Completed sherpa segments are discarded so its ring buffer does not fill.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        window_size: int = 512,
    ) -> None:
        if not Path(model_path).is_file():
            raise ModelNotReady(
                f"找不到 Silero VAD 模型：{model_path}。先运行：vibe-voice download"
            )
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.threshold = float(threshold)
        config.silero_vad.min_silence_duration = 0.03
        config.silero_vad.min_speech_duration = 0.03
        config.silero_vad.window_size = int(window_size)
        config.silero_vad.max_speech_duration = 60
        config.sample_rate = int(sample_rate)
        if hasattr(config, "num_threads"):
            config.num_threads = 1
        self.sample_rate = sample_rate
        self.window_size = int(config.silero_vad.window_size or window_size)
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        self._pending = np.zeros(0, dtype=np.float32)

    def classify(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._pending = np.concatenate([self._pending, incoming])
        labeled: list[tuple[np.ndarray, bool]] = []
        while self._pending.size >= self.window_size:
            window = np.ascontiguousarray(self._pending[: self.window_size])
            self._pending = self._pending[self.window_size :]
            self._vad.accept_waveform(window)
            while not self._vad.empty():
                self._vad.pop()
            labeled.append((window, bool(self._vad.is_speech_detected())))
        return labeled

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._vad.reset()
