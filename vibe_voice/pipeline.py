"""Microphone loop: wake word, then one utterance, then a spoken reply."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from vibe_voice.audio import resample_linear
from vibe_voice.llm import ChatMessage, append_history
from vibe_voice.vad import UtteranceSegmenter, ms_to_samples


class MicPipeline:
    """Half-duplex state machine. Playback happens inside ``feed``, so the
    microphone loop does not capture the assistant's own voice.
    """

    def __init__(
        self,
        *,
        kws: object,
        classifier: object,
        segmenter: UtteranceSegmenter,
        engine: object,
        player: object,
        sample_rate: int,
        wake_suppress_samples: int,
        wake_timeout_samples: int,
        max_history_messages: int = 16,
    ) -> None:
        self.kws = kws
        self.classifier = classifier
        self.segmenter = segmenter
        self.engine = engine
        self.player = player
        self.sample_rate = sample_rate
        self.wake_suppress_samples = wake_suppress_samples
        self.wake_timeout_samples = wake_timeout_samples
        self.max_history_messages = max_history_messages
        self.state = "armed"
        self.history: list[ChatMessage] = []
        self._suppress = 0
        self._waited_without_speech = 0
        self._heard_speech = False

    def feed(self, samples: np.ndarray) -> list[str]:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        if self.state == "armed":
            hit = self.kws.accept(audio)
            if not hit:
                return []
            self._begin_listening()
            logs = [f"唤醒：{hit}"]
            logs.extend(self._listen(audio))
            return logs
        return self._listen(audio)

    def _begin_listening(self) -> None:
        self.state = "listening"
        self._suppress = self.wake_suppress_samples
        self._waited_without_speech = 0
        self._heard_speech = False
        self.segmenter.reset()
        reset = getattr(self.classifier, "reset", None)
        if callable(reset):
            reset()

    def _listen(self, audio: np.ndarray) -> list[str]:
        incoming = int(audio.size)
        if self._suppress > 0:
            drop = min(self._suppress, audio.size)
            self._suppress -= drop
            audio = audio[drop:]
        if not self._heard_speech:
            self._waited_without_speech += incoming
        logs: list[str] = []
        utterances: list[np.ndarray] = []
        if audio.size:
            for frame, is_speech in self.classifier.classify(audio):
                if is_speech:
                    self._heard_speech = True
                utterances.extend(self.segmenter.push(frame, bool(is_speech)))
        if utterances:
            for utterance in utterances:
                logs.extend(self._respond(utterance))
            return logs
        if (
            not self._heard_speech
            and self.wake_timeout_samples > 0
            and self._waited_without_speech >= self.wake_timeout_samples
        ):
            logs.append("没有听到后续语音，继续等待唤醒")
            self._arm()
        return logs

    def _respond(self, utterance: np.ndarray) -> list[str]:
        logs: list[str] = []
        try:
            text = self.engine.transcribe(utterance, self.sample_rate).strip()
            if not text:
                logs.append("没有识别到内容，继续等待唤醒")
                return logs
            logs.append(f"你：{text}")
            answer = self.engine.reply(text, list(self.history))
            logs.append(f"小助：{answer}")
            append_history(self.history, text, answer, self.max_history_messages)
            synthesized = self.engine.synthesize(answer)
            if synthesized is None:
                logs.append("未加载中文 TTS 模型，只显示文字。运行 vibe-voice download 可下载语音。")
            else:
                samples, rate = synthesized
                self.player.play(samples, rate)
        except Exception as exc:  # pipeline keeps listening after a bad turn
            logs.append(f"出错：{exc}")
        finally:
            self._arm()
        return logs

    def _arm(self) -> None:
        self.state = "armed"
        self._suppress = 0
        self._heard_speech = False
        self.segmenter.reset()
        reset = getattr(self.classifier, "reset", None)
        if callable(reset):
            reset()


def microphone_chunks(
    capture_rate: int,
    target_rate: int,
    block_ms: int,
    device: int | None,
) -> Iterator[np.ndarray]:
    import sounddevice as sd

    block = max(1, int(capture_rate * block_ms / 1000))
    with sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=capture_rate,
        blocksize=block,
        device=device,
    ) as stream:
        while True:
            data, _overflowed = stream.read(block)
            audio = np.asarray(data, dtype=np.float32).reshape(-1)
            yield resample_linear(audio, capture_rate, target_rate)


def segmenter_from_settings(settings: object) -> UtteranceSegmenter:
    return UtteranceSegmenter(
        sample_rate=16000,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
        max_speech_ms=settings.vad_max_speech_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )


def wake_sample_counts(settings: object) -> tuple[int, int]:
    return (
        ms_to_samples(settings.wake_suppress_ms, 16000),
        ms_to_samples(settings.wake_timeout_ms, 16000),
    )
