import numpy as np

from vibe_voice.errors import LlmError
from vibe_voice.llm import ChatMessage
from vibe_voice.pipeline import MicPipeline
from vibe_voice.vad import UtteranceSegmenter


class ScriptedKws:
    def __init__(self, fire_on: set[int]):
        self.fire_on = fire_on
        self.calls = 0

    def accept(self, samples):
        self.calls += 1
        if self.calls in self.fire_on:
            return "你好小助"
        return None

    def reset(self):
        return None


class LevelClassifier:
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self.seen = 0

    def classify(self, samples):
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.seen += int(audio.size)
        speech = bool(audio.size) and float(np.max(np.abs(audio))) >= self.threshold
        return [(audio, speech)]

    def reset(self):
        return None


class FakeEngine:
    def __init__(self, with_audio: bool = True):
        self.with_audio = with_audio
        self.transcripts = []
        self.replies = []
        self.history_sizes = []

    def transcribe(self, samples, sample_rate):
        self.transcripts.append((sample_rate, int(samples.size)))
        return "今天天气怎么样"

    def reply(self, text, history):
        self.replies.append(text)
        self.history_sizes.append(len(history))
        return "今天晴天。"

    def synthesize(self, text):
        if not self.with_audio:
            return None
        return np.full(8, 0.1, dtype=np.float32), 16000


class BoomEngine(FakeEngine):
    def reply(self, text, history):
        raise LlmError("连接失败")


class ListPlayer:
    def __init__(self):
        self.played = []

    def play(self, samples, sample_rate):
        self.played.append((sample_rate, int(np.asarray(samples).size)))


def _pipeline(engine, *, fire_on=(1,), timeout=10_000, suppress=0):
    player = ListPlayer()
    pipeline = MicPipeline(
        kws=ScriptedKws(set(fire_on)),
        classifier=LevelClassifier(),
        segmenter=UtteranceSegmenter(
            sample_rate=1000,
            min_speech_ms=100,
            min_silence_ms=50,
            max_speech_ms=5000,
            speech_pad_ms=0,
        ),
        engine=engine,
        player=player,
        sample_rate=1000,
        wake_suppress_samples=suppress,
        wake_timeout_samples=timeout,
    )
    return pipeline, player


def test_wake_then_speech_then_silence_replies_once():
    engine = FakeEngine()
    pipeline, player = _pipeline(engine)
    assert pipeline.feed(np.zeros(10)) == ["唤醒：你好小助"]
    assert pipeline.state == "listening"
    assert pipeline.feed(np.full(100, 0.5)) == []
    logs = pipeline.feed(np.zeros(50))
    assert logs[0] == "你：今天天气怎么样"
    assert logs[1] == "小助：今天晴天。"
    assert pipeline.state == "armed"
    assert player.played == [(16000, 8)]
    assert engine.history_sizes == [0]


def test_second_turn_sees_history_and_missing_tts_is_text_only():
    engine = FakeEngine(with_audio=False)
    pipeline, player = _pipeline(engine, fire_on=(1, 2))
    pipeline.feed(np.zeros(5))
    pipeline.feed(np.full(100, 0.5))
    first = pipeline.feed(np.zeros(50))
    assert any("未加载中文 TTS" in line for line in first)
    assert player.played == []
    pipeline.feed(np.zeros(5))
    pipeline.feed(np.full(100, 0.5))
    pipeline.feed(np.zeros(50))
    assert engine.history_sizes == [0, 2]
    assert len(pipeline.history) == 4
    assert pipeline.history[0] == ChatMessage("user", "今天天气怎么样")
    assert pipeline.history[-1] == ChatMessage("assistant", "今天晴天。")


def test_wake_timeout_returns_to_armed_without_calling_the_model():
    engine = FakeEngine()
    pipeline, player = _pipeline(engine, timeout=30)
    logs = pipeline.feed(np.zeros(10))
    assert logs == ["唤醒：你好小助"]
    logs = pipeline.feed(np.zeros(25))
    assert logs == ["没有听到后续语音，继续等待唤醒"]
    assert pipeline.state == "armed"
    assert engine.replies == []
    assert player.played == []


def test_llm_error_is_reported_and_listening_resumes():
    pipeline, _player = _pipeline(BoomEngine())
    pipeline.feed(np.zeros(5))
    pipeline.feed(np.full(100, 0.5))
    logs = pipeline.feed(np.zeros(50))
    assert any(line.startswith("出错：") for line in logs)
    assert pipeline.state == "armed"
