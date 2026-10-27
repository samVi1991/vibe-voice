"""sherpa-onnx keyword spotting, SenseVoice, and Chinese VITS."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from vibe_voice.audio import resample_linear
from vibe_voice.errors import ModelNotReady
from vibe_voice.keywords import Keyword, match_phrase

_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")


def clean_sensevoice_text(text: str) -> str:
    """Drop SenseVoice language / emotion / event tags."""

    return _SENSEVOICE_TAG.sub("", text or "").strip()


class SherpaKeywordSpotter:
    """Streaming keyword spotter. One ``accept`` call may return a phrase."""

    def __init__(self, spotter: object, entries: list[Keyword], sample_rate: int = 16000) -> None:
        self._kws = spotter
        self._entries = entries
        self.sample_rate = sample_rate
        self._stream = spotter.create_stream()

    def accept(self, samples: np.ndarray) -> str | None:
        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None
        if self.sample_rate != 16000:
            audio = np.ascontiguousarray(resample_linear(audio, self.sample_rate, 16000))
        self._stream.accept_waveform(16000, audio)
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
        text = _keyword_text(self._kws.get_result(self._stream))
        if not text:
            return None
        self._kws.reset_stream(self._stream)
        return match_phrase(text, self._entries) or text

    def reset(self) -> None:
        self._kws.reset_stream(self._stream)


class SenseVoiceTranscriber:
    def __init__(self, recognizer: object) -> None:
        self._recognizer = recognizer

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = resample_linear(samples, sample_rate, 16000)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self._recognizer.decode_stream(stream)
        return clean_sensevoice_text(getattr(stream.result, "text", ""))


class ChineseTts:
    """sherpa-onnx VITS Chinese voices. ``sid`` selects the speaker."""

    def __init__(self, tts: object, speaker_id: int, speed: float) -> None:
        self._tts = tts
        self.speaker_id = speaker_id
        self.speed = speed

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        spoken = text.strip()
        if not spoken:
            return None
        audio = self._tts.generate(spoken, sid=self.speaker_id, speed=self.speed)
        samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        return samples, int(audio.sample_rate)


def create_keyword_spotter(
    *,
    tokens: Path,
    encoder: Path,
    decoder: Path,
    joiner: Path,
    keywords_file: Path,
    num_threads: int,
    keywords_score: float,
    keywords_threshold: float,
    num_trailing_blanks: int,
    entries: list[Keyword],
) -> SherpaKeywordSpotter:
    _require(tokens, encoder, decoder, joiner, keywords_file)
    import sherpa_onnx

    spotter = sherpa_onnx.KeywordSpotter(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        keywords_file=str(keywords_file),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        max_active_paths=4,
        keywords_score=float(keywords_score),
        keywords_threshold=float(keywords_threshold),
        num_trailing_blanks=int(num_trailing_blanks),
        provider="cpu",
    )
    return SherpaKeywordSpotter(spotter, entries)


def create_sensevoice(
    *,
    model: Path,
    tokens: Path,
    num_threads: int,
    language: str,
    use_itn: bool,
) -> SenseVoiceTranscriber:
    _require(model, tokens)
    import sherpa_onnx

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
        provider="cpu",
        language=language,
        use_itn=use_itn,
    )
    return SenseVoiceTranscriber(recognizer)


def create_chinese_tts(
    *,
    model: Path,
    lexicon: Path,
    tokens: Path,
    rule_fsts: list[Path],
    num_threads: int,
    speaker_id: int,
    speed: float,
    max_sentences: int,
) -> ChineseTts:
    _require(model, lexicon, tokens)
    import sherpa_onnx

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model),
        lexicon=str(lexicon),
        tokens=str(tokens),
    )
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits,
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=",".join(str(path) for path in rule_fsts),
        max_num_sentences=max_sentences,
    )
    validate = getattr(config, "validate", None)
    if callable(validate) and not validate():
        raise ModelNotReady("中文 TTS 配置无效，请检查模型、lexicon 和 tokens")
    return ChineseTts(sherpa_onnx.OfflineTts(config), speaker_id, speed)


def _require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(missing)
        raise ModelNotReady(f"缺少模型文件：\n{joined}\n先运行：vibe-voice download")


def _keyword_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    for attr in ("keyword", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = str(result).strip()
    return "" if text in {"None", ""} else text
