"""Wire settings to the conversation engine and the microphone pipeline."""

from __future__ import annotations

import atexit
import tempfile
import threading
from pathlib import Path

import numpy as np

from vibe_voice.config import Settings
from vibe_voice.errors import ModelNotReady
from vibe_voice.keywords import Keyword, load_keywords
from vibe_voice.llm import ChatMessage, OpenAICompatibleClient
from vibe_voice.speech import (
    ChineseTts,
    SherpaKeywordSpotter,
    create_chinese_tts,
    create_keyword_spotter,
    create_sensevoice,
)
from vibe_voice.vad import SileroSpeechClassifier


class LocalConversationEngine:
    """SenseVoice, an OpenAI-compatible chat model, and optional Chinese TTS."""

    def __init__(
        self,
        transcriber: object | None,
        llm: OpenAICompatibleClient,
        tts: ChineseTts | None,
        *,
        missing_stt: str = "",
    ) -> None:
        self._transcriber = transcriber
        self._llm = llm
        self._tts = tts
        self._missing_stt = missing_stt
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._transcriber is None:
            raise ModelNotReady(self._missing_stt or "未找到 SenseVoice 模型。运行：vibe-voice download")
        with self._lock:
            return self._transcriber.transcribe(samples, sample_rate)

    def reply(self, user_text: str, history: list[ChatMessage]) -> str:
        return self._llm.complete(user_text, history)

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        if self._tts is None:
            return None
        with self._lock:
            return self._tts.synthesize(text)


def make_llm(settings: Settings) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.temperature,
        system_prompt=settings.system_prompt,
        timeout_s=settings.openai_timeout_s,
        max_tokens=settings.openai_max_tokens,
    )


def stt_ready(settings: Settings) -> bool:
    return settings.sensevoice_model.is_file() and settings.sensevoice_tokens.is_file()


def kws_ready(settings: Settings) -> bool:
    return all(
        path.is_file()
        for path in (
            settings.kws_tokens,
            settings.kws_encoder,
            settings.kws_decoder,
            settings.kws_joiner,
        )
    )


def vad_ready(settings: Settings) -> bool:
    return settings.vad_model.is_file()


def tts_ready(settings: Settings) -> bool:
    return (
        settings.tts_model.is_file()
        and settings.tts_tokens.is_file()
        and settings.tts_lexicon.is_file()
    )


def tts_incomplete(settings: Settings) -> bool:
    present = [
        path.is_file()
        for path in (settings.tts_model, settings.tts_tokens, settings.tts_lexicon)
    ]
    return any(present) and not all(present)


def missing_required(settings: Settings) -> list[str]:
    required = {
        "关键词 tokens": settings.kws_tokens,
        "关键词 encoder": settings.kws_encoder,
        "关键词 decoder": settings.kws_decoder,
        "关键词 joiner": settings.kws_joiner,
        "SenseVoice 模型": settings.sensevoice_model,
        "SenseVoice tokens": settings.sensevoice_tokens,
        "Silero VAD": settings.vad_model,
    }
    return [f"{name}：{path}" for name, path in required.items() if not path.is_file()]


def build_conversation_engine(settings: Settings) -> LocalConversationEngine:
    llm = make_llm(settings)
    transcriber = None
    missing = ""
    if stt_ready(settings):
        transcriber = create_sensevoice(
            model=settings.sensevoice_model,
            tokens=settings.sensevoice_tokens,
            num_threads=settings.num_threads,
            language=settings.stt_language,
            use_itn=settings.stt_use_itn,
        )
    else:
        missing = "未找到 SenseVoice 模型。运行：vibe-voice download"
    tts = load_tts(settings)
    return LocalConversationEngine(transcriber, llm, tts, missing_stt=missing)


def load_tts(settings: Settings) -> ChineseTts | None:
    if tts_incomplete(settings):
        raise ModelNotReady(
            "中文 TTS 模型不完整，需要 model.onnx、lexicon.txt 和 tokens.txt。"
            "重新运行：vibe-voice download"
        )
    if not tts_ready(settings):
        return None
    return create_chinese_tts(
        model=settings.tts_model,
        lexicon=settings.tts_lexicon,
        tokens=settings.tts_tokens,
        rule_fsts=settings.tts_rule_fsts,
        num_threads=settings.num_threads,
        speaker_id=settings.tts_sid,
        speed=settings.tts_speed,
        max_sentences=settings.tts_max_sentences,
    )


def load_keyword_spotter(settings: Settings, entries: list[Keyword]) -> SherpaKeywordSpotter:
    if not kws_ready(settings):
        raise ModelNotReady("未找到关键词模型。运行：vibe-voice download")
    runtime_keywords = materialize_keywords(entries)
    return create_keyword_spotter(
        tokens=settings.kws_tokens,
        encoder=settings.kws_encoder,
        decoder=settings.kws_decoder,
        joiner=settings.kws_joiner,
        keywords_file=runtime_keywords,
        num_threads=settings.num_threads,
        keywords_score=settings.kws_score,
        keywords_threshold=settings.kws_threshold,
        num_trailing_blanks=settings.kws_num_trailing_blanks,
        entries=entries,
    )


def load_silero(settings: Settings) -> SileroSpeechClassifier:
    if not vad_ready(settings):
        raise ModelNotReady("未找到 Silero VAD 模型。运行：vibe-voice download")
    return SileroSpeechClassifier(
        settings.vad_model,
        sample_rate=16000,
        threshold=settings.vad_threshold,
    )


def load_keyword_entries(settings: Settings) -> list[Keyword]:
    return load_keywords(settings.keywords_file)


def materialize_keywords(entries: list[Keyword]) -> Path:
    """Write a comment-free keywords file for sherpa-onnx and delete it on exit."""

    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="vibe-keywords-",
        suffix=".txt",
        delete=False,
    )
    path = Path(handle.name)
    with handle:
        for entry in entries:
            handle.write(entry.to_sherpa_line() + "\n")
    atexit.register(lambda: path.unlink(missing_ok=True))
    return path
