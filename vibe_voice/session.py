"""Audio-in / audio-out session for the WebSocket server.

Two dialects share one state machine:

* ``vibe`` is the small JSON protocol used by a later open-xiaoai forwarder.
* ``xiaozhi`` speaks the conversation messages from xiaozhi-esp32 (hello, listen,
  abort, stt, llm, tts) plus binary audio. The device keeps wake word and AEC.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np

from vibe_voice.audio import (
    AudioFormatError,
    OpusCodec,
    decode_wav,
    float_to_pcm16,
    opus_available,
    pcm16le_to_float,
    resample_linear,
    unwrap_binary_frame,
    wrap_binary_frame,
)
from vibe_voice.errors import VibeVoiceError
from vibe_voice.llm import ChatMessage, append_history
from vibe_voice.vad import UtteranceSegmenter

_FORMATS = {"pcm", "wav", "opus"}
_XIAOZHI_LISTEN = {"start", "stop", "detect"}


@dataclass
class OutEvent:
    text: dict | None = None
    binary: bytes | None = None


class VoiceSession:
    def __init__(
        self,
        engine: object,
        *,
        segmenter: UtteranceSegmenter | None = None,
        classifier: object | None = None,
        sample_rate: int = 16000,
        max_history_messages: int = 16,
        max_speech_ms: int = 15000,
    ) -> None:
        self.engine = engine
        self.segmenter = segmenter or UtteranceSegmenter(sample_rate=sample_rate)
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.max_history_messages = max_history_messages
        self.max_samples = max(1, sample_rate * max_speech_ms // 1000)
        self.session_id = uuid.uuid4().hex
        self.dialect: str | None = None
        self.listening = False
        self.endpoint = "manual"
        self.input_format = "pcm"
        self.input_rate = sample_rate
        self.binary_version = 1
        self.frame_ms = 60
        self.source = ""
        self.wake_keyword = ""
        self.history: list[ChatMessage] = []
        self._chunks: list[np.ndarray] = []
        self._buffered = 0
        self._wav = bytearray()
        self._opus: OpusCodec | None = None
        self._vad_warned = False
        self._timestamp_ms = 0

    def handle_text(self, raw: str) -> list[OutEvent]:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return self._error("无法解析 JSON")
        if not isinstance(message, dict):
            return self._error("JSON 必须是对象")
        kind = message.get("type")
        if not isinstance(kind, str) or not kind:
            return self._error("消息缺少 type")
        if kind == "hello":
            return self._hello(message)
        if kind == "listen":
            return self._listen(message)
        if kind == "abort":
            return self._abort()
        if kind == "session.start":
            return self._session_start(message)
        if kind == "wake":
            return self._wake(message)
        if kind == "audio.end":
            return self._end_audio()
        if kind == "session.stop":
            return self._session_stop()
        if kind == "ping":
            return self._pong()
        if kind == "mcp":
            return self._mcp(message)
        return self._error(f"不认识的消息类型：{kind}")

    def handle_binary(self, data: bytes) -> list[OutEvent]:
        if self.dialect is None:
            self.dialect = "vibe"
            self.listening = True
            self.input_format = "pcm"
        if not self.listening:
            if self.dialect == "xiaozhi":
                return []
            self.listening = True
            self._reset_audio()
        try:
            return self._accept_payload(data)
        except (VibeVoiceError, AudioFormatError) as exc:
            return self._error(str(exc))

    def _hello(self, message: dict) -> list[OutEvent]:
        self.dialect = "xiaozhi"
        params = message.get("audio_params") or {}
        audio_format = str(params.get("format") or "opus").lower()
        if audio_format not in _FORMATS:
            return self._error(f"不支持的音频格式：{audio_format}")
        self.input_format = audio_format
        self.input_rate = int(params.get("sample_rate") or self.sample_rate)
        self.frame_ms = int(params.get("frame_duration") or 60)
        self.binary_version = int(message.get("version") or 1)
        self.listening = False
        out_format = "opus" if audio_format == "opus" else "pcm"
        events = [
            self._json(
                {
                    "type": "hello",
                    "transport": "websocket",
                    "session_id": self.session_id,
                    "audio_params": {
                        "format": out_format,
                        "sample_rate": self.sample_rate,
                        "channels": 1,
                        "frame_duration": self.frame_ms,
                    },
                }
            )
        ]
        if out_format == "opus" and not opus_available():
            events.extend(
                self._error("服务器没有 Opus 编解码（需要 libopus 和 opuslib），设备音频无法收发。")
            )
        return events

    def _listen(self, message: dict) -> list[OutEvent]:
        self.dialect = self.dialect or "xiaozhi"
        state = str(message.get("state") or "")
        if state not in _XIAOZHI_LISTEN:
            return self._error(f"listen.state 无效：{state}")
        if state == "detect":
            self.wake_keyword = str(message.get("text") or "")
            return []
        if state == "start":
            mode = str(message.get("mode") or "manual")
            self.endpoint = "vad" if mode in {"auto", "realtime"} else "manual"
            self._reset_audio()
            self.listening = True
            events: list[OutEvent] = []
            if self.endpoint == "vad" and self.classifier is None and not self._vad_warned:
                self._vad_warned = True
                events.extend(
                    self._error("没有 Silero VAD 模型，auto 模式要等 listen stop，或先运行 vibe-voice download。")
                )
            return events
        return self._end_audio()

    def _session_start(self, message: dict) -> list[OutEvent]:
        self.dialect = "vibe"
        audio = message.get("audio") or {}
        audio_format = str(audio.get("format") or "pcm").lower()
        if audio_format not in _FORMATS:
            return self._error(f"不支持的音频格式：{audio_format}")
        self.input_format = audio_format
        self.input_rate = int(audio.get("sample_rate") or self.sample_rate)
        self.endpoint = "vad" if str(message.get("endpoint") or "manual") == "vad" else "manual"
        self.source = str(message.get("source") or "")
        self.binary_version = 1
        self._reset_audio()
        self.listening = True
        ready = {
            "type": "session.ready",
            "session_id": self.session_id,
            "audio_out": {
                "format": "pcm",
                "encoding": "int16",
                "channels": 1,
            },
        }
        if self.source:
            ready["source"] = self.source
        events = [self._json(ready)]
        if self.endpoint == "vad" and self.classifier is None and not self._vad_warned:
            self._vad_warned = True
            events.extend(self._error("没有 Silero VAD 模型。请发送 audio.end，或先运行 vibe-voice download。"))
        return events

    def _wake(self, message: dict) -> list[OutEvent]:
        self.dialect = self.dialect or "vibe"
        self.wake_keyword = str(message.get("keyword") or "")
        self.source = str(message.get("source") or self.source)
        body = {"type": "wake.ack", "session_id": self.session_id, "keyword": self.wake_keyword}
        if self.source:
            body["source"] = self.source
        return [self._json(body)]

    def _abort(self) -> list[OutEvent]:
        self._reset_audio()
        self.listening = False
        if self.dialect == "xiaozhi":
            return [self._json({"type": "tts", "state": "stop", "session_id": self.session_id})]
        return [self._json({"type": "aborted", "session_id": self.session_id})]

    def _session_stop(self) -> list[OutEvent]:
        self._reset_audio()
        self.listening = False
        self.history.clear()
        return [self._json({"type": "session.stopped", "session_id": self.session_id})]

    def _pong(self) -> list[OutEvent]:
        if self.dialect == "xiaozhi":
            return [self._json({"type": "pong", "session_id": self.session_id})]
        return [self._json({"type": "pong", "session_id": self.session_id})]

    def _mcp(self, message: dict) -> list[OutEvent]:
        self.dialect = self.dialect or "xiaozhi"
        payload = message.get("payload") or {}
        if not isinstance(payload, dict):
            return self._error("mcp.payload 必须是对象")
        response = _mcp_response(payload)
        if response is None:
            return []
        return [
            self._json(
                {
                    "type": "mcp",
                    "session_id": self.session_id,
                    "payload": response,
                }
            )
        ]

    def _accept_payload(self, data: bytes) -> list[OutEvent]:
        payload = data
        if self.dialect == "xiaozhi":
            payload = unwrap_binary_frame(data, self.binary_version)
        if self.input_format == "wav":
            self._wav.extend(payload)
            return []
        samples = self._decode_samples(payload)
        return self._push_samples(samples, self.input_rate if self.input_format == "pcm" else self.sample_rate)

    def _decode_samples(self, payload: bytes) -> np.ndarray:
        if self.input_format == "pcm":
            return pcm16le_to_float(payload)
        if self.input_format == "opus":
            if self._opus is None:
                rate = self.input_rate if self.input_rate in {8000, 12000, 16000, 24000, 48000} else 16000
                frame_ms = self.frame_ms if self.frame_ms in {10, 20, 40, 60} else 60
                self._opus = OpusCodec(rate, frame_ms)
            return self._opus.decode_packet(payload)
        raise AudioFormatError(f"不支持的音频格式：{self.input_format}")

    def _push_samples(self, samples: np.ndarray, source_rate: int) -> list[OutEvent]:
        audio = resample_linear(samples, source_rate, self.sample_rate)
        if audio.size == 0:
            return []
        if self.endpoint == "vad" and self.classifier is not None:
            events: list[OutEvent] = []
            for frame, is_speech in self.classifier.classify(audio):
                for utterance in self.segmenter.push(frame, bool(is_speech)):
                    events.extend(self._finish(utterance))
            return events
        self._chunks.append(audio)
        self._buffered += int(audio.size)
        if self._buffered >= self.max_samples:
            return self._finish(self._take_buffer())
        return []

    def _end_audio(self) -> list[OutEvent]:
        self.listening = False
        if self.input_format == "wav" and self._wav:
            samples, rate = decode_wav(bytes(self._wav))
            self._wav.clear()
            samples = resample_linear(samples, rate, self.sample_rate)
            return self._finish(samples)
        if self.endpoint == "vad" and self.classifier is not None:
            pending = self.segmenter.flush()
            events: list[OutEvent] = []
            for utterance in pending:
                events.extend(self._finish(utterance))
            self._reset_audio()
            if self.dialect == "xiaozhi" and self.endpoint == "vad":
                self.listening = True
            return events
        audio = self._take_buffer()
        if audio.size == 0:
            if self.dialect == "xiaozhi":
                return []
            return self._error("没有收到音频")
        events = self._finish(audio)
        if self.dialect == "xiaozhi" and self.endpoint == "vad":
            self.listening = True
        return events

    def _finish(self, samples: np.ndarray) -> list[OutEvent]:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._chunks = []
        self._buffered = 0
        if audio.size < self.sample_rate // 10:
            if self.dialect == "xiaozhi":
                return []
            return self._error("音频太短，没有送去识别")
        try:
            transcript = self.engine.transcribe(audio, self.sample_rate).strip()
        except VibeVoiceError as exc:
            return self._error(str(exc))
        if not transcript:
            if self.dialect == "xiaozhi":
                return []
            return self._error("没有识别到文字")
        try:
            answer = self.engine.reply(transcript, list(self.history))
        except VibeVoiceError as exc:
            events = self._transcript(transcript)
            events.extend(self._error(str(exc)))
            return events
        append_history(self.history, transcript, answer, self.max_history_messages)
        try:
            synthesized = self.engine.synthesize(answer)
        except VibeVoiceError as exc:
            events = self._reply_events(transcript, answer, None)
            events.extend(self._error(str(exc)))
            return events
        return self._reply_events(transcript, answer, synthesized)

    def _reply_events(
        self,
        transcript: str,
        answer: str,
        synthesized: tuple[np.ndarray, int] | None,
    ) -> list[OutEvent]:
        events = self._transcript(transcript)
        if self.dialect == "xiaozhi":
            events.append(
                self._json(
                    {
                        "type": "llm",
                        "session_id": self.session_id,
                        "emotion": "neutral",
                        "text": "🙂",
                    }
                )
            )
            events.append(self._json({"type": "tts", "state": "start", "session_id": self.session_id}))
            events.append(
                self._json(
                    {
                        "type": "tts",
                        "state": "sentence_start",
                        "session_id": self.session_id,
                        "text": answer,
                    }
                )
            )
            events.extend(self._tts_audio(synthesized, xiaozhi=True))
            events.append(self._json({"type": "tts", "state": "stop", "session_id": self.session_id}))
            return events
        events.append(self._json({"type": "llm.final", "session_id": self.session_id, "text": answer}))
        if synthesized is None:
            events.append(
                self._json(
                    {
                        "type": "tts.unavailable",
                        "session_id": self.session_id,
                        "message": "未找到中文 TTS 模型，仅返回文本。运行 vibe-voice download 可下载语音。",
                    }
                )
            )
            return events
        samples, rate = synthesized
        events.append(
            self._json(
                {
                    "type": "tts.start",
                    "session_id": self.session_id,
                    "audio": {
                        "format": "pcm",
                        "encoding": "int16",
                        "sample_rate": rate,
                        "channels": 1,
                    },
                }
            )
        )
        events.extend(self._tts_audio(synthesized, xiaozhi=False))
        events.append(self._json({"type": "tts.end", "session_id": self.session_id}))
        return events

    def _tts_audio(
        self,
        synthesized: tuple[np.ndarray, int] | None,
        *,
        xiaozhi: bool,
    ) -> list[OutEvent]:
        if synthesized is None:
            return []
        samples, rate = synthesized
        if xiaozhi:
            if self.input_format != "opus":
                pcm = float_to_pcm16(resample_linear(samples, rate, self.sample_rate))
                return [OutEvent(binary=chunk) for chunk in _chunks(pcm, self.sample_rate * 2 // 10)]
            if not opus_available():
                return self._error("无法把 TTS 编成 Opus。安装 libopus 和 opuslib 后再接小智设备。")
            codec = OpusCodec(self.sample_rate, 60)
            frames = codec.encode(resample_linear(samples, rate, self.sample_rate))
            frames.extend(codec.flush())
            events = []
            for frame in frames:
                wrapped = wrap_binary_frame(frame, self.binary_version, self._timestamp_ms)
                self._timestamp_ms += 60
                events.append(OutEvent(binary=wrapped))
            return events
        pcm = float_to_pcm16(samples)
        width = max(2, rate * 2 // 10)
        return [OutEvent(binary=chunk) for chunk in _chunks(pcm, width)]

    def _transcript(self, transcript: str) -> list[OutEvent]:
        if self.dialect == "xiaozhi":
            return [self._json({"type": "stt", "session_id": self.session_id, "text": transcript})]
        return [self._json({"type": "stt.final", "session_id": self.session_id, "text": transcript})]

    def _take_buffer(self) -> np.ndarray:
        if not self._chunks:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(self._chunks)
        self._chunks = []
        self._buffered = 0
        return audio

    def _reset_audio(self) -> None:
        self._chunks = []
        self._buffered = 0
        self._wav.clear()
        self._opus = None
        self.segmenter.reset()
        reset = getattr(self.classifier, "reset", None)
        if callable(reset):
            reset()

    def _error(self, message: str) -> list[OutEvent]:
        if self.dialect == "xiaozhi":
            return [
                self._json(
                    {
                        "type": "alert",
                        "session_id": self.session_id,
                        "status": "出错",
                        "message": message,
                        "emotion": "neutral",
                    }
                )
            ]
        return [self._json({"type": "error", "session_id": self.session_id, "message": message})]

    def _json(self, body: dict) -> OutEvent:
        return OutEvent(text=body)


def _mcp_response(payload: dict) -> dict | None:
    method = payload.get("method")
    request_id = payload.get("id")
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "initialize":
        params = payload.get("params") or {}
        result = {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vibe-voice", "version": "0.1.0"},
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if request_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "vibe-voice 没有实现这个设备工具"},
    }


def _chunks(data: bytes, size: int) -> list[bytes]:
    step = max(2, size - (size % 2))
    return [data[index : index + step] for index in range(0, len(data), step)] or []


def dumps(event: OutEvent) -> str:
    return json.dumps(event.text, ensure_ascii=False)
