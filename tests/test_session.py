import asyncio
import json

import numpy as np
import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from vibe_voice.audio import OpusCodec, encode_wav, float_to_pcm16, opus_available, wrap_binary_frame
from vibe_voice.config import Settings
from vibe_voice.server import start_server
from vibe_voice.session import VoiceSession
from vibe_voice.vad import UtteranceSegmenter


class FakeEngine:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, sample_rate):
        self.calls.append(("stt", sample_rate, int(samples.size)))
        return "今天天气怎么样"

    def reply(self, text, history):
        self.calls.append(("llm", text, len(history)))
        return "今天晴天。"

    def synthesize(self, text):
        self.calls.append(("tts", text))
        return np.full(160, 0.25, dtype=np.float32), 8000


def _texts(events):
    return [event.text for event in events if event.text is not None]


def _bins(events):
    return [event.binary for event in events if event.binary is not None]


def test_vibe_session_pcm_roundtrip_and_second_turn_history():
    engine = FakeEngine()
    session = VoiceSession(engine, sample_rate=16000, max_history_messages=16)
    ready = _texts(session.handle_text(json.dumps({
        "type": "session.start",
        "source": "open-xiaoai",
        "audio": {"format": "pcm", "sample_rate": 16000, "channels": 1, "encoding": "int16"},
    })))
    assert ready[0]["type"] == "session.ready"
    assert ready[0]["source"] == "open-xiaoai"
    ack = _texts(session.handle_text(json.dumps({"type": "wake", "keyword": "你好小助", "source": "open-xiaoai"})))
    assert ack[0]["keyword"] == "你好小助"
    pcm = float_to_pcm16(np.full(2000, 0.2, dtype=np.float32))
    assert session.handle_binary(pcm[:1000]) == []
    assert session.handle_binary(pcm[1000:]) == []
    events = session.handle_text('{"type":"audio.end"}')
    bodies = _texts(events)
    kinds = [body["type"] for body in bodies]
    assert kinds == ["stt.final", "llm.final", "tts.start", "tts.end"]
    assert bodies[0]["text"] == "今天天气怎么样"
    assert bodies[1]["text"] == "今天晴天。"
    assert bodies[2]["audio"]["sample_rate"] == 8000
    assert b"".join(_bins(events)) == float_to_pcm16(np.full(160, 0.25, dtype=np.float32))
    assert engine.calls[0][0] == "stt"
    assert engine.calls[1] == ("llm", "今天天气怎么样", 0)

    session.handle_binary(pcm)
    session.handle_text('{"type":"audio.end"}')
    assert engine.calls[4] == ("llm", "今天天气怎么样", 2)


def test_wav_blob_and_abort_and_bad_json():
    engine = FakeEngine()
    session = VoiceSession(engine)
    session.handle_text(json.dumps({"type": "session.start", "audio": {"format": "wav", "sample_rate": 16000}}))
    wav = encode_wav(np.full(1800, 0.1, dtype=np.float32), 16000)
    assert session.handle_binary(wav) == []
    events = session.handle_text('{"type":"audio.end"}')
    assert _texts(events)[0]["type"] == "stt.final"
    assert engine.calls[-3][0] == "stt"

    session.handle_text(json.dumps({"type": "session.start", "audio": {"format": "pcm"}}))
    session.handle_binary(float_to_pcm16(np.ones(500, dtype=np.float32)))
    aborted = _texts(session.handle_text('{"type":"abort"}'))
    assert aborted[0]["type"] == "aborted"
    assert session.handle_text('{"type":"audio.end"}')[0].text["type"] == "error"
    assert len(engine.calls) == 3  # only the wav turn reached the engine
    bad = session.handle_text("not-json")
    assert bad[0].text["type"] == "error"


def test_tts_missing_still_returns_text():
    class TextOnly(FakeEngine):
        def synthesize(self, text):
            return None

    session = VoiceSession(TextOnly())
    session.handle_text('{"type":"session.start","audio":{"format":"pcm"}}')
    session.handle_binary(float_to_pcm16(np.ones(1600, dtype=np.float32)))
    bodies = _texts(session.handle_text('{"type":"audio.end"}'))
    assert [body["type"] for body in bodies] == ["stt.final", "llm.final", "tts.unavailable"]
    assert "TTS" in bodies[-1]["message"]


def test_xiaozhi_manual_pcm_and_mcp_and_binary_header():
    engine = FakeEngine()
    session = VoiceSession(engine)
    hello = _texts(session.handle_text(json.dumps({
        "type": "hello",
        "version": 2,
        "transport": "websocket",
        "audio_params": {"format": "pcm", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
    })))
    assert hello[0]["type"] == "hello"
    assert hello[0]["transport"] == "websocket"
    assert hello[0]["audio_params"]["format"] == "pcm"
    session.handle_text(json.dumps({"type": "listen", "state": "detect", "text": "你好小智"}))
    assert session.wake_keyword == "你好小智"
    session.handle_text(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
    pcm = float_to_pcm16(np.full(2000, 0.2, dtype=np.float32))
    ignored = session.handle_binary(wrap_binary_frame(b"\x00\x00", 2))
    # listening is true, but this tiny payload is buffered, not finished
    assert ignored == []
    session.handle_binary(wrap_binary_frame(pcm, 2))
    events = session.handle_text(json.dumps({"type": "listen", "state": "stop"}))
    kinds = [body["type"] for body in _texts(events)]
    assert kinds[:4] == ["stt", "llm", "tts", "tts"]
    assert _texts(events)[0]["text"] == "今天天气怎么样"
    assert _texts(events)[2]["state"] == "start"
    assert _texts(events)[3]["state"] == "sentence_start"
    assert _texts(events)[3]["text"] == "今天晴天。"
    assert _texts(events)[-1] == {"type": "tts", "state": "stop", "session_id": session.session_id}
    assert _bins(events)

    listed = _texts(session.handle_text(json.dumps({
        "type": "mcp",
        "payload": {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
    })))
    assert listed[0]["payload"]["result"] == {"tools": []}
    init = _texts(session.handle_text(json.dumps({
        "type": "mcp",
        "payload": {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    })))
    assert init[0]["payload"]["result"]["serverInfo"]["name"] == "vibe-voice"
    unknown = _texts(session.handle_text(json.dumps({
        "type": "mcp",
        "payload": {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {}},
    })))
    assert unknown[0]["payload"]["error"]["code"] == -32601


def test_vad_endpoint_closes_without_audio_end():
    engine = FakeEngine()

    class Always:
        def classify(self, samples):
            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
            return [(audio, float(np.max(np.abs(audio))) > 0.2)]

        def reset(self):
            return None

    session = VoiceSession(
        engine,
        segmenter=UtteranceSegmenter(
            sample_rate=16000,
            min_speech_ms=100,
            min_silence_ms=50,
            max_speech_ms=5000,
            speech_pad_ms=0,
        ),
        classifier=Always(),
        max_speech_ms=5000,
    )
    session.handle_text(json.dumps({"type": "session.start", "endpoint": "vad", "audio": {"format": "pcm"}}))
    assert session.handle_binary(float_to_pcm16(np.full(1600, 0.4, dtype=np.float32))) == []
    events = session.handle_binary(float_to_pcm16(np.zeros(800, dtype=np.float32)))
    assert _texts(events)[0]["type"] == "stt.final"
    assert engine.calls[0][0] == "stt"


def test_opus_session_when_library_is_present():
    if not opus_available():
        pytest.skip("opuslib is not installed")
    engine = FakeEngine()
    session = VoiceSession(engine)
    session.handle_text(json.dumps({
        "type": "session.start",
        "audio": {"format": "opus", "sample_rate": 16000},
    }))
    codec = OpusCodec(16000, 60)
    tone = (0.2 * np.sin(np.linspace(0, 80, 1920, dtype=np.float32))).astype(np.float32)
    packets = codec.encode(tone)
    for packet in packets:
        session.handle_binary(packet)
    events = session.handle_text('{"type":"audio.end"}')
    assert _texts(events)[0]["type"] == "stt.final"
    assert engine.calls[0][2] >= 1600


def test_websocket_server_speaks_the_vibe_session():
    asyncio.run(_websocket_roundtrip())


async def _websocket_roundtrip():
    engine = FakeEngine()
    settings = Settings()
    settings.ws_host = "127.0.0.1"
    settings.ws_port = 0
    settings.ws_token = "secret"
    server = await start_server(settings, engine, None)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}/v1/voice"
    try:
        with pytest.raises(ConnectionClosed):
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"ping"}')
                await ws.recv()
        async with websockets.connect(url, additional_headers={"Authorization": "Bearer secret"}) as ws:
            await ws.send(json.dumps({"type": "session.start", "audio": {"format": "pcm", "sample_rate": 16000}}))
            ready = json.loads(await ws.recv())
            assert ready["type"] == "session.ready"
            await ws.send(float_to_pcm16(np.full(1800, 0.2, dtype=np.float32)))
            await ws.send('{"type":"audio.end"}')
            seen = []
            binary = 0
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=2)
                if isinstance(message, bytes):
                    binary += len(message)
                    continue
                body = json.loads(message)
                seen.append(body["type"])
                if body["type"] == "tts.end":
                    break
            assert seen == ["stt.final", "llm.final", "tts.start", "tts.end"]
            assert binary > 0
    finally:
        server.close()
        await server.wait_closed()
