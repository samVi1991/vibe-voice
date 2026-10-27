import numpy as np

from vibe_voice.audio import (
    OpusCodec,
    decode_wav,
    encode_wav,
    float_to_pcm16,
    opus_available,
    pcm16le_to_float,
    resample_linear,
    unwrap_binary_frame,
    wrap_binary_frame,
)
from vibe_voice.speech import clean_sensevoice_text


def test_clean_sensevoice_tags():
    raw = "<|zh|><|NEUTRAL|><|Speech|>今天天气怎么样<|woitn|>"
    assert clean_sensevoice_text(raw) == "今天天气怎么样"
    assert clean_sensevoice_text("") == ""


def test_pcm_and_wav_roundtrip():
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm = float_to_pcm16(samples)
    back = pcm16le_to_float(pcm)
    assert back.shape == samples.shape
    assert np.allclose(back, samples, atol=1 / 32767)
    wav = encode_wav(samples, 16000)
    decoded, rate = decode_wav(wav)
    assert rate == 16000
    assert np.allclose(decoded, samples, atol=1 / 32767)


def test_resample_halves_a_second():
    tone = np.linspace(-1, 1, 16000, dtype=np.float32)
    halved = resample_linear(tone, 16000, 8000)
    assert halved.shape == (8000,)
    assert resample_linear(tone, 16000, 16000) is tone or np.shares_memory(
        resample_linear(tone, 16000, 16000), tone
    )


def test_xiaozhi_binary_v2_and_v3_roundtrip():
    import struct

    payload = b"\x01\x02\x03\x04"
    assert unwrap_binary_frame(wrap_binary_frame(payload, 2, timestamp_ms=60), 2) == payload
    little_endian = struct.pack("<HHIII", 2, 0, 0, 60, len(payload)) + payload
    assert unwrap_binary_frame(little_endian, 2) == payload
    assert unwrap_binary_frame(wrap_binary_frame(payload, 3), 3) == payload
    assert unwrap_binary_frame(payload, 1) == payload


def test_opus_frame_roundtrip():
    if not opus_available():
        return
    codec = OpusCodec(16000, 60)
    samples = (0.2 * np.sin(np.linspace(0, 40, 960, dtype=np.float32))).astype(np.float32)
    frames = codec.encode(samples)
    assert len(frames) == 1
    decoded = codec.decode_packet(frames[0])
    assert decoded.shape == (960,)
    assert np.max(np.abs(decoded)) > 0.01
