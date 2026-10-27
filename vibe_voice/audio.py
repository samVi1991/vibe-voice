"""PCM, WAV, resampling, and optional Opus framing."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

from vibe_voice.errors import AudioFormatError

_OPUS_IMPORT_ERROR: Exception | None = None


def opus_available() -> bool:
    try:
        import opuslib  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the host
        global _OPUS_IMPORT_ERROR
        _OPUS_IMPORT_ERROR = exc
        return False
    return True


def opus_import_error() -> str:
    if opus_available():
        return ""
    reason = _OPUS_IMPORT_ERROR
    return f"{type(reason).__name__}: {reason}" if reason else "opuslib 未安装"


def pcm16le_to_float(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]
    pcm = np.frombuffer(data, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(array, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise AudioFormatError(f"无法解析 WAV：{exc}") from exc
    if width != 2:
        raise AudioFormatError("只支持 16-bit PCM WAV")
    if rate <= 0:
        raise AudioFormatError("WAV 采样率无效")
    pcm = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        usable = pcm.size - (pcm.size % channels)
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    return np.asarray(pcm, dtype=np.float32) / 32768.0, rate


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    pcm = float_to_pcm16(samples)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or array.size == 0:
        return array
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioFormatError(f"采样率无效：{orig_sr} -> {target_sr}")
    target_len = max(1, int(round(array.size * target_sr / orig_sr)))
    if array.size == 1 or target_len == 1:
        return np.full(target_len, array[0], dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, array.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(new_x, old_x, array).astype(np.float32)


def wrap_binary_frame(payload: bytes, version: int, timestamp_ms: int = 0) -> bytes:
    """Wrap one Opus packet the way xiaozhi-esp32 version 2 or 3 expects."""

    if version <= 1:
        return payload
    if version == 2:
        header = struct.pack(">HHIII", version, 0, 0, timestamp_ms, len(payload))
        return header + payload
    if version == 3:
        header = struct.pack(">BBH", 0, 0, len(payload))
        return header + payload
    raise AudioFormatError(f"不支持的二进制协议版本：{version}")


def unwrap_binary_frame(data: bytes, version: int) -> bytes:
    """Return the audio payload inside a xiaozhi binary frame.

    Version 1 is the raw payload. Versions 2 and 3 use the packed headers from
    the firmware. Both big-endian (``htons`` / ``htonl``) and little-endian
    sizes are accepted when the size field matches the remaining bytes.
    """

    if version <= 1 or not data:
        return data
    if version == 2 and len(data) >= 16:
        payload = _slice_sized(data, 12, 16, 4)
        if payload is not None:
            return payload
        return data[16:]
    if version == 3 and len(data) >= 4:
        payload = _slice_sized(data, 2, 4, 2)
        if payload is not None:
            return payload
        return data[4:]
    return data


def _slice_sized(data: bytes, size_offset: int, header_len: int, width: int) -> bytes | None:
    for endian in ("big", "little"):
        size = int.from_bytes(data[size_offset : size_offset + width], endian)
        if size == len(data) - header_len:
            return data[header_len : header_len + size]
    return None


class OpusCodec:
    """Encode and decode mono Opus frames. Requires ``opuslib`` and libopus."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 60) -> None:
        if not opus_available():
            detail = opus_import_error()
            raise AudioFormatError(
                "Opus 编解码需要系统库 libopus 和 Python 包 opuslib"
                + (f"（{detail}）" if detail else "")
                + "。电脑麦克风和 PCM/WAV 会话不需要它。"
            )
        import opuslib

        if frame_ms not in {10, 20, 40, 60} or sample_rate not in {8000, 12000, 16000, 24000, 48000}:
            raise AudioFormatError(
                f"Opus 不支持 {sample_rate} Hz / {frame_ms} ms。设备链路使用 16000 Hz、60 ms。"
            )
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self._decoder = opuslib.Decoder(sample_rate, 1)
        self._encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_VOIP)
        self._pending = np.zeros(0, dtype=np.float32)

    def decode_packet(self, packet: bytes) -> np.ndarray:
        if not packet:
            return np.zeros(0, dtype=np.float32)
        pcm = self._decoder.decode(packet, self.frame_samples)
        return pcm16le_to_float(pcm)

    def encode(self, samples: np.ndarray) -> list[bytes]:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._pending = np.concatenate([self._pending, incoming])
        frames: list[bytes] = []
        while self._pending.size >= self.frame_samples:
            frame = self._pending[: self.frame_samples]
            self._pending = self._pending[self.frame_samples :]
            frames.append(self._encoder.encode(float_to_pcm16(frame), self.frame_samples))
        return frames

    def flush(self) -> list[bytes]:
        if self._pending.size == 0:
            return []
        pad = self.frame_samples - (self._pending.size % self.frame_samples)
        if pad != self.frame_samples:
            self._pending = np.concatenate(
                [self._pending, np.zeros(pad, dtype=np.float32)]
            )
        return self.encode(np.zeros(0, dtype=np.float32))
