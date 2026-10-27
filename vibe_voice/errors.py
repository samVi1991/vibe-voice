"""Errors raised by the voice pipeline."""


class VibeVoiceError(Exception):
    """Base error for vibe-voice."""


class KeywordFileError(VibeVoiceError):
    """The sherpa-onnx keywords file is missing or malformed."""


class ModelNotReady(VibeVoiceError):
    """A required model file is not on disk."""


class AudioFormatError(VibeVoiceError):
    """An audio payload is not PCM, WAV, or Opus we can decode."""


class LlmError(VibeVoiceError):
    """The OpenAI-compatible chat endpoint failed."""
