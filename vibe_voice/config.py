"""Settings from an optional TOML file and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from vibe_voice.keywords import packaged_keywords_path

DEFAULT_SYSTEM_PROMPT = (
    "你是语音助手小助。用户正在通过语音和你说话。"
    "请用简短的口语中文回答，通常一到两句，适合直接朗读。"
    "不要使用 Markdown、编号列表、表情符号或网址，除非用户明确要你念出来。"
)

KWS_DIR_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
KWS_ENCODER_NAME = "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
KWS_DECODER_NAME = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
KWS_JOINER_NAME = "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
SENSEVOICE_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SENSEVOICE_MODEL_NAME = "model.int8.onnx"
TTS_DIR_NAME = "sherpa-onnx-vits-zh-ll"
VAD_MODEL_NAME = "silero_vad.onnx"


@dataclass
class Settings:
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout_s: float = 60.0
    openai_max_tokens: int | None = None
    temperature: float = 0.7
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    keywords_file: Path = field(default_factory=packaged_keywords_path)
    models_dir: Path = Path("models")
    sample_rate: int = 16000
    mic_sample_rate: int = 16000
    mic_block_ms: int = 100
    mic_device: int | None = None
    speaker_device: int | None = None
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    ws_token: str = ""
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 600
    vad_max_speech_ms: int = 15000
    vad_speech_pad_ms: int = 200
    wake_suppress_ms: int = 300
    wake_timeout_ms: int = 8000
    kws_score: float = 1.0
    kws_threshold: float = 0.25
    kws_num_trailing_blanks: int = 1
    stt_language: str = "zh"
    stt_use_itn: bool = True
    tts_sid: int = 2
    tts_speed: float = 1.0
    tts_max_sentences: int = 4
    num_threads: int = 2
    max_history_messages: int = 16

    @property
    def kws_dir(self) -> Path:
        return self.models_dir / KWS_DIR_NAME

    @property
    def kws_tokens(self) -> Path:
        return self.kws_dir / "tokens.txt"

    @property
    def kws_encoder(self) -> Path:
        return self.kws_dir / KWS_ENCODER_NAME

    @property
    def kws_decoder(self) -> Path:
        return self.kws_dir / KWS_DECODER_NAME

    @property
    def kws_joiner(self) -> Path:
        return self.kws_dir / KWS_JOINER_NAME

    @property
    def sensevoice_model(self) -> Path:
        return self.models_dir / SENSEVOICE_DIR_NAME / SENSEVOICE_MODEL_NAME

    @property
    def sensevoice_tokens(self) -> Path:
        return self.models_dir / SENSEVOICE_DIR_NAME / "tokens.txt"

    @property
    def vad_model(self) -> Path:
        return self.models_dir / VAD_MODEL_NAME

    @property
    def tts_dir(self) -> Path:
        return self.models_dir / TTS_DIR_NAME

    @property
    def tts_model(self) -> Path:
        return self.tts_dir / "model.onnx"

    @property
    def tts_tokens(self) -> Path:
        return self.tts_dir / "tokens.txt"

    @property
    def tts_lexicon(self) -> Path:
        return self.tts_dir / "lexicon.txt"

    @property
    def tts_rule_fsts(self) -> list[Path]:
        names = ("phone.fst", "date.fst", "number.fst")
        return [path for name in names if (path := self.tts_dir / name).is_file()]


def load_settings(
    config_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    root: Path | None = None,
) -> Settings:
    """Load defaults, then TOML, then environment variables.

    ``environ`` replaces ``os.environ`` when given, which keeps tests isolated.
    A ``.env`` file in ``root`` fills only keys that are still missing.
    """

    root = Path(root or Path.cwd())
    env: dict[str, str] = dict(os.environ if environ is None else environ)
    _apply_dotenv(root / ".env", env)

    data: dict = {}
    chosen = config_path
    if chosen is None and env.get("VIBE_CONFIG"):
        chosen = Path(env["VIBE_CONFIG"])
    if chosen is None:
        default_file = root / "config.toml"
        if default_file.is_file():
            chosen = default_file
    if chosen is not None:
        path = chosen if chosen.is_absolute() else root / chosen
        data = _load_toml(path)

    settings = Settings()
    _apply_toml(settings, data, root)
    _apply_env(settings, env, root)
    settings.keywords_file = _resolve_keywords(settings.keywords_file, env, data, root)
    settings.models_dir = _resolve_path(settings.models_dir, root)
    return settings


def _load_toml(path: Path) -> dict:
    import tomllib

    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件必须是表：{path}")
    return loaded


def _apply_toml(settings: Settings, data: dict, root: Path) -> None:
    openai = data.get("openai") or {}
    server = data.get("server") or {}
    vad = data.get("vad") or {}
    tts = data.get("tts") or {}
    kws = data.get("kws") or {}
    audio = data.get("audio") or {}
    if "base_url" in openai:
        settings.openai_base_url = str(openai["base_url"])
    if "api_key" in openai:
        settings.openai_api_key = str(openai["api_key"])
    if "model" in openai:
        settings.openai_model = str(openai["model"])
    if "timeout_s" in openai:
        settings.openai_timeout_s = float(openai["timeout_s"])
    if "max_tokens" in openai:
        settings.openai_max_tokens = int(openai["max_tokens"])
    if "temperature" in openai:
        settings.temperature = float(openai["temperature"])
    if "system_prompt" in openai:
        settings.system_prompt = str(openai["system_prompt"])
    if "host" in server:
        settings.ws_host = str(server["host"])
    if "port" in server:
        settings.ws_port = int(server["port"])
    if "token" in server:
        settings.ws_token = str(server["token"])
    if "threshold" in vad:
        settings.vad_threshold = float(vad["threshold"])
    if "min_speech_ms" in vad:
        settings.vad_min_speech_ms = int(vad["min_speech_ms"])
    if "min_silence_ms" in vad:
        settings.vad_min_silence_ms = int(vad["min_silence_ms"])
    if "max_speech_ms" in vad:
        settings.vad_max_speech_ms = int(vad["max_speech_ms"])
    if "speech_pad_ms" in vad:
        settings.vad_speech_pad_ms = int(vad["speech_pad_ms"])
    if "sid" in tts:
        settings.tts_sid = int(tts["sid"])
    if "speed" in tts:
        settings.tts_speed = float(tts["speed"])
    if "max_sentences" in tts:
        settings.tts_max_sentences = int(tts["max_sentences"])
    if "keywords_file" in kws:
        settings.keywords_file = Path(str(kws["keywords_file"]))
    if "score" in kws:
        settings.kws_score = float(kws["score"])
    if "threshold" in kws:
        settings.kws_threshold = float(kws["threshold"])
    if "models_dir" in data:
        settings.models_dir = Path(str(data["models_dir"]))
    if "sample_rate" in audio:
        settings.sample_rate = int(audio["sample_rate"])
    if "mic_sample_rate" in audio:
        settings.mic_sample_rate = int(audio["mic_sample_rate"])
    if "mic_device" in audio:
        settings.mic_device = int(audio["mic_device"])
    if "speaker_device" in audio:
        settings.speaker_device = int(audio["speaker_device"])
    if "num_threads" in data:
        settings.num_threads = int(data["num_threads"])
    if "stt_language" in data:
        settings.stt_language = str(data["stt_language"])


def _apply_env(settings: Settings, env: dict[str, str], root: Path) -> None:
    _set_str(env, "OPENAI_BASE_URL", settings, "openai_base_url")
    _set_str(env, "OPENAI_API_KEY", settings, "openai_api_key")
    _set_str(env, "OPENAI_MODEL", settings, "openai_model")
    _set_float(env, "OPENAI_TIMEOUT_S", settings, "openai_timeout_s")
    _set_optional_int(env, "OPENAI_MAX_TOKENS", settings, "openai_max_tokens")
    _set_float(env, "VIBE_TEMPERATURE", settings, "temperature")
    _set_str(env, "VIBE_SYSTEM_PROMPT", settings, "system_prompt")
    _set_path(env, "VIBE_MODELS_DIR", settings, "models_dir", root)
    _set_int(env, "VIBE_SAMPLE_RATE", settings, "sample_rate")
    _set_int(env, "VIBE_MIC_SAMPLE_RATE", settings, "mic_sample_rate")
    _set_int(env, "VIBE_MIC_BLOCK_MS", settings, "mic_block_ms")
    _set_optional_int(env, "VIBE_MIC_DEVICE", settings, "mic_device")
    _set_optional_int(env, "VIBE_SPEAKER_DEVICE", settings, "speaker_device")
    _set_str(env, "VIBE_WS_HOST", settings, "ws_host")
    _set_int(env, "VIBE_WS_PORT", settings, "ws_port")
    _set_str(env, "VIBE_WS_TOKEN", settings, "ws_token")
    _set_float(env, "VIBE_VAD_THRESHOLD", settings, "vad_threshold")
    _set_int(env, "VIBE_VAD_MIN_SPEECH_MS", settings, "vad_min_speech_ms")
    _set_int(env, "VIBE_VAD_MIN_SILENCE_MS", settings, "vad_min_silence_ms")
    _set_int(env, "VIBE_VAD_MAX_SPEECH_MS", settings, "vad_max_speech_ms")
    _set_int(env, "VIBE_VAD_SPEECH_PAD_MS", settings, "vad_speech_pad_ms")
    _set_int(env, "VIBE_WAKE_SUPPRESS_MS", settings, "wake_suppress_ms")
    _set_int(env, "VIBE_WAKE_TIMEOUT_MS", settings, "wake_timeout_ms")
    _set_float(env, "VIBE_KWS_SCORE", settings, "kws_score")
    _set_float(env, "VIBE_KWS_THRESHOLD", settings, "kws_threshold")
    _set_int(env, "VIBE_KWS_TRAILING_BLANKS", settings, "kws_num_trailing_blanks")
    _set_str(env, "VIBE_STT_LANGUAGE", settings, "stt_language")
    _set_bool(env, "VIBE_STT_USE_ITN", settings, "stt_use_itn")
    _set_int(env, "VIBE_TTS_SID", settings, "tts_sid")
    _set_float(env, "VIBE_TTS_SPEED", settings, "tts_speed")
    _set_int(env, "VIBE_TTS_MAX_SENTENCES", settings, "tts_max_sentences")
    _set_int(env, "VIBE_NUM_THREADS", settings, "num_threads")
    _set_int(env, "VIBE_MAX_HISTORY", settings, "max_history_messages")
    if env.get("VIBE_KEYWORDS_FILE"):
        settings.keywords_file = _resolve_path(Path(env["VIBE_KEYWORDS_FILE"]), root)


def _resolve_keywords(
    current: Path,
    env: dict[str, str],
    data: dict,
    root: Path,
) -> Path:
    if env.get("VIBE_KEYWORDS_FILE"):
        return _resolve_path(Path(env["VIBE_KEYWORDS_FILE"]), root)
    kws = data.get("kws") or {}
    if kws.get("keywords_file"):
        return _resolve_path(Path(str(kws["keywords_file"])), root)
    local = root / "keywords.txt"
    if local.is_file():
        return local
    if current.is_file():
        return current
    return packaged_keywords_path()


def _resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _apply_dotenv(path: Path, env: dict[str, str]) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env.setdefault(key, value)


def _set_str(env: dict[str, str], key: str, settings: Settings, attr: str) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, env[key])


def _set_path(
    env: dict[str, str], key: str, settings: Settings, attr: str, root: Path
) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, _resolve_path(Path(env[key]), root))


def _set_int(env: dict[str, str], key: str, settings: Settings, attr: str) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, int(env[key]))


def _set_optional_int(
    env: dict[str, str], key: str, settings: Settings, attr: str
) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, int(env[key]))


def _set_float(env: dict[str, str], key: str, settings: Settings, attr: str) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, float(env[key]))


def _set_bool(env: dict[str, str], key: str, settings: Settings, attr: str) -> None:
    if key in env and env[key] != "":
        setattr(settings, attr, env[key].strip().lower() in {"1", "true", "yes", "on"})
