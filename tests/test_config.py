from pathlib import Path

from vibe_voice.cli import main
from vibe_voice.config import load_settings


def test_env_overrides_toml_and_dotenv(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        """
[openai]
base_url = "http://toml.example/v1"
api_key = "toml-key"
model = "toml-model"

[server]
port = 9000

[tts]
sid = 4
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "OPENAI_MODEL=dotenv-model\nVIBE_WS_PORT=1111\n",
        encoding="utf-8",
    )
    settings = load_settings(
        environ={"OPENAI_MODEL": "env-model", "VIBE_TTS_SID": "3"},
        root=tmp_path,
    )
    assert settings.openai_base_url == "http://toml.example/v1"
    assert settings.openai_api_key == "toml-key"
    assert settings.openai_model == "env-model"
    assert settings.ws_port == 1111  # .env fills the gap left by the process env
    assert settings.tts_sid == 3
    assert settings.keywords_file.name == "keywords.txt"


def test_keywords_command_prints_default_phrase(capsys):
    assert main(["keywords"]) == 0
    out = capsys.readouterr().out
    assert "你好小助" in out
    assert "n ǐ h ǎo x iǎo zh ù" in out


def test_check_reports_missing_models_without_downloading(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VIBE_MODELS_DIR", raising=False)
    monkeypatch.delenv("VIBE_KEYWORDS_FILE", raising=False)
    monkeypatch.delenv("VIBE_CONFIG", raising=False)
    assert main(["check"]) == 2
    out = capsys.readouterr().out
    assert "SenseVoice" in out
    assert "未找到" in out
    assert "你好小助" in out
