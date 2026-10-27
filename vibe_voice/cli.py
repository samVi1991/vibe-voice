"""Command line: microphone pipeline, WebSocket server, model download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vibe_voice.config import Settings, load_settings
from vibe_voice.engine import (
    build_conversation_engine,
    kws_ready,
    load_keyword_entries,
    load_keyword_spotter,
    load_silero,
    missing_required,
    stt_ready,
    tts_incomplete,
    tts_ready,
    vad_ready,
)
from vibe_voice.errors import ModelNotReady, VibeVoiceError
from vibe_voice.keywords import Keyword
from vibe_voice.pipeline import (
    MicPipeline,
    microphone_chunks,
    segmenter_from_settings,
    wake_sample_counts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vibe-voice",
        description="中文语音助手：电脑麦克风流水线，或只启动 WebSocket 供后续设备接入。",
    )
    parser.add_argument("--config", type=Path, help="TOML 配置文件，默认 ./config.toml（如果存在）")
    parser.add_argument("--host", help="WebSocket 监听地址，覆盖配置")
    parser.add_argument("--port", type=int, help="WebSocket 端口，覆盖配置")
    parser.add_argument(
        "--websocket-only",
        action="store_true",
        help="只启动 WebSocket，不打开麦克风",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("mic", help="电脑麦克风：唤醒 → 识别 → 大模型 → 语音")
    serve = sub.add_parser("serve", help="只启动 WebSocket 音频会话")
    serve.add_argument("--host", help="覆盖 VIBE_WS_HOST")
    serve.add_argument("--port", type=int, help="覆盖 VIBE_WS_PORT")

    download = sub.add_parser("download", help="下载 sherpa-onnx 模型（测试不会下载）")
    download.add_argument(
        "--only",
        help="逗号分隔：kws,stt,vad,tts",
    )
    download.add_argument("--skip-tts", action="store_true", help="不下载中文 TTS")
    download.add_argument("--force", action="store_true", help="已存在也重新下载")

    sub.add_parser("keywords", help="打印关键词文件里解析出的唤醒词")
    sub.add_parser("check", help="检查配置和模型文件，不加载模型")

    args = parser.parse_args(argv)
    if args.websocket_only and args.command not in (None, "serve"):
        parser.error("--websocket-only 只表示启动 WebSocket，不能和其它子命令一起用")
    command = args.command or ("serve" if args.websocket_only else "mic")
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    _apply_cli_overrides(settings, args)
    try:
        if command == "mic":
            return _cmd_mic(settings)
        if command == "serve":
            return _cmd_serve(settings)
        if command == "download":
            return _cmd_download(settings, args)
        if command == "keywords":
            return _cmd_keywords(settings)
        if command == "check":
            return _cmd_check(settings)
    except ModelNotReady as exc:
        print(exc, file=sys.stderr)
        return 2
    except VibeVoiceError as exc:
        print(exc, file=sys.stderr)
        return 1
    parser.error(f"未知命令：{command}")
    return 2


def readiness_report(settings: Settings) -> tuple[str, bool]:
    entries = _safe_keywords(settings)
    lines = [
        f"关键词文件：{settings.keywords_file}",
        "唤醒词：" + ("、".join(_display(entry) for entry in entries) if entries else "（无法解析）"),
        f"大模型：{settings.openai_base_url}  model={settings.openai_model}  key={'已设置' if settings.openai_api_key else '未设置'}",
        f"WebSocket：ws://{settings.ws_host}:{settings.ws_port}/v1/voice  以及 /xiaozhi",
        "模型目录：" + str(settings.models_dir),
        _mark("关键词模型", kws_ready(settings)),
        _mark("SenseVoice", stt_ready(settings)),
        _mark("Silero VAD", vad_ready(settings)),
        _mark("中文 TTS（可选）", tts_ready(settings)),
    ]
    if tts_incomplete(settings):
        lines.append("中文 TTS 文件不完整。")
    missing = missing_required(settings)
    if missing:
        lines.append("麦克风流水线还缺：")
        lines.extend(f"  {item}" for item in missing)
        lines.append("下载：vibe-voice download")
    elif entries:
        lines.append("麦克风流水线需要的模型文件都在。")
    if not entries:
        lines.append("关键词文件无法解析。")
    if not tts_ready(settings):
        lines.append("没有 TTS 模型时仍会返回文字，只是不播放语音。默认音色是 sherpa-onnx-vits-zh-ll 的说话人 2。")
    required_ok = bool(entries) and not missing and not tts_incomplete(settings)
    return "\n".join(lines), required_ok


def _cmd_mic(settings: Settings) -> int:
    report, ok = readiness_report(settings)
    print(report)
    if not ok:
        return 2
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:
        print(
            "无法加载电脑麦克风。需要 PortAudio（Debian/Ubuntu：sudo apt install libportaudio2）"
            f"以及 pip 包 sounddevice。{exc}",
            file=sys.stderr,
        )
        return 1
    pipeline = _build_mic_pipeline(settings)
    phrases = "、".join(_display(entry) for entry in load_keyword_entries(settings))
    print(f"对着麦克风说唤醒词（{phrases}），然后说一句话。Ctrl+C 结束。播放回复时不会同时听麦。")
    try:
        chunks = microphone_chunks(
            settings.mic_sample_rate,
            16000,
            settings.mic_block_ms,
            settings.mic_device,
        )
        for chunk in chunks:
            for line in pipeline.feed(chunk):
                print(line, flush=True)
    except KeyboardInterrupt:
        print("\n已停止")
        return 0
    except Exception as exc:
        print(f"麦克风出错：{exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_serve(settings: Settings) -> int:
    report, _ok = readiness_report(settings)
    print(report)
    if tts_incomplete(settings):
        return 2
    try:
        engine = build_conversation_engine(settings)
        classifier = load_silero(settings) if vad_ready(settings) else None
    except ImportError as exc:
        print(f"缺少 Python 依赖（sherpa-onnx 或其它）：{exc}", file=sys.stderr)
        return 1
    if classifier is None:
        print("没有 Silero 模型：WebSocket 的 auto/vad 模式不能靠静音断句，需要客户端发送结束。")
    if not tts_ready(settings):
        print("没有中文 TTS：会话会返回文字，不返回语音。")
    from vibe_voice.server import run_server

    print(
        f"WebSocket 监听 {settings.ws_host}:{settings.ws_port}。"
        "自有协议路径 /v1/voice，小智对话路径 /xiaozhi。Ctrl+C 结束。"
    )
    try:
        run_server(settings, engine, classifier)
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


def _cmd_download(settings: Settings, args: argparse.Namespace) -> int:
    from vibe_voice.download_models import download_models

    only = None
    if args.only:
        only = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = only - {"kws", "stt", "vad", "tts"}
        if unknown:
            print(f"不认识的模型：{', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    download_models(settings, only=only, force=args.force, skip_tts=args.skip_tts)
    print(readiness_report(settings)[0])
    return 0


def _cmd_keywords(settings: Settings) -> int:
    entries = load_keyword_entries(settings)
    print(f"文件：{settings.keywords_file}")
    for entry in entries:
        label = entry.phrase or " ".join(entry.tokens)
        extra = []
        if entry.boost is not None:
            extra.append(f"加速分 {entry.boost:g}")
        if entry.threshold is not None:
            extra.append(f"阈值 {entry.threshold:g}")
        suffix = f"（{'，'.join(extra)}）" if extra else ""
        print(f"{label}{suffix}")
        print(f"  {entry.to_sherpa_line()}")
    return 0


def _cmd_check(settings: Settings) -> int:
    text, ok = readiness_report(settings)
    print(text)
    return 0 if ok else 2


def _build_mic_pipeline(settings: Settings) -> MicPipeline:
    entries = load_keyword_entries(settings)
    try:
        kws = load_keyword_spotter(settings, entries)
        classifier = load_silero(settings)
        engine = build_conversation_engine(settings)
    except ImportError as exc:
        raise ModelNotReady(f"缺少 Python 依赖：{exc}。运行：pip install -e .") from exc
    suppress, timeout = wake_sample_counts(settings)
    player = _SoundDevicePlayer(settings.speaker_device)
    return MicPipeline(
        kws=kws,
        classifier=classifier,
        segmenter=segmenter_from_settings(settings),
        engine=engine,
        player=player,
        sample_rate=16000,
        wake_suppress_samples=suppress,
        wake_timeout_samples=timeout,
        max_history_messages=settings.max_history_messages,
    )


class _SoundDevicePlayer:
    def __init__(self, device: int | None) -> None:
        self.device = device

    def play(self, samples, sample_rate: int) -> None:
        import numpy as np
        import sounddevice as sd

        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        sd.play(audio, samplerate=sample_rate, device=self.device)
        sd.wait()


def _apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> None:
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    if host:
        settings.ws_host = host
    if port:
        settings.ws_port = port


def _safe_keywords(settings: Settings) -> list[Keyword]:
    try:
        return load_keyword_entries(settings)
    except VibeVoiceError:
        return []


def _display(entry: Keyword) -> str:
    return entry.phrase or " ".join(entry.tokens)


def _mark(name: str, ready: bool) -> str:
    return f"{name}：{'已找到' if ready else '未找到'}"


if __name__ == "__main__":
    raise SystemExit(main())
