"""Download sherpa-onnx models used by the pipeline. Not used by tests."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from vibe_voice.config import Settings
from vibe_voice.engine import kws_ready, stt_ready, tts_ready, vad_ready

_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# Official sherpa-onnx release assets. Archives land under models/ with the
# upstream directory names. None of these are multi-gigabyte, but they are
# large enough to stay behind this command instead of pip or the test suite.
ASSETS = (
    {
        "name": "kws",
        "url": f"{_RELEASE}/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2",
        "kind": "archive",
        "ready": kws_ready,
    },
    {
        "name": "stt",
        "url": f"{_RELEASE}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "kind": "archive",
        "ready": stt_ready,
    },
    {
        "name": "vad",
        "url": f"{_RELEASE}/asr-models/silero_vad.onnx",
        "kind": "file",
        "ready": vad_ready,
    },
    {
        "name": "tts",
        "url": f"{_RELEASE}/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2",
        "kind": "archive",
        "ready": tts_ready,
    },
)


def download_models(
    settings: Settings,
    *,
    only: set[str] | None = None,
    force: bool = False,
    skip_tts: bool = False,
) -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        name = asset["name"]
        if only is not None and name not in only:
            continue
        if skip_tts and name == "tts":
            print("跳过 TTS")
            continue
        if asset["ready"](settings) and not force:
            print(f"已存在，跳过 {name}")
            continue
        partial = settings.models_dir / (Path(asset["url"]).name + ".partial")
        print(f"下载 {name}：{asset['url']}")
        _download(asset["url"], partial)
        if asset["kind"] == "archive":
            print(f"解压 {partial.name}")
            with tarfile.open(partial, "r:bz2") as archive:
                try:
                    archive.extractall(settings.models_dir, filter="data")
                except TypeError:  # Python 3.11 without the filter argument
                    archive.extractall(settings.models_dir)
            partial.unlink(missing_ok=True)
        else:
            settings.vad_model.parent.mkdir(parents=True, exist_ok=True)
            partial.replace(settings.vad_model)
        print(f"完成 {name}")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "vibe-voice"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        got = 0
        next_mark = 0
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                got += len(chunk)
                if total and got * 100 // total >= next_mark:
                    print(f"  {name_of(destination)} {got * 100 // total}%", flush=True)
                    next_mark += 10
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"下载结果为空：{url}")


def name_of(path: Path) -> str:
    return path.name
