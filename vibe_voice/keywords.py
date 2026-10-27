"""Parse sherpa-onnx keyword files (ppinyin tokens plus optional scores)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vibe_voice.errors import KeywordFileError

_BOOST = re.compile(r"^:([0-9]*\.?[0-9]+)$")
_THRESHOLD = re.compile(r"^#([0-9]*\.?[0-9]+)$")


@dataclass(frozen=True)
class Keyword:
    """One wake word in the form sherpa-onnx reads."""

    tokens: tuple[str, ...]
    phrase: str
    boost: float | None = None
    threshold: float | None = None

    def to_sherpa_line(self) -> str:
        line = " ".join(self.tokens)
        if self.boost is not None:
            line += f" :{_format_score(self.boost)}"
        if self.threshold is not None:
            line += f" #{_format_score(self.threshold)}"
        if self.phrase:
            line += f" @{self.phrase}"
        return line


def packaged_keywords_path() -> Path:
    return Path(__file__).resolve().parent / "keywords.txt"


def parse_keywords(text: str) -> list[Keyword]:
    """Parse a keywords file. Blank lines and ``# `` comments are ignored."""

    entries: list[Keyword] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("# "):
            continue
        parts = stripped.split()
        if parts[0].startswith("#") and _THRESHOLD.match(parts[0]) is None:
            continue
        entries.append(_parse_line(parts, lineno))
    if not entries:
        raise KeywordFileError("关键词文件里没有可用的唤醒词")
    return entries


def load_keywords(path: Path) -> list[Keyword]:
    if not path.is_file():
        raise KeywordFileError(f"找不到关键词文件：{path}")
    return parse_keywords(path.read_text(encoding="utf-8"))


def match_phrase(result: str, entries: list[Keyword]) -> str:
    """Map a spotter result back to the ``@`` phrase when we know it."""

    text = result.strip()
    if not text:
        return ""
    for entry in entries:
        tokens = " ".join(entry.tokens)
        if text == entry.phrase or text == tokens:
            return entry.phrase or text
    return text


def _parse_line(parts: list[str], lineno: int) -> Keyword:
    phonetic: list[str] = []
    boost: float | None = None
    threshold: float | None = None
    phrase: str | None = None
    for token in parts:
        if token.startswith("@"):
            if phrase is not None:
                raise KeywordFileError(f"第 {lineno} 行有多个 @原文")
            phrase = token[1:]
            if not phrase:
                raise KeywordFileError(
                    f"第 {lineno} 行：@ 后面要直接写唤醒词，例如 @你好小助"
                )
        elif token.startswith(":"):
            match = _BOOST.match(token)
            if match is None or boost is not None:
                raise KeywordFileError(
                    f"第 {lineno} 行的加速分无效：{token}（应为 :1.5，冒号后不要空格）"
                )
            boost = float(match.group(1))
        elif token.startswith("#"):
            match = _THRESHOLD.match(token)
            if match is None or threshold is not None:
                raise KeywordFileError(
                    f"第 {lineno} 行的阈值无效：{token}（应为 #0.25，# 后不要空格）"
                )
            threshold = float(match.group(1))
        else:
            if boost is not None or threshold is not None or phrase is not None:
                raise KeywordFileError(
                    f"第 {lineno} 行：拼音 token 必须写在 :、#、@ 之前"
                )
            phonetic.append(token)
    if not phonetic:
        raise KeywordFileError(f"第 {lineno} 行没有拼音 token")
    return Keyword(tuple(phonetic), phrase or "", boost, threshold)


def _format_score(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"
