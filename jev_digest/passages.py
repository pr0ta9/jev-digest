"""Mechanical passage segmentation: 2-4 sentences, roughly 60-120 words (character-based sizing for CJK text)."""

from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT.split(text) if p.strip()]


def size(sentence: str) -> int:
    return max(len(sentence.split()), len(sentence) // 4)


def passages(text: str, target: int = 60, hard: int = 120, max_sentences: int = 4, min_size: int = 12) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    current_size = 0
    for s in sentences(text):
        if current and (current_size + size(s) > hard or len(current) >= max_sentences or current_size >= target):
            out.append(" ".join(current))
            current, current_size = [], 0
        current.append(s)
        current_size += size(s)
    if current:
        out.append(" ".join(current))
    return [p for p in out if size(p) >= min_size]


def contextual_passages(text: str) -> list[dict]:
    sections, body, out = [], [], []

    def flush():
        out.extend({"text": p, "section": " > ".join(t for _, t in sections)} for p in passages("\n".join(body), min_size=1))
        body.clear()

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            flush()
            depth = len(heading[1])
            sections = [(d, t) for d, t in sections if d < depth]
            sections.append((depth, heading[2]))
        else:
            body.append(line)
    flush()
    return out
