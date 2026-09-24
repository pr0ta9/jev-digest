"""Local documents for read_documents: files or folders of Markdown, text, HTML and PDF, loaded as pages for judging."""

from __future__ import annotations

import os
from pathlib import Path

from .fetch import extract, pdf_text

TEXT = {".md", ".markdown", ".txt", ".rst"}
HTML = {".html", ".htm"}
PDF = {".pdf"}
MAX_FILES = 50

SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv"}


def walk(folder: Path) -> list[Path]:
    found: list[Path] = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        found += [Path(root) / name for name in sorted(names) if Path(name).suffix.lower() in TEXT | HTML | PDF]
        if len(found) >= MAX_FILES:
            break
    return found


def expand(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files += walk(path)
        else:
            files.append(path)
    return list(dict.fromkeys(p.resolve() for p in files))[:MAX_FILES]


def load_document(path: Path, max_chars: int) -> dict:
    url = str(path)
    if not path.exists():
        return {"url": url, "title": path.name, "status": "not found"}
    suffix = path.suffix.lower()
    if suffix not in TEXT | HTML | PDF:
        return {"url": url, "title": path.name, "status": "unsupported file type"}
    try:
        if suffix in PDF:
            text, title = pdf_text(path.read_bytes(), max_chars)
        elif suffix in HTML:
            text, title = extract(path.read_text(encoding="utf-8-sig", errors="replace"), max_chars)
        else:
            text, title = path.read_text(encoding="utf-8-sig", errors="replace")[:max_chars], ""
    except Exception as exc:  # noqa: BLE001 - a malformed file is a per-file outcome
        return {"url": url, "title": path.name, "status": f"error: {type(exc).__name__}"}
    if not text.strip():
        return {"url": url, "title": path.name, "status": "no text"}
    return {"url": url, "title": title or path.name, "text": text, "status": "read", "truncated": len(text) >= max_chars}
