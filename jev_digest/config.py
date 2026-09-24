"""Configuration from environment variables (a .env in the working directory is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _home() -> Path:
    return Path(os.environ.get("JEV_DIGEST_HOME", Path.home() / ".jev-digest"))


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("TYPESAFE_API_KEY", "").strip())
    api_url: str = field(default_factory=lambda: os.environ.get("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"))
    model: str = field(default_factory=lambda: os.environ.get("JEV_DIGEST_MODEL", "jev-1.13.0"))
    searxng_url: str = field(default_factory=lambda: os.environ.get("JEV_DIGEST_SEARXNG_URL", "http://localhost:8089"))
    home: Path = field(default_factory=_home)
    fetch_cutoff_s: float = field(default_factory=lambda: float(os.environ.get("JEV_DIGEST_FETCH_CUTOFF", "2.0")))
    browser_fallback: bool = field(default_factory=lambda: os.environ.get("JEV_DIGEST_BROWSER_FALLBACK", "1") not in ("0", "false", "no"))
    browser_budget_s: float = field(default_factory=lambda: float(os.environ.get("JEV_DIGEST_BROWSER_BUDGET", "3.0")))
    # Primary and reference sources are recovered even when secondary pages reach this threshold.
    browser_min_pages: int = field(default_factory=lambda: int(os.environ.get("JEV_DIGEST_BROWSER_MIN_PAGES", "6")))
    questions_per_request: int = 40
    max_pages: int = field(default_factory=lambda: int(os.environ.get("JEV_DIGEST_MAX_PAGES", "20")))
    jev_concurrency: int = 12
    representatives_per_aspect: int = field(default_factory=lambda: int(os.environ.get("JEV_DIGEST_REPRESENTATIVES", "3")))
    max_chars_per_page: int = 50000
    max_chars_per_document: int = 400000

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"
