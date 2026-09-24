"""Minimal TypeSafe System One client: Choice questions only, strict response validation, one retry on an inconsistent answer."""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

import httpx


class JevError(RuntimeError):
    pass


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    if not 1 <= len(criteria) <= 255:
        raise ValueError(f"Choice needs 1-255 options, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def validate(result: Any, questions: dict, model: str) -> None:
    """Reject anything that is not a complete, self-consistent set of Choice answers."""
    if not isinstance(result, dict) or result.get("model") != model:
        raise ValueError("model missing or differs from the requested model")
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("answers do not match the questions")
    for key, question in questions.items():
        a = answers[key]
        probs = a.get("probabilities") if isinstance(a, dict) else None
        if a.get("type") != "choice" or not isinstance(probs, dict) or set(probs) != set(question["criteria"]):
            raise ValueError(f"malformed answer for {key}")
        if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values()):
            raise ValueError(f"bad probabilities for {key}")
        chosen = a.get("choice")
        if chosen not in probs:
            raise ValueError(f"illegal choice for {key}")
        if probs[chosen] < max(probs.values()) - 1e-9:
            raise ValueError(f"choice is not the highest-probability option for {key}")


class JevClient:
    """Async client with a concurrency cap and running accounting. Not thread-safe; one per pipeline run."""

    def __init__(self, api_key: str, model: str = "jev-1.13.0", url: str = "https://api.typesafe.ai/v1/systemone",
                 concurrency: int = 12, timeout: float = 30.0, trace_dir=None):
        if not api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self.api_key, self.model, self.url = api_key, model, url
        self._sem = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.trace_dir = trace_dir
        self._account_error = None
        self.stats = {"requests": 0, "questions": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "invalid": 0, "failed": 0}

    async def decide(self, state: Any, questions: dict[str, dict]) -> dict[str, dict] | None:
        """Return {question_id: {choice, confidence, probabilities}} or None when the API failed twice."""
        payload = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(2):
            async with self._sem:
                if self._account_error:
                    raise JevError(self._account_error)
                started = time.perf_counter()
                self.stats["requests"] += 1
                self.stats["questions"] += len(questions)
                try:
                    resp = await self._http.post(self.url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
                except httpx.HTTPError:
                    self.stats["seconds"] += time.perf_counter() - started
                    continue
                self.stats["seconds"] += time.perf_counter() - started
                if self.trace_dir is not None:
                    self._trace(payload, resp)
                if resp.status_code != 200:
                    if resp.status_code in (401, 402, 403):
                        self.stats["failed"] += 1
                        self._account_error = f"TypeSafe HTTP {resp.status_code}: account authentication, access or billing prevents research. Restore API access before retrying."
                        raise JevError(self._account_error)
                    if resp.status_code not in (408, 429, 500, 502, 503, 504, 529):
                        break
                    await asyncio.sleep(0.5)
                    continue
                try:
                    result = resp.json()
                    validate(result, questions, self.model)
                except ValueError:
                    self.stats["invalid"] += 1
                    continue
                usage = result.get("usage") or {}
                self.stats["input_tokens"] += int(usage.get("input_tokens") or 0)
                self.stats["output_tokens"] += int(usage.get("output_tokens") or 0)
                return result["answers"]
        self.stats["failed"] += 1
        return None

    def _trace(self, payload, resp) -> None:
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            n = self.stats["requests"]
            (self.trace_dir / f"{n:04d}.request.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            (self.trace_dir / f"{n:04d}.response.json").write_bytes(resp.content)
        except OSError:
            pass

    async def close(self) -> None:
        await self._http.aclose()
