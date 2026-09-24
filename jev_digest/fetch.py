"""Fetch stage: parallel HTTP-first wave with a hard cutoff and a JSON cache by URL, plus an optional headless-browser
second wave for pages that returned no text or hit the cutoff. The pipeline overlaps the second wave with judging."""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
RETRYABLE = ("cutoff", "insufficient text")
EXTRACT_BUDGET_S = 4.0
PDF_MAX_BYTES = 20_000_000
HTML_MAX_CHARS = 3_000_000


def cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def extract(html: str, max_chars: int) -> tuple[str, str]:
    import trafilatura
    text = trafilatura.extract(html, include_links=False, include_tables=True, output_format="markdown") or ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip() if m else ""
    return text[:max_chars], title


def pdf_text(data: bytes, max_chars: int) -> tuple[str, str]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages, size = [], 0
    for page in reader.pages:
        # PDF lines break mid-sentence; keep blank-line paragraph breaks only.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", page.extract_text() or "")
        pages.append(text)
        size += len(text)
        if size >= max_chars:
            break
    title = str((reader.metadata or {}).get("/Title") or "")
    return "\n\n".join(pages)[:max_chars], title


def _store(cache_dir: Path, url: str, title: str, text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, url).write_text(json.dumps({"url": url, "title": title, "text": text, "extraction_format": "markdown-v1", "fetched_at": time.time()},
                                                     ensure_ascii=False), encoding="utf-8")


async def _fetch_http(session, url: str, cutoff: float, max_chars: int, cache_dir: Path) -> dict:
    started = time.perf_counter()
    try:
        resp = await asyncio.wait_for(session.get(url), timeout=cutoff)
    except asyncio.TimeoutError:
        return {"url": url, "status": "cutoff", "seconds": round(time.perf_counter() - started, 3)}
    except Exception as exc:  # noqa: BLE001 - per-URL outcome
        return {"url": url, "status": f"error: {type(exc).__name__}", "seconds": round(time.perf_counter() - started, 3)}
    ctype = resp.headers.get("content-type", "")
    status = f"fetched http {resp.status_code}"
    try:
        if "pdf" in ctype:
            if len(resp.content) > PDF_MAX_BYTES:
                return {"url": url, "status": "skipped: pdf too large", "seconds": round(time.perf_counter() - started, 3)}
            work = (pdf_text, resp.content)
        elif "html" in ctype or "xml" in ctype:
            work = (extract, resp.text[:HTML_MAX_CHARS])
        else:
            return {"url": url, "status": f"skipped: {ctype[:40]}", "seconds": round(time.perf_counter() - started, 3)}
        # Extraction is CPU-bound and runs after the download cutoff; in the event loop one whole-novel page froze a digest for 208 s.
        try:
            text, title = await asyncio.wait_for(asyncio.to_thread(work[0], work[1], max_chars), timeout=EXTRACT_BUDGET_S)
        except asyncio.TimeoutError:
            return {"url": url, "status": "cutoff: text extraction", "seconds": round(time.perf_counter() - started, 3)}
        if "pdf" in ctype:
            title = title or unquote(urlparse(url).path.rsplit("/", 1)[-1])
            status = "fetched pdf"
    except Exception as exc:  # noqa: BLE001 - a malformed PDF or page is a per-URL outcome
        return {"url": url, "status": f"error: {type(exc).__name__}", "seconds": round(time.perf_counter() - started, 3)}
    if len(text) < 200:
        return {"url": url, "status": "insufficient text", "seconds": round(time.perf_counter() - started, 3)}
    _store(cache_dir, url, title, text)
    return {"url": url, "title": title, "text": text, "status": status, "seconds": round(time.perf_counter() - started, 3)}


async def http_wave(urls: list[str], cache_dir: Path, cutoff: float, max_chars: int, use_cache: bool) -> dict[str, dict]:
    """Cache hits plus one parallel HTTP pass. Returns {url: page-or-failure} for every url."""
    from curl_cffi.requests import AsyncSession
    out: dict[str, dict] = {}
    todo = []
    for u in urls:
        cp = cache_path(cache_dir, u)
        if use_cache and cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            if d.get("extraction_format") == "markdown-v1":
                out[u] = {"url": u, "title": d["title"], "text": d["text"], "status": "cached", "seconds": 0.0}
                continue
        todo.append(u)
    if todo:
        async with AsyncSession(allow_redirects=True, timeout=cutoff + 1, headers=HEADERS, impersonate="chrome") as session:
            for r in await asyncio.gather(*(_fetch_http(session, u, cutoff, max_chars, cache_dir) for u in todo)):
                out[r["url"]] = r
    return out


def retryable(out: dict[str, dict]) -> list[str]:
    return [u for u, r in out.items() if not r.get("text") and r["status"] in RETRYABLE]


async def browser_wave(urls: list[str], budget: float, max_chars: int, cache_dir: Path) -> dict[str, dict]:
    """One headless Chromium, all pages in parallel, whole wave time-boxed. Returns {url: page-or-failure}."""
    started = time.perf_counter()
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {u: {"url": u, "status": "browser fallback unavailable (pip install playwright)", "seconds": 0.0} for u in urls}
    results: dict[str, dict] = {}

    async def one(browser, url):
        t = time.perf_counter()
        page = await browser.new_page(user_agent=HEADERS["User-Agent"])
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=int(budget * 1000))
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:  # noqa: BLE001 - best effort
                pass
            html = (await page.content())[:HTML_MAX_CHARS]
            text, title = await asyncio.wait_for(asyncio.to_thread(extract, html, max_chars), timeout=EXTRACT_BUDGET_S)
            if len(text) < 200:
                results[url] = {"url": url, "status": "browser: insufficient text", "seconds": round(time.perf_counter() - t, 3)}
                return
            _store(cache_dir, url, title, text)
            results[url] = {"url": url, "title": title, "text": text, "status": "fetched via browser", "seconds": round(time.perf_counter() - t, 3)}
        except Exception as exc:  # noqa: BLE001
            results[url] = {"url": url, "status": f"browser error: {type(exc).__name__}", "seconds": round(time.perf_counter() - t, 3)}
        finally:
            await page.close()

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                await asyncio.wait_for(asyncio.gather(*(one(browser, u) for u in urls)), timeout=budget + 3)
            except asyncio.TimeoutError:
                pass
            finally:
                await browser.close()
    except Exception as exc:  # noqa: BLE001
        return {u: {"url": u, "status": f"browser launch failed: {type(exc).__name__}", "seconds": round(time.perf_counter() - started, 3)} for u in urls}
    return {u: results.get(u, {"url": u, "status": "browser: wave budget exceeded", "seconds": round(time.perf_counter() - started, 3)}) for u in urls}
