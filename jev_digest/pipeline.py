"""End-to-end run with per-stage timings and accounting. The browser second wave overlaps Jev round 1 on the pages
that arrived in the HTTP wave, so slow pages add only their own judging time. Writes a run directory so passages
can be opened later."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import tiktoken

from .aspects import aspects_from_query
from .client import JevClient
from .config import Settings
from .digest import build_blocks, render_evidence
from .documents import HTML, MAX_FILES, PDF, TEXT, expand, load_document
from .fetch import browser_wave, http_wave, retryable
from .judge import KEEP, group_by_aspect, judge_aspects, judge_coverage, judge_relevance, judge_scope
from .passages import contextual_passages
from .search import link_only, on_domain, search, search_many, select_sources

_ENC = tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _prepare(page: dict, index: int) -> dict:
    page["id"] = f"d{index}"
    page["passages"] = [dict(p, id=f"d{index}p{j}", page=page["id"]) for j, p in enumerate(contextual_passages(page["text"]))]
    return page


async def judge_kept(client, query: str, aspects: dict[str, str], kept: list[dict], cfg: Settings, timings: dict):
    t = time.perf_counter()
    (kept, scope_audit), _ = await asyncio.gather(judge_scope(client, query, kept, cfg.questions_per_request),
                                                  judge_aspects(client, query, aspects, kept, cfg.questions_per_request))
    timings["scope_check"] = round(time.perf_counter() - t, 3)
    groups = group_by_aspect(kept)
    t = time.perf_counter()
    verdicts, r2_requests = await judge_coverage(client, query, aspects, groups, cfg.representatives_per_aspect, cfg.questions_per_request)
    timings["jev_round2"] = round(time.perf_counter() - t, 3)
    return kept, scope_audit, groups, verdicts, r2_requests


async def run_digest(query: str, search_query: str | None = None, urls: list[str] | None = None, top: int = 10,
                     settings: Settings | None = None, use_cache: bool = True, trace: bool = False,
                     search_queries: list[str] | None = None, allowed_domains: list[str] | None = None,
                     blocked_domains: list[str] | None = None) -> dict:
    cfg = settings or Settings()
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = cfg.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    t_all = time.perf_counter()

    t = time.perf_counter()
    if allowed_domains and not urls:
        search_queries = [f"site:{d} {q}" for q in (search_queries or [search_query or query]) for d in allowed_domains]
    if search_queries and not urls:
        results, search_meta = await search_many(search_queries, top, max(cfg.max_pages, top), cfg.searxng_url)
    else:
        results, search_meta = await search(search_query or query, top, cfg.searxng_url, urls)
    timings["search"] = round(time.perf_counter() - t, 3)
    if allowed_domains:
        results = [r for r in results if on_domain(r["url"], allowed_domains)]
    if blocked_domains:
        results = [r for r in results if not on_domain(r["url"], blocked_domains)]
    if not results:
        return {"run_id": run_id, "error": "no search results", "search": search_meta, "timings": timings}
    client = JevClient(cfg.api_key, cfg.model, cfg.api_url, cfg.jev_concurrency, trace_dir=out_dir / "trace" if trace else None)
    try:
        t = time.perf_counter()
        results, selection = await select_sources(client, query, results)
        links = [r for r in results if link_only(r["url"])][:5]
        results = [r for r in results if not link_only(r["url"])]
        timings["source_selection"] = round(time.perf_counter() - t, 3)
        order = [r["url"] for r in results]
        metadata = {r["url"]: r for r in results}
        t = time.perf_counter()
        fetched = await http_wave(order, cfg.cache_dir, cfg.fetch_cutoff_s, cfg.max_chars_per_page, use_cache)
        timings["fetch_http_wave"] = round(time.perf_counter() - t, 3)
        readable = sum(1 for u in order if fetched[u].get("text"))
        retry = [u for u in retryable(fetched) if cfg.browser_fallback and
                 (readable < cfg.browser_min_pages or metadata[u].get("source_role") in ("PRIMARY", "REFERENCE"))]
        timings["browser_wave_skipped_urls"] = len(retryable(fetched)) - len(retry)
        browser_task = asyncio.create_task(browser_wave(retry, cfg.browser_budget_s, cfg.max_chars_per_page, cfg.cache_dir)) if retry else None
        aspects = aspects_from_query(query)
        pages = [_prepare(dict(metadata[u], **fetched[u]), i) for i, u in enumerate(order) if fetched[u].get("text")]
        t = time.perf_counter()
        kept, r1 = await judge_relevance(client, query, pages, cfg.questions_per_request)
        timings["jev_round1"] = round(time.perf_counter() - t, 3)
        if browser_task is not None:
            t = time.perf_counter()
            late = await browser_task
            fetched.update(late)
            timings["fetch_browser_wave_extra_wait"] = round(time.perf_counter() - t, 3)
            timings["browser_urls"] = len(retry)
            late_pages = [_prepare(dict(metadata[u], **fetched[u]), i) for i, u in enumerate(order) if u in late and late[u].get("text")]
            if late_pages:
                t = time.perf_counter()
                kept2, r1b = await judge_relevance(client, query, late_pages, cfg.questions_per_request)
                timings["jev_round1_late_pages"] = round(time.perf_counter() - t, 3)
                kept += kept2
                r1 = {"requests": r1["requests"] + r1b["requests"], "pages_on_topic": r1["pages_on_topic"] + r1b["pages_on_topic"],
                      "pages_judged": r1["pages_judged"] + r1b["pages_judged"]}
                pages += late_pages
        kept, scope_audit, groups, verdicts, r2_requests = await judge_kept(client, query, aspects, kept, cfg, timings)
    finally:
        await client.close()

    pages_by_id = {p["id"]: p for p in pages}
    blocks = build_blocks(aspects, groups, verdicts, pages_by_id, cfg.representatives_per_aspect)
    md_evidence = render_evidence(query, aspects, blocks, page_outcomes(order, fetched, pages, {p["id"] for p in kept}), links)
    timings["total"] = round(time.perf_counter() - t_all, 3)
    result = {
        "run_id": run_id, "query": query, "search_query": search_query or query, "search_queries": search_queries or [], "aspects": aspects, "search": search_meta,
        "fetch": [{k: v for k, v in fetched[u].items() if k not in ("text", "passages")} for u in order],
        "source_selection": selection,
        "scope_check": scope_audit,
        "timings": timings,
        "counts": {"urls": len(order), "pages": len(pages), "passages": sum(len(p["passages"]) for p in pages), "kept": len(kept),
                   "aspects": len(aspects), "representatives": sum(len(b["representatives"]) for b in blocks.values()),
                   "covered": sum(len(b["also_reported_by"]) for b in blocks.values()), "adds": sum(len(b["adds"]) for b in blocks.values()),
                   "conflicts": sum(len(b["conflicts"]) for b in blocks.values()), "pages_on_topic": r1["pages_on_topic"]},
        "tokens": {"raw_pages": sum(tokens(p["text"]) for p in pages), "kept_passages": sum(tokens(p["text"]) for p in kept),
                   "digest_evidence": tokens(md_evidence)},
        "jev": dict(client.stats, round1_requests=r1["requests"], round2_requests=r2_requests),
        "digest_evidence": md_evidence, "blocks": blocks,
    }
    (out_dir / "digest_evidence.md").write_text(md_evidence, encoding="utf-8")
    (out_dir / "passages.json").write_text(json.dumps({x["id"]: {"text": x["text"], "section": x.get("section", ""), "url": p["url"], "title": p.get("title", "")}
                                                       for p in pages for x in p["passages"]}, ensure_ascii=False), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "digest_evidence"},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    return result


async def run_read(paths: list[str], question: str, settings: Settings | None = None) -> dict:
    """Judge local documents against a question with the same chain as web digests; the file path stands in for the URL."""
    cfg = settings or Settings()
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = cfg.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    t_all = time.perf_counter()
    t = time.perf_counter()
    docs = [dict(load_document(p, cfg.max_chars_per_document), source_role="PRIMARY") for p in expand(paths)]
    timings["load"] = round(time.perf_counter() - t, 3)
    files = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    pages = [_prepare(dict(d), i) for i, d in enumerate(docs) if d.get("text")]
    if not pages:
        return {"run_id": run_id, "error": "no readable files", "paths": paths, "supported": sorted(TEXT | HTML | PDF),
                "files": files, "timings": timings}
    aspects = aspects_from_query(question)
    client = JevClient(cfg.api_key, cfg.model, cfg.api_url, cfg.jev_concurrency)
    try:
        t = time.perf_counter()
        kept, _ = await judge_relevance(client, question, pages, cfg.questions_per_request)
        timings["jev_round1"] = round(time.perf_counter() - t, 3)
        kept, scope_audit, groups, verdicts, _ = await judge_kept(client, question, aspects, kept, cfg, timings)
    finally:
        await client.close()
    blocks = build_blocks(aspects, groups, verdicts, {p["id"]: p for p in pages}, cfg.representatives_per_aspect)
    order = [d["url"] for d in docs]
    evidence = render_evidence(question, aspects, blocks, page_outcomes(order, {d["url"]: d for d in docs}, pages, {p["id"] for p in kept}),
                               local=True)
    notices = []
    unread = [d for d in docs if not d.get("text")]
    if unread and any(block["representatives"] for block in blocks.values()):
        notices += ["Files not read:"] + [f"- {d['url']}: {d['status']}" for d in unread]
    if len(docs) == MAX_FILES:
        notices.append(f"Only the first {MAX_FILES} files were read; name subfolders or files to read others.")
    notices += [f"Cut at {cfg.max_chars_per_document} characters: {d['url']}" for d in docs if d.get("truncated")]
    if notices:
        evidence_lines = evidence.split("\n")
        evidence = "\n".join(evidence_lines[:-1] + notices + evidence_lines[-1:])
    timings["total"] = round(time.perf_counter() - t_all, 3)
    (out_dir / "passages.json").write_text(json.dumps({x["id"]: {"text": x["text"], "section": x.get("section", ""), "url": p["url"], "title": p.get("title", "")}
                                                       for p in pages for x in p["passages"]}, ensure_ascii=False), encoding="utf-8")
    result = {"run_id": run_id, "question": question, "paths": paths, "files": files, "aspects": aspects, "scope_check": scope_audit,
              "timings": timings, "counts": {"files": len(docs), "read": len(pages), "passages": sum(len(p["passages"]) for p in pages),
                                             "kept": len(kept)},
              "jev": dict(client.stats), "evidence": evidence}
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def page_outcomes(order: list[str], fetched: dict, pages: list[dict], final_ids: set[str]) -> list[dict]:
    by_url = {p["url"]: p for p in pages}
    out = []
    for url in order:
        page = by_url.get(url)
        if page is None:
            out.append({"url": url, "title": fetched[url].get("title", ""), "outcome": f"unreadable: {fetched[url]['status']}"})
            continue
        n = len(page["passages"])
        relevant = [p for p in page["passages"] if p.get("level") in KEEP]
        final = sum(p["id"] in final_ids for p in page["passages"])
        if page.get("suitable") is False:
            outcome = f"read {n} passages; page judged unsuitable for the question"
        elif not relevant:
            outcome = f"read {n} passages; none judged relevant"
        elif not final:
            outcome = f"read {n} passages; {len(relevant)} relevant, all removed by the scope check"
        else:
            outcome = f"read {n} passages; {final} kept"
        out.append({"url": url, "title": page.get("title", ""), "outcome": outcome})
    return out


async def run_browse(url: str, purpose: str, settings: Settings | None = None, use_cache: bool = True) -> dict:
    """Read one known page and keep only its passages that bear on the purpose: no search, source selection, scope or merging."""
    cfg = settings or Settings()
    timings: dict = {}
    t = time.perf_counter()
    fetched = await http_wave([url], cfg.cache_dir, cfg.fetch_cutoff_s, cfg.max_chars_per_page, use_cache)
    if cfg.browser_fallback and retryable(fetched):
        fetched.update(await browser_wave([url], cfg.browser_budget_s, cfg.max_chars_per_page, cfg.cache_dir))
    timings["fetch"] = round(time.perf_counter() - t, 3)
    if not fetched[url].get("text"):
        return {"url": url, "evidence": f"Could not read {url}: {fetched[url]['status']}.", "timings": timings}
    page = _prepare(dict(fetched[url], url=url), 0)
    client = JevClient(cfg.api_key, cfg.model, cfg.api_url, cfg.jev_concurrency)
    try:
        t = time.perf_counter()
        kept, _ = await judge_relevance(client, purpose, [page], cfg.questions_per_request)
        timings["jev_relevance"] = round(time.perf_counter() - t, 3)
    finally:
        await client.close()
    lines = [f"Page: {page.get('title') or url}", url, ""]
    for p in sorted(kept, key=lambda p: int(p["id"].rsplit("p", 1)[1])):
        lines += [f"[{p['id']}] {p.get('section', '')}".rstrip(), p["text"], ""]
    if not kept:
        lines.append(f"Read {len(page['passages'])} passages; none bear on: {purpose}")
    lines.append("Passages above are original text from this URL; cite it.")
    return {"url": url, "evidence": "\n".join(lines), "timings": timings, "jev": dict(client.stats),
            "counts": {"passages": len(page["passages"]), "kept": len(kept)}}


def open_passage(run_id: str, passage_id: str, settings: Settings | None = None) -> dict | None:
    cfg = settings or Settings()
    path = cfg.runs_dir / run_id / "passages.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    entry = saved.get(passage_id)
    if entry is None:
        return None
    doc, number = passage_id.rsplit("p", 1)
    entry["context"] = [dict(saved[key], id=key) for i in range(max(0, int(number) - 1), int(number) + 2)
                        if (key := f"{doc}p{i}") in saved]
    return entry
