import asyncio
import json

from jev_digest import digest, fetch, judge, pipeline, search
from jev_digest.aspects import aspects_from_query
from jev_digest.config import Settings


def registered_tools(monkeypatch, **fakes):
    from mcp.server.fastmcp import FastMCP
    from jev_digest import mcp_server
    captured = {}
    monkeypatch.setattr(FastMCP, "run", lambda self, *a, **k: captured.setdefault("server", self))
    for name, fake in fakes.items():
        monkeypatch.setattr(mcp_server, name, fake)
    mcp_server.main()
    return captured["server"]


def tools_of(server):
    return {t.name: t for t in asyncio.run(server.list_tools())}


def test_digest_takes_the_same_input_as_common_agent_web_search(monkeypatch):
    # Common agent web search tools take one keyword `query` plus optional domain filters; agents call new tools the same way.
    tools = tools_of(registered_tools(monkeypatch))
    schema = tools["digest"].inputSchema
    assert set(schema["properties"]) == {"query", "allowed_domains", "blocked_domains"}
    assert schema["required"] == ["query"]
    assert all(p.get("description") for p in schema["properties"].values())


def test_tool_descriptions_state_use_returns_and_errors(monkeypatch):
    # Agents choose tools from their descriptions, so each must say what it does, when to use it, what it returns and how it fails.
    tools = tools_of(registered_tools(monkeypatch))
    assert set(tools) == {"digest", "browse", "read_documents"}
    for tool in tools.values():
        assert "Use it" in tool.description and "Returns" in tool.description and "Errors" in tool.description
    assert "original page text" in tools["digest"].description


async def test_digest_passes_query_and_domain_filters_to_the_pipeline(monkeypatch):
    requests = []

    async def run(question, search_query=None, **kwargs):
        requests.append((question, kwargs))
        return {"digest_evidence": "Evidence", "run_id": "r", "counts": {"pages": 1, "kept": 1}, "timings": {"total": 0.1}}

    server = registered_tools(monkeypatch, run_digest=run)
    await server.call_tool("digest", {"query": "ETIAS fee 2026"})
    await server.call_tool("digest", {"query": "ETIAS fee 2026", "allowed_domains": ["europa.eu"], "blocked_domains": ["etias.com"]})
    assert requests == [("ETIAS fee 2026", {"search_queries": ["ETIAS fee 2026"], "allowed_domains": None, "blocked_domains": None}),
                        ("ETIAS fee 2026", {"search_queries": ["ETIAS fee 2026"], "allowed_domains": ["europa.eu"], "blocked_domains": ["etias.com"]})]


async def test_domain_filters_restrict_searches_and_results(tmp_path, monkeypatch):
    searched = []
    rows = [{"url": "https://travel-europe.europa.eu/etias", "title": "Official"}, {"url": "https://www.etias.com/fees", "title": "Agency"},
            {"url": "https://news.example/etias", "title": "News"}]

    async def many(queries, *args):
        searched.extend(queries)
        return rows, {"used": "fixture"}

    async def select(client, query, results):
        return [dict(r, source_role="PRIMARY") for r in results], [dict(r, decision="PRIMARY", usefulness="LIKELY") for r in results]

    downloaded = []

    async def download(urls, *args):
        downloaded.extend(urls)
        return {u: {"url": u, "status": "cutoff"} for u in urls}

    monkeypatch.setattr(pipeline, "search_many", many)
    monkeypatch.setattr(pipeline, "select_sources", select)
    monkeypatch.setattr(pipeline, "http_wave", download)
    monkeypatch.setattr(pipeline, "JevClient", lambda *a, **k: _Closing(Answers(lambda k, s: "YES")))
    cfg = Settings(home=tmp_path, browser_fallback=False)
    await pipeline.run_digest("ETIAS fee", search_queries=["ETIAS fee"], allowed_domains=["europa.eu"], settings=cfg)
    assert searched == ["site:europa.eu ETIAS fee"] and downloaded == ["https://travel-europe.europa.eu/etias"]
    searched.clear(); downloaded.clear()
    await pipeline.run_digest("ETIAS fee", search_queries=["ETIAS fee"], blocked_domains=["etias.com"], settings=cfg)
    assert searched == ["ETIAS fee"] and "https://www.etias.com/fees" not in downloaded and len(downloaded) == 2


async def test_browse_accepts_one_url_or_several_read_in_parallel(monkeypatch):
    # Some agents send one tool call at a time, so several pages must fit in one call to overlap.
    running, peak, requests = 0, 0, []

    async def browse(url, purpose):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        requests.append((url, purpose))
        return {"evidence": f"Evidence from {url}"}

    server = registered_tools(monkeypatch, run_browse=browse)
    tools = {t.name: t for t in await server.list_tools()}
    assert tools["browse"].inputSchema["required"] == ["url", "purpose"]
    await server.call_tool("browse", {"url": "https://a.example", "purpose": "fee"})
    result = await server.call_tool("browse", {"url": ["https://b.example", "https://c.example"], "purpose": "dates"})
    assert sorted(requests) == [("https://a.example", "fee"), ("https://b.example", "dates"), ("https://c.example", "dates")]
    assert peak == 2
    assert "Evidence from https://c.example" in json.dumps(result, default=str)


async def test_browse_requires_a_purpose(monkeypatch):
    # Unfiltered whole pages dominate the returned text and stay in the agent's context for every later round.
    server = registered_tools(monkeypatch)
    tools = {t.name: t for t in await server.list_tools()}
    assert tools["browse"].inputSchema["required"] == ["url", "purpose"]


def test_relevance_requests_do_not_ask_the_topic_of_every_passage():
    # The topic answer for rejected passages was discarded; asking it only for kept passages halves the questions.
    page = {"id": "d0", "url": "https://x.example", "passages": [{"id": f"d0p{i}", "text": "t"} for i in range(50)]}
    requests = list(judge.relevance_requests("q", page, 40))
    assert [len(ids) for _, _, ids in requests] == [39, 11]
    assert not any(k.startswith("aspect__") for _, questions, _ in requests for k in questions)


class Answers:
    def __init__(self, pick):
        self.pick, self.seen = pick, []

    async def decide(self, state, questions):
        self.seen.append((state, questions))
        return {k: {"choice": self.pick(k, state), "confidence": 1.0} for k in questions}


async def test_page_is_suitable_when_any_of_its_batches_says_so():
    # A single batch answering NO used to discard that batch's direct answers from an otherwise suitable page.
    page = {"id": "d0", "url": "https://x.example", "passages": [{"id": f"d0p{i}", "text": "t"} for i in range(50)]}
    first = {p["id"] for p in page["passages"][:39]}
    client = Answers(lambda k, s: ("NO" if s["passages"][0]["id"] in first else "YES") if k == "page" else "DIRECT_ANSWER")
    kept, _ = await judge.judge_relevance(client, "q", [page], 40)
    assert len(kept) == 50


async def test_kept_passages_get_their_topic_from_a_separate_request():
    kept = [{"id": "d0p1", "text": "fee is 20 euros"}, {"id": "d0p2", "text": "starts in Q4"}]
    client = Answers(lambda k, s: "T1" if k == "aspect__d0p1" else "T2")
    await judge.judge_aspects(client, "fee; start", {"T1": "fee", "T2": "start"}, kept, 40)
    assert [p["aspect"] for p in kept] == ["T1", "T2"]
    assert all(k.startswith("aspect__") for _, questions in client.seen for k in questions)


def test_question_word_pairs_stay_one_topic():
    aspects = aspects_from_query("Oldeus: revenge against Hitogami; how and where he learned time travel; cause of death")
    assert "how and where he learned time travel" in aspects.values()
    assert "how" not in aspects.values()


class Session:
    def __init__(self, ctype, body):
        self.ctype, self.body, self.urls = ctype, body, []

    async def get(self, url):
        self.urls.append(url)
        body, ctype = self.body, self.ctype

        class Response:
            status_code = 200
            headers = {"content-type": ctype}
            content = body if isinstance(body, bytes) else body.encode()
            text = body if isinstance(body, str) else ""
        return Response()


def pdf_with(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


async def test_pdf_results_are_read_instead_of_skipped(tmp_path):
    # Both empty Rudeus calls skipped the one source rated primary because it was a PDF.
    sentence = "The diary records that the old man travelled back in time to warn his younger self. " * 4
    session = Session("application/pdf", pdf_with(sentence))
    page = await fetch._fetch_http(session, "https://archive.example/volume.pdf", 2.0, 50000, tmp_path)
    assert "travelled back in time" in page["text"]
    assert page["status"] == "fetched pdf"


def test_empty_evidence_lists_every_page_read_and_why_it_was_not_used():
    pages = [{"url": "https://a.example", "title": "Summary", "outcome": "read 102 passages; 12 relevant, all removed by the scope check"},
             {"url": "https://b.example/v.pdf", "title": "", "outcome": "unreadable: insufficient text"}]
    rendered = digest.render_evidence("q", {"T1": "q"}, {}, pages)
    for page in pages:
        assert page["url"] in rendered and page["outcome"] in rendered
    assert "open_passage" not in rendered


def test_evidence_calls_passages_original_text_and_points_to_browse_for_more():
    entry = {"id": "d0p0", "level": "DIRECT_ANSWER", "text": "The fee is 20 euros.", "site": "a.example", "url": "https://a.example",
             "title": "Fees", "section": "Cost", "source_role": "PRIMARY"}
    blocks = {"T1": {"aspect": "fee", "representatives": [entry], "adds": [], "conflicts": [], "also_reported_by": []}}
    rendered = digest.render_evidence("fee?", {"T1": "fee"}, blocks)
    assert "original text" in rendered and "browse" in rendered
    assert "open_passage" not in rendered and "not verified facts" not in rendered


async def test_browse_rates_only_the_requested_page_against_the_purpose(tmp_path, monkeypatch):
    url = "https://novel.example/chapter-15"
    calls = []

    async def download(urls, *args):
        calls.append(urls)
        return {url: {"url": url, "title": "Chapter 15", "status": "fetched http 200",
                      "text": "# Diary\n\nThe old man wrote how he travelled back in time.\n\n# Recipes\n\nBake the bread for an hour at high heat."}}

    client = Answers(lambda k, s: "YES" if k == "page" else "DIRECT_ANSWER" if k == "level__d0p0" else "IRRELEVANT")
    monkeypatch.setattr(pipeline, "http_wave", download)
    monkeypatch.setattr(pipeline, "JevClient", lambda *a, **k: _Closing(client))
    result = await pipeline.run_browse(url, "how he travelled back", settings=Settings(home=tmp_path, browser_fallback=False))
    assert calls == [[url]]
    assert all(not k.startswith(("aspect__", "cov__")) and not k.startswith("d0p") for _, q in client.seen for k in q)
    assert "travelled back in time" in result["evidence"] and url in result["evidence"]
    assert "Bake the bread" not in result["evidence"]


class _Closing:
    def __init__(self, inner):
        self.inner, self.stats = inner, {}

    async def decide(self, state, questions):
        return await self.inner.decide(state, questions)

    async def close(self):
        pass


def test_usefulness_levels_are_defined_once_per_request_with_three_options():
    # Replay: definitions in shared state plus merged reject options cut relevance input 57% without losing agreed passages.
    page = {"id": "d0", "url": "https://x.example", "passages": [{"id": f"d0p{i}", "text": "t"} for i in range(3)]}
    state, questions, _ = next(judge.relevance_requests("q", page, 40))
    assert set(state["usefulness_levels"]) == {"IRRELEVANT", "RELEVANT_DETAIL", "DIRECT_ANSWER"}
    levels = [q for k, q in questions.items() if k.startswith("level__")]
    assert len(levels) == 3
    assert all(set(q["criteria"]) == set(state["usefulness_levels"]) for q in levels)
    assert all(len(q["instructions"]) + sum(map(len, q["criteria"].values())) < 200 for q in levels)


async def test_results_unlikely_to_answer_are_not_downloaded():
    # Replay: skipping snippet-level UNLIKELY results removed 54% of Rudeus passage judging and lost only a duplicated quote.
    class Client:
        def __init__(self):
            self.seen = []

        async def decide(self, state, questions):
            self.seen.append(state)
            usefulness = {"0": "UNLIKELY", "1": "POSSIBLE", "2": "LIKELY"}
            return {k: {"choice": usefulness[k.split("__")[1]] if k.startswith("use__") else "REFERENCE"} for k in questions}

    rows = [{"url": f"https://source{i}.example", "title": t, "snippet": s} for i, (t, s) in enumerate(
        [("Series overview", "A Japanese light novel series"), ("Character page", "Future self"), ("The Diary", "The future self's diary")])]
    client = Client()
    selected, audit = await search.select_sources(client, "What did the future self's diary say?", rows)
    assert {r["url"] for r in selected} == {rows[1]["url"], rows[2]["url"]}
    assert [a["usefulness"] for a in audit] == ["UNLIKELY", "POSSIBLE", "LIKELY"]
    assert set(client.seen[0]["usefulness_levels"]) == {"LIKELY", "POSSIBLE", "UNLIKELY"}


async def test_native_site_prefix_is_rewritten_for_the_search_engine(monkeypatch):
    # Agents often write site.domain; SearXNG would search for the word "site" and return dictionaries.
    seen = []

    async def fake(query, top, url):
        seen.append(query)
        return [{"url": "https://x.example"}], []

    async def no_brave(query, top):
        return []

    monkeypatch.setattr(search, "searxng", fake)
    monkeypatch.setattr(search, "brave", no_brave)
    await search.search('site.mushokutensei.fandom.com "Rudeus Greyrat/Future"', 10, "http://s.test")
    assert seen == ['site:mushokutensei.fandom.com "Rudeus Greyrat/Future"']


async def test_video_and_social_results_are_listed_not_downloaded(tmp_path, monkeypatch):
    # YouTube and Reddit pages produced only menus and CAPTCHA text; they are shown as links, as search engines show videos.
    rows = [{"url": "https://www.youtube.com/watch?v=abc", "title": "Oldeus explained (video)", "snippet": "A video about the diary"},
            {"url": "https://article.example/diary", "title": "The diary", "snippet": "What the diary says"}]
    downloaded = []

    async def find(*args, **kwargs):
        return rows, {"used": "fixture"}

    async def select(client, query, results):
        return [dict(r, source_role="SECONDARY") for r in results], [dict(r, decision="SECONDARY", usefulness="POSSIBLE") for r in results]

    async def download(urls, *args):
        downloaded.extend(urls)
        return {u: {"url": u, "title": "The diary", "status": "fetched http 200",
                    "text": "# Diary\n\nThe future self wrote that Roxy died of the stone disease after eating tainted food."} for u in urls}

    client = Answers(lambda k, s: "YES" if k == "page" else "DIRECT_ANSWER" if k.startswith("level__") else "IN_SCOPE" if k.startswith("d") else "T1")
    monkeypatch.setattr(pipeline, "search", find)
    monkeypatch.setattr(pipeline, "select_sources", select)
    monkeypatch.setattr(pipeline, "http_wave", download)
    monkeypatch.setattr(pipeline, "JevClient", lambda *a, **k: _Closing(client))
    result = await pipeline.run_digest("What did the diary say about Roxy?", settings=Settings(home=tmp_path, browser_fallback=False))
    assert downloaded == ["https://article.example/diary"]
    assert "https://www.youtube.com/watch?v=abc" in result["digest_evidence"]
    assert "Oldeus explained (video)" in result["digest_evidence"]


async def test_overloaded_jev_response_is_retried():
    import httpx
    from jev_digest.client import JevClient
    questions = {"check": {"type": "choice", "instructions": "Relevant?", "criteria": {"YES": "y", "NO": "n"}}}
    replies = [httpx.Response(529, json={"detail": "overloaded"}),
               httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"check": {"type": "choice", "choice": "YES", "confidence": 0.9,
                                                                                     "probabilities": {"YES": 0.9, "NO": 0.1}}}})]
    client = JevClient("key")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: replies.pop(0)))
    try:
        answers = await client.decide({}, questions)
        assert answers["check"]["choice"] == "YES"
        assert replies == []
    finally:
        await client.close()


BRAVE_PAGE = """<html><body><section id="mixed-main">
<div class="snippet svelte-jmfu5f" data-pos="0" data-type="web"><div class="result-content"><a href="https://travel-europe.europa.eu/etias" class="l1">
<div class="site-name-content"><cite class="snippet-url">travel-europe.europa.eu</cite></div>
<div class="title search-snippet-title line-clamp-1" title="ETIAS | Official site">ETIAS | Official site</div></a>
<div class="generic-snippet"><div class="content desktop-default-regular"><span class="t-secondary">July 16, 2026 -</span> ETIAS launches in Q4 2026 and costs <strong>20</strong> euros.</div></div></div></div>
<div class="snippet" data-pos="1" data-type="web"><div class="result-content"><a href="https://news.example/etias" class="l1">
<div class="title search-snippet-title" title="ETIAS explained">ETIAS explained</div></a>
<div class="generic-snippet"><div class="content">Who needs to apply.</div></div></div></div>
<div class="snippet" data-type="video"><a href="https://video.example/x"><div class="title search-snippet-title" title="Video">Video</div></a></div>
</section></body></html>"""


def test_brave_results_page_is_parsed_into_search_rows():
    rows = search.brave_results(BRAVE_PAGE)
    assert [r["url"] for r in rows] == ["https://travel-europe.europa.eu/etias", "https://news.example/etias"]
    assert rows[0]["title"] == "ETIAS | Official site"
    assert "costs 20 euros" in rows[0]["snippet"] and rows[0]["published"] == "July 16, 2026"
    assert rows[0]["engine"] == "brave"


async def test_brave_is_searched_directly_beside_searxng(monkeypatch):
    # SearXNG's requests to Brave were refused as "too many requests"; the same search from a browser-like client succeeded.
    async def fake_brave(query, top):
        return [{"url": "https://travel-europe.europa.eu/etias", "title": "Official", "engine": "brave"}]

    async def fake_searxng(query, top, url):
        return [{"url": "https://news.example/etias", "title": "News", "engine": "google"},
                {"url": "https://travel-europe.europa.eu/etias", "title": "Official", "engine": "google"}], []

    monkeypatch.setattr(search, "brave", fake_brave)
    monkeypatch.setattr(search, "searxng", fake_searxng)
    rows, meta = await search.search("ETIAS fee", 10, "http://s.test")
    assert [r["url"] for r in rows] == ["https://travel-europe.europa.eu/etias", "https://news.example/etias"]
    assert meta["brave"]["results"] == 1 and meta["used"] == "searxng+brave"


async def test_searxng_results_survive_a_brave_failure(monkeypatch):
    async def failing_brave(query, top):
        raise RuntimeError("blocked")

    async def fake_searxng(query, top, url):
        return [{"url": "https://news.example/etias", "title": "News", "engine": "google"}], []

    monkeypatch.setattr(search, "brave", failing_brave)
    monkeypatch.setattr(search, "searxng", fake_searxng)
    rows, meta = await search.search("ETIAS fee", 10, "http://s.test")
    assert [r["url"] for r in rows] == ["https://news.example/etias"]
    assert "blocked" in meta["brave"]["error"]


async def test_slow_text_extraction_is_cut_off_without_blocking(tmp_path, monkeypatch):
    # Extracting a whole novel (PDF or one huge page) froze every other request in a digest for 208 s.
    import time as clock

    def slow(data, max_chars):
        clock.sleep(2)
        return "text", "title"

    monkeypatch.setattr(fetch, "pdf_text", slow)
    monkeypatch.setattr(fetch, "extract", slow)
    monkeypatch.setattr(fetch, "EXTRACT_BUDGET_S", 0.2)
    for ctype, body in (("application/pdf", b"%PDF-1.4"), ("text/html", "<html>whole novel</html>")):
        started = clock.perf_counter()
        page = await fetch._fetch_http(Session(ctype, body), "https://archive.example/v", 2.0, 50000, tmp_path)
        assert page["status"] == "cutoff: text extraction"
        assert clock.perf_counter() - started < 1.5


def test_cli_status_line_reports_fetch_time_from_real_timing_keys(monkeypatch, capsys):
    """The command line prints the same evidence the MCP tool returns, and its status line reads the pipeline's own timing keys."""
    import sys
    from jev_digest import cli

    async def fake_run_digest(*a, **k):
        return {"run_id": "r1", "digest_evidence": "evidence",
                "counts": {"pages": 3, "passages": 40, "kept": 5},
                "timings": {"total": 9.0, "search": 1.0, "source_selection": 0.5, "fetch_http_wave": 2.5,
                            "jev_round1": 3.0, "scope_check": 1.0, "jev_round2": 0.7},
                "tokens": {"raw_pages": 9000, "digest_evidence": 400},
                "jev": {"requests": 6, "input_tokens": 50000}}

    monkeypatch.setattr(cli, "run_digest", fake_run_digest)
    monkeypatch.setattr(sys, "argv", ["jev-digest", "question"])
    cli.main()
    out = capsys.readouterr()
    assert out.out.strip() == "evidence"
    assert "fetch 2.5" in out.err
