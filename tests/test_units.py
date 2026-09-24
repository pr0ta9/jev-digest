import pytest

from jev_digest import aspects, client, digest, judge, passages, search


def test_passages_group_two_to_four_sentences():
    text = " ".join(f"Sentence number {i} says something about the topic at hand." for i in range(12))
    ps = passages.passages(text)
    assert 3 <= len(ps) <= 6 and all(1 <= p.count("Sentence number") <= 4 for p in ps)
    assert " ".join(ps).count("Sentence number") == 12


def test_passages_size_cjk_by_characters():
    ps = passages.passages("ルーデウスは地下室に行った。" * 3 + "ロキシーは病気になった。" * 3, target=20, hard=40)
    assert ps and all(len(p) >= 12 for p in ps)


def test_aspects_split_on_semicolons_and_question_clauses():
    q = "How do I enable free-threaded mode in Python 3.13, how do I check it is active, and what are its current limitations?"
    a = list(aspects.aspects_from_query(q).values())
    assert a == ["How do I enable free-threaded mode in Python 3.13", "how do I check it is active", "what are its current limitations"]
    b = list(aspects.aspects_from_query("who needs it; how much it costs; when it starts").values())
    assert b == ["who needs it", "how much it costs", "when it starts"]


def test_aspects_expand_name_lists_and_keep_single_queries_whole():
    q = "Future Rudeus: what happened to Roxy, Sylphy, Eris and his family in his timeline; how he died."
    a = list(aspects.aspects_from_query(q).values())
    assert a[:4] == ["what happened to Roxy in his timeline", "what happened to Sylphy in his timeline",
                     "what happened to Eris in his timeline", "what happened to his family in his timeline"]
    assert a[-1] == "how he died"
    assert aspects.aspects_from_query("who is Future Rudeus") == {"T1": "who is Future Rudeus"}


def test_relevance_requests_cap_questions_and_carry_only_their_passages():
    page = {"id": "d0", "url": "https://x.test/p", "title": "t", "passages": [{"id": f"d0p{i}", "text": f"passage {i}", "page": "d0"} for i in range(45)]}
    reqs = list(judge.relevance_requests("q", page, 40))
    assert len(reqs) == 2
    for state, questions, ids in reqs:
        assert len(questions) <= 40 and len(questions) == len(ids) + 1
        assert [p["id"] for p in state["passages"]] == ids


def _p(i, page, text, level="RELEVANT_DETAIL", conf=0.8, aspect="T1"):
    return {"id": f"{page}p{i}", "page": page, "text": text, "level": level, "confidence": conf, "aspect": aspect}


def test_coverage_uses_top_passages_as_representatives():
    kept = [_p(1, "d0", "Cliff was poisoned and died.", "DIRECT_ANSWER", 0.9), _p(2, "d1", "Cliff died after the escape.", conf=0.7),
            _p(3, "d2", "The escape cost Cliff his life.", conf=0.6), _p(4, "d3", "Cliff was poisoned during the escape.", conf=0.5),
            _p(5, "d4", "Zanoba survived.", "DIRECT_ANSWER", 0.9, aspect="T2")]
    groups = judge.group_by_aspect(kept)
    assert [p["id"] for p in groups["T1"]] == ["d0p1", "d1p2", "d2p3", "d3p4"]
    reqs = list(judge.coverage_requests("q", {"T1": "Cliff", "T2": "Zanoba"}, groups, 3, 40))
    assert len(reqs) == 1
    state, questions, key, batch = reqs[0]
    assert key == "T1" and set(state["representatives"]) == {"d0p1", "d1p2", "d2p3"} and [c["id"] for c in batch] == ["d3p4"]


def test_blocks_and_evidence_mark_disagreement_and_bare_site():
    pages = {"d0": {"url": "https://a.example/x"}, "d1": {"url": "https://www.b.example/y"}, "d2": {"url": "https://c.example/z"}}
    kept = [_p(1, "d0", "Fiber target is 25-38 g.", "DIRECT_ANSWER"), _p(1, "d1", "Adults need 25 to 38 grams.", conf=0.7),
            _p(1, "d2", "Aim for 14 g per 1,000 kcal.", conf=0.6), _p(2, "d2", "Only 5 g is enough.", conf=0.5)]
    groups = judge.group_by_aspect(kept)
    blocks = digest.build_blocks({"T1": "daily fiber intake"}, groups, {"d2p2": "CONFLICTS"}, pages, 3)
    b = blocks["T1"]
    assert [r["id"] for r in b["representatives"]] == ["d0p1", "d1p1", "d2p1"] and [c["id"] for c in b["conflicts"]] == ["d2p2"]
    assert b["representatives"][1]["site"] == "b.example"
    rendered = digest.render_evidence("q", {"T1": "daily fiber intake"}, blocks)
    assert "[d2p2; T1; possible disagreement]" in rendered and "https://www.b.example/y" in rendered


def test_validate_rejects_non_argmax_and_accepts_consistent_answers():
    q = {"k": client.choice("pick", {"A": "a", "B": "b"})}
    ok = {"model": "jev-1.13.0", "answers": {"k": {"type": "choice", "choice": "A", "confidence": 0.6, "probabilities": {"A": 0.8, "B": 0.2}}}}
    client.validate(ok, q, "jev-1.13.0")
    bad = {"model": "jev-1.13.0", "answers": {"k": {"type": "choice", "choice": "B", "confidence": 0.1, "probabilities": {"A": 0.51, "B": 0.49}}}}
    with pytest.raises(ValueError):
        client.validate(bad, q, "jev-1.13.0")
    with pytest.raises(ValueError):
        client.choice("too many", {str(i): "x" for i in range(256)})


def test_merge_results_interleaves_queries_dedupes_and_caps():
    """A batch of searches must let every query contribute its best hits before any query's tail, with no URL twice."""
    a = [{"url": "a1"}, {"url": "shared"}, {"url": "a3"}]
    b = [{"url": "shared"}, {"url": "b2"}, {"url": "b3"}]
    merged = search.merge_results([a, b], total=4)
    assert [r["url"] for r in merged] == ["a1", "shared", "b2", "a3"]


def test_page_title_decodes_html_entities():
    """Titles are shown to the agent as source headings, so "&amp;" must read as "&"."""
    from jev_digest import fetch
    _, title = fetch.extract("<html><title>Launch Date, Cost &amp; Requirements</title><body><p>x</p></body></html>", 1000)
    assert title == "Launch Date, Cost & Requirements"
