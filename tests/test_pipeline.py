import json

import pytest

from jev_digest import pipeline
from jev_digest.config import Settings


@pytest.mark.parametrize("failed", [{0}, {1}, {0, 2}])
async def test_browser_recovery_preserves_each_passages_text_and_source(tmp_path, monkeypatch, failed):
    urls = [f"https://source{i}.example/page" for i in range(4)]
    texts = {url: f"Source {i} reports its unique finding about the topic." for i, url in enumerate(urls)}

    async def search(*args):
        return [{"url": url} for url in urls], {"used": "fixture"}

    async def http_wave(*args):
        return {url: {"url": url, "status": "cutoff"} if i in failed else
                {"url": url, "text": texts[url], "title": str(i), "status": "fetched http 200"}
                for i, url in enumerate(urls)}

    async def browser_wave(requested, *args):
        return {url: {"url": url, "text": texts[url], "title": "recovered", "status": "fetched via browser"}
                for url in requested}

    class Client:
        def __init__(self, *args, **kwargs):
            self.stats = {}

        async def decide(self, state, questions):
            answers = {}
            for key, question in questions.items():
                value = "YES" if key == "page" else "DIRECT_ANSWER" if key.startswith("level__") else "T1" if key.startswith("aspect__") else "ADDS"
                answers[key] = {"choice": value, "confidence": 1.0, "type": "choice",
                                "probabilities": {option: float(option == value) for option in question["criteria"]}}
            return answers

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, "search", search)
    monkeypatch.setattr(pipeline, "http_wave", http_wave)
    monkeypatch.setattr(pipeline, "browser_wave", browser_wave)
    monkeypatch.setattr(pipeline, "JevClient", Client)
    settings = Settings(home=tmp_path, browser_fallback=True, browser_min_pages=6)
    result = await pipeline.run_digest("findings about the topic", settings=settings)
    saved = json.loads((settings.runs_dir / result["run_id"] / "passages.json").read_text(encoding="utf-8"))

    assert {(entry["text"], entry["url"]) for entry in saved.values()} == {(text, url) for url, text in texts.items()}
    entries = [entry for block in result["blocks"].values() for bucket in ("representatives", "adds") for entry in block[bucket]]
    assert len({entry["id"] for entry in entries}) == len(urls)
    for entry in entries:
        assert entry["text"] == texts[entry["url"]]
        opened = pipeline.open_passage(result["run_id"], entry["id"], settings)
        assert (opened["text"], opened["url"]) == (entry["text"], entry["url"])
        assert entry["text"] in result["digest_evidence"]
