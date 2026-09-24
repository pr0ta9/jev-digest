import json

from jev_digest import documents


def test_folders_are_expanded_recursively_to_supported_files(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "a.md").write_text("# A\n\nAlpha.", encoding="utf-8")
    (tmp_path / "notes" / "b.txt").write_text("Beta.", encoding="utf-8")
    (tmp_path / "notes" / "image.png").write_bytes(b"\x89PNG")
    files = documents.expand([str(tmp_path), str(tmp_path / "a.md")])
    assert [f.name for f in files] == ["a.md", "b.txt"]


def test_explicit_files_are_kept_even_when_unsupported_so_the_reason_can_be_reported(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    files = documents.expand([str(tmp_path / "image.png"), str(tmp_path / "missing.md")])
    assert [f.name for f in files] == ["image.png", "missing.md"]


def test_folder_expansion_is_capped(tmp_path):
    for i in range(60):
        (tmp_path / f"{i:02d}.md").write_text("x", encoding="utf-8")
    assert len(documents.expand([str(tmp_path)])) == documents.MAX_FILES


def test_hidden_and_vendored_folders_are_skipped(tmp_path):
    # Reading a project folder must reach its documents, not its virtualenv, VCS or dependency folders.
    for folder in (".venv/lib", ".git", "node_modules/pkg", "__pycache__", "docs"):
        (tmp_path / folder).mkdir(parents=True)
    for folder in (".venv/lib", ".git", "node_modules/pkg", "__pycache__"):
        (tmp_path / folder / "junk.md").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("x", encoding="utf-8")
    assert [f.name for f in documents.expand([str(tmp_path)])] == ["guide.md"]


def test_utf8_byte_order_mark_is_not_part_of_the_text(tmp_path):
    path = tmp_path / "bom.md"
    path.write_bytes("\ufeff# Title\n\nBody text.".encode("utf-8"))
    assert documents.load_document(path, 1000)["text"].startswith("# Title")


def test_markdown_is_loaded_with_its_path_as_the_source(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("# Method\n\nWe train on 3,044 sentences.", encoding="utf-8")
    page = documents.load_document(path, 1000)
    assert page["url"] == str(path)
    assert page["title"] == "paper.md"
    assert "3,044 sentences" in page["text"] and page["status"] == "read" and page["truncated"] is False


def test_unreadable_documents_report_why(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    assert documents.load_document(tmp_path / "missing.md", 1000)["status"] == "not found"
    assert documents.load_document(tmp_path / "image.png", 1000)["status"] == "unsupported file type"
    assert documents.load_document(tmp_path / "empty.md", 1000)["status"] == "no text"


def test_long_documents_are_cut_and_marked_truncated(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("word " * 1000, encoding="utf-8")
    page = documents.load_document(path, 100)
    assert len(page["text"]) == 100 and page["truncated"] is True


from jev_digest import pipeline
from jev_digest.config import Settings


class Judge:
    """Keeps passages mentioning 'sentences'; answers every other question plainly."""

    def __init__(self, *args, seen=None, **kwargs):
        self.stats, self.seen = {"requests": 0}, seen if seen is not None else []

    async def decide(self, state, questions):
        self.seen.append(questions)
        answers = {}
        for key in questions:
            if key == "page":
                choice = "YES"
            elif key.startswith("level__"):
                pid = key.split("__")[1]
                text = next(p["text"] for p in state["passages"] if p["id"] == pid)
                choice = "DIRECT_ANSWER" if "sentences" in text else "IRRELEVANT"
            elif key.startswith("aspect__"):
                choice = "T1"
            elif key.startswith("cov__"):
                choice = "ADDS"
            else:
                choice = "IN_SCOPE"
            answers[key] = {"choice": choice, "confidence": 1.0}
        return answers

    async def close(self):
        pass


async def test_read_returns_only_passages_that_answer_with_their_file(tmp_path, monkeypatch):
    paper = tmp_path / "paper.md"
    paper.write_text("# Data\n\nThe corpus has 3,044 sentences in 100 dialogs.\n\n# Thanks\n\nWe thank our reviewers.", encoding="utf-8")
    judge_calls = []
    monkeypatch.setattr(pipeline, "JevClient", lambda *a, **k: Judge(seen=judge_calls))
    result = await pipeline.run_read([str(paper)], "How large is the corpus?", settings=Settings(home=tmp_path / "home"))
    assert str(paper) in result["evidence"]
    assert "3,044 sentences" in result["evidence"]
    assert "thank our reviewers" not in result["evidence"]
    assert result["counts"]["kept"] == 1
    assert "cite those files" in result["evidence"] and "browse" not in result["evidence"]
    assert "Source type" not in result["evidence"]
    assert any(key.startswith("d0p") for questions in judge_calls for key in questions), "the scope check must run for files too"
    assert pipeline.open_passage(result["run_id"], "d0p0", Settings(home=tmp_path / "home"))["url"] == str(paper)


async def test_read_lists_every_file_and_why_when_nothing_answers(tmp_path, monkeypatch):
    note = tmp_path / "note.md"
    note.write_text("# Thanks\n\nWe thank our reviewers.", encoding="utf-8")
    monkeypatch.setattr(pipeline, "JevClient", Judge)
    result = await pipeline.run_read([str(note), str(tmp_path / "gone.md")], "How large is the corpus?",
                                     settings=Settings(home=tmp_path / "home"))
    assert "none judged relevant" in result["evidence"]
    assert "gone.md" in result["evidence"] and "not found" in result["evidence"]
    assert "No passage from these files answered the question" in result["evidence"]


async def test_read_with_no_readable_files_does_not_call_jev(tmp_path, monkeypatch):
    def no_jev(*a, **k):
        raise AssertionError("nothing to judge")

    monkeypatch.setattr(pipeline, "JevClient", no_jev)
    result = await pipeline.run_read([str(tmp_path / "gone.md")], "anything?", settings=Settings(home=tmp_path / "home"))
    assert result["error"] == "no readable files"
    assert "gone.md" in json.dumps(result["files"])


async def test_read_reports_folders_cut_at_the_file_cap_and_truncated_documents(tmp_path, monkeypatch):
    # Silently reading only part of what was asked for would look like "the documents do not say".
    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(documents.MAX_FILES + 5):
        (folder / f"{i:02d}.md").write_text("The corpus has 3,044 sentences.", encoding="utf-8")
    long = tmp_path / "long.md"
    long.write_text("The corpus has 3,044 sentences. " * 20, encoding="utf-8")
    monkeypatch.setattr(pipeline, "JevClient", Judge)
    result = await pipeline.run_read([str(folder)], "How large is the corpus?", settings=Settings(home=tmp_path / "home"))
    assert f"Only the first {documents.MAX_FILES} files were read" in result["evidence"]
    result = await pipeline.run_read([str(long)], "How large is the corpus?", settings=Settings(home=tmp_path / "home", max_chars_per_document=100))
    assert "Cut at 100 characters: " + str(long) in result["evidence"]


import asyncio


def served(monkeypatch, **fakes):
    from mcp.server.fastmcp import FastMCP
    from jev_digest import mcp_server
    captured = {}
    monkeypatch.setattr(FastMCP, "run", lambda self, *a, **k: captured.setdefault("server", self))
    for name, fake in fakes.items():
        monkeypatch.setattr(mcp_server, name, fake)
    mcp_server.main()
    return captured["server"]


def test_read_documents_takes_paths_and_a_question(monkeypatch):
    tools = {t.name: t for t in asyncio.run(served(monkeypatch).list_tools())}
    schema = tools["read_documents"].inputSchema
    assert schema["required"] == ["paths", "question"]
    assert schema["properties"]["paths"]["type"] == "array"
    assert all(p.get("description") for p in schema["properties"].values())
    text = tools["read_documents"].description
    assert "Use it" in text and "Returns" in text and "Errors" in text


async def test_read_documents_passes_paths_and_question_to_the_pipeline(monkeypatch):
    requests = []

    async def read(paths, question):
        requests.append((paths, question))
        return {"evidence": "Evidence from files", "run_id": "r", "counts": {"read": 1, "kept": 1}, "timings": {"total": 0.1}}

    server = served(monkeypatch, run_read=read)
    result = await server.call_tool("read_documents", {"paths": ["C:/docs/paper.md"], "question": "How large is the corpus?"})
    assert requests == [(["C:/docs/paper.md"], "How large is the corpus?")]
    assert "Evidence from files" in json.dumps(result, default=str)


async def test_unread_files_are_listed_even_when_another_file_answers(tmp_path, monkeypatch):
    # A mistyped path must not look like "that document says nothing".
    paper = tmp_path / "paper.md"
    paper.write_text("# Data\n\nThe corpus has 3,044 sentences.", encoding="utf-8")
    monkeypatch.setattr(pipeline, "JevClient", Judge)
    result = await pipeline.run_read([str(paper), str(tmp_path / "gone.md")], "How large is the corpus?",
                                     settings=Settings(home=tmp_path / "home"))
    assert "3,044 sentences" in result["evidence"]
    assert "Files not read:" in result["evidence"] and "gone.md: not found" in result["evidence"]


async def test_a_folder_with_no_supported_files_says_what_was_asked_and_what_is_supported(tmp_path, monkeypatch):
    folder = tmp_path / "office"
    folder.mkdir()
    (folder / "report.docx").write_bytes(b"PK")
    monkeypatch.setattr(pipeline, "JevClient", Judge)
    result = await pipeline.run_read([str(folder)], "anything?", settings=Settings(home=tmp_path / "home"))
    assert result["error"] == "no readable files"
    assert result["paths"] == [str(folder)] and ".pdf" in result["supported"]


async def test_read_documents_error_output_names_the_files_and_supported_types(monkeypatch):
    async def read(paths, question):
        return {"error": "no readable files", "paths": paths, "supported": [".md", ".pdf"],
                "files": [{"url": "C:/docs/gone.md", "status": "not found"}], "timings": {}}

    server = served(monkeypatch, run_read=read)
    result = json.dumps(await server.call_tool("read_documents", {"paths": ["C:/docs/gone.md"], "question": "anything?"}), default=str)
    assert "C:/docs/gone.md" in result and "not found" in result and ".pdf" in result
