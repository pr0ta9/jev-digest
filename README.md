# jev-digest

**Too much to read? Let your agent digest it.**

Turn web pages and local documents into bite-sized evidence, with the original wording and sources intact.

jev-digest searches the web or reads local documents, then uses TypeSafe's Jev model to select the passages relevant to your question. Your agent gets original source text with URLs or file paths attached, ready to reason over and cite.

**In one Codex research run: 67% fewer input tokens and half as many tool rounds.** See the [early results](#early-results) for the full comparison and its limits.

[Quick start](#quick-start) | [Example output](#example-output) | [Tools](#tools) | [Early results](#early-results) | [Contribute](#contribute)

**Watch the 20-second demo**

https://github.com/user-attachments/assets/cac9a98d-f7a0-4a88-89a8-2721a50964f0

## Why use it?

Research can fill an agent's context with navigation, unrelated sections and repeated information. jev-digest does the first pass across sources so the agent can focus on the passages that matter.

- **Keep the source wording.** Returned passages are verbatim, with source references, sections and passage IDs. jev-digest does not generate a summary.
- **Research across sources.** Search the web, inspect known URLs, or ask questions across local Markdown, text, HTML and PDF files.
- **See gaps and possible disagreements.** Output identifies topics without evidence and can flag passages that may disagree with other sources.
- **Use your existing agent.** Connect through the Model Context Protocol (MCP), or call the Python API and command-line tool.

**Status: working proof of concept.** Interfaces and prompts may change. Passage selection is not fact verification, and the benchmarks are small.

## Example output

An agent calls:

```python
digest(query="ETIAS validity, fee and who must apply")
```

The existing project example reports ten pages processed in 3.9 seconds on 23 September 2026. Here is a shortened excerpt showing an original passage and a possible disagreement from another source; it is a sample of tool output, not current travel guidance.

```text
Evidence for: ETIAS validity, fee and who must apply

Topics: T1: ETIAS validity, fee; T2: who must apply

## Frequently asked questions - ETIAS
https://travel-europe.europa.eu/etias/faq
Source type: primary (selection estimate, not fact verification)
[d1p54; T1] Fees and Payment
When applying using this official ETIAS website, you will be charged a fee of EUR 20. Applicants who are under 18 or over 70 years of age are exempt from this payment. You are also exempt if you are a family member of an EU national and you fulfil the conditions to qualify for family member status.

[d1p68; T1] Validity and renewal
Your travel authorisation will be valid for three years or until the end of validity of your travel document - whichever comes first. If your passport is valid for two years, your ETIAS will also be valid for two years. You will receive an email regarding the upcoming expiration of your ETIAS travel authorisation.

## Overview of ETIAS Requirements for Travelers
https://etias.com/etias-requirements/
Source type: secondary (selection estimate, not fact verification)
[d3p39; T1; possible disagreement] Step-by-Step Guide to Applying for ETIAS > Step 4: Pay the application fee
After filling out the form, you’ll be directed to a secure payment page. Most people will pay €7, but applicants under 18 or over 70 often don’t have to pay. Make sure your card is enabled for international transactions.
```

Passage IDs such as `d1p68` identify the source and passage; `T1` identifies the question's topic. The agent can cite the original URLs or use `browse` to read more.

## Tools

| Tool | Use it to |
| --- | --- |
| `digest(query, allowed_domains?, blocked_domains?)` | Search the web and return relevant passages grouped by source. Optionally allow or block domains. |
| `browse(url, purpose)` | Read a known page, or up to five URLs in parallel, for a specific purpose. |
| `read_documents(paths, question)` | Ask a question across local files or folders, up to 50 Markdown, text, HTML or PDF files. |

Separate sub-questions with semicolons. When nothing answers, the result lists the pages or files read and why they were not used.

## Quick start

You need **Python 3.11+**, a **TypeSafe API key**, and **Docker** for the bundled SearXNG web-search service. Run these commands from the repository directory. The examples below use Windows PowerShell.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[browser,mcp,dev]"
.venv/Scripts/python.exe -m playwright install chromium
$env:TYPESAFE_API_KEY = "your-key"
docker compose -f docker/docker-compose.searxng.yml up -d
```

Docker Desktop must be running on Windows. The search service listens on port 8089 by default; set `JEV_DIGEST_SEARXNG_URL` to use another instance. Check it before relying on results:

```powershell
Invoke-RestMethod http://localhost:8089/healthz
Invoke-RestMethod 'http://localhost:8089/search?q=test&format=json'
```

### Connect your MCP client

For clients that accept `mcpServers` JSON configuration, use the full path to the installed server:

```json
{
  "mcpServers": {
    "jev-digest": {
      "command": "C:/path/to/jev-digest/.venv/Scripts/jev-digest-mcp.exe",
      "env": {
        "TYPESAFE_API_KEY": "your-key"
      }
    }
  }
}
```

Once connected, ask your agent:

> Use jev-digest to research this question. Cite the source URLs, preserve disagreements, and tell me where evidence is missing.

### Use Python

```python
import asyncio

from jev_digest.pipeline import run_digest, run_read

result = asyncio.run(run_digest("How long is ETIAS valid; who must apply", search_queries=["ETIAS validity who must apply"]))
print(result["digest_evidence"])

result = asyncio.run(run_read(["C:/docs/paper.md"], "Which datasets do they use?"))
print(result["evidence"])
```

### Use the command line

Web search only, using the executable in the virtual environment:

```powershell
.venv/Scripts/jev-digest.exe "Question; second aspect" --search-query "keywords"
```

## Early results

### Codex

Codex CLI (`gpt-6-astra`, high reasoning) answered the same research question twice, from prompt to cited answer: once with its built-in web search, once with jev-digest. The question asks what happened in the "Oldeus" future timeline of *Mushoku Tensei*: the fates of the main characters, how he learned time travel, and how he died.

|                             | Codex built-in search | Codex + jev-digest        |
| --------------------------- | --------------------- | ------------------------- |
| Total time                  | 150.1 s               | **140.3 s** (10 s faster) |
| Codex input tokens          | 757,825               | **251,320** (67% fewer)   |
| Codex uncached input tokens | 107,457               | **30,520** (72% fewer)    |
| Codex output tokens         | 2,863                 | 2,816                     |
| Tool rounds                 | 12                    | 6                         |

In this run, jev-digest reduced cumulative input tokens and tool rounds. These are results from one question, not a general performance guarantee.

### Claude Code

Claude Code (Sonnet 5) answered two questions with its built-in WebSearch and with jev-digest as its only web tool. Cost is the Claude Code CLI's list price.

| Question       | Built-in search | jev-digest      |
| -------------- | --------------- | --------------- |
| ETIAS          | **26 s**, $0.19 | 32 s, **$0.14** |
| Mushoku Tensei | 135 s, $0.63    | **74 s, $0.23** |

These are small, reported experiments, not a comprehensive benchmark. Results vary with the agent, question and search engine. Benchmark scripts and run records are currently kept outside this repository.

## How it works

1. **Search.** This demo searches through a local SearXNG instance and Brave. Video and social results are listed as links instead of being downloaded.
2. **Screening.** Jev labels each search result by source type and by how likely its title and snippet are to answer the question. Results unlikely to help are not downloaded.
3. **Download.** HTTP with a 2-second cutoff, PDF and HTML text extraction with a time budget, and a headless-browser retry for important pages that fail.
4. **Judging.** The text is split into short passages. Jev rates each passage's usefulness, checks that kept passages match the exact subject asked about, sorts them into the question's topics, and marks passages that only repeat others.
5. **Rendering.** Kept passages are returned verbatim, in page order, with their source, section and passage ID.

Every judgment is a choice among fixed options; the returned text is always original source text.

## Limitations

- Jev calls consume API tokens; agent costs and search infrastructure are additional considerations.
- SearXNG relies on public search engines, so availability and latency vary.
- Web pages are cut at 50,000 characters and local documents at 400,000. `read_documents` reports truncation; `digest` does not yet.
- Passage selection and disagreement flags are model judgments, not guarantees of completeness or correctness.
- For a single short document the agent already knows, reading it directly can be cheaper than calling a tool.

## Development

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Tests run without network access, Jev credits or an agent.

## Contribute

Everyone is welcome to help shape jev-digest. Whether you are building agents, exploring MCP, making your first open-source contribution, or just curious about the idea, there is a place for you here.

Bring an idea, ask a question, try it on your own research, improve the docs, or contribute code. Small contributions matter, and you do not need to be an expert to get involved.

Open an issue to start a conversation or send a pull request when you have something to share. This project is still taking shape; let's build it together.

## License

[MIT](LICENSE)
