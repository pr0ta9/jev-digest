"""Assemble aspect blocks and render the evidence the tools return. Every passage keeps its id so its neighbors can be opened."""

from __future__ import annotations

from .judge import site
from .search import SOURCE_RANK


def build_blocks(aspects: dict[str, str], groups: dict[str, list[dict]], verdicts: dict[str, str], pages_by_id: dict[str, dict], reps_n: int) -> dict:
    blocks: dict[str, dict] = {}
    for key, members in groups.items():
        reps, rest = members[:reps_n], members[reps_n:]
        block = {"aspect": aspects.get(key, "Other relevant material"), "representatives": [], "also_reported_by": [],
                 "adds": [], "conflicts": [], "off_topic": []}
        for r in reps:
            block["representatives"].append(_entry(r, pages_by_id))
        for c in rest:
            bucket = {"COVERED": "also_reported_by", "ADDS": "adds", "CONFLICTS": "conflicts", "OFF_TOPIC": "off_topic"}[verdicts.get(c["id"], "ADDS")]
            block[bucket].append(_entry(c, pages_by_id))
        blocks[key] = block
    return blocks


def _entry(p: dict, pages_by_id: dict[str, dict]) -> dict:
    page = pages_by_id[p["page"]]
    return {"id": p["id"], "level": p["level"], "text": p["text"], "site": site(page["url"]), "url": page["url"],
            "title": page.get("title", ""), "section": p.get("section", ""),
            "source_role": p.get("source_role", page.get("source_role", "UNCERTAIN")), "published": page.get("published")}


def render_evidence(query: str, aspects: dict[str, str], blocks: dict, pages: list[dict] | None = None, links: list[dict] | None = None,
                    local: bool = False) -> str:
    sources, additional = {}, []
    for key, block in blocks.items():
        visible_adds = max(0, 5 - len(block["representatives"]))
        for bucket in ("representatives", "conflicts", "adds"):
            entries = block[bucket]
            if bucket == "adds":
                additional.extend(e["id"] for e in entries[visible_adds:])
                entries = entries[:visible_adds]
            for entry in entries:
                source = sources.setdefault(entry["url"], {"entry": entry, "passages": {}})
                source["passages"].setdefault(entry["id"], (entry, key, bucket == "conflicts"))
    lines = [f"Evidence for: {query}", "", "Topics: " + "; ".join(f"{k}: {v}" for k, v in aspects.items()), ""]
    for source in sorted(sources.values(), key=lambda s: -SOURCE_RANK.get(s["entry"].get("source_role", "UNCERTAIN"), 1)):
        head = source["entry"]
        lines += [f"## {head.get('title') or head['site']}", head["url"]]
        if not local:
            lines.append(f"Source type: {head.get('source_role', 'UNCERTAIN').lower()} (selection estimate, not fact verification)")
        if head.get("published"):
            lines.append(f"Search-reported date: {head['published']}")
        for entry, key, conflict in sorted(source["passages"].values(), key=lambda v: int(v[0]["id"].rsplit("p", 1)[1])):
            label = f"[{entry['id']}; {key}" + ("; possible disagreement" if conflict else "") + "]"
            if entry.get("section"):
                label += " " + entry["section"]
            lines += [label, entry["text"], ""]
    missing = [v for k, v in aspects.items() if not blocks.get(k, {}).get("representatives")]
    if missing:
        lines.append("No retained evidence for: " + "; ".join(missing))
    if not sources and pages:
        lines.append("No passage from these files answered the question:" if local else
                     "No passage from these pages answered the question:")
        lines += [f"- {p['title'] or p['url']} {p['url']}: {p['outcome']}" for p in pages]
    if links:
        lines.append("Video and social results (not read):")
        lines += [f"- {r.get('title') or r['url']} {r['url']}" + (f": {r['snippet']}" if r.get("snippet") else "") for r in links]
    if additional:
        lines.append(f"{len(additional)} more relevant passages from the listed files are not shown; ask a narrower question to see them."
                     if local else
                     f"{len(additional)} more relevant passages from the listed pages are not shown; browse a URL to read more of that page.")
    lines.append("Passages above are original text from the listed files; cite those files." if local else
                 "Passages above are original text from the listed URLs; cite those URLs. To read more of a listed page, browse its URL.")
    return "\n".join(lines)
