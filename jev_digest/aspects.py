"""Split the stated purpose into aspects without any semantics: separators, question-word clauses, and 'A, B and C' lists."""

from __future__ import annotations

import re

QWORD = r"(?:how|what|which|who|whom|whose|when|where|why|is|are|does|do|did|can|could|should|will|would)\b"
_SEP = re.compile(r";|\s/\s|\n|,\s+(?=(?:and\s+)?" + QWORD + r")|\s+and\s+(?=" + QWORD + r")", re.I)
_LIST = re.compile(r"((?:[A-Z][\w']+, )+[A-Z][\w']+,? and (?:his |her |their |the )?[\w' ]+?)(?=\s+(?:in|during|after|before|on|at)\b|$)")


def aspects_from_query(query: str) -> dict[str, str]:
    body = query
    if ":" in query and len(query.split(":", 1)[1].strip()) > 20:
        body = query.split(":", 1)[1]
    parts = [re.sub(r"^(?:and\s+)", "", p.strip(" .;?"), flags=re.I) for p in _SEP.split(body)]
    parts = [p for p in parts if p]
    # "how and where X" splits on "and where"; a bare question word belongs with the clause after it.
    for i in range(len(parts) - 2, -1, -1):
        if re.fullmatch(QWORD, parts[i], re.I):
            parts[i:i + 2] = [f"{parts[i]} and {parts[i + 1]}"]
    if len(parts) < 2:
        return {"T1": query.strip()}
    out: list[str] = []
    for part in parts:
        m = _LIST.search(part)
        if m:
            head, tail = part[: m.start()], part[m.end():]
            items = [x.strip() for x in re.split(r",\s*|\s+and\s+", m.group(1)) if x.strip()]
            out.extend(f"{head}{item}{tail}".strip() for item in items)
        else:
            out.append(part)
    return {f"T{i + 1}": a for i, a in enumerate(out)}
