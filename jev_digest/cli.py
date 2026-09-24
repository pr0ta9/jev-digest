"""Command line: jev-digest "question" [--search-query ...] [--urls file] [--top N] [--json] [--no-cache] [--trace] [--open RUN_ID PASSAGE_ID]"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .pipeline import open_passage, run_digest


def main() -> None:
    ap = argparse.ArgumentParser(prog="jev-digest", description="Purpose-driven passage digest for agents (no generation).")
    ap.add_argument("query", nargs="?", help="the information need, in plain language; separate aspects with ';'")
    ap.add_argument("--search-query", action="append", help="search-engine query; repeat the flag to run several in parallel and merge (default: the question)")
    ap.add_argument("--urls", help="file with one URL per line; skips search")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="print the result as JSON (without the markdown)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--trace", action="store_true", help="keep every Jev request/response under the run directory")
    ap.add_argument("--open", nargs=2, metavar=("RUN_ID", "PASSAGE_ID"), help="print one passage's exact text and URL")
    args = ap.parse_args()
    if args.open:
        p = open_passage(*args.open)
        print(json.dumps(p, ensure_ascii=False, indent=1) if p else "not found")
        return
    if not args.query:
        ap.error("query is required")
    urls = [u.strip() for u in open(args.urls, encoding="utf-8") if u.strip()] if args.urls else None
    sq = args.search_query or []
    result = asyncio.run(run_digest(args.query, sq[0] if len(sq) == 1 else None, urls, args.top, use_cache=not args.no_cache, trace=args.trace,
                                    search_queries=sq if len(sq) > 1 else None))
    if "error" in result:
        print(json.dumps(result, ensure_ascii=False, indent=1), file=sys.stderr)
        sys.exit(2)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k not in ("digest_evidence", "blocks")}, ensure_ascii=False, indent=1))
    else:
        print(result["digest_evidence"])
        c, t, j = result["counts"], result["timings"], result["jev"]
        print(f"\n[run {result['run_id']}] {c['pages']} pages, {c['passages']} passages, {c['kept']} kept | "
              f"{t['total']} s total (search {t['search']}, fetch {t['fetch_http_wave']}, jev {t['jev_round1']}+{t['jev_round2']}) | "
              f"tokens raw {result['tokens']['raw_pages']} -> evidence {result['tokens']['digest_evidence']} | "
              f"jev {j['requests']} requests, {j['input_tokens']} input tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
