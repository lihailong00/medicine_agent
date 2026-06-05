from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import ResearchRequest
from .orchestrator import run_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medicine-agent", description="Offline-first bioinformatics research agent")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run a research workflow")
    run.add_argument("--question", required=True, help="Research question")
    run.add_argument("--data-dir", default="data", help="Input data directory (read-only)")
    run.add_argument("--output-dir", default="generated/medicine_agent", help="Generated output directory")
    run.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Force deterministic offline/mock providers; this is also the default unless --live-api is set",
    )
    run.add_argument(
        "--live-api",
        action="store_true",
        help="Use real allowlisted PubMed/NCBI, arXiv, and Semantic Scholar API queries",
    )
    run.add_argument(
        "--full-text",
        action="store_true",
        help="After live metadata search, retrieve approved full-text/snippet artifacts where available",
    )
    run.add_argument("--include-preprints", action="store_true", help="Include preprint sources in source plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help(sys.stderr)
        return 2
    if args.full_text and not args.live_api:
        parser.error("--full-text requires --live-api because full-text retrieval is a live-network operation")
    if args.full_text and args.offline:
        parser.error("--full-text cannot be combined with --offline")
    try:
        result = run_research(ResearchRequest(
            question=args.question,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            offline=args.offline or not args.live_api,
            live_api=args.live_api,
            include_preprints=args.include_preprints,
            full_text=args.full_text,
        ))
    except Exception as exc:
        print(f"medicine-agent failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
