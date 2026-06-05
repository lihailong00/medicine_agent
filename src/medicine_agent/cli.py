from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import ResearchRequest
from .orchestrator import run_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medicine-agent", description="默认联网、仅科研用途的生信科研 agent")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="运行科研工作流")
    run.add_argument("--question", required=True, help="科研问题")
    run.add_argument("--data-dir", default="data", help="输入数据目录（只读，默认 data）")
    run.add_argument("--output-dir", default="generated/medicine_agent", help="生成结果目录")
    run.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="强制使用确定性的离线/模拟提供器；默认会使用受 allowlist 限制的联网检索",
    )
    run.add_argument(
        "--live-api",
        action="store_true",
        help="兼容旧用法；现在默认已启用受 allowlist 限制的联网检索",
    )
    run.add_argument(
        "--full-text",
        action="store_true",
        help="在实时元数据检索后，尽可能获取获批路径上的全文/片段产物",
    )
    run.add_argument("--include-preprints", action="store_true", help="在来源计划中包含预印本来源")
    run.add_argument("--no-debug-steps", action="store_true", help="不向 stderr 打印逐步调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help(sys.stderr)
        return 2
    if args.full_text and args.offline:
        parser.error("--full-text 不能与 --offline 同时使用，因为全文检索属于实时网络操作")
    live_api = not args.offline
    try:
        result = run_research(ResearchRequest(
            question=args.question,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            offline=args.offline,
            live_api=live_api,
            include_preprints=args.include_preprints,
            full_text=args.full_text,
            debug_steps=not args.no_debug_steps,
        ))
    except Exception as exc:
        print(f"medicine-agent 执行失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
