from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import LLMConfigurationError, require_deepseek_config
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
        help="已禁用：当前项目只支持联网 + LLM 模式",
    )
    run.add_argument(
        "--live-api",
        action="store_true",
        help="兼容旧用法；当前项目始终使用联网检索",
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
    if args.offline:
        parser.error("当前项目只支持联网 + LLM 模式；--offline 已禁用")
    try:
        require_deepseek_config()
        result = run_research(ResearchRequest(
            question=args.question,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            offline=False,
            live_api=True,
            include_preprints=args.include_preprints,
            full_text=args.full_text,
            debug_steps=not args.no_debug_steps,
        ))
    except LLMConfigurationError as exc:
        print(f"medicine-agent 配置错误: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"medicine-agent 执行失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
