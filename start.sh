#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
  cat <<'USAGE'
用法：
  ./start.sh

可选覆盖：
  ./start.sh "新的科研问题"
  ./start.sh --question "新的科研问题" --output-dir generated/custom_review
  QUESTION="新的科研问题" ./start.sh

默认内置问题：
  请先读取根目录 data/ 下的 CSV/LIANA 结果，找出最显著的 ligand-receptor 互作、source/target 细胞对和 top genes，然后结合 PubMed 和 Semantic Scholar 检索相关文献，解释这些互作在生信疾病研究中的可能意义。

脚本内置参数：
  - 问题：请先读取根目录 data/ 下的 CSV/LIANA 结果，找出最显著的 ligand-receptor 互作、source/target 细胞对和 top genes，然后结合 PubMed 和 Semantic Scholar 检索相关文献，解释这些互作在生信疾病研究中的可能意义。
  - data 目录：./data
  - 输出目录：generated/medicine_agent
  - 联网检索：开启
  - 全文/片段检索：开启
  - 运行结束后重点查看 generated/medicine_agent/report.md

.env 至少需要包含：
  DEEPSEEK_API_KEY=你的key
  DEEPSEEK_MODEL=deepseek-v4-pro
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -z "${DEEPSEEK_API_KEY:-${MEDICINE_AGENT_DEEPSEEK_API_KEY:-}}" ]]; then
  echo "错误：缺少 DEEPSEEK_API_KEY 或 MEDICINE_AGENT_DEEPSEEK_API_KEY。请先在 .env 中配置。" >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "错误：缺少 DEEPSEEK_MODEL。请先在 .env 中配置，例如 DEEPSEEK_MODEL=deepseek-v4-pro。" >&2
  exit 2
fi

DEFAULT_QUESTION="请先读取根目录 data/ 下的 CSV/LIANA 结果，找出最显著的 ligand-receptor 互作、source/target 细胞对和 top genes，然后结合 PubMed 和 Semantic Scholar 检索相关文献，解释这些互作在生信疾病研究中的可能意义。"
DEFAULT_DATA_DIR="data"
DEFAULT_OUTPUT_DIR="generated/medicine_agent"

ARGS=("$@")

if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--question "${QUESTION:-$DEFAULT_QUESTION}")
elif [[ ${#ARGS[@]} -gt 0 && "${ARGS[0]}" != -* ]]; then
  QUESTION_TEXT="${ARGS[0]}"
  ARGS=(--question "$QUESTION_TEXT" "${ARGS[@]:1}")
elif [[ -n "${QUESTION:-}" ]]; then
  ARGS=(--question "$QUESTION" "${ARGS[@]}")
fi

HAS_QUESTION=0
HAS_DATA_DIR=0
HAS_OUTPUT_DIR=0
HAS_LIVE_API=0
HAS_FULL_TEXT_MODE=0
for arg in "${ARGS[@]}"; do
  case "$arg" in
    --question|--question=*) HAS_QUESTION=1 ;;
    --data-dir|--data-dir=*) HAS_DATA_DIR=1 ;;
    --output-dir|--output-dir=*) HAS_OUTPUT_DIR=1 ;;
    --live-api) HAS_LIVE_API=1 ;;
    --full-text|--no-full-text) HAS_FULL_TEXT_MODE=1 ;;
  esac
done

if [[ "$HAS_QUESTION" -ne 1 ]]; then
  ARGS=(--question "$DEFAULT_QUESTION" "${ARGS[@]}")
fi

if [[ "$HAS_DATA_DIR" -ne 1 ]]; then
  ARGS+=(--data-dir "$DEFAULT_DATA_DIR")
fi

if [[ "$HAS_OUTPUT_DIR" -ne 1 ]]; then
  ARGS+=(--output-dir "$DEFAULT_OUTPUT_DIR")
fi

if [[ "$HAS_LIVE_API" -ne 1 ]]; then
  ARGS+=(--live-api)
fi

if [[ "$HAS_FULL_TEXT_MODE" -ne 1 ]]; then
  ARGS+=(--full-text)
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m medicine_agent.cli run "${ARGS[@]}"
