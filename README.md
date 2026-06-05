# Medicine Agent

默认联网、仅科研用途的生信科研 agent CLI/库原型。真实文献检索默认开启，但始终严格限制到 PubMed/NCBI E-utilities、arXiv API 与 Semantic Scholar API；如需完全离线/fixture 模式，可显式传入 `--offline`。

## 运行

最常用命令只需要问题和输出目录；`data` 是默认输入目录，联网检索也是默认行为：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "帮我调研糖尿病研究的最新进展" \
  --output-dir generated/medicine_agent
```

默认流程会尝试真实联网检索，不需要显式传 `--live-api`。如果配置了 `DEEPSEEK_API_KEY`，还会使用 LLM 做 query 改写、证据抽取和结构化综述；没有 key 时会自动使用确定性降级逻辑。派生的报告、manifest 与表格会写入你指定的输出目录。

运行时会默认向 stderr 打印逐步调试日志，例如“开始文献检索”“跳过本地数据扫描”“开始写入产物”等；stdout 仍然只输出最终 JSON，方便脚本解析。如需关闭逐步日志，可增加 `--no-debug-steps`。

兼容旧命令的 `--live-api` 仍可传入，但现在它只是显式说明“使用联网模式”，不再是必需参数：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "帮我调研糖尿病研究的最新进展" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --live-api
```

如需完全离线 fixture 模式，请显式传入：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "帮我调研糖尿病研究的最新进展" \
  --output-dir generated/medicine_agent \
  --offline
```

## 何时读取 data 目录

agent 不会因为默认存在 `data/` 或传入了 `--data-dir data` 就自动读取本地数据。只有当 `--question` 明确要求查看 data 目录或本地数据文件时，才会扫描并解析 `data/`。例如：

- `帮我调研糖尿病研究的最新进展`：只做文献调研，不读取 `data/`。
- `请使用 data 目录中的 CSV 分析配体受体互作`：会读取 `data/` 并运行 LIANA/CSV 数据通道。
- `请分析我的 csv 文件并结合文献解释`：会读取本地数据文件。

原始 `data/` 文件始终只作为只读输入；派生输出只会写入配置的输出目录。

## 联网范围

默认联网模式会触发真实 HTTPS 请求，但证据性文献检索仍只会访问以下来源：

- `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`：用于 PubMed/NCBI ESearch 与 EFetch
- `https://export.arxiv.org/api/query`：用于 arXiv
- `https://api.semanticscholar.org/graph/v1/paper/search`：用于 Semantic Scholar

### 可选：使用 DeepSeek 做 query 改写、证据抽取与结构化综述

如果设置了 `DEEPSEEK_API_KEY`（或 `MEDICINE_AGENT_DEEPSEEK_API_KEY`），agent 会在联网模式下调用 DeepSeek OpenAI 兼容的 Chat Completions 接口完成更适合 LLM 的语义任务。未设置 key、接口失败或使用 `--offline` 时，会自动降级为确定性规则，不影响主流程。

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek key"
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "帮我调研糖尿病研究的最新进展" \
  --output-dir generated/medicine_agent
```

可选环境变量：

- `DEEPSEEK_MODEL`：默认 `deepseek-chat`
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`，会请求 `/chat/completions`
- `DEEPSEEK_TIMEOUT_SECONDS`：默认与其他实时 API 一致

DeepSeek key 只从环境变量读取，不会写入报告、manifest、搜索日志或安全决策日志。DeepSeek 参与的是语义规划与写作，不替代证据来源：论文元数据、摘要和全文证据仍来自 PubMed/NCBI、arXiv 与 Semantic Scholar 的 allowlist 路径。

LLM 当前用于：

- **query 改写/来源规划**：把中文或口语化科研问题改写成英文检索主题，并判断是否需要 arXiv。
- **证据主张抽取**：从本轮检索到的论文摘要/获批全文片段/本地数据摘要中抽取可引用主张。
- **结构化综述生成**：生成执行摘要、关键发现、证据表、机制综述、可检验假设、冲突/局限与复现说明。

LLM 不用于替代：

- PubMed/NCBI、arXiv、Semantic Scholar 的真实 API 检索。
- URL allowlist、安全门、文件读写边界、CSV/LIANA 统计排序。
- 未检索到的论文引用；LLM 只能引用本次运行允许的 `paper_id` 或数据行号，非法引用会被过滤，没有引用的支持性主张会降级为 `hypothesis`。

LLM 综述会写入 `artifacts/review_synthesis.json`，并同步进入 `run_manifest.json` 与最终 `report.md`。

## 全文/片段检索

如需在实时元数据检索后尝试获取获批路径上的全文证据，请增加 `--full-text`。因为联网现在是默认行为，不需要再额外传 `--live-api`：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "Intercellular communication analysis of single-cell transcriptomics data" \
  --output-dir generated/medicine_agent \
  --full-text
```

`--full-text` 不能与 `--offline` 同时使用。它仍然只使用获批来源路径：

- 当存在 PMCID/PMC 链接时，通过 NCBI E-utilities PubMed→PMC ELink 与 PMC EFetch XML 获取文本
- 对 arXiv 记录构造 `https://arxiv.org/pdf/<arxiv-id>` PDF 产物 URL
- 对 Semantic Scholar 记录使用 `https://api.semanticscholar.org/graph/v1/snippet/search` 片段接口

agent 会写出 `artifacts/full_text_results.json`，并在检索成功时写出逐篇论文的文本/PDF 产物。arXiv PDF 会作为审计产物保存，但不会在无额外依赖的前提下解析为全文文本。

`--offline` 始终强制 fixture 模式并阻止真实网络调用，即使同时传入了 `--live-api`。

## 安全与证据策略

- 所有有副作用的动作都必须经过 `SafetyGate`。
- 证据性文献检索只允许访问 PubMed/NCBI、arXiv 与 Semantic Scholar API 主机；可选 DeepSeek 端点仅用于 query 改写、证据抽取与结构化综述。
- 全文检索绝不跟随出版社链接或任意 `openAccessPdf` URL。
- 依赖安装、API key 使用、长任务、脚本执行与覆盖写入在非交互模式下都需要确认。
- 生物医学输出仅限科研用途，不是临床决策支持。
- 综合性主张必须带有 `ClaimStatus` 与证据引用，除非明确标记为假设或超出范围的临床拒答。
