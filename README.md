# Medicine Agent

离线优先的生信科研 agent CLI/库原型。真实文献检索只能通过显式参数开启，并且严格限制到 PubMed/NCBI E-utilities、arXiv API 与 Semantic Scholar API。

## 运行

```bash
python -m medicine_agent.cli run \
  --question "Which ligand-receptor interactions are most relevant?" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --offline
```

默认/离线流程是确定性的，不需要 API key，也不会安装新依赖。原始 `data/` 文件只作为只读输入；派生的报告、manifest 与表格会写入你指定的输出目录。

如需真实联网文献检索，请显式开启：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "Which ligand-receptor interactions are most relevant in tumor immune communication?" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --live-api
```

`--live-api` 只会对以下地址发起真实 HTTPS 请求：

- `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`：用于 PubMed/NCBI ESearch 与 EFetch
- `https://export.arxiv.org/api/query`：用于 arXiv
- `https://api.semanticscholar.org/graph/v1/paper/search`：用于 Semantic Scholar

如需在实时元数据检索后尝试获取获批路径上的全文证据，请增加 `--full-text`：

```bash
PYTHONPATH=src python -m medicine_agent.cli run \
  --question "Intercellular communication analysis of single-cell transcriptomics data" \
  --data-dir data \
  --output-dir generated/medicine_agent \
  --live-api \
  --full-text
```

`--full-text` 必须与 `--live-api` 一起使用，且不能与 `--offline` 同时使用。它仍然只使用获批来源路径：

- 当存在 PMCID/PMC 链接时，通过 NCBI E-utilities PubMed→PMC ELink 与 PMC EFetch XML 获取文本
- 对 arXiv 记录构造 `https://arxiv.org/pdf/<arxiv-id>` PDF 产物 URL
- 对 Semantic Scholar 记录使用 `https://api.semanticscholar.org/graph/v1/snippet/search` 片段接口

agent 会写出 `artifacts/full_text_results.json`，并在检索成功时写出逐篇论文的文本/PDF 产物。arXiv PDF 会作为审计产物保存，但不会在无额外依赖的前提下解析为全文文本。

`--offline` 始终强制 fixture 模式并阻止真实网络调用，即使同时传入了 `--live-api`。

## 安全与证据策略

- 所有有副作用的动作都必须经过 `SafetyGate`。
- 网络调用只允许访问 PubMed/NCBI、arXiv 与 Semantic Scholar API 主机；其他主机全部阻断。
- 全文检索绝不跟随出版社链接或任意 `openAccessPdf` URL。
- 依赖安装、API key 使用、长任务、脚本执行与覆盖写入在非交互模式下都需要确认。
- 生物医学输出仅限科研用途，不是临床决策支持。
- 综合性主张必须带有 `ClaimStatus` 与证据引用，除非明确标记为假设或超出范围的临床拒答。
