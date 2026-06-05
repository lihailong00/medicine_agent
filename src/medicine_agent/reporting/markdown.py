from __future__ import annotations

from typing import Mapping

from medicine_agent.models import DataFileRecord, EvidenceItem, PaperRecord, SourcePlan, SourceStatus

RESEARCH_ONLY_STATEMENT = (
    "仅限科研的安全声明：本报告仅用于生信科研解读，不是临床决策支持，"
    "也不得用于诊断、治疗、处方或患者管理。"
)


def render_report(
    question: str,
    subquestions: list[str],
    source_plans: list[SourcePlan],
    source_statuses: list[SourceStatus],
    papers: list[PaperRecord],
    data_records: list[DataFileRecord],
    liana_summary: dict,
    evidence: list[EvidenceItem],
    artifact_paths: list[str],
    manifest_path: str,
    full_text_records: list[Mapping[str, object]] | None = None,
    full_text_summary: Mapping[str, object] | None = None,
    review_synthesis: Mapping[str, object] | None = None,
) -> str:
    citation_map = _build_citation_map(papers, liana_summary, evidence, review_synthesis)
    lines: list[str] = [
        "# 生信科研综述报告",
        "",
        RESEARCH_ONLY_STATEMENT,
        "",
        "## 摘要",
        "",
    ]
    lines.extend(_format_article_abstract(question, review_synthesis, citation_map))
    lines += [
        "",
        "## 关键词",
        "",
        "- 生信科研；文献综述；证据表；全文检索；可复现分析",
        "",
        "## 引言",
        "",
        "生信科研问题通常同时涉及文献证据、数据证据和可复现实验假设。"
        "本报告把用户问题转化为可审计的检索与证据综合流程，并把每条支持性结论"
        "尽量绑定到本次运行检索到的论文或本地数据行号。",
        "",
        "## 研究问题与方法",
        "",
        f"**问题：** {question}",
        "",
        "本报告采用 query 分解、获批学术 API 检索、获批全文/片段检索、"
        "本地数据按需只读解析和 LLM 结构化证据综合的流程生成。"
        "所有支持性主张必须追溯到本次运行检索到的论文 ID 或本地数据行号；"
        "无法追溯的语句保守标记为假设。",
        "",
        "### 查询分解",
        "",
    ]
    lines.extend(f"- {subq}" for subq in subquestions)
    lines += ["", "### 检索来源", ""]
    lines.extend(_format_source_methods(source_plans, source_statuses))
    lines += ["", "## 主要发现", ""]
    lines.extend(_format_cited_findings(review_synthesis, evidence, citation_map))
    lines += ["", "## 机制讨论与可检验假设", ""]
    mechanism_lines = _format_mechanism_and_hypotheses(review_synthesis, citation_map)
    if mechanism_lines:
        lines.extend(mechanism_lines)
    else:
        lines.extend(f"- [{item.status}] {item.claim}{_citation_suffix(item.evidence_refs, citation_map)}" for item in evidence)
    lines += ["", "## 局限性与冲突证据", ""]
    synthesis_limits = _string_list(_as_mapping(review_synthesis).get("limitations_conflicts"))
    if synthesis_limits:
        lines.extend(f"- {item}" for item in synthesis_limits)
    lines.append("- 文献检索仅限获批 API 来源；检索日志与安全决策必须接受审计。")
    lines.append("- 全文检索仅限获批路径：NCBI/PMC XML、构造的 arXiv PDF 产物 URL 或 Semantic Scholar API 片段。")
    lines.append("- LIANA 排名是保留溯源的关联摘要，不是因果证明。")
    lines += ["", "## 结构化综述元信息", ""]
    lines.extend(_format_review_synthesis(review_synthesis, citation_map))
    lines += ["", "## 全文检索摘要", ""]
    lines.extend(_format_full_text_records(full_text_records, full_text_summary))
    lines += ["", "## 数据输入清单与方法", ""]
    if data_records:
        for rec in data_records:
            lines.append(f"- `{rec.path}` ({rec.file_type}) — {rec.parser_status}; sha256={rec.sha256}; 警告={rec.warnings}")
    else:
        lines.append("- 未读取本地数据文件；只有 query 明确要求查看 data 目录或数据文件时才会扫描。")
    lines += ["", "## LIANA 排名互作", "", f"排序方法: {liana_summary.get('ranking_method')}", ""]
    top_interactions = liana_summary.get("top_interactions", [])[:10]
    if top_interactions:
        for item in top_interactions:
            lines.append(
                f"- `{item['source_file']}#row-{item['row_index']}`: {item['source_cell']} -> {item['target_cell']} "
                f"经由 {item['ligand']} -> {item['receptor']} (pvalue={item['pvalue']}, lr.mean={item['lr_mean']})"
            )
    else:
        lines.append("- 未生成 LIANA 排名互作。")
    for warning in liana_summary.get("warnings", []):
        lines.append(
            f"- 数据/LIANA 说明：{warning}"
        )
    lines += ["", "## 证据表", "", "| ClaimStatus | 主张 | 证据引用 | 局限 |", "| --- | --- | --- | --- |"]
    for item in evidence:
        lines.append(
            f"| {item.status} | {item.claim}{_citation_suffix(item.evidence_refs, citation_map)} | "
            f"{_citation_cell(item.evidence_refs, citation_map)} | {'; '.join(item.limitations)} |"
        )
    lines += ["", "## 参考文献与证据来源", ""]
    lines.extend(_format_references(citation_map, papers, liana_summary))
    lines += ["", "## 可复现性说明", "", f"- 运行清单: `{manifest_path}`", "- 原始数据文件只读；派生输出只写入配置的输出目录。"]
    lines.extend(f"- {item}" for item in _string_list(_as_mapping(review_synthesis).get("reproducibility_notes")))
    lines += ["", "## 附录：检索日志与生成产物", ""]
    for plan in source_plans:
        lines.append(f"- 计划 `{plan.source}` 查询 `{plan.query}` — {plan.rationale}")
    for status in source_statuses:
        lines.append(f"- 状态 `{status.provider}`: {status.status}; ids={status.result_ids}; 原因={status.reason}")
    lines += ["", "### 生成产物", ""]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines.append("")
    return "\n".join(lines)


def _format_full_text_records(
    records: list[Mapping[str, object]] | None,
    summary: Mapping[str, object] | None = None,
) -> list[str]:
    if records is None:
        if summary and summary.get("requested"):
            reason = summary.get("reason") or "全文检索未启用。"
            return [f"- 已请求全文检索但未运行: {reason}"]
        return ["- 未请求全文检索。"]
    if not records:
        return ["- 已请求全文检索，但没有可用文献记录。"]
    lines: list[str] = []
    for record in records[:20]:
        candidate = _as_mapping(record.get("candidate"))
        status = _as_mapping(record.get("status"))
        title = str(candidate.get("title") or "未命名记录")
        paper_id = str(candidate.get("paper_id") or status.get("paper_id") or "未知")
        provider = str(status.get("provider") or candidate.get("provider") or "未知")
        status_value = str(status.get("status") or "未知")
        scope = str(status.get("scope") or "未知")
        artifact = status.get("artifact_path")
        reason = status.get("reason")
        artifact_text = f"; 产物=`{artifact}`" if artifact else ""
        reason_text = f"; 原因={reason}" if reason else ""
        lines.append(f"- `{paper_id}` {title} — 提供器={provider}; 状态={status_value}; 范围={scope}{artifact_text}{reason_text}")
    return lines


def _format_article_abstract(
    question: str,
    review_synthesis: Mapping[str, object] | None,
    citation_map: Mapping[str, int],
) -> list[str]:
    synthesis = _as_mapping(review_synthesis)
    executive = synthesis.get("executive_summary")
    if isinstance(executive, str) and executive.strip():
        abstract = executive.strip()
    else:
        abstract = f"本报告围绕“{question}”生成可审计的生信科研综述。"
    cited_rows = _mapping_list(synthesis.get("key_findings")) or _mapping_list(synthesis.get("evidence_table"))
    first_refs: list[str] = []
    for row in cited_rows:
        first_refs.extend(_string_list(row.get("evidence_refs")))
        if len(first_refs) >= 3:
            break
    suffix = _citation_suffix(first_refs, citation_map)
    if suffix:
        abstract = f"{abstract} 代表性证据来源见{suffix}。"
    return [abstract]


def _format_source_methods(source_plans: list[SourcePlan], source_statuses: list[SourceStatus]) -> list[str]:
    lines: list[str] = []
    if source_plans:
        planned = "；".join(f"{plan.source}: `{plan.query}`" for plan in source_plans)
        lines.append(f"- 检索计划：{planned}。")
    if source_statuses:
        status_text = "；".join(
            f"{status.provider}={status.status}, 返回 {len(status.result_ids)} 个 ID"
            for status in source_statuses
        )
        lines.append(f"- 检索状态：{status_text}。")
    return lines or ["- 未记录可用检索来源。"]


def _format_cited_findings(
    review_synthesis: Mapping[str, object] | None,
    evidence: list[EvidenceItem],
    citation_map: Mapping[str, int],
) -> list[str]:
    synthesis = _as_mapping(review_synthesis)
    findings = _mapping_list(synthesis.get("key_findings")) or _mapping_list(synthesis.get("evidence_table"))
    if findings:
        return [_format_synthesis_row(item, citation_map) for item in findings]
    return [
        f"- {item.claim}{_citation_suffix(item.evidence_refs, citation_map)}"
        f"（证据等级：{item.status}；置信度：{item.confidence}）"
        for item in evidence
    ]


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _format_review_synthesis(
    review_synthesis: Mapping[str, object] | None,
    citation_map: Mapping[str, int],
) -> list[str]:
    synthesis = _as_mapping(review_synthesis)
    if not synthesis:
        return ["- 未生成结构化综述。"]
    lines: list[str] = []
    source = synthesis.get("source") or "unknown"
    reason = synthesis.get("reason") or "未提供。"
    lines.append(f"- 综述生成器：`{source}`；说明：{reason}")
    executive = synthesis.get("executive_summary")
    if isinstance(executive, str) and executive.strip():
        lines += ["", "### 执行摘要", "", executive.strip()]
    findings = _mapping_list(synthesis.get("key_findings"))
    if findings:
        lines += ["", "### 关键发现", ""]
        for item in findings:
            lines.append(_format_synthesis_row(item, citation_map))
    return lines


def _format_mechanism_and_hypotheses(
    review_synthesis: Mapping[str, object] | None,
    citation_map: Mapping[str, int],
) -> list[str]:
    synthesis = _as_mapping(review_synthesis)
    lines: list[str] = []
    mechanisms = _string_list(synthesis.get("mechanism_review"))
    finding_refs = _refs_from_rows(_mapping_list(synthesis.get("key_findings")) or _mapping_list(synthesis.get("evidence_table")))
    mechanism_suffix = _citation_suffix(finding_refs[:4], citation_map)
    if mechanisms:
        lines.append("### 机制综述")
        lines.extend(f"- {item}{mechanism_suffix}" for item in mechanisms)
    hypotheses = _string_list(synthesis.get("hypotheses"))
    if hypotheses:
        if lines:
            lines.append("")
        lines.append("### 可检验假设")
        lines.extend(f"- {item}" for item in hypotheses)
    return lines


def _format_synthesis_row(item: Mapping[str, object], citation_map: Mapping[str, int]) -> str:
    claim = str(item.get("claim") or "未命名主张")
    status = str(item.get("status") or "hypothesis")
    refs = _string_list(item.get("evidence_refs"))
    confidence = str(item.get("confidence") or "medium")
    limitations = _string_list(item.get("limitations"))
    refs_text = _citation_cell(refs, citation_map)
    limitation_text = f"；局限：{'; '.join(limitations)}" if limitations else ""
    return f"- {claim}{_citation_suffix(refs, citation_map)}（证据等级：{status}；置信度：{confidence}；引用：{refs_text}{limitation_text}）"


def _build_citation_map(
    papers: list[PaperRecord],
    liana_summary: Mapping[str, object],
    evidence: list[EvidenceItem],
    review_synthesis: Mapping[str, object] | None,
) -> dict[str, int]:
    refs: list[str] = []

    def add(ref: object) -> None:
        if isinstance(ref, str) and ref.strip() and ref not in refs:
            refs.append(ref)

    for paper in papers:
        add(paper.evidence_id)
    for interaction in _mapping_list(liana_summary.get("top_interactions")):
        source_file = interaction.get("source_file")
        row_index = interaction.get("row_index")
        if source_file is not None and row_index is not None:
            add(f"{source_file}#row-{row_index}")
    synthesis = _as_mapping(review_synthesis)
    for row in _mapping_list(synthesis.get("key_findings")) + _mapping_list(synthesis.get("evidence_table")):
        for ref in _string_list(row.get("evidence_refs")):
            add(ref)
    for evidence_item in evidence:
        for ref in evidence_item.evidence_refs:
            add(ref)
    return {ref: index + 1 for index, ref in enumerate(refs)}


def _citation_suffix(refs: list[str], citation_map: Mapping[str, int]) -> str:
    numbers = _citation_numbers(refs, citation_map)
    if not numbers:
        return ""
    return f" [{', '.join(str(number) for number in numbers)}]"


def _citation_cell(refs: list[str], citation_map: Mapping[str, int]) -> str:
    numbers = _citation_numbers(refs, citation_map)
    if not numbers:
        return "无直接引用/假设"
    return ", ".join(f"[{number}]" for number in numbers)


def _citation_numbers(refs: list[str], citation_map: Mapping[str, int]) -> list[int]:
    numbers: list[int] = []
    for ref in refs:
        number = citation_map.get(ref)
        if number is not None and number not in numbers:
            numbers.append(number)
    return numbers


def _format_references(
    citation_map: Mapping[str, int],
    papers: list[PaperRecord],
    liana_summary: Mapping[str, object],
) -> list[str]:
    if not citation_map:
        return ["- 未生成可编号参考来源。"]
    paper_by_ref = {paper.evidence_id: paper for paper in papers}
    data_by_ref = _data_reference_lookup(liana_summary)
    lines: list[str] = []
    for ref, number in sorted(citation_map.items(), key=lambda item: item[1]):
        paper = paper_by_ref.get(ref)
        if paper is not None:
            lines.append(f"- [{number}] {_format_paper_reference(paper)}")
            continue
        data_item = data_by_ref.get(ref)
        if data_item is not None:
            lines.append(f"- [{number}] {_format_data_reference(ref, data_item)}")
            continue
        lines.append(f"- [{number}] 运行证据 `{ref}`。")
    return lines


def _format_paper_reference(paper: PaperRecord) -> str:
    authors = ", ".join(paper.authors[:6]) if paper.authors else paper.source
    year = str(paper.year) if paper.year else "n.d."
    venue = f" {paper.venue}." if paper.venue else ""
    identifiers = []
    if paper.pmid:
        identifiers.append(f"PMID: {paper.pmid}")
    if paper.pmcid:
        identifiers.append(f"PMCID: {paper.pmcid}")
    if paper.doi:
        identifiers.append(f"DOI: {paper.doi}")
    if paper.arxiv_id:
        identifiers.append(f"arXiv: {paper.arxiv_id}")
    if paper.semantic_scholar_id:
        identifiers.append(f"Semantic Scholar: {paper.semantic_scholar_id}")
    id_text = f" {'; '.join(identifiers)}." if identifiers else ""
    url_text = f" {paper.source_url}" if paper.source_url else ""
    return f"{authors} ({year}). {paper.title}.{venue}{id_text}{url_text}".strip()


def _data_reference_lookup(liana_summary: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    lookup: dict[str, Mapping[str, object]] = {}
    for item in _mapping_list(liana_summary.get("top_interactions")):
        source_file = item.get("source_file")
        row_index = item.get("row_index")
        if source_file is not None and row_index is not None:
            lookup[f"{source_file}#row-{row_index}"] = item
    return lookup


def _format_data_reference(ref: str, item: Mapping[str, object]) -> str:
    ligand = item.get("ligand") or "未知 ligand"
    receptor = item.get("receptor") or "未知 receptor"
    source_cell = item.get("source_cell") or "未知 source"
    target_cell = item.get("target_cell") or "未知 target"
    pvalue = item.get("pvalue")
    lr_mean = item.get("lr_mean")
    return (
        f"本地数据 `{ref}`：{source_cell} -> {target_cell} 的 {ligand} -> {receptor} "
        f"互作；pvalue={pvalue}, lr.mean={lr_mean}。"
    )


def _refs_from_rows(rows: list[Mapping[str, object]]) -> list[str]:
    refs: list[str] = []
    for row in rows:
        for ref in _string_list(row.get("evidence_refs")):
            if ref not in refs:
                refs.append(ref)
    return refs


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]
