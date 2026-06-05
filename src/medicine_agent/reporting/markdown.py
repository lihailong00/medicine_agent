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
) -> str:
    lines: list[str] = [
        "# 生信科研 Agent 报告",
        "",
        RESEARCH_ONLY_STATEMENT,
        "",
        "## 科研问题与查询分解",
        "",
        f"**问题：** {question}",
        "",
    ]
    lines.extend(f"- {subq}" for subq in subquestions)
    lines += ["", "## 检索日志", ""]
    for plan in source_plans:
        lines.append(f"- 计划 `{plan.source}` 查询 `{plan.query}` — {plan.rationale}")
    for status in source_statuses:
        lines.append(f"- 状态 `{status.provider}`: {status.status}; ids={status.result_ids}; 原因={status.reason}")
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
        lines.append(f"| {item.status} | {item.claim} | {', '.join(item.evidence_refs) or '假设标签'} | {'; '.join(item.limitations)} |")
    lines += ["", "## 机制综合与可检验假设", ""]
    lines.extend(f"- [{item.status}] {item.claim}" for item in evidence)
    lines += ["", "## 冲突证据与局限", ""]
    lines.append("- 离线/模拟文献模式不能建立真实文献支持；实时学术检索必须显式启用并接受审计。")
    lines.append("- 全文检索仅限获批路径：NCBI/PMC XML、构造的 arXiv PDF 产物 URL 或 Semantic Scholar API 片段。")
    lines.append("- LIANA 排名是保留溯源的关联摘要，不是因果证明。")
    lines += ["", "## 生成产物", ""]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines += ["", "## 可复现性说明", "", f"- 运行清单: `{manifest_path}`", "- 原始数据文件只读；派生输出只写入配置的输出目录。"]
    lines += ["", "## 文献记录", ""]
    for paper in papers[:20]:
        lines.append(f"- `{paper.evidence_id}` {paper.title} ({paper.source}, {paper.year})")
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


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
