from __future__ import annotations

from typing import Mapping

from medicine_agent.models import DataFileRecord, EvidenceItem, PaperRecord, SourcePlan, SourceStatus

RESEARCH_ONLY_STATEMENT = (
    "Research-only safety statement: This report is for bioinformatics research interpretation only. "
    "It is not clinical decision support and must not be used for diagnosis, treatment, prescribing, or patient management."
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
        "# Bioinformatics Research Agent Report",
        "",
        RESEARCH_ONLY_STATEMENT,
        "",
        "## Research Question and Decomposition",
        "",
        f"**Question:** {question}",
        "",
    ]
    lines.extend(f"- {subq}" for subq in subquestions)
    lines += ["", "## Search Log", ""]
    for plan in source_plans:
        lines.append(f"- Planned `{plan.source}` query `{plan.query}` — {plan.rationale}")
    for status in source_statuses:
        lines.append(f"- Status `{status.provider}`: {status.status}; ids={status.result_ids}; reason={status.reason}")
    lines += ["", "## Full-text Retrieval Summary", ""]
    lines.extend(_format_full_text_records(full_text_records, full_text_summary))
    lines += ["", "## Data Input Manifest and Methods", ""]
    for rec in data_records:
        lines.append(f"- `{rec.path}` ({rec.file_type}) — {rec.parser_status}; sha256={rec.sha256}; warnings={rec.warnings}")
    lines += ["", "## LIANA Ranked Interactions", "", f"Ranking method: {liana_summary.get('ranking_method')}", ""]
    for item in liana_summary.get("top_interactions", [])[:10]:
        lines.append(
            f"- `{item['source_file']}#row-{item['row_index']}`: {item['source_cell']} -> {item['target_cell']} "
            f"via {item['ligand']} -> {item['receptor']} (pvalue={item['pvalue']}, lr.mean={item['lr_mean']})"
        )
    lines += ["", "## Evidence Table", "", "| ClaimStatus | Claim | Evidence refs | Limitations |", "| --- | --- | --- | --- |"]
    for item in evidence:
        lines.append(f"| {item.status} | {item.claim} | {', '.join(item.evidence_refs) or 'hypothesis label'} | {'; '.join(item.limitations)} |")
    lines += ["", "## Mechanism Synthesis and Testable Hypotheses", ""]
    lines.extend(f"- [{item.status}] {item.claim}" for item in evidence)
    lines += ["", "## Conflicting Evidence and Limitations", ""]
    lines.append("- Offline/mock literature mode cannot establish real literature support; live scholarly retrieval must be explicitly enabled and audited.")
    lines.append("- Full-text retrieval is scoped to approved routes only: NCBI/PMC XML, constructed arXiv PDF artifact URLs, or Semantic Scholar API snippets.")
    lines.append("- LIANA rankings are association summaries with preserved provenance, not causal proof.")
    lines += ["", "## Generated Artifacts", ""]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines += ["", "## Reproducibility Notes", "", f"- Manifest: `{manifest_path}`", "- Original data files were read only; derived outputs were written only under the configured output directory."]
    lines += ["", "## Literature Records", ""]
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
            reason = summary.get("reason") or "Full-text retrieval was not enabled."
            return [f"- Full-text retrieval was requested but not run: {reason}"]
        return ["- Full-text retrieval was not requested."]
    if not records:
        return ["- Full-text retrieval was requested, but no literature records were available."]
    lines: list[str] = []
    for record in records[:20]:
        candidate = _as_mapping(record.get("candidate"))
        status = _as_mapping(record.get("status"))
        title = str(candidate.get("title") or "Untitled record")
        paper_id = str(candidate.get("paper_id") or status.get("paper_id") or "unknown")
        provider = str(status.get("provider") or candidate.get("provider") or "unknown")
        status_value = str(status.get("status") or "unknown")
        scope = str(status.get("scope") or "unknown")
        artifact = status.get("artifact_path")
        reason = status.get("reason")
        artifact_text = f"; artifact=`{artifact}`" if artifact else ""
        reason_text = f"; reason={reason}" if reason else ""
        lines.append(f"- `{paper_id}` {title} — provider={provider}; status={status_value}; scope={scope}{artifact_text}{reason_text}")
    return lines


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
