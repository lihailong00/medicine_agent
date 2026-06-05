from __future__ import annotations

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
    lines.append("- LIANA rankings are association summaries with preserved provenance, not causal proof.")
    lines += ["", "## Generated Artifacts", ""]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines += ["", "## Reproducibility Notes", "", f"- Manifest: `{manifest_path}`", "- Original data files were read only; derived outputs were written only under the configured output directory."]
    lines += ["", "## Literature Records", ""]
    for paper in papers[:20]:
        lines.append(f"- `{paper.evidence_id}` {paper.title} ({paper.source}, {paper.year})")
    lines.append("")
    return "\n".join(lines)
