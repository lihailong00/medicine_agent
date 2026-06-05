from __future__ import annotations

from medicine_agent.models import EvidenceItem, PaperRecord


def build_evidence(question: str, papers: list[PaperRecord], interactions: list[dict]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    if interactions:
        top = interactions[0]
        ref = f"{top['source_file']}#row-{top['row_index']}"
        evidence.append(EvidenceItem(
            claim=(
                f"The strongest ranked LIANA interaction in the provided data is "
                f"{top['ligand']} -> {top['receptor']} from {top['source_cell']} to {top['target_cell']}."
            ),
            status="data_supported",
            evidence_refs=[ref],
            confidence="medium",
            limitations=["Ranking is statistical/provenance evidence only and does not establish mechanism causality."],
        ))
    if papers:
        evidence.append(EvidenceItem(
            claim="Offline literature search produced traceable placeholder records for reproducible workflow validation.",
            status="literature_supported",
            evidence_refs=[p.evidence_id for p in papers[:5]],
            confidence="low",
            limitations=["Offline mock records are not scientific evidence; enable reviewed live providers for real literature retrieval."],
        ))
    evidence.append(EvidenceItem(
        claim=f"Mechanistic interpretation for '{question}' remains a testable hypothesis until supported by direct literature and experimental validation.",
        status="hypothesis",
        evidence_refs=[],
        confidence="low",
        limitations=["Generated for research planning only; not diagnostic or therapeutic advice."],
    ))
    return evidence


def validate_evidence(items: list[EvidenceItem]) -> None:
    for item in items:
        EvidenceItem(item.claim, item.status, list(item.evidence_refs), item.confidence, list(item.limitations))
