from __future__ import annotations

from typing import Mapping

from medicine_agent.models import EvidenceItem, PaperRecord


def build_evidence(
    question: str,
    papers: list[PaperRecord],
    interactions: list[dict],
    full_text_records: list[Mapping[str, object]] | None = None,
    full_text_enabled: bool = False,
    live_literature_enabled: bool = False,
) -> list[EvidenceItem]:
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
        if full_text_enabled:
            scopes = _evidence_scopes(full_text_records or [])
            claim = "Live literature retrieval produced traceable records with scoped evidence boundaries."
            confidence = "medium" if "full_text_xml" in scopes else "low"
            limitations = [
                "Evidence scope is per record: abstract metadata, Semantic Scholar snippets, PMC XML text, or arXiv PDF artifact bytes.",
                "arXiv PDF artifacts are stored for audit but are not parsed as dependency-free full text.",
            ]
            if not scopes:
                limitations.append("No approved full-text route succeeded; synthesis should rely on metadata/abstract evidence only.")
        elif live_literature_enabled:
            claim = "Live literature metadata/abstract retrieval produced traceable records from approved scholarly APIs."
            confidence = "medium"
            limitations = [
                "Live metadata/abstract records are not full-text evidence; enable --full-text to attempt approved full-text/snippet artifacts.",
            ]
        else:
            claim = "Offline literature search produced traceable placeholder records for reproducible workflow validation."
            confidence = "low"
            limitations = ["Offline mock records are not scientific evidence; enable reviewed live providers for real literature retrieval."]
        evidence.append(EvidenceItem(
            claim=claim,
            status="literature_supported",
            evidence_refs=[p.evidence_id for p in papers[:5]],
            confidence=confidence,
            limitations=limitations,
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


def _evidence_scopes(records: list[Mapping[str, object]]) -> set[str]:
    scopes: set[str] = set()
    for record in records:
        status = record.get("status")
        if not isinstance(status, Mapping):
            continue
        if isinstance(status.get("scope"), str):
            scopes.add(str(status["scope"]))
    return scopes
