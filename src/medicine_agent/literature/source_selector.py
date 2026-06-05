"""Adaptive source selection for literature search plans."""

from __future__ import annotations

from .base import QueryDecomposition, SearchQueryRecord

BIOMEDICAL_TERMS = {
    "cancer",
    "tumor",
    "tumour",
    "cell",
    "immune",
    "ligand",
    "receptor",
    "gene",
    "protein",
    "pathway",
    "single-cell",
    "scrna",
    "liana",
    "biomarker",
}
PREPRINT_TERMS = {"preprint", "emerging", "latest", "novel", "recent"}
COMPUTATIONAL_TERMS = {
    "machine learning",
    "deep learning",
    "model",
    "algorithm",
    "statistical",
    "bayesian",
    "embedding",
    "transformer",
}


def select_sources(question: str) -> tuple[str, ...]:
    """Select source names for a question using deterministic keyword rules."""

    normalized = question.lower()
    selected: list[str] = []
    if any(term in normalized for term in BIOMEDICAL_TERMS):
        selected.extend(["pubmed", "semantic_scholar"])
    if any(term in normalized for term in PREPRINT_TERMS):
        selected.append("arxiv")
    if any(term in normalized for term in COMPUTATIONAL_TERMS):
        selected.append("arxiv")
    if not selected:
        selected.extend(["pubmed", "semantic_scholar"])
    return tuple(dict.fromkeys(selected))


def decompose_question(question: str, sources: tuple[str, ...] | None = None) -> QueryDecomposition:
    """Create a simple deterministic search decomposition for MVP offline use."""

    cleaned = " ".join(question.strip().split())
    if not cleaned:
        raise ValueError("question is required")
    selected = sources or select_sources(cleaned)
    subquestions = (
        f"What primary literature addresses: {cleaned}?",
        "Which findings are metadata/abstract-supported versus hypotheses?",
    )
    queries = tuple(
        SearchQueryRecord(
            provider=source,
            query=_provider_query(cleaned, source),
            rationale=_rationale_for_source(source),
            endpoint_family=_endpoint_family_for_source(source),
        )
        for source in selected
    )
    return QueryDecomposition(question=cleaned, subquestions=subquestions, queries=queries)


def _provider_query(question: str, source: str) -> str:
    if source == "pubmed":
        return f"({question}) AND (review OR mechanism OR single-cell)"
    if source == "arxiv":
        return f"{question} computational biology"
    if source == "semantic_scholar":
        return question
    return question


def _rationale_for_source(source: str) -> str:
    return {
        "pubmed": "Biomedical/life-science source for peer-reviewed metadata.",
        "arxiv": "Allowlisted arXiv API source for preprint and computational/statistical questions.",
        "semantic_scholar": "Broad metadata and citation enrichment source.",
    }.get(source, "Selected by source policy.")


def _endpoint_family_for_source(source: str) -> str:
    return {
        "pubmed": "ncbi_eutils",
        "arxiv": "arxiv_atom",
        "semantic_scholar": "s2_graph",
    }.get(source, "unknown")
