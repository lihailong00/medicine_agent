"""文献检索计划的自适应来源选择。"""

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
    """使用确定性关键词规则为问题选择来源名称。"""

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
    """为 MVP 离线用例创建简单且确定性的检索分解。"""

    cleaned = " ".join(question.strip().split())
    if not cleaned:
        raise ValueError("必须提供科研问题")
    selected = sources or select_sources(cleaned)
    subquestions = (
        f"哪些原始文献讨论了：{cleaned}？",
        "哪些发现由元数据/摘要支持，哪些仍属于假设？",
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
        "pubmed": "用于同行评议元数据的生物医学/生命科学来源。",
        "arxiv": "allowlist 中的 arXiv API 来源，适合预印本与计算/统计问题。",
        "semantic_scholar": "覆盖面较广的元数据与引用增强来源。",
    }.get(source, "由来源策略选定。")


def _endpoint_family_for_source(source: str) -> str:
    return {
        "pubmed": "ncbi_eutils",
        "arxiv": "arxiv_atom",
        "semantic_scholar": "s2_graph",
    }.get(source, "unknown")
