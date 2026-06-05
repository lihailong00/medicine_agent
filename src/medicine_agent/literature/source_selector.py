"""文献检索计划的自适应来源选择。"""

from __future__ import annotations

import re

from medicine_agent.safety import SafetyGate

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
    "糖尿病",
    "肿瘤",
    "癌症",
    "免疫",
    "细胞",
    "受体",
    "配体",
    "基因",
    "蛋白",
}
PREPRINT_TERMS = {"preprint", "emerging", "novel", "预印本"}
COMPUTATIONAL_TERMS = {
    "machine learning",
    "deep learning",
    "model",
    "algorithm",
    "statistical",
    "bayesian",
    "embedding",
    "transformer",
    "computational biology",
}

CHINESE_QUERY_EXPANSIONS = (
    ("2型糖尿病", "type 2 diabetes mellitus"),
    ("二型糖尿病", "type 2 diabetes mellitus"),
    ("1型糖尿病", "type 1 diabetes mellitus"),
    ("一型糖尿病", "type 1 diabetes mellitus"),
    ("糖尿病", "diabetes mellitus"),
    ("最新进展", "recent advances"),
    ("研究进展", "research advances"),
    ("最新", "recent advances"),
    ("调研", "review"),
)


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


def decompose_question(
    question: str,
    sources: tuple[str, ...] | None = None,
    *,
    allow_llm: bool = False,
    network_gate: SafetyGate | None = None,
) -> QueryDecomposition:
    """创建检索分解；配置 key 且允许时优先使用 DeepSeek query 规划。"""

    cleaned = " ".join(question.strip().split())
    if not cleaned:
        raise ValueError("必须提供科研问题")
    selected_override = tuple(dict.fromkeys(sources)) if sources else None
    llm_plan = None
    if allow_llm:
        from medicine_agent.llm import plan_query_with_llm  # 延迟导入，离线规则路径不加载 LLM 组件。

        allowed = selected_override or select_sources(cleaned)
        llm_plan = plan_query_with_llm(cleaned, allowed_sources=allowed, network_gate=network_gate)

    if llm_plan is not None:
        selected = selected_override or llm_plan.sources
        search_topic = llm_plan.search_topic
        subquestions = llm_plan.subquestions
        planner = llm_plan.planner
        planner_reason = llm_plan.rationale
    else:
        selected = selected_override or select_sources(cleaned)
        search_topic = normalize_search_topic(cleaned)
        subquestions = (
            f"哪些原始文献讨论了：{cleaned}？",
            "哪些发现由元数据/摘要支持，哪些仍属于假设？",
        )
        planner = "deterministic"
        planner_reason = "未启用 DeepSeek 环境变量或 LLM 规划失败；使用确定性关键词/中英映射规则。"

    queries = tuple(
        SearchQueryRecord(
            provider=source,
            query=_provider_query(search_topic, source),
            rationale=_rationale_for_source(source),
            endpoint_family=_endpoint_family_for_source(source),
        )
        for source in selected
    )
    return QueryDecomposition(
        question=cleaned,
        subquestions=subquestions,
        queries=queries,
        search_topic=search_topic,
        planner=planner,
        planner_reason=planner_reason,
    )


def normalize_search_topic(question: str) -> str:
    """把中文科研问题映射为更适合英文文献 API 的检索主题。"""

    expansions: list[str] = []
    for marker, term in CHINESE_QUERY_EXPANSIONS:
        if marker in question:
            expansions.append(term)
    ascii_phrases = [
        " ".join(match.group(0).split())
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9+/ -]{2,}", question)
    ]
    terms = ascii_phrases + expansions
    if not terms:
        return question
    return " ".join(dict.fromkeys(term.strip() for term in terms if term.strip()))


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
