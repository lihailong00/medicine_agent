"""文献提供器共享的契约。

首版实现刻意只使用标准库，并以离线/模拟优先。提供器网络调用必须通过显式
SourceStatus 记录呈现；除非调用方显式开启 live 标志，否则绝不尝试联网。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Protocol, Sequence


class SourceStatusValue(str, Enum):
    """每次提供器/查询尝试的可观测状态。"""

    ATTEMPTED = "attempted"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    NEEDS_CONFIRMATION = "needs_confirmation"


def utc_now_iso() -> str:
    """返回带稳定 Z 后缀的 ISO-8601 UTC 时间戳。"""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceStatus:
    """PRD 检索日志契约要求的提供器可观测记录。"""

    provider: str
    endpoint_family: str
    query: str
    status: SourceStatusValue
    timestamp: str = field(default_factory=utc_now_iso)
    result_ids: tuple[str, ...] = ()
    error_class: str | None = None
    retry_backoff_seconds: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "endpoint_family": self.endpoint_family,
            "query": self.query,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "result_ids": list(self.result_ids),
            "error_class": self.error_class,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SearchQueryRecord:
    """分解后的、面向特定提供器的文献查询。"""

    provider: str
    query: str
    rationale: str
    endpoint_family: str = "offline_fixture"

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "query": self.query,
            "rationale": self.rationale,
            "endpoint_family": self.endpoint_family,
        }


@dataclass(frozen=True)
class QueryDecomposition:
    """由科研问题派生出的检索计划。"""

    question: str
    subquestions: tuple[str, ...]
    queries: tuple[SearchQueryRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "subquestions": list(self.subquestions),
            "queries": [query.to_dict() for query in self.queries],
        }


@dataclass(frozen=True)
class PaperRecord:
    """跨 PubMed/PMC、arXiv 与 Semantic Scholar 的规范化元数据契约。"""

    provider: str
    title: str
    source_url: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: tuple[str, ...] = ()
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    citation_count: int | None = None
    open_access_url: str | None = None

    @property
    def stable_id(self) -> str:
        for value in (
            self.pmid,
            self.pmcid,
            self.doi,
            self.arxiv_id,
            self.semantic_scholar_id,
            self.source_url,
        ):
            if value:
                return value
        return self.title

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "title": self.title,
            "source_url": self.source_url,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "authors": list(self.authors),
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "semantic_scholar_id": self.semantic_scholar_id,
            "citation_count": self.citation_count,
            "open_access_url": self.open_access_url,
        }


@dataclass(frozen=True)
class CitationEvidenceRecord:
    """用于仅科研报告综合的引用安全证据记录。"""

    paper_id: str
    provider: str
    citation_label: str
    evidence_note: str
    source_url: str

    def to_dict(self) -> dict[str, str]:
        return {
            "paper_id": self.paper_id,
            "provider": self.provider,
            "citation_label": self.citation_label,
            "evidence_note": self.evidence_note,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class ProviderSearchResult:
    """提供器查询返回的结果包。"""

    provider: str
    query: str
    papers: tuple[PaperRecord, ...]
    statuses: tuple[SourceStatus, ...]

    def evidence_records(self) -> tuple[CitationEvidenceRecord, ...]:
        return tuple(
            CitationEvidenceRecord(
                paper_id=paper.stable_id,
                provider=paper.provider,
                citation_label=_citation_label(paper),
                evidence_note="仅为元数据/摘要证据；不假定已获得全文访问。",
                source_url=paper.source_url,
            )
            for paper in self.papers
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "query": self.query,
            "papers": [paper.to_dict() for paper in self.papers],
            "statuses": [status.to_dict() for status in self.statuses],
            "evidence_records": [record.to_dict() for record in self.evidence_records()],
        }


class LiteratureProvider(Protocol):
    """所有文献来源适配器实现的协议。"""

    provider_name: str
    endpoint_family: str

    def search(self, query: str, *, allow_live: bool = False) -> ProviderSearchResult:
        """针对 ``query`` 检索文献。

        实现默认必须是确定性的离线行为。如果 live 模式不可用或未被显式允许，
        必须返回 SourceStatus 记录，而不是静默省略该提供器。
        """


def _citation_label(paper: PaperRecord) -> str:
    if paper.authors:
        first = paper.authors[0].split()[-1]
    else:
        first = paper.provider
    if paper.year:
        return f"{first} et al., {paper.year}"
    return f"{first} et al."


def merge_search_logs(results: Sequence[ProviderSearchResult]) -> list[dict[str, object]]:
    """将提供器状态展平成可序列化的检索日志。"""

    return [status.to_dict() for result in results for status in result.statuses]


def records_to_jsonable(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """为 manifest/报告写入器规范化映射序列。"""

    return [dict(record) for record in records]
