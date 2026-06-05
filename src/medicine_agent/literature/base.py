"""Contracts shared by literature providers.

The first implementation is deliberately stdlib-only and offline/mock-first.
Provider network calls are represented by explicit SourceStatus records and are
never attempted unless a caller opts in with an explicit live flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Protocol, Sequence


class SourceStatusValue(str, Enum):
    """Observable status for every provider/query attempt."""

    ATTEMPTED = "attempted"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    NEEDS_CONFIRMATION = "needs_confirmation"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable Z suffix."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceStatus:
    """Provider observability record required by the PRD search log contract."""

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
    """A decomposed provider-specific literature query."""

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
    """Search plan derived from a research question."""

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
    """Normalized metadata contract across PubMed/PMC, arXiv, and Semantic Scholar."""

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
    """Citation-safe evidence record for research-only report synthesis."""

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
    """Result bundle from a provider query."""

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
                evidence_note="Metadata/abstract evidence only; full-text access is not assumed.",
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
    """Protocol implemented by all literature source adapters."""

    provider_name: str
    endpoint_family: str

    def search(self, query: str, *, allow_live: bool = False) -> ProviderSearchResult:
        """Search literature for ``query``.

        Implementations must be deterministic and offline by default. If live
        mode is unavailable or not explicitly allowed, they must return a
        SourceStatus record instead of silently omitting the provider.
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
    """Flatten provider statuses into a serializable search log."""

    return [status.to_dict() for result in results for status in result.statuses]


def records_to_jsonable(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Normalize mapping sequences for manifest/report writers."""

    return [dict(record) for record in records]
