"""Offline-first literature provider implementations.

Live network access is intentionally not performed by default. Setting
``MEDICINE_AGENT_LIVE_API=1`` or passing ``allow_live=True`` only enables the
live scaffold path; unsupported live fetches still return SourceStatus records
instead of requiring API keys or dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping
from urllib.parse import quote_plus

from .base import PaperRecord, ProviderSearchResult, SourceStatus, SourceStatusValue
from .source_selector import decompose_question


@dataclass(frozen=True)
class OfflineFixtureProvider:
    provider_name: str
    endpoint_family: str
    fixture_records: tuple[PaperRecord, ...]

    def search(self, query: str, *, allow_live: bool = False) -> ProviderSearchResult:
        live_requested = allow_live or os.environ.get("MEDICINE_AGENT_LIVE_API") == "1"
        if live_requested:
            return self._live_scaffold(query)
        matches = _filter_fixture_records(self.fixture_records, query)
        status = SourceStatus(
            provider=self.provider_name,
            endpoint_family="offline_fixture",
            query=query,
            status=SourceStatusValue.SUCCEEDED,
            result_ids=tuple(paper.stable_id for paper in matches),
            reason="deterministic offline fixture provider used; no network attempted",
        )
        return ProviderSearchResult(
            provider=self.provider_name,
            query=query,
            papers=matches,
            statuses=(status,),
        )

    def _live_scaffold(self, query: str) -> ProviderSearchResult:
        status = SourceStatus(
            provider=self.provider_name,
            endpoint_family=self.endpoint_family,
            query=query,
            status=SourceStatusValue.NEEDS_CONFIRMATION,
            reason=(
                "live provider scaffold requires an explicit SafetyGate NETWORK_CALL "
                "decision by the orchestrator; no API key is required by default and "
                "no network call was attempted"
            ),
        )
        return ProviderSearchResult(
            provider=self.provider_name,
            query=query,
            papers=(),
            statuses=(status,),
        )

    def build_live_url(self, query: str) -> str:
        raise NotImplementedError


class PubMedProvider(OfflineFixtureProvider):
    def __init__(self) -> None:
        super().__init__("pubmed", "ncbi_eutils", _PUBMED_FIXTURES)

    def build_live_url(self, query: str) -> str:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        return f"{base}?db=pubmed&retmode=json&term={quote_plus(query)}"


class BioRxivProvider(OfflineFixtureProvider):
    def __init__(self) -> None:
        super().__init__("biorxiv", "biorxiv_details", _BIORXIV_FIXTURES)

    def build_live_url(self, query: str) -> str:
        return "https://api.biorxiv.org/details/biorxiv/2020-01-01/3000-01-01/0"


class ArxivProvider(OfflineFixtureProvider):
    def __init__(self) -> None:
        super().__init__("arxiv", "arxiv_atom", _ARXIV_FIXTURES)

    def build_live_url(self, query: str) -> str:
        base = "https://export.arxiv.org/api/query"
        return f"{base}?search_query=all:{quote_plus(query)}&start=0&max_results=5"


class SemanticScholarProvider(OfflineFixtureProvider):
    def __init__(self) -> None:
        super().__init__("semantic_scholar", "s2_graph", _SEMANTIC_SCHOLAR_FIXTURES)

    def build_live_url(self, query: str) -> str:
        fields = "paperId,title,abstract,year,authors,citationCount,externalIds,openAccessPdf"
        base = "https://api.semanticscholar.org/graph/v1/paper/search"
        return f"{base}?limit=5&fields={fields}&query={quote_plus(query)}"


@dataclass(frozen=True)
class LiteratureSearchCoordinator:
    providers: Mapping[str, OfflineFixtureProvider]

    def search_question(self, question: str, *, allow_live: bool = False) -> dict[str, object]:
        decomposition = decompose_question(question)
        results: list[ProviderSearchResult] = []
        for query in decomposition.queries:
            provider = self.providers.get(query.provider)
            if provider is None:
                status = SourceStatus(
                    provider=query.provider,
                    endpoint_family=query.endpoint_family,
                    query=query.query,
                    status=SourceStatusValue.SKIPPED,
                    reason="provider is not configured",
                )
                results.append(ProviderSearchResult(query.provider, query.query, (), (status,)))
                continue
            results.append(provider.search(query.query, allow_live=allow_live))
        papers = tuple(_dedupe_papers(paper for result in results for paper in result.papers))
        return {
            "decomposition": decomposition.to_dict(),
            "results": [result.to_dict() for result in results],
            "papers": [paper.to_dict() for paper in papers],
            "search_log": [status.to_dict() for result in results for status in result.statuses],
            "evidence_records": [
                evidence.to_dict()
                for result in results
                for evidence in result.evidence_records()
            ],
        }


def build_default_coordinator() -> LiteratureSearchCoordinator:
    providers = (PubMedProvider(), BioRxivProvider(), ArxivProvider(), SemanticScholarProvider())
    return LiteratureSearchCoordinator({provider.provider_name: provider for provider in providers})


def _filter_fixture_records(records: tuple[PaperRecord, ...], query: str) -> tuple[PaperRecord, ...]:
    tokens = {
        token.strip("()[]{}.,;:!?\"'").lower()
        for token in query.split()
        if len(token.strip("()[]{}.,;:!?\"'")) >= 4
    }
    if not tokens:
        return records[:1]
    scored: list[tuple[int, PaperRecord]] = []
    for record in records:
        haystack = " ".join(
            value for value in (record.title, record.abstract or "", record.venue or "")
        ).lower()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            scored.append((score, record))
    if not scored:
        return records[:1]
    scored.sort(key=lambda item: (-item[0], item[1].year or 0, item[1].title))
    return tuple(record for _, record in scored[:5])


def _dedupe_papers(papers: Iterable[PaperRecord]) -> tuple[PaperRecord, ...]:
    seen: set[str] = set()
    deduped: list[PaperRecord] = []
    for paper in papers:
        key = paper.stable_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return tuple(deduped)


_PUBMED_FIXTURES = (
    PaperRecord(
        provider="pubmed",
        pmid="34763053",
        pmcid="PMC8576925",
        doi="10.1038/s41586-021-03929-7",
        title="Intercellular communication analysis of single-cell transcriptomics data",
        abstract="A benchmark and framework for ligand-receptor inference in single-cell data.",
        year=2021,
        venue="Nature",
        authors=("Dimitrov D", "Türei D"),
        source_url="https://pubmed.ncbi.nlm.nih.gov/34763053/",
        open_access_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8576925/",
    ),
    PaperRecord(
        provider="pubmed",
        pmid="30664773",
        doi="10.1038/s41586-019-0938-7",
        title="Single-cell transcriptomics of human tumors reveals immune cell states",
        abstract="Single-cell RNA sequencing characterizes tumor microenvironment cell states and interactions.",
        year=2019,
        venue="Nature",
        authors=("Sade-Feldman M",),
        source_url="https://pubmed.ncbi.nlm.nih.gov/30664773/",
    ),
)

_BIORXIV_FIXTURES = (
    PaperRecord(
        provider="biorxiv",
        doi="10.1101/2023.01.01.000001",
        title="Emerging preprint on ligand receptor signaling in tumor immune niches",
        abstract="Preprint metadata fixture for emerging cell-cell communication hypotheses.",
        year=2023,
        venue="bioRxiv",
        authors=("Fixture A",),
        source_url="https://www.biorxiv.org/content/10.1101/2023.01.01.000001v1",
    ),
)

_ARXIV_FIXTURES = (
    PaperRecord(
        provider="arxiv",
        arxiv_id="2301.00001",
        title="Graph neural models for computational biology interaction ranking",
        abstract="Computational biology fixture covering machine learning ranking of interactions.",
        year=2023,
        venue="arXiv",
        authors=("Fixture B",),
        source_url="https://arxiv.org/abs/2301.00001",
        open_access_url="https://arxiv.org/pdf/2301.00001",
    ),
)

_SEMANTIC_SCHOLAR_FIXTURES = (
    PaperRecord(
        provider="semantic_scholar",
        semantic_scholar_id="S2-FIXTURE-LIANA",
        doi="10.1038/s41586-021-03929-7",
        title="Intercellular communication resources integrated for ligand receptor analysis",
        abstract="Semantic Scholar metadata fixture with citation enrichment for LIANA-style analysis.",
        year=2021,
        venue="Nature",
        authors=("Dimitrov D",),
        citation_count=500,
        source_url="https://www.semanticscholar.org/paper/S2-FIXTURE-LIANA",
        open_access_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8576925/",
    ),
)
