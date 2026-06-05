"""仅联网、带 allowlist 的文献提供器。

联网模式刻意保持狭窄：本模块唯一允许的证据检索目的地是 NCBI/PubMed
E-utilities、arXiv Atom API 与 Semantic Scholar Graph API。不需要 API key
或标准库之外的依赖。
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode

from medicine_agent.network_policy import ALLOWED_LIVE_HOSTS as NETWORK_ALLOWED_LIVE_HOSTS
from medicine_agent.network_policy import DEFAULT_TIMEOUT_SECONDS, fetch_url_bytes
from medicine_agent.safety import SafetyGate

from .base import PaperRecord, ProviderSearchResult, QueryDecomposition, SourceStatus, SourceStatusValue
from .source_selector import decompose_question

DEFAULT_MAX_RESULTS = 5
ALLOWED_LIVE_HOSTS = NETWORK_ALLOWED_LIVE_HOSTS


@dataclass(frozen=True)
class LiveApiProvider:
    provider_name: str
    endpoint_family: str

    def search(
        self,
        query: str,
        *,
        allow_live: bool = True,
        network_gate: SafetyGate | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> ProviderSearchResult:
        if allow_live:
            return self._search_live(query, network_gate=network_gate, max_results=max_results)
        status = SourceStatus(
            provider=self.provider_name,
            endpoint_family=self.endpoint_family,
            query=query,
            status=SourceStatusValue.SKIPPED,
            reason="当前项目只支持联网检索；显式禁用 live 时不会使用离线 fixture。",
        )
        return ProviderSearchResult(
            provider=self.provider_name,
            query=query,
            papers=(),
            statuses=(status,),
        )

    def _search_live(
        self,
        query: str,
        *,
        network_gate: SafetyGate | None,
        max_results: int,
    ) -> ProviderSearchResult:
        status = SourceStatus(
            provider=self.provider_name,
            endpoint_family=self.endpoint_family,
            query=query,
            status=SourceStatusValue.FAILED,
            reason="该提供器尚未实现联网检索",
        )
        return ProviderSearchResult(self.provider_name, query, (), (status,))

    def build_live_url(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        raise NotImplementedError


class PubMedProvider(LiveApiProvider):
    def __init__(self) -> None:
        super().__init__("pubmed", "ncbi_eutils")

    def build_live_url(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": str(max_results),
            "sort": _pubmed_sort_for_query(query),
            "tool": "medicine_agent",
            "term": query,
        }
        email = os.environ.get("NCBI_EMAIL")
        if email:
            params["email"] = email
        return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(params)

    def _search_live(
        self,
        query: str,
        *,
        network_gate: SafetyGate | None,
        max_results: int,
    ) -> ProviderSearchResult:
        try:
            search_url = self.build_live_url(query, max_results=max_results)
            search_payload = json.loads(_fetch_url(search_url, network_gate=network_gate).decode("utf-8"))
            ids = tuple(search_payload.get("esearchresult", {}).get("idlist", [])[:max_results])
            if not ids:
                return _empty_success(
                    self.provider_name,
                    self.endpoint_family,
                    query,
                    "NCBI ESearch 未返回 PubMed ID",
                )

            fetch_url = _build_pubmed_efetch_url(ids)
            xml_payload = _fetch_url(fetch_url, network_gate=network_gate).decode("utf-8", errors="replace")
            papers = _parse_pubmed_efetch(xml_payload)
            status = SourceStatus(
                provider=self.provider_name,
                endpoint_family=self.endpoint_family,
                query=query,
                status=SourceStatusValue.SUCCEEDED,
                result_ids=tuple(paper.stable_id for paper in papers),
                reason=f"实时 NCBI ESearch+EFetch 已完成，得到 {len(papers)} 条 PubMed 记录",
            )
            return ProviderSearchResult(self.provider_name, query, papers, (status,))
        except Exception as exc:  # noqa: BLE001 - 提供器异常必须转为 SourceStatus。
            return _failed_result(self.provider_name, self.endpoint_family, query, exc)


class ArxivProvider(LiveApiProvider):
    def __init__(self) -> None:
        super().__init__("arxiv", "arxiv_atom")

    def build_live_url(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return "https://export.arxiv.org/api/query?" + urlencode(params)

    def _search_live(
        self,
        query: str,
        *,
        network_gate: SafetyGate | None,
        max_results: int,
    ) -> ProviderSearchResult:
        try:
            url = self.build_live_url(query, max_results=max_results)
            xml_payload = _fetch_url(url, network_gate=network_gate).decode("utf-8", errors="replace")
            papers = _parse_arxiv_atom(xml_payload)
            status = SourceStatus(
                provider=self.provider_name,
                endpoint_family=self.endpoint_family,
                query=query,
                status=SourceStatusValue.SUCCEEDED,
                result_ids=tuple(paper.stable_id for paper in papers),
                reason=f"实时 arXiv API 查询已完成，得到 {len(papers)} 条记录",
            )
            return ProviderSearchResult(self.provider_name, query, papers, (status,))
        except Exception as exc:  # noqa: BLE001
            return _failed_result(self.provider_name, self.endpoint_family, query, exc)


class SemanticScholarProvider(LiveApiProvider):
    def __init__(self) -> None:
        super().__init__("semantic_scholar", "s2_graph")

    def build_live_url(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        fields = "paperId,title,abstract,year,authors,citationCount,externalIds,openAccessPdf,url,venue"
        params = {"limit": str(max_results), "fields": fields, "query": query}
        return "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(params)

    def _search_live(
        self,
        query: str,
        *,
        network_gate: SafetyGate | None,
        max_results: int,
    ) -> ProviderSearchResult:
        try:
            url = self.build_live_url(query, max_results=max_results)
            payload = json.loads(_fetch_url(url, network_gate=network_gate).decode("utf-8"))
            papers = _parse_semantic_scholar(payload)
            status = SourceStatus(
                provider=self.provider_name,
                endpoint_family=self.endpoint_family,
                query=query,
                status=SourceStatusValue.SUCCEEDED,
                result_ids=tuple(paper.stable_id for paper in papers),
                reason=f"实时 Semantic Scholar Graph API 查询已完成，得到 {len(papers)} 条记录",
            )
            return ProviderSearchResult(self.provider_name, query, papers, (status,))
        except Exception as exc:  # noqa: BLE001
            return _failed_result(self.provider_name, self.endpoint_family, query, exc)


@dataclass(frozen=True)
class LiteratureSearchCoordinator:
    providers: Mapping[str, LiveApiProvider]

    def search_question(
        self,
        question: str,
        *,
        allow_live: bool = True,
        network_gate: SafetyGate | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        decomposition: QueryDecomposition | None = None,
        allow_llm: bool = False,
    ) -> dict[str, object]:
        if decomposition is None:
            decomposition = decompose_question(question, allow_llm=allow_llm, network_gate=network_gate)
        results: list[ProviderSearchResult] = []
        for query in decomposition.queries:
            provider = self.providers.get(query.provider)
            if provider is None:
                status = SourceStatus(
                    provider=query.provider,
                    endpoint_family=query.endpoint_family,
                    query=query.query,
                    status=SourceStatusValue.SKIPPED,
                    reason="提供器未配置，或不在允许的实时来源集合中",
                )
                results.append(ProviderSearchResult(query.provider, query.query, (), (status,)))
                continue
            results.append(
                provider.search(
                    query.query,
                    allow_live=allow_live,
                    network_gate=network_gate,
                    max_results=max_results,
                )
            )
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
    providers = (PubMedProvider(), ArxivProvider(), SemanticScholarProvider())
    return LiteratureSearchCoordinator({provider.provider_name: provider for provider in providers})


def _fetch_url(
    url: str,
    *,
    network_gate: SafetyGate | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int | None = None,
) -> bytes:
    return fetch_url_bytes(url, network_gate=network_gate, timeout=timeout, max_bytes=max_bytes)


def _build_pubmed_efetch_url(ids: tuple[str, ...]) -> str:
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(ids),
        "tool": "medicine_agent",
    }
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)


def _pubmed_sort_for_query(query: str) -> str:
    normalized = query.lower()
    if any(term in normalized for term in ("recent", "latest", "最新", "研究进展")):
        return "pub_date"
    return "relevance"


def _parse_pubmed_efetch(xml_payload: str) -> tuple[PaperRecord, ...]:
    root = ET.fromstring(xml_payload)
    papers: list[PaperRecord] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_node = medline.find("Article") if medline is not None else None
        if medline is None or article_node is None:
            continue
        pmid = _text(medline.find("PMID"))
        title = _text(article_node.find("ArticleTitle")) or "未命名 PubMed 记录"
        abstract = " ".join(
            part.strip()
            for part in (_element_text(node) for node in article_node.findall("Abstract/AbstractText"))
            if part.strip()
        )
        journal = _text(article_node.find("Journal/Title"))
        year = _pubmed_year(article_node)
        authors = tuple(_pubmed_author_name(node) for node in article_node.findall("AuthorList/Author"))
        article_ids = {
            (node.attrib.get("IdType") or "").lower(): (node.text or "").strip()
            for node in article.findall("PubmedData/ArticleIdList/ArticleId")
            if (node.text or "").strip()
        }
        papers.append(
            PaperRecord(
                provider="pubmed",
                title=title,
                abstract=abstract or None,
                year=year,
                venue=journal or "PubMed",
                authors=tuple(author for author in authors if author),
                pmid=pmid or None,
                pmcid=article_ids.get("pmc"),
                doi=article_ids.get("doi"),
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
                open_access_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{article_ids['pmc']}/" if article_ids.get("pmc") else None,
            )
        )
    return tuple(papers)


def _parse_arxiv_atom(xml_payload: str) -> tuple[PaperRecord, ...]:
    root = ET.fromstring(xml_payload)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers: list[PaperRecord] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _text(entry.find("atom:id", ns))
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        year = _year_from_prefix(_text(entry.find("atom:published", ns)))
        papers.append(
            PaperRecord(
                provider="arxiv",
                title=" ".join(_text(entry.find("atom:title", ns)).split()) or "未命名 arXiv 记录",
                abstract=" ".join(_text(entry.find("atom:summary", ns)).split()) or None,
                year=year,
                venue="arXiv",
                authors=tuple(_text(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)),
                doi=_text(entry.find("arxiv:doi", ns)) or None,
                arxiv_id=arxiv_id,
                source_url=entry_id or "https://arxiv.org/",
                open_access_url=pdf_url,
            )
        )
    return tuple(papers)


def _parse_semantic_scholar(payload: Mapping[str, object]) -> tuple[PaperRecord, ...]:
    papers: list[PaperRecord] = []
    data_items = payload.get("data", [])
    if not isinstance(data_items, list):
        return ()
    for item in data_items:
        if not isinstance(item, dict):
            continue
        item_map: dict[Any, Any] = item
        external_raw = item_map.get("externalIds")
        open_pdf_raw = item_map.get("openAccessPdf")
        external: dict[Any, Any] = external_raw if isinstance(external_raw, dict) else {}
        open_pdf: dict[Any, Any] = open_pdf_raw if isinstance(open_pdf_raw, dict) else {}
        author_items = item_map.get("authors", [])
        if not isinstance(author_items, list):
            author_items = []
        authors = tuple(
            str(author.get("name"))
            for author in author_items
            if isinstance(author, dict) and author.get("name")
        )
        paper_id = item_map.get("paperId")
        papers.append(
            PaperRecord(
                provider="semantic_scholar",
                title=str(item_map.get("title") or "未命名 Semantic Scholar 记录"),
                abstract=str(item_map.get("abstract")) if item_map.get("abstract") else None,
                year=item_map.get("year") if isinstance(item_map.get("year"), int) else None,
                venue=str(item_map.get("venue")) if item_map.get("venue") else "Semantic Scholar",
                authors=authors,
                pmid=str(external.get("PubMed")) if external.get("PubMed") else None,
                pmcid=str(external.get("PubMedCentral")) if external.get("PubMedCentral") else None,
                doi=str(external.get("DOI")) if external.get("DOI") else None,
                arxiv_id=str(external.get("ArXiv")) if external.get("ArXiv") else None,
                semantic_scholar_id=str(paper_id) if paper_id else None,
                citation_count=item_map.get("citationCount")
                if isinstance(item_map.get("citationCount"), int)
                else None,
                source_url=str(item_map.get("url"))
                if item_map.get("url")
                else f"https://www.semanticscholar.org/paper/{paper_id}",
                open_access_url=str(open_pdf.get("url")) if open_pdf.get("url") else None,
            )
        )
    return tuple(papers)


def _empty_success(provider: str, endpoint_family: str, query: str, reason: str) -> ProviderSearchResult:
    return ProviderSearchResult(
        provider,
        query,
        (),
        (
            SourceStatus(
                provider=provider,
                endpoint_family=endpoint_family,
                query=query,
                status=SourceStatusValue.SUCCEEDED,
                reason=reason,
            ),
        ),
    )


def _failed_result(provider: str, endpoint_family: str, query: str, exc: Exception) -> ProviderSearchResult:
    status_value = SourceStatusValue.RATE_LIMITED if str(exc) == "rate_limited" else SourceStatusValue.FAILED
    return ProviderSearchResult(
        provider,
        query,
        (),
        (
            SourceStatus(
                provider=provider,
                endpoint_family=endpoint_family,
                query=query,
                status=status_value,
                error_class=type(exc).__name__,
                reason=str(exc),
            ),
        ),
    )


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


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _pubmed_year(article_node: ET.Element) -> int | None:
    for xpath in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year"):
        value = _text(article_node.find(xpath))
        year = _year_from_prefix(value)
        if year is not None:
            return year
    return None


def _pubmed_author_name(node: ET.Element) -> str:
    collective = _text(node.find("CollectiveName"))
    if collective:
        return collective
    last = _text(node.find("LastName"))
    initials = _text(node.find("Initials"))
    return " ".join(part for part in (last, initials) if part)


def _year_from_prefix(value: str) -> int | None:
    if len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None
