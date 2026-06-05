"""Full-text retrieval contracts and allowlisted scholarly fetchers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

from medicine_agent.models import OperationClass
from medicine_agent.network_policy import DEFAULT_TIMEOUT_SECONDS, fetch_url_bytes
from medicine_agent.safety import SafetyGate
from medicine_agent.utils.io import write_text

DEFAULT_MAX_FULL_TEXT_BYTES = 2_000_000
DEFAULT_MAX_FULL_TEXT_CHARS = 50_000
FetchUrl = Callable[..., bytes]


class EvidenceScope(str, Enum):
    ABSTRACT = "abstract"
    SNIPPET = "snippet"
    FULL_TEXT_XML = "full_text_xml"
    FULL_TEXT_PDF_ARTIFACT = "full_text_pdf_artifact"
    UNAVAILABLE = "unavailable"


class FullTextStatusValue(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FullTextCandidate:
    provider: str
    paper_id: str
    title: str
    source_url: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    open_access_url: str | None = None
    abstract: str | None = None

    @classmethod
    def from_provider_payload(cls, payload: Mapping[str, Any]) -> "FullTextCandidate":
        data = dict(payload)
        paper_id = next(
            (
                str(value)
                for value in (
                    data.get("pmid"),
                    data.get("pmcid"),
                    data.get("doi"),
                    data.get("arxiv_id"),
                    data.get("semantic_scholar_id"),
                    data.get("source_url"),
                    data.get("title"),
                )
                if value
            ),
            "unknown-paper",
        )
        return cls(
            provider=str(data.get("provider") or data.get("source") or "unknown"),
            paper_id=paper_id,
            title=str(data.get("title") or "Untitled record"),
            source_url=data.get("source_url") if isinstance(data.get("source_url"), str) else None,
            pmid=data.get("pmid") if isinstance(data.get("pmid"), str) else None,
            pmcid=data.get("pmcid") if isinstance(data.get("pmcid"), str) else None,
            doi=data.get("doi") if isinstance(data.get("doi"), str) else None,
            arxiv_id=data.get("arxiv_id") if isinstance(data.get("arxiv_id"), str) else None,
            semantic_scholar_id=data.get("semantic_scholar_id")
            if isinstance(data.get("semantic_scholar_id"), str)
            else None,
            open_access_url=data.get("open_access_url") if isinstance(data.get("open_access_url"), str) else None,
            abstract=data.get("abstract") if isinstance(data.get("abstract"), str) else None,
        )


@dataclass(frozen=True)
class FullTextStatus:
    provider: str
    paper_id: str
    status: FullTextStatusValue
    scope: EvidenceScope
    source_url: str | None = None
    artifact_path: str | None = None
    parser: str | None = None
    byte_count: int = 0
    char_count: int = 0
    sha256: str | None = None
    truncated: bool = False
    parser_limitations: tuple[str, ...] = ()
    reason: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "paper_id": self.paper_id,
            "status": self.status.value,
            "scope": self.scope.value,
            "source_url": self.source_url,
            "artifact_path": self.artifact_path,
            "parser": self.parser,
            "byte_count": self.byte_count,
            "char_count": self.char_count,
            "sha256": self.sha256,
            "truncated": self.truncated,
            "parser_limitations": list(self.parser_limitations),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class FullTextRecord:
    candidate: FullTextCandidate
    status: FullTextStatus
    text_preview: str = ""
    attempts: tuple[FullTextStatus, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": {
                "provider": self.candidate.provider,
                "paper_id": self.candidate.paper_id,
                "title": self.candidate.title,
                "source_url": self.candidate.source_url,
                "pmid": self.candidate.pmid,
                "pmcid": self.candidate.pmcid,
                "doi": self.candidate.doi,
                "arxiv_id": self.candidate.arxiv_id,
                "semantic_scholar_id": self.candidate.semantic_scholar_id,
                "open_access_url": self.candidate.open_access_url,
            },
            "status": self.status.to_dict(),
            "text_preview": self.text_preview,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class ParsedFullText:
    text: str
    parser: str
    char_count: int
    truncated: bool = False
    parser_limitations: tuple[str, ...] = ()


def build_pubmed_to_pmc_elink_url(pmid: str) -> str:
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "linkname": "pubmed_pmc",
        "retmode": "json",
        "tool": "medicine_agent",
    }
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?" + urlencode(params)


def build_pmc_efetch_url(pmcid_or_uid: str) -> str:
    normalized = normalize_pmcid(pmcid_or_uid)
    if normalized is None:
        raise ValueError(f"invalid PMCID/PMC UID: {pmcid_or_uid}")
    params = {
        "db": "pmc",
        "id": normalized.uid,
        "retmode": "xml",
        "tool": "medicine_agent",
    }
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)


@dataclass(frozen=True)
class NormalizedPmcId:
    display: str
    uid: str


def normalize_pmcid(value: str | None) -> NormalizedPmcId | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    match = re.fullmatch(r"(?i)PMC?(\d+)", cleaned)
    if match:
        uid = match.group(1)
        return NormalizedPmcId(display=f"PMC{uid}", uid=uid)
    if cleaned.isdigit():
        return NormalizedPmcId(display=f"PMC{cleaned}", uid=cleaned)
    return None


def parse_elink_pmc_uid(payload: bytes | str) -> str | None:
    data = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    for linkset in data.get("linksets", []):
        for linksetdb in linkset.get("linksetdbs", []):
            if linksetdb.get("linkname") == "pubmed_pmc":
                links = linksetdb.get("links", [])
                if links:
                    return str(links[0])
    return None


def parse_pmc_xml(xml_payload: str | bytes, *, max_chars: int = DEFAULT_MAX_FULL_TEXT_CHARS) -> ParsedFullText:
    xml_text = xml_payload.decode("utf-8", errors="replace") if isinstance(xml_payload, bytes) else xml_payload
    root = ET.fromstring(xml_text)
    sections: list[str] = []
    title = _first_text(root, "article-title")
    if title:
        sections.append(f"Title: {title}")
    abstract_parts = _texts_under_first(root, "abstract", include_tags={"title", "p", "abstract"})
    if abstract_parts:
        sections.append("Abstract:\n" + "\n".join(abstract_parts))
    body = _first_element(root, "body")
    if body is not None:
        body_parts: list[str] = []
        for element in body.iter():
            local = _local_name(element.tag)
            text = _collapse_whitespace(" ".join(element.itertext()))
            if not text:
                continue
            if local == "title":
                body_parts.append(f"\n## {text}")
            elif local == "p":
                body_parts.append(text)
        if body_parts:
            sections.append("Body:\n" + "\n".join(body_parts))
    license_text = _first_text(root, "license-p")
    if license_text:
        sections.append(f"License: {license_text}")
    ref_count = sum(1 for element in root.iter() if _local_name(element.tag) == "ref")
    if ref_count:
        sections.append(f"References count: {ref_count}")
    text = _collapse_whitespace("\n\n".join(sections))
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
    limitations = ("PMC XML parsed with stdlib ElementTree; tables/figures/supplements are not fully extracted.",)
    return ParsedFullText(
        text=text,
        parser="pmc_jats_elementtree",
        char_count=len(text),
        truncated=truncated,
        parser_limitations=limitations,
    )


def retrieve_pmc_full_text(
    candidate: FullTextCandidate,
    *,
    artifacts_dir: Path,
    safety_gate: SafetyGate,
    network_gate: SafetyGate | None = None,
    fetcher: FetchUrl = fetch_url_bytes,
    max_chars: int = DEFAULT_MAX_FULL_TEXT_CHARS,
) -> FullTextRecord:
    normalized = normalize_pmcid(candidate.pmcid)
    source_url: str | None = None
    try:
        if normalized is None and candidate.pmid:
            elink_url = build_pubmed_to_pmc_elink_url(candidate.pmid)
            uid = parse_elink_pmc_uid(
                fetcher(
                    elink_url,
                    network_gate=network_gate,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                    max_bytes=DEFAULT_MAX_FULL_TEXT_BYTES,
                )
            )
            normalized = normalize_pmcid(uid)
        if normalized is None:
            return _unavailable_record(candidate, "No PMCID/PMC link available from approved NCBI path.")

        efetch_url = build_pmc_efetch_url(normalized.uid)
        source_url = efetch_url
        raw_xml = fetcher(
            efetch_url,
            network_gate=network_gate,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            max_bytes=DEFAULT_MAX_FULL_TEXT_BYTES,
        )
        parsed = parse_pmc_xml(raw_xml, max_chars=max_chars)
        artifact_path = artifacts_dir / f"{_safe_artifact_id(candidate.paper_id)}-pmc-full-text.txt"
        write_text(artifact_path, parsed.text, safety_gate)
        encoded = parsed.text.encode("utf-8")
        status = FullTextStatus(
            provider="pmc",
            paper_id=candidate.paper_id,
            status=FullTextStatusValue.SUCCEEDED,
            scope=EvidenceScope.FULL_TEXT_XML,
            source_url=source_url,
            artifact_path=str(artifact_path),
            parser=parsed.parser,
            byte_count=len(encoded),
            char_count=parsed.char_count,
            sha256=hashlib.sha256(encoded).hexdigest(),
            truncated=parsed.truncated,
            parser_limitations=parsed.parser_limitations,
            reason=f"Retrieved PMC full text for {normalized.display} via NCBI EFetch.",
        )
        return FullTextRecord(candidate, status, parsed.text[:500])
    except Exception as exc:  # noqa: BLE001 - full-text retrieval should degrade per paper.
        return _failed_record(candidate, source_url, exc)


def retrieve_best_available_text(
    candidate: FullTextCandidate,
    *,
    artifacts_dir: Path,
    safety_gate: SafetyGate,
    network_gate: SafetyGate | None = None,
    fetcher: FetchUrl = fetch_url_bytes,
    max_chars: int = DEFAULT_MAX_FULL_TEXT_CHARS,
) -> FullTextRecord:
    attempts: list[FullTextStatus] = []
    if candidate.pmcid or candidate.pmid:
        pmc_record = retrieve_pmc_full_text(
            candidate,
            artifacts_dir=artifacts_dir,
            safety_gate=safety_gate,
            network_gate=network_gate,
            fetcher=fetcher,
            max_chars=max_chars,
        )
        if pmc_record.status.status == FullTextStatusValue.SUCCEEDED:
            return pmc_record
        attempts.append(pmc_record.status)
        if candidate.provider == "pubmed":
            if candidate.abstract:
                return _abstract_record(candidate, pmc_record.status.reason, attempts=tuple(attempts))
            return pmc_record
    if candidate.arxiv_id:
        arxiv_record = retrieve_arxiv_pdf_artifact(
            candidate,
            artifacts_dir=artifacts_dir,
            safety_gate=safety_gate,
            network_gate=network_gate,
            fetcher=fetcher,
        )
        if arxiv_record.status.status == FullTextStatusValue.SUCCEEDED:
            return _with_attempts(arxiv_record, attempts)
        attempts.append(arxiv_record.status)
        if not (candidate.semantic_scholar_id or candidate.provider == "semantic_scholar"):
            if candidate.abstract:
                return _abstract_record(candidate, arxiv_record.status.reason, attempts=tuple(attempts))
            return _with_attempts(arxiv_record, attempts[:-1])
    if candidate.semantic_scholar_id or candidate.provider == "semantic_scholar":
        snippet_record = retrieve_semantic_scholar_snippets(
            candidate,
            artifacts_dir=artifacts_dir,
            safety_gate=safety_gate,
            network_gate=network_gate,
            fetcher=fetcher,
            max_chars=max_chars,
        )
        if snippet_record.status.status == FullTextStatusValue.SUCCEEDED:
            return _with_attempts(snippet_record, attempts)
        attempts.append(snippet_record.status)
        if candidate.abstract:
            return _abstract_record(candidate, snippet_record.status.reason, attempts=tuple(attempts))
        return _with_attempts(snippet_record, attempts[:-1])
    if candidate.abstract:
        reason = attempts[-1].reason if attempts else "No approved full-text route is available; using provider abstract scope."
        return _abstract_record(candidate, reason, attempts=tuple(attempts))
    return _unavailable_record(candidate, "No approved full-text or snippet route is available for this record.")


def retrieve_full_text_for_payloads(
    paper_payloads: list[Mapping[str, Any]],
    *,
    artifacts_dir: Path,
    safety_gate: SafetyGate,
    network_gate: SafetyGate | None = None,
    fetcher: FetchUrl = fetch_url_bytes,
    max_chars: int = DEFAULT_MAX_FULL_TEXT_CHARS,
) -> dict[str, object]:
    """Retrieve best available approved full-text evidence for provider payloads.

    The returned structure is intentionally manifest/report friendly and records
    every per-paper success, skip, rate limit, or failure without aborting the
    whole research run.
    """

    records: list[FullTextRecord] = []
    for payload in paper_payloads:
        candidate = FullTextCandidate.from_provider_payload(payload)
        records.append(
            retrieve_best_available_text(
                candidate,
                artifacts_dir=artifacts_dir,
                safety_gate=safety_gate,
                network_gate=network_gate,
                fetcher=fetcher,
                max_chars=max_chars,
            )
        )
    record_payloads = [record.to_dict() for record in records]
    statuses = [record.status.to_dict() for record in records]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": record_payloads,
        "statuses": statuses,
        "scope_counts": _scope_counts(records),
    }


def build_arxiv_pdf_url(arxiv_id: str) -> str:
    cleaned = arxiv_id.strip().removeprefix("arXiv:").removeprefix("https://arxiv.org/abs/")
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", cleaned):
        raise ValueError(f"unsupported or unsafe arXiv id for constructed PDF URL: {arxiv_id}")
    return f"https://arxiv.org/pdf/{cleaned}"


def retrieve_arxiv_pdf_artifact(
    candidate: FullTextCandidate,
    *,
    artifacts_dir: Path,
    safety_gate: SafetyGate,
    network_gate: SafetyGate | None = None,
    fetcher: FetchUrl = fetch_url_bytes,
) -> FullTextRecord:
    source_url: str | None = None
    try:
        if not candidate.arxiv_id:
            return _unavailable_record(candidate, "No arXiv ID is available for approved arXiv full-text retrieval.")
        source_url = build_arxiv_pdf_url(candidate.arxiv_id)
        payload = fetcher(
            source_url,
            network_gate=network_gate,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            max_bytes=DEFAULT_MAX_FULL_TEXT_BYTES,
        )
        artifact_path = artifacts_dir / f"{_safe_artifact_id(candidate.paper_id)}-arxiv.pdf"
        _write_bytes(artifact_path, payload, safety_gate)
        status = FullTextStatus(
            provider="arxiv",
            paper_id=candidate.paper_id,
            status=FullTextStatusValue.SUCCEEDED,
            scope=EvidenceScope.FULL_TEXT_PDF_ARTIFACT,
            source_url=source_url,
            artifact_path=str(artifact_path),
            parser="pdf_bytes_only",
            byte_count=len(payload),
            char_count=0,
            sha256=hashlib.sha256(payload).hexdigest(),
            parser_limitations=(
                "arXiv PDF bytes were retrieved from an allowlisted arXiv path; no dependency-free full-text PDF parsing is claimed.",
            ),
            reason="Retrieved arXiv PDF artifact from constructed arXiv HTTPS URL.",
        )
        return FullTextRecord(candidate, status)
    except Exception as exc:  # noqa: BLE001
        return _failed_record(candidate, source_url, exc)


def build_semantic_scholar_snippet_url(query: str, *, max_results: int = 3) -> str:
    params = {"query": query, "limit": str(max_results)}
    return "https://api.semanticscholar.org/graph/v1/snippet/search?" + urlencode(params)


def retrieve_semantic_scholar_snippets(
    candidate: FullTextCandidate,
    *,
    artifacts_dir: Path,
    safety_gate: SafetyGate,
    network_gate: SafetyGate | None = None,
    fetcher: FetchUrl = fetch_url_bytes,
    max_chars: int = DEFAULT_MAX_FULL_TEXT_CHARS,
) -> FullTextRecord:
    source_url: str | None = None
    try:
        query = candidate.title or candidate.paper_id
        source_url = build_semantic_scholar_snippet_url(query)
        payload = fetcher(
            source_url,
            network_gate=network_gate,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            max_bytes=DEFAULT_MAX_FULL_TEXT_BYTES,
        )
        snippets = _parse_semantic_scholar_snippets(payload, candidate_id=candidate.semantic_scholar_id)
        if not snippets:
            return _unavailable_record(candidate, "Semantic Scholar returned no body snippets for this record.")
        text = "\n\n".join(snippets)
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
        artifact_path = artifacts_dir / f"{_safe_artifact_id(candidate.paper_id)}-semantic-scholar-snippets.txt"
        write_text(artifact_path, text, safety_gate)
        encoded = text.encode("utf-8")
        status = FullTextStatus(
            provider="semantic_scholar",
            paper_id=candidate.paper_id,
            status=FullTextStatusValue.SUCCEEDED,
            scope=EvidenceScope.SNIPPET,
            source_url=source_url,
            artifact_path=str(artifact_path),
            parser="s2_snippet_json",
            byte_count=len(encoded),
            char_count=len(text),
            sha256=hashlib.sha256(encoded).hexdigest(),
            truncated=truncated,
            parser_limitations=("Semantic Scholar snippets are API excerpts matched to the candidate when a paperId is available, not complete full text.",),
            reason="Retrieved Semantic Scholar API snippets without following external openAccessPdf links.",
        )
        return FullTextRecord(candidate, status, text[:500])
    except Exception as exc:  # noqa: BLE001
        return _failed_record(candidate, source_url, exc)


def _unavailable_record(candidate: FullTextCandidate, reason: str) -> FullTextRecord:
    status = FullTextStatus(
        provider=candidate.provider,
        paper_id=candidate.paper_id,
        status=FullTextStatusValue.UNAVAILABLE,
        scope=EvidenceScope.UNAVAILABLE,
        source_url=candidate.source_url,
        reason=reason,
    )
    return FullTextRecord(candidate, status)


def _abstract_record(
    candidate: FullTextCandidate,
    reason: str | None,
    *,
    attempts: tuple[FullTextStatus, ...] = (),
) -> FullTextRecord:
    text = _collapse_whitespace(candidate.abstract or "")
    limitations = ["Provider abstract/metadata only; no approved full-text body was retrieved for this record."]
    if attempts:
        limitations.append(
            "One or more approved route attempts did not yield body evidence; see attempts for per-route status."
        )
    status = FullTextStatus(
        provider=candidate.provider,
        paper_id=candidate.paper_id,
        status=FullTextStatusValue.SKIPPED,
        scope=EvidenceScope.ABSTRACT,
        source_url=candidate.source_url,
        parser="provider_metadata_abstract",
        char_count=len(text),
        parser_limitations=tuple(limitations),
        reason=reason or "No approved full-text route succeeded; using abstract evidence scope.",
    )
    return FullTextRecord(candidate, status, text[:500], attempts)


def _with_attempts(record: FullTextRecord, attempts: list[FullTextStatus]) -> FullTextRecord:
    if not attempts:
        return record
    return FullTextRecord(record.candidate, record.status, record.text_preview, tuple(attempts))


def _failed_record(candidate: FullTextCandidate, source_url: str | None, exc: Exception) -> FullTextRecord:
    status_value = FullTextStatusValue.RATE_LIMITED if str(exc) == "rate_limited" else FullTextStatusValue.FAILED
    status = FullTextStatus(
        provider=candidate.provider,
        paper_id=candidate.paper_id,
        status=status_value,
        scope=EvidenceScope.UNAVAILABLE,
        source_url=source_url,
        reason=str(exc),
        parser_limitations=(type(exc).__name__,),
    )
    return FullTextRecord(candidate, status)


def _parse_semantic_scholar_snippets(payload: bytes | str, *, candidate_id: str | None = None) -> list[str]:
    data = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    snippets: list[str] = []
    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue
        result_id = _semantic_scholar_snippet_paper_id(item)
        if candidate_id and result_id != candidate_id:
            continue
        snippet = item.get("snippet")
        text = ""
        if isinstance(snippet, dict):
            text = str(snippet.get("text") or snippet.get("snippet") or "")
        elif isinstance(snippet, str):
            text = snippet
        if not text and item.get("text"):
            text = str(item.get("text"))
        text = _collapse_whitespace(text)
        if text:
            snippets.append(text)
    return snippets


def _semantic_scholar_snippet_paper_id(item: Mapping[str, Any]) -> str | None:
    paper = item.get("paper")
    if isinstance(paper, Mapping) and paper.get("paperId"):
        return str(paper["paperId"])
    if item.get("paperId"):
        return str(item["paperId"])
    return None


def _write_bytes(path: Path, payload: bytes, safety: SafetyGate) -> None:
    safety.assert_allowed(OperationClass.WRITE_DERIVED_OUTPUT, path, "write derived binary artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _safe_artifact_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe[:80] or "paper"


def _scope_counts(records: list[FullTextRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        scope = record.status.scope.value
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_element(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _first_text(root: ET.Element, name: str) -> str:
    element = _first_element(root, name)
    return _collapse_whitespace(" ".join(element.itertext())) if element is not None else ""


def _texts_under_first(root: ET.Element, name: str, *, include_tags: set[str]) -> list[str]:
    element = _first_element(root, name)
    if element is None:
        return []
    values: list[str] = []
    for child in element.iter():
        if _local_name(child.tag) in include_tags:
            text = _collapse_whitespace(" ".join(child.itertext()))
            if text and text not in values:
                values.append(text)
    return values


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())
