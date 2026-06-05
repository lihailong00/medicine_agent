"""Shared live-network policy for allowlisted scholarly API access."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from medicine_agent.models import OperationClass, SafetyDecisionStatus

ALLOWED_LIVE_HOSTS = frozenset(
    {
        "eutils.ncbi.nlm.nih.gov",
        "export.arxiv.org",
        "api.semanticscholar.org",
        "arxiv.org",
    }
)
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = "medicine-agent/0.1 (+research-only)"


@dataclass(frozen=True)
class UrlPolicyDecision:
    url: str
    allowed: bool
    endpoint_family: str
    reason: str


def classify_allowed_url(url: str) -> UrlPolicyDecision:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return UrlPolicyDecision(url, False, "blocked", "network calls require HTTPS")
    if parsed.netloc == "eutils.ncbi.nlm.nih.gov" and parsed.path.startswith("/entrez/eutils/"):
        return UrlPolicyDecision(url, True, "ncbi_eutils", "NCBI E-utilities endpoint is allowlisted")
    if parsed.netloc == "export.arxiv.org" and parsed.path == "/api/query":
        return UrlPolicyDecision(url, True, "arxiv_atom", "arXiv Atom API endpoint is allowlisted")
    if parsed.netloc == "api.semanticscholar.org" and parsed.path in {
        "/graph/v1/paper/search",
        "/graph/v1/snippet/search",
    }:
        return UrlPolicyDecision(url, True, "s2_graph", "Semantic Scholar API endpoint is allowlisted")
    if parsed.netloc == "arxiv.org" and _is_allowlisted_arxiv_full_text_path(parsed.path):
        return UrlPolicyDecision(url, True, "arxiv_full_text", "arXiv full-text endpoint is allowlisted")
    return UrlPolicyDecision(
        url,
        False,
        "blocked",
        "network calls are restricted to PubMed/NCBI E-utilities, arXiv API/full-text paths, and Semantic Scholar API",
    )


def assert_url_allowed(url: str) -> UrlPolicyDecision:
    decision = classify_allowed_url(url)
    if not decision.allowed:
        parsed = urlparse(url)
        raise PermissionError(f"live literature request blocked by URL policy: {parsed.scheme}://{parsed.netloc}{parsed.path}")
    return decision


def fetch_url_bytes(
    url: str,
    *,
    network_gate: Any | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    rationale: str = "live literature API request to allowlisted provider",
    max_bytes: int | None = None,
) -> bytes:
    assert_url_allowed(url)
    if network_gate is not None:
        decision = network_gate.decide(OperationClass.NETWORK_CALL, url, rationale)
        if decision.status != SafetyDecisionStatus.ALLOWED:
            raise PermissionError(decision.rationale)

    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    opener = build_opener(AllowlistRedirectHandler)
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = getattr(response, "geturl", lambda: url)()
            assert_url_allowed(final_url)
            return _read_capped(response, max_bytes=max_bytes)
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("rate_limited") from exc
        raise
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


class AllowlistRedirectHandler(HTTPRedirectHandler):
    """Redirect handler that blocks redirects outside the shared URL policy."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        assert_url_allowed(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_capped(response: Any, *, max_bytes: int | None) -> bytes:
    if max_bytes is None:
        return response.read()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"response exceeded configured byte cap ({max_bytes})")
        chunks.append(chunk)
    return b"".join(chunks)


def _is_allowlisted_arxiv_full_text_path(path: str) -> bool:
    if not path.startswith("/pdf/"):
        return False
    arxiv_id = path.removeprefix("/pdf/").removesuffix(".pdf")
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id))
