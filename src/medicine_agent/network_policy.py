"""获批学术 API 访问共用的实时网络策略。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping
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
        "api.deepseek.com",
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
        return UrlPolicyDecision(url, False, "blocked", "网络调用必须使用 HTTPS")
    if parsed.netloc == "eutils.ncbi.nlm.nih.gov" and parsed.path.startswith("/entrez/eutils/"):
        return UrlPolicyDecision(url, True, "ncbi_eutils", "NCBI E-utilities 端点已列入 allowlist")
    if parsed.netloc == "export.arxiv.org" and parsed.path == "/api/query":
        return UrlPolicyDecision(url, True, "arxiv_atom", "arXiv Atom API 端点已列入 allowlist")
    if parsed.netloc == "api.semanticscholar.org" and parsed.path in {
        "/graph/v1/paper/search",
        "/graph/v1/snippet/search",
    }:
        return UrlPolicyDecision(url, True, "s2_graph", "Semantic Scholar API 端点已列入 allowlist")
    if parsed.netloc == "arxiv.org" and _is_allowlisted_arxiv_full_text_path(parsed.path):
        return UrlPolicyDecision(url, True, "arxiv_full_text", "arXiv 全文端点已列入 allowlist")
    if parsed.netloc == "api.deepseek.com" and parsed.path in {"/chat/completions", "/v1/chat/completions"}:
        return UrlPolicyDecision(url, True, "deepseek_chat", "DeepSeek Chat Completions 端点已列入 query 规划 allowlist")
    return UrlPolicyDecision(
        url,
        False,
        "blocked",
        "网络调用仅限 PubMed/NCBI E-utilities、arXiv API/全文路径、Semantic Scholar API 与 DeepSeek query 规划端点",
    )


def assert_url_allowed(url: str) -> UrlPolicyDecision:
    decision = classify_allowed_url(url)
    if not decision.allowed:
        parsed = urlparse(url)
        raise PermissionError(f"实时文献请求被 URL 策略阻断: {parsed.scheme}://{parsed.netloc}{parsed.path}")
    return decision


def fetch_url_bytes(
    url: str,
    *,
    network_gate: Any | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    rationale: str = "向 allowlist 中的提供器发起实时文献 API 请求",
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


def post_json_bytes(
    url: str,
    payload: Mapping[str, object],
    *,
    headers: Mapping[str, str] | None = None,
    network_gate: Any | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    rationale: str = "向 allowlist 中的 JSON API 发起实时请求",
    max_bytes: int | None = None,
) -> bytes:
    """向 allowlist JSON API 发送 POST 请求，避免把敏感 header 写入安全日志。"""

    assert_url_allowed(url)
    if network_gate is not None:
        decision = network_gate.decide(OperationClass.NETWORK_CALL, url, rationale)
        if decision.status != SafetyDecisionStatus.ALLOWED:
            raise PermissionError(decision.rationale)

    request_headers = {"User-Agent": DEFAULT_USER_AGENT, "Content-Type": "application/json"}
    if headers:
        request_headers.update(dict(headers))
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method="POST")
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
    """阻断共享 URL 策略之外跳转的重定向处理器。"""

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
            raise ValueError(f"响应超过配置的字节上限 byte cap ({max_bytes})")
        chunks.append(chunk)
    return b"".join(chunks)


def _is_allowlisted_arxiv_full_text_path(path: str) -> bool:
    if not path.startswith("/pdf/"):
        return False
    arxiv_id = path.removeprefix("/pdf/").removesuffix(".pdf")
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id))
