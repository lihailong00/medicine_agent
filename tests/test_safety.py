from pathlib import Path
from urllib.request import Request

import pytest

from medicine_agent.models import OperationClass, SafetyDecisionStatus
from medicine_agent.network_policy import AllowlistRedirectHandler, assert_url_allowed, classify_allowed_url, fetch_url_bytes
from medicine_agent.safety import SafetyGate


def test_safety_gate_allows_only_derived_output_writes(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    allowed = gate.decide(OperationClass.WRITE_DERIVED_OUTPUT, tmp_path / "out" / "report.md", "derived")
    blocked = gate.decide(OperationClass.WRITE_DERIVED_OUTPUT, tmp_path / "elsewhere.md", "outside")
    assert allowed.status == SafetyDecisionStatus.ALLOWED
    assert blocked.status == SafetyDecisionStatus.BLOCKED


def test_safety_gate_requires_confirmation_for_high_risk_ops(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    for op in [
        OperationClass.INSTALL_DEP,
        OperationClass.USE_API_KEY,
        OperationClass.OVERWRITE_INPUT,
        OperationClass.LONG_JOB,
        OperationClass.RUN_SCRIPT,
    ]:
        assert gate.decide(op, "target", "risk").status == SafetyDecisionStatus.NEEDS_CONFIRMATION


def test_safety_gate_allows_only_approved_live_api_hosts(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")

    allowed = gate.decide(
        OperationClass.NETWORK_CALL,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=x",
        "live PubMed search",
    )
    blocked_host = gate.decide(OperationClass.NETWORK_CALL, "https://api.biorxiv.org/details", "disallowed")
    blocked_scheme = gate.decide(OperationClass.NETWORK_CALL, "http://export.arxiv.org/api/query", "plaintext")
    allowed_arxiv_pdf = gate.decide(OperationClass.NETWORK_CALL, "https://arxiv.org/pdf/2301.00001", "arxiv pdf")
    blocked_arxiv_page = gate.decide(OperationClass.NETWORK_CALL, "https://arxiv.org/abs/2301.00001", "arxiv page")

    assert allowed.status == SafetyDecisionStatus.ALLOWED
    assert allowed.target.startswith("https://eutils.ncbi.nlm.nih.gov/")
    assert allowed_arxiv_pdf.status == SafetyDecisionStatus.ALLOWED
    assert blocked_host.status == SafetyDecisionStatus.BLOCKED
    assert blocked_scheme.status == SafetyDecisionStatus.BLOCKED
    assert blocked_arxiv_page.status == SafetyDecisionStatus.BLOCKED


def test_shared_url_policy_classifies_allowlisted_paths():
    assert classify_allowed_url("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=123").allowed
    assert classify_allowed_url("https://api.semanticscholar.org/graph/v1/snippet/search?query=x").allowed
    assert classify_allowed_url("https://arxiv.org/pdf/2301.00001").allowed
    assert not classify_allowed_url("https://arxiv.org/abs/2301.00001").allowed
    assert not classify_allowed_url("https://arxiv.org/pdf/not-an-arxiv-id").allowed
    assert not classify_allowed_url("https://publisher.example/file.pdf").allowed


def test_redirect_handler_blocks_cross_host_redirect_before_body_read():
    handler = AllowlistRedirectHandler()
    request = Request("https://arxiv.org/pdf/2301.00001")

    with pytest.raises(PermissionError):
        handler.redirect_request(request, None, 302, "Found", {}, "https://publisher.example/file.pdf")

    allowed = handler.redirect_request(request, None, 302, "Found", {}, "https://arxiv.org/pdf/2301.00001v2")
    assert allowed is not None
    with pytest.raises(PermissionError):
        assert_url_allowed("http://arxiv.org/pdf/2301.00001")


def test_fetch_url_bytes_enforces_byte_cap(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

        def geturl(self):
            return "https://arxiv.org/pdf/2301.00001"

        def read(self, _size=None):
            if not hasattr(self, "_sent"):
                self._sent = True
                return b"x" * 5
            return b""

    class FakeOpener:
        def open(self, _request, timeout=20):
            del timeout
            return FakeResponse()

    monkeypatch.setattr("medicine_agent.network_policy.build_opener", lambda *_args: FakeOpener())

    with pytest.raises(ValueError, match="byte cap"):
        fetch_url_bytes("https://arxiv.org/pdf/2301.00001", max_bytes=4)


def test_clinical_question_is_blocked_as_out_of_scope(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    decision = gate.screen_question("How should I treat this patient?")
    assert decision is not None
    assert decision.status == SafetyDecisionStatus.BLOCKED
    assert "超出范围" in decision.rationale
