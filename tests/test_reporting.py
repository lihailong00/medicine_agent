import pytest

from medicine_agent.models import EvidenceItem, SourceStatus
from medicine_agent.reporting.markdown import RESEARCH_ONLY_STATEMENT


def test_supported_evidence_requires_reference():
    with pytest.raises(ValueError):
        EvidenceItem("unsupported definitive claim", "data_supported")


def test_hypothesis_evidence_can_be_labeled_without_refs():
    item = EvidenceItem("needs validation", "hypothesis")
    assert item.status == "hypothesis"


def test_research_only_statement_is_strict():
    assert "not clinical decision support" in RESEARCH_ONLY_STATEMENT
    assert "diagnosis" in RESEARCH_ONLY_STATEMENT


def test_source_status_observability_fields():
    status = SourceStatus("pubmed", "offline_mock", "query", "succeeded", "now", ["id1"], reason="mock")
    assert status.provider == "pubmed"
    assert status.result_ids == ["id1"]
