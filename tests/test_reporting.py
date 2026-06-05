import pytest

from medicine_agent.models import EvidenceItem, PaperRecord, SourceStatus
from medicine_agent.reporting.evidence import build_evidence
from medicine_agent.reporting.markdown import RESEARCH_ONLY_STATEMENT


def test_supported_evidence_requires_reference():
    with pytest.raises(ValueError):
        EvidenceItem("unsupported definitive claim", "data_supported")


def test_hypothesis_evidence_can_be_labeled_without_refs():
    item = EvidenceItem("needs validation", "hypothesis")
    assert item.status == "hypothesis"


def test_research_only_statement_is_strict():
    assert "不是临床决策支持" in RESEARCH_ONLY_STATEMENT
    assert "诊断" in RESEARCH_ONLY_STATEMENT


def test_source_status_observability_fields():
    status = SourceStatus("pubmed", "offline_mock", "query", "succeeded", "now", ["id1"], reason="mock")
    assert status.provider == "pubmed"
    assert status.result_ids == ["id1"]


def test_live_metadata_only_evidence_is_not_called_offline_mock():
    evidence = build_evidence(
        "ligand receptor communication",
        [PaperRecord(title="Live PubMed paper", abstract="abstract", source="pubmed", pmid="123")],
        [],
        live_literature_enabled=True,
    )

    literature = next(item for item in evidence if item.status == "literature_supported")
    assert "实时文献元数据/摘要" in literature.claim
    assert "离线模拟" not in literature.claim
