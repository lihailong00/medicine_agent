import pytest

from medicine_agent.models import EvidenceItem, PaperRecord, SourceStatus
from medicine_agent.reporting.evidence import build_evidence
from medicine_agent.reporting.markdown import RESEARCH_ONLY_STATEMENT
from medicine_agent.reporting.synthesis import evidence_from_synthesis, sanitize_review_synthesis


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


def test_synthesis_sanitizer_filters_unretrieved_references():
    raw = {
        "source": "llm_deepseek",
        "executive_summary": "摘要",
        "evidence_table": [
            {
                "claim": "有引用支持的主张",
                "status": "literature_supported",
                "evidence_refs": ["PMID-1", "NOT-RETRIEVED"],
                "confidence": "high",
                "limitations": ["仅摘要证据"],
            },
            {
                "claim": "没有可验证引用的主张",
                "status": "literature_supported",
                "evidence_refs": ["NOT-RETRIEVED"],
                "confidence": "high",
            },
        ],
        "key_findings": [],
    }

    synthesis = sanitize_review_synthesis(raw, allowed_refs={"PMID-1"})

    assert synthesis is not None
    rows = synthesis["evidence_table"]
    assert rows[0]["evidence_refs"] == ["PMID-1"]
    assert rows[1]["status"] == "hypothesis"
    assert rows[1]["evidence_refs"] == []


def test_evidence_from_synthesis_replaces_rule_evidence_with_clean_llm_items():
    fallback = [EvidenceItem("fallback hypothesis", "hypothesis")]
    synthesis = {
        "evidence_table": [
            {
                "claim": "LLM 证据主张",
                "status": "literature_supported",
                "evidence_refs": ["PMID-1"],
                "confidence": "medium",
                "limitations": ["摘要证据"],
            }
        ]
    }

    evidence = evidence_from_synthesis(synthesis, fallback, allowed_refs={"PMID-1"})

    assert evidence[0].claim == "LLM 证据主张"
    assert evidence[0].evidence_refs == ["PMID-1"]
