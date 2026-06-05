import pytest

from medicine_agent.models import EvidenceItem, PaperRecord, SourcePlan, SourceStatus
from medicine_agent.reporting.evidence import build_evidence
from medicine_agent.reporting.markdown import RESEARCH_ONLY_STATEMENT, render_report
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
    status = SourceStatus("pubmed", "ncbi_eutils", "query", "succeeded", "now", ["id1"], reason="mock")
    assert status.provider == "pubmed"
    assert status.result_ids == ["id1"]


def test_live_metadata_only_evidence_has_no_fixture_language():
    evidence = build_evidence(
        "ligand receptor communication",
        [PaperRecord(title="Live PubMed paper", abstract="abstract", source="pubmed", pmid="123")],
        [],
        live_literature_enabled=True,
    )

    literature = next(item for item in evidence if item.status == "literature_supported")
    assert "实时文献元数据/摘要" in literature.claim
    assert "fixture" not in literature.claim.lower()


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


def test_render_report_is_paper_like_with_inline_citations():
    papers = [
        PaperRecord(
            title="Beta Cell Stress in Diabetes",
            abstract="Recent diabetes studies discuss beta cell stress and immune-metabolic mechanisms.",
            source="pubmed",
            year=2026,
            authors=["Curie M"],
            pmid="PMID-1",
            source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
        )
    ]
    evidence = [
        EvidenceItem(
            "近期糖尿病研究强调 beta cell stress 与免疫代谢机制。",
            "literature_supported",
            evidence_refs=["PMID-1"],
            confidence="high",
            limitations=["示例证据仅覆盖一篇摘要。"],
        )
    ]
    synthesis = {
        "source": "llm_deepseek",
        "reason": "测试用结构化综述。",
        "executive_summary": "糖尿病研究正在关注 beta cell stress 与免疫代谢交互。",
        "key_findings": [
            {
                "claim": "近期糖尿病研究强调 beta cell stress 与免疫代谢机制。",
                "status": "literature_supported",
                "evidence_refs": ["PMID-1"],
                "confidence": "high",
                "limitations": ["示例证据仅覆盖一篇摘要。"],
            }
        ],
        "evidence_table": [
            {
                "claim": "近期糖尿病研究强调 beta cell stress 与免疫代谢机制。",
                "status": "literature_supported",
                "evidence_refs": ["PMID-1"],
                "confidence": "high",
                "limitations": ["示例证据仅覆盖一篇摘要。"],
            }
        ],
        "mechanism_review": ["beta cell stress 可能连接代谢压力与炎症信号。"],
        "hypotheses": ["可在单细胞数据中检验应激通路是否与免疫细胞浸润相关。"],
        "limitations_conflicts": ["需要更多全文与数据验证。"],
        "reproducibility_notes": ["测试报告保留 run manifest。"],
    }

    report = render_report(
        question="帮我调研糖尿病研究的最新进展",
        subquestions=["糖尿病研究有哪些最新机制方向？"],
        source_plans=[SourcePlan("pubmed", "diabetes recent advances", "测试检索计划")],
        source_statuses=[
            SourceStatus(
                "pubmed",
                "ncbi_eutils",
                "diabetes recent advances",
                "succeeded",
                "2026-06-05T00:00:00Z",
                ["PMID-1"],
            )
        ],
        papers=papers,
        data_records=[],
        liana_summary={"ranking_method": "not_requested", "top_interactions": [], "warnings": []},
        evidence=evidence,
        artifact_paths=["generated/medicine_agent/report.md"],
        manifest_path="generated/medicine_agent/run_manifest.json",
        full_text_records=[],
        full_text_summary={"requested": True, "enabled": True},
        review_synthesis=synthesis,
    )

    assert "# 生信科研综述报告" in report
    assert "## 摘要" in report
    assert "## 引言" in report
    assert "## 主要发现" in report
    assert "近期糖尿病研究强调 beta cell stress 与免疫代谢机制。 [1]" in report
    assert "## 参考文献与证据来源" in report
    assert "[1] Curie M (2026). Beta Cell Stress in Diabetes." in report
    assert "PMID: PMID-1" in report
