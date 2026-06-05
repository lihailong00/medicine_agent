import json
import subprocess
import sys
from pathlib import Path

import pytest

from medicine_agent import cli
from medicine_agent.llm import LLMQueryPlan
from medicine_agent.models import ResearchRequest


def _configure_dummy_llm(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")


def _fake_query_plan(_question: str, **_kwargs) -> LLMQueryPlan:
    return LLMQueryPlan(
        search_topic="diabetes mellitus recent advances",
        subquestions=(
            "What recent mechanisms are supported by the retrieved literature?",
            "Which findings remain hypotheses or need stronger evidence?",
        ),
        sources=("pubmed", "semantic_scholar"),
        rationale="测试用 LLM query 规划结果。",
    )


def _fake_review_synthesis(ref: str, claim: str = "测试综述主张。") -> dict[str, object]:
    item = {
        "claim": claim,
        "status": "literature_supported",
        "evidence_refs": [ref],
        "confidence": "medium",
        "limitations": ["测试替身。"],
    }
    return {
        "schema_version": "review_synthesis.v1",
        "source": "llm_deepseek",
        "reason": "mock LLM synthesis",
        "executive_summary": "LLM 测试替身生成的结构化综述摘要。",
        "key_findings": [item],
        "evidence_table": [item],
        "mechanism_review": ["测试机制综述。"],
        "hypotheses": ["测试可检验假设。"],
        "limitations_conflicts": ["测试局限。"],
        "reproducibility_notes": ["测试复现说明。"],
    }


def test_cli_offline_mode_is_disabled_without_mutating_data(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    out = tmp_path / "agent-out"
    cmd = [
        sys.executable,
        "-m",
        "medicine_agent.cli",
        "run",
        "--question",
        "请使用 data 目录中的 CSV 分析 tumor immune communication 中最相关的 ligand receptor interactions",
        "--data-dir",
        "data",
        "--output-dir",
        str(out),
        "--offline",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert proc.returncode == 2
    assert "--offline 已禁用" in proc.stderr
    assert not out.exists()
    after = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    assert before == after


def test_cli_requires_llm_api_key_and_model_before_running(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    out = tmp_path / "literature-only"
    env = {
        key: value
        for key, value in __import__("os").environ.items()
        if key not in {"DEEPSEEK_API_KEY", "MEDICINE_AGENT_DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"}
    }
    env["PYTHONPATH"] = "src"
    cmd = [
        sys.executable,
        "-m",
        "medicine_agent.cli",
        "run",
        "--question",
        "帮我调研糖尿病研究的最新进展",
        "--data-dir",
        "data",
        "--output-dir",
        str(out),
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)

    assert proc.returncode == 2
    assert "缺少必需的大模型配置" in proc.stderr
    assert "DEEPSEEK_MODEL" in proc.stderr
    assert not out.exists()
    after = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    assert before == after


def test_cli_missing_question_exits_nonzero():
    proc = subprocess.run([sys.executable, "-m", "medicine_agent.cli", "run"], text=True, capture_output=True, check=False, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert proc.returncode != 0


def test_cli_live_api_flag_disables_offline_mode(monkeypatch):
    _configure_dummy_llm(monkeypatch)
    captured = {}

    def fake_run_research(request):
        captured["request"] = request
        return {
            "report_path": "report.md",
            "manifest_path": "run_manifest.json",
            "artifact_manifest_path": "artifact_manifest.json",
            "search_log_path": "search_log.json",
            "output_dir": "generated/medicine_agent",
        }

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    exit_code = cli.main(["run", "--question", "tumor immune communication", "--live-api"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is True


def test_cli_full_text_defaults_to_live_api(monkeypatch):
    _configure_dummy_llm(monkeypatch)
    captured = {}

    def fake_run_research(request):
        captured["request"] = request
        return {
            "report_path": "report.md",
            "manifest_path": "run_manifest.json",
            "artifact_manifest_path": "artifact_manifest.json",
            "search_log_path": "search_log.json",
            "full_text_results_path": "full_text_results.json",
            "review_synthesis_path": "review_synthesis.json",
            "output_dir": "generated/medicine_agent",
        }

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    exit_code = cli.main(["run", "--question", "tumor immune communication"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is True


def test_cli_no_full_text_disables_default_full_text(monkeypatch):
    _configure_dummy_llm(monkeypatch)
    captured = {}

    def fake_run_research(request):
        captured["request"] = request
        return {
            "report_path": "report.md",
            "manifest_path": "run_manifest.json",
            "artifact_manifest_path": "artifact_manifest.json",
            "search_log_path": "search_log.json",
            "full_text_results_path": "full_text_results.json",
            "review_synthesis_path": "review_synthesis.json",
            "output_dir": "generated/medicine_agent",
        }

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    exit_code = cli.main(["run", "--question", "tumor immune communication", "--no-full-text"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is False


def test_cli_full_text_rejects_offline_mode():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--question", "tumor immune communication", "--full-text", "--offline"])

    assert exc.value.code == 2


def test_research_request_rejects_full_text_without_live_mode(tmp_path):
    with pytest.raises(ValueError):
        ResearchRequest(
            question="tumor immune communication",
            output_dir=tmp_path / "out",
            full_text=True,
            live_api=False,
            offline=True,
        )


def test_research_request_rejects_offline_mode(tmp_path):
    with pytest.raises(ValueError, match="只支持联网模式"):
        ResearchRequest(
            question="tumor immune communication",
            output_dir=tmp_path / "out",
            offline=True,
        )


def test_research_request_rejects_disabled_live_api(tmp_path):
    with pytest.raises(ValueError, match="live_api 必须为 True"):
        ResearchRequest(
            question="tumor immune communication",
            output_dir=tmp_path / "out",
            live_api=False,
        )


def test_cli_full_text_flag_sets_live_full_text_request(monkeypatch):
    _configure_dummy_llm(monkeypatch)
    captured = {}

    def fake_run_research(request):
        captured["request"] = request
        return {
            "report_path": "report.md",
            "manifest_path": "run_manifest.json",
            "artifact_manifest_path": "artifact_manifest.json",
            "search_log_path": "search_log.json",
            "full_text_results_path": "full_text_results.json",
            "output_dir": "generated/medicine_agent",
        }

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    exit_code = cli.main(["run", "--question", "tumor immune communication", "--live-api", "--full-text"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is True


def test_cli_defaults_to_live_api_and_data_dir_without_flags(monkeypatch):
    _configure_dummy_llm(monkeypatch)
    captured = {}

    def fake_run_research(request):
        captured["request"] = request
        return {
            "report_path": "report.md",
            "manifest_path": "run_manifest.json",
            "artifact_manifest_path": "artifact_manifest.json",
            "search_log_path": "search_log.json",
            "review_synthesis_path": "review_synthesis.json",
            "output_dir": "generated/medicine_agent",
        }

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    exit_code = cli.main(["run", "--question", "tumor immune communication"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is True
    assert captured["request"].data_dir == Path("data")


def test_orchestrator_full_text_writes_manifest_and_report(monkeypatch, tmp_path):
    from medicine_agent import orchestrator

    _configure_dummy_llm(monkeypatch)

    class FakeCoordinator:
        def search_question(self, question, *, allow_live=False, network_gate=None, max_results=5, **_kwargs):
            assert allow_live is True
            return {
                "decomposition": {"question": question, "subquestions": [], "queries": []},
                "results": [],
                "papers": [
                    {
                        "provider": "pubmed",
                        "title": "Intercellular communication analysis",
                        "abstract": "Ligand receptor communication.",
                        "year": 2021,
                        "authors": ["Dimitrov D"],
                        "pmid": "34763053",
                        "pmcid": "PMC8576925",
                        "source_url": "https://pubmed.ncbi.nlm.nih.gov/34763053/",
                    }
                ],
                "search_log": [
                    {
                        "provider": "pubmed",
                        "endpoint_family": "ncbi_eutils",
                        "query": "ligand receptor",
                        "status": "succeeded",
                        "timestamp": "2026-06-05T00:00:00Z",
                        "result_ids": ["34763053"],
                    }
                ],
                "evidence_records": [
                    {
                        "paper_id": "34763053",
                        "provider": "pubmed",
                        "citation_label": "Dimitrov et al., 2021",
                        "evidence_note": "mock",
                        "source_url": "https://pubmed.ncbi.nlm.nih.gov/34763053/",
                    }
                ],
            }

    def fake_full_text(paper_payloads, **_kwargs):
        assert paper_payloads[0]["pmcid"] == "PMC8576925"
        return {
            "generated_at": "2026-06-05T00:00:00Z",
            "records": [
                {
                    "candidate": {
                        "provider": "pubmed",
                        "paper_id": "34763053",
                        "title": "Intercellular communication analysis",
                    },
                    "status": {
                        "provider": "pmc",
                        "paper_id": "34763053",
                        "status": "succeeded",
                        "scope": "full_text_xml",
                        "artifact_path": str(tmp_path / "out" / "artifacts" / "full_text" / "paper.txt"),
                        "reason": "mock PMC XML full text",
                    },
                    "text_preview": "Ligand receptor full text preview",
                }
            ],
            "statuses": [
                {
                    "provider": "pmc",
                    "paper_id": "34763053",
                    "status": "succeeded",
                    "scope": "full_text_xml",
                }
            ],
            "scope_counts": {"full_text_xml": 1},
        }

    monkeypatch.setattr(orchestrator, "build_default_coordinator", lambda: FakeCoordinator())
    monkeypatch.setattr(orchestrator, "retrieve_full_text_for_payloads", fake_full_text)
    monkeypatch.setattr("medicine_agent.llm.plan_query_with_llm", _fake_query_plan)
    monkeypatch.setattr(
        orchestrator,
        "synthesize_review_with_llm",
        lambda *_args, **_kwargs: _fake_review_synthesis("34763053", "配体受体通信由测试文献支持。"),
    )

    result = orchestrator.run_research(
        ResearchRequest(
            question="ligand receptor communication",
            data_dir=Path("data"),
            output_dir=tmp_path / "out",
            live_api=True,
            offline=False,
            full_text=True,
        )
    )

    full_text_path = Path(result["full_text_results_path"])
    assert full_text_path.exists()
    full_text_payload = json.loads(full_text_path.read_text(encoding="utf-8"))
    assert full_text_payload["scope_counts"] == {"full_text_xml": 1}
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["request"]["full_text"] is True
    assert manifest["full_text_results"]["records"][0]["status"]["scope"] == "full_text_xml"
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "全文检索摘要" in report
    assert "full_text_xml" in report


def test_orchestrator_report_explains_full_text_safety_skip(monkeypatch, tmp_path):
    _configure_dummy_llm(monkeypatch)
    monkeypatch.setattr("medicine_agent.llm.plan_query_with_llm", _fake_query_plan)

    result = cli.run_research(
        ResearchRequest(
            question="How should I treat this patient?",
            data_dir=Path("data"),
            output_dir=tmp_path / "out",
            live_api=True,
            offline=False,
            full_text=True,
        )
    )

    report = Path(result["report_path"]).read_text(encoding="utf-8")
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "已请求全文检索但未运行" in report
    assert manifest["full_text_results"]["requested"] is True
    assert manifest["full_text_results"]["enabled"] is False


def test_orchestrator_uses_llm_review_synthesis_when_available(monkeypatch, tmp_path):
    from medicine_agent import orchestrator

    _configure_dummy_llm(monkeypatch)

    class FakeCoordinator:
        def search_question(self, question, *, allow_live=False, network_gate=None, max_results=5, **_kwargs):
            assert allow_live is True
            return {
                "decomposition": {"question": question, "subquestions": [], "queries": []},
                "results": [],
                "papers": [
                    {
                        "provider": "pubmed",
                        "title": "Diabetes beta cell review",
                        "abstract": "Beta cell stress and insulin resistance are discussed.",
                        "year": 2026,
                        "authors": ["Curie M"],
                        "pmid": "PMID-1",
                        "source_url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                    }
                ],
                "search_log": [
                    {
                        "provider": "pubmed",
                        "endpoint_family": "ncbi_eutils",
                        "query": "diabetes",
                        "status": "succeeded",
                        "timestamp": "2026-06-05T00:00:00Z",
                        "result_ids": ["PMID-1"],
                    }
                ],
                "evidence_records": [],
            }

    def fake_synthesis(*_args, **_kwargs):
        return {
            "schema_version": "review_synthesis.v1",
            "source": "llm_deepseek",
            "reason": "mock LLM synthesis",
            "executive_summary": "LLM 生成的糖尿病结构化综述摘要。",
            "key_findings": [
                {
                    "claim": "近期糖尿病研究强调 beta cell stress。",
                    "status": "literature_supported",
                    "evidence_refs": ["PMID-1"],
                    "confidence": "medium",
                    "limitations": ["mock"],
                }
            ],
            "evidence_table": [
                {
                    "claim": "近期糖尿病研究强调 beta cell stress。",
                    "status": "literature_supported",
                    "evidence_refs": ["PMID-1"],
                    "confidence": "medium",
                    "limitations": ["mock"],
                }
            ],
            "mechanism_review": ["beta cell stress 与胰岛素抵抗共同构成机制线索。"],
            "hypotheses": ["可检验 beta cell stress 标志物。"],
            "limitations_conflicts": ["仅 mock 文献。"],
            "reproducibility_notes": ["引用 PMID-1 来自本次运行。"],
        }

    monkeypatch.setattr(orchestrator, "build_default_coordinator", lambda: FakeCoordinator())
    monkeypatch.setattr("medicine_agent.llm.plan_query_with_llm", _fake_query_plan)
    monkeypatch.setattr(orchestrator, "synthesize_review_with_llm", fake_synthesis)

    result = orchestrator.run_research(
        ResearchRequest(
            question="帮我调研糖尿病研究的最新进展",
            data_dir=Path("data"),
            output_dir=tmp_path / "out",
            live_api=True,
            offline=False,
            full_text=False,
        )
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    review_path = Path(result["review_synthesis_path"])

    assert review_path.exists()
    assert manifest["review_synthesis"]["source"] == "llm_deepseek"
    assert manifest["evidence"][0]["claim"] == "近期糖尿病研究强调 beta cell stress。"
    debug_messages = "\n".join(item["message"] for item in manifest["debug_steps"])
    assert "查询改写结果：search_topic=`diabetes mellitus recent advances`" in debug_messages
    assert "来源计划预览[1]：pubmed query=`(diabetes mellitus recent advances) AND (review OR mechanism OR single-cell)`" in debug_messages
    assert "论文预览[1/1]：pubmed；id=PMID-1；Diabetes beta cell review" in debug_messages
    assert "基础证据预览[1/" in debug_messages
    assert "LLM 输入规模：papers=1/1" in debug_messages
    assert "LLM 预算参数：context_tokens=" in debug_messages
    assert "LLM 结构化综述 API 调用返回：耗时" in debug_messages
    assert "LLM 摘要预览：LLM 生成的糖尿病结构化综述摘要。" in debug_messages
    assert "LLM 关键发现预览[1/1]" in debug_messages
    assert "综述生成器：`llm_deepseek`" in report
    assert "LLM 生成的糖尿病结构化综述摘要" in report
