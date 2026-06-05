import json
import subprocess
import sys
from pathlib import Path

import pytest

from medicine_agent import cli
from medicine_agent.models import ResearchRequest


def test_cli_offline_e2e_generates_required_artifacts_without_mutating_data(tmp_path):
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
    assert proc.returncode == 0, proc.stderr
    assert "[medicine-agent]" in proc.stderr
    assert "检测到本地数据读取请求" in proc.stderr
    result = json.loads(proc.stdout)
    for key in ["report_path", "manifest_path", "artifact_manifest_path", "search_log_path"]:
        assert Path(result[key]).exists(), key
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "仅限科研的安全声明" in report
    assert "证据表" in report
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source_statuses"]
    assert manifest["data_files"]
    assert all(dec["operation"] != "OVERWRITE_INPUT" for dec in manifest["safety_decisions"])
    after = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    assert before == after


def test_cli_literature_only_question_skips_data_directory_and_prints_steps(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    out = tmp_path / "literature-only"
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
        "--offline",
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env={**__import__("os").environ, "PYTHONPATH": "src"})

    assert proc.returncode == 0, proc.stderr
    assert "[medicine-agent]" in proc.stderr
    assert "已跳过本地数据扫描" in proc.stderr
    result = json.loads(proc.stdout)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert manifest["data_files"] == []
    assert any("已跳过本地数据扫描" in step["message"] for step in manifest["debug_steps"])
    assert not any(decision["operation"] == "READ_FILE" for decision in manifest["safety_decisions"])
    assert "未读取本地数据文件" in report
    after = {p: p.stat().st_mtime_ns for p in Path("data").glob("*.csv")}
    assert before == after


def test_cli_missing_question_exits_nonzero():
    proc = subprocess.run([sys.executable, "-m", "medicine_agent.cli", "run"], text=True, capture_output=True, check=False, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert proc.returncode != 0


def test_cli_live_api_flag_disables_offline_mode(monkeypatch):
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


def test_cli_full_text_defaults_to_live_api(monkeypatch):
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

    exit_code = cli.main(["run", "--question", "tumor immune communication", "--full-text"])

    assert exit_code == 0
    assert captured["request"].live_api is True
    assert captured["request"].offline is False
    assert captured["request"].full_text is True


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


def test_cli_full_text_flag_sets_live_full_text_request(monkeypatch):
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
    assert captured["request"].data_dir == Path("data")


def test_orchestrator_full_text_writes_manifest_and_report(monkeypatch, tmp_path):
    from medicine_agent import orchestrator

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MEDICINE_AGENT_DEEPSEEK_API_KEY", raising=False)

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
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MEDICINE_AGENT_DEEPSEEK_API_KEY", raising=False)

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

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MEDICINE_AGENT_DEEPSEEK_API_KEY", raising=False)

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
    monkeypatch.setattr(orchestrator, "synthesize_review_with_llm", fake_synthesis)

    result = orchestrator.run_research(
        ResearchRequest(
            question="帮我调研糖尿病研究的最新进展",
            data_dir=Path("data"),
            output_dir=tmp_path / "out",
            live_api=True,
            offline=False,
        )
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    review_path = Path(result["review_synthesis_path"])

    assert review_path.exists()
    assert manifest["review_synthesis"]["source"] == "llm_deepseek"
    assert manifest["evidence"][0]["claim"] == "近期糖尿病研究强调 beta cell stress。"
    assert "综述生成器：`llm_deepseek`" in report
    assert "LLM 生成的糖尿病结构化综述摘要" in report
