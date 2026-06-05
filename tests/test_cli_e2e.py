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
    cmd = [sys.executable, "-m", "medicine_agent.cli", "run", "--question", "Which ligand receptor interactions are most relevant in tumor immune communication?", "--data-dir", "data", "--output-dir", str(out), "--offline"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    for key in ["report_path", "manifest_path", "artifact_manifest_path", "search_log_path"]:
        assert Path(result[key]).exists(), key
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Research-only safety statement" in report
    assert "Evidence Table" in report
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source_statuses"]
    assert manifest["data_files"]
    assert all(dec["operation"] != "OVERWRITE_INPUT" for dec in manifest["safety_decisions"])
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


def test_cli_full_text_requires_live_api():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--question", "tumor immune communication", "--full-text"])

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


def test_cli_defaults_to_offline_without_live_api(monkeypatch):
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

    exit_code = cli.main(["run", "--question", "tumor immune communication"])

    assert exit_code == 0
    assert captured["request"].live_api is False
    assert captured["request"].offline is True


def test_orchestrator_full_text_writes_manifest_and_report(monkeypatch, tmp_path):
    from medicine_agent import orchestrator

    class FakeCoordinator:
        def search_question(self, question, *, allow_live=False, network_gate=None, max_results=5):
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
    assert "Full-text Retrieval Summary" in report
    assert "full_text_xml" in report


def test_orchestrator_report_explains_full_text_safety_skip(tmp_path):
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
    assert "Full-text retrieval was requested but not run" in report
    assert manifest["full_text_results"]["requested"] is True
    assert manifest["full_text_results"]["enabled"] is False
