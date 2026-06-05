import json
import subprocess
import sys
from pathlib import Path


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
