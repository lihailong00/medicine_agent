import csv
from pathlib import Path

from medicine_agent.data.discovery import discover_data_files
from medicine_agent.data.liana import summarize_liana
from medicine_agent.safety import SafetyGate


def _write_csv(path: Path, rows):
    fieldnames = ["source", "target", "ligand.complex", "ligand", "receptor.complex", "receptor", "receptor.prop", "ligand.prop", "ligand.expr", "receptor.expr", "lr.mean", "pvalue"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_liana_ranking_is_deterministic_and_preserves_provenance(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [
        {"source": "B", "target": "T", "ligand.complex": "", "ligand": "L2", "receptor.complex": "", "receptor": "R2", "receptor.prop": "1", "ligand.prop": "1", "ligand.expr": "2", "receptor.expr": "3", "lr.mean": "9", "pvalue": "0.01"},
        {"source": "A", "target": "T", "ligand.complex": "", "ligand": "L1", "receptor.complex": "", "receptor": "R1", "receptor.prop": "1", "ligand.prop": "1", "ligand.expr": "2", "receptor.expr": "3", "lr.mean": "10", "pvalue": "0.01"},
        {"source": "A", "target": "T", "ligand.complex": "", "ligand": "BAD", "receptor.complex": "", "receptor": "BAD", "receptor.prop": "1", "ligand.prop": "1", "ligand.expr": "2", "receptor.expr": "3", "lr.mean": "", "pvalue": "nan-ish"},
    ])
    gate = SafetyGate(data, tmp_path / "out")
    records = discover_data_files(data, gate)
    summary = summarize_liana(records, gate)
    top = summary["top_interactions"]
    assert top[0]["ligand"] == "L1"
    assert top[0]["row_index"] == 3
    assert top[0]["source_file"].endswith("a.csv")
    assert summary["unranked_interactions"][0]["reason"] == "invalid pvalue or lr.mean"
    assert records[0].parser_status == "parsed_liana_csv"


def test_existing_data_fixtures_are_liana_shape():
    data = Path("data")
    gate = SafetyGate(data, Path("/tmp/medicine-agent-test-out"))
    records = discover_data_files(data, gate)
    summary = summarize_liana(records, gate, top_n=5)
    assert len([r for r in records if r.file_type == "csv"]) >= 1
    assert summary["top_interactions"]
    assert all("source_file" in row and "row_index" in row for row in summary["top_interactions"])
