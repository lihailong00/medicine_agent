from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from medicine_agent.data.liana import process_data_dir, summarize_liana_files
from medicine_agent.safety import SafetyGate, SafetyStatus

HEADER = [
    "source",
    "target",
    "ligand.complex",
    "ligand",
    "receptor.complex",
    "receptor",
    "receptor.prop",
    "ligand.prop",
    "ligand.expr",
    "receptor.expr",
    "lr.mean",
    "pvalue",
]


def write_csv(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or HEADER)
        writer.writerows(rows)


class LianaDataTests(unittest.TestCase):
    def test_stable_ranking_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            write_csv(
                data_dir / "b.csv",
                [
                    ["B", "T", "L2", "L2", "R2", "R2", "", "", "", "", "7", "0.01"],
                    ["B", "T", "Lbad", "Lbad", "Rbad", "Rbad", "", "", "", "", "8", "bad"],
                ],
            )
            write_csv(
                data_dir / "a.csv",
                [["A", "T", "L1", "L1", "R1", "R1", "", "", "", "", "5", "0.01"]],
            )

            summary = summarize_liana_files(data_dir, top_n=10)

            ranked = summary.ranked_interactions
            self.assertEqual([item.ligand for item in ranked], ["L2", "L1"])
            self.assertEqual(ranked[0].provenance["relative_file"], "b.csv")
            self.assertEqual(ranked[0].provenance["csv_row_number"], 2)
            self.assertEqual(len(summary.unranked_rows), 1)
            self.assertEqual(summary.unranked_rows[0]["reason"], "missing_or_invalid_ranking_value")

    def test_missing_required_columns_warns_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            write_csv(data_dir / "not_liana.csv", [["A", "B"]], header=["source", "target"])

            summary = summarize_liana_files(data_dir)

            self.assertEqual(summary.ranked_interactions, [])
            self.assertIn("missing LIANA required columns", summary.files[0].warnings[0])

    def test_process_writes_only_generated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            output_dir = Path(tmp) / "generated"
            data_dir.mkdir()
            write_csv(data_dir / "liana.csv", [["A", "B", "L", "L", "R", "R", "0.1", "0.2", "3", "4", "9", "0.001"]])
            before = (data_dir / "liana.csv").read_bytes()

            artifacts = process_data_dir(data_dir, output_dir, top_n=5)

            self.assertEqual((data_dir / "liana.csv").read_bytes(), before)
            self.assertEqual(set(artifacts), {"data_manifest", "liana_interactions", "liana_summary"})
            for path in artifacts.values():
                self.assertTrue(path.exists())
                self.assertTrue(path.resolve().is_relative_to(output_dir.resolve()))
            interactions = json.loads(artifacts["liana_interactions"].read_text(encoding="utf-8"))
            self.assertEqual(interactions["ranked_interactions"][0]["provenance"]["csv_row_number"], 2)
            manifest = json.loads(artifacts["data_manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["research_only"])
            self.assertTrue(any(d["operation"] == "WRITE_DERIVED_OUTPUT" for d in manifest["safety_decisions"]))

    def test_safety_gate_blocks_output_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = SafetyGate(output_dir=Path(tmp) / "generated", data_dir=Path(tmp) / "data")
            decision = gate.allow_write_derived_output(Path(tmp) / "data" / "mutate.csv")
            self.assertEqual(decision.status, SafetyStatus.BLOCKED)

    def test_excel_word_placeholders_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "input.xlsx").write_bytes(b"placeholder")
            (data_dir / "notes.docx").write_bytes(b"placeholder")

            summary = summarize_liana_files(data_dir)

            statuses = {record.file_type: record.parser_status.status for record in summary.files}
            self.assertEqual(statuses["excel"], "unsupported_optional_dependency")
            self.assertEqual(statuses["word"], "unsupported_optional_dependency")
            self.assertGreaterEqual(len(summary.warnings), 2)


if __name__ == "__main__":
    unittest.main()
