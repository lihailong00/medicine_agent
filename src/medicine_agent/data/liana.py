"""仅用标准库实现的 CSV/LIANA 摄取、画像、排序与溯源。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from medicine_agent.safety import SafetyGate, SafetyStatus

LIANA_REQUIRED_COLUMNS = (
    "source",
    "target",
    "ligand.complex",
    "ligand",
    "receptor.complex",
    "receptor",
    "lr.mean",
    "pvalue",
)

LIANA_OPTIONAL_NUMERIC_COLUMNS = (
    "receptor.prop",
    "ligand.prop",
    "ligand.expr",
    "receptor.expr",
)

RANKING_COLUMNS = ("pvalue", "lr.mean")


@dataclass(frozen=True)
class ParserStatus:
    """已发现输入解析器的状态。"""

    status: str
    file_type: str
    reason: str = ""


@dataclass(frozen=True)
class ColumnProfile:
    """单列的简要 schema 画像。"""

    name: str
    present_count: int = 0
    missing_count: int = 0
    numeric_valid_count: int = 0
    numeric_invalid_count: int = 0
    numeric_min: float | None = None
    numeric_max: float | None = None
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataFileRecord:
    """已发现数据文件及其解析器/schema 元数据。"""

    path: str
    file_type: str
    parser_status: ParserStatus
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    schema_profile: dict[str, ColumnProfile] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class LianaInteraction:
    """带源行溯源、可排序的 LIANA 互作。"""

    source: str
    target: str
    ligand: str
    receptor: str
    ligand_complex: str
    receptor_complex: str
    pvalue: float
    lr_mean: float
    metrics: dict[str, float]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LianaSummary:
    """用于序列化与报告的数据通道完整摘要。"""

    generated_at: str
    input_dir: str
    files: list[DataFileRecord]
    ranked_interactions: list[LianaInteraction]
    unranked_rows: list[dict[str, Any]]
    warnings: list[str]
    safety_decisions: list[dict[str, str]]


def discover_data_files(data_dir: str | Path) -> list[Path]:
    """返回确定性的受支持/占位输入文件候选。"""

    root = Path(data_dir)
    suffixes = {".csv", ".tsv", ".xlsx", ".xls", ".docx", ".doc"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def summarize_liana_files(data_dir: str | Path, *, top_n: int = 50, safety_gate: SafetyGate | None = None) -> LianaSummary:
    """发现数据文件、画像 CSV，并对 LIANA 互作排序。

    当可选解析器不可用时，Excel 与 Word 文件会被表示为可操作的解析器占位记录；
    不会安装或要求任何依赖。
    """

    data_root = Path(data_dir).resolve()
    gate = safety_gate or SafetyGate(output_dir=data_root / "generated", data_dir=data_root)
    records: list[DataFileRecord] = []
    interactions: list[LianaInteraction] = []
    unranked_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in discover_data_files(data_root):
        read_decision = gate.allow_read_file(path)
        if read_decision.status != SafetyStatus.ALLOWED:
            warning = f"已跳过读取 {path}: {read_decision.rationale}"
            warnings.append(warning)
            records.append(_placeholder_record(path, "blocked", warning))
            continue

        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            reason = "Excel 解析需要可选解析依赖；CSV 路径仍然可用。"
            records.append(_placeholder_record(path, "unsupported_optional_dependency", reason, file_type="excel"))
            warnings.append(f"{path}: {reason}")
            continue
        if suffix in {".docx", ".doc"}:
            reason = "Word 解析需要可选解析依赖；CSV 路径仍然可用。"
            records.append(_placeholder_record(path, "unsupported_optional_dependency", reason, file_type="word"))
            warnings.append(f"{path}: {reason}")
            continue

        record, file_interactions, file_unranked = _read_csv_file(path, data_root)
        records.append(record)
        warnings.extend(f"{path}: {warning}" for warning in record.warnings)
        interactions.extend(file_interactions)
        unranked_rows.extend(file_unranked)

    ranked = sorted(interactions, key=_ranking_key)[:top_n]
    return LianaSummary(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        input_dir=str(data_root),
        files=records,
        ranked_interactions=ranked,
        unranked_rows=unranked_rows,
        warnings=warnings,
        safety_decisions=gate.decision_log(),
    )


def process_data_dir(data_dir: str | Path, output_dir: str | Path, *, top_n: int = 50) -> dict[str, Path]:
    """处理数据，并且只在 output_dir 下写入 JSON/Markdown 产物。"""

    data_root = Path(data_dir).resolve()
    output_root = Path(output_dir).resolve()
    gate = SafetyGate(output_dir=output_root, data_dir=data_root)
    summary = summarize_liana_files(data_root, top_n=top_n, safety_gate=gate)

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "data_manifest": output_root / "data_manifest.json",
        "liana_interactions": output_root / "liana_interactions.json",
        "liana_summary": output_root / "liana_summary.md",
    }

    # 在序列化 manifest 前记录写入审批，确保溯源包含本次运行的每个派生输出副作用。
    for path in artifacts.values():
        _assert_write_allowed(path, gate)

    _write_json_unchecked(artifacts["data_manifest"], _manifest_payload(summary, gate))
    _write_json_unchecked(artifacts["liana_interactions"], _interactions_payload(summary))
    _write_text_unchecked(artifacts["liana_summary"], render_markdown_summary(summary))
    return artifacts


def render_markdown_summary(summary: LianaSummary) -> str:
    """渲染简洁的仅科研 LIANA 摘要。"""

    lines = [
        "# LIANA 数据处理摘要",
        "",
        "仅科研产物：本摘要只支持生物学科研解读，不是诊断或治疗建议。",
        "",
        f"生成时间: `{summary.generated_at}`",
        f"输入目录: `{summary.input_dir}`",
        "",
        "## 输入文件",
        "",
        "| 文件 | 类型 | 解析状态 | 行数 | 列数 | 警告 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in summary.files:
        lines.append(
            "| {path} | {file_type} | {status} | {rows} | {cols} | {warnings} |".format(
                path=record.path,
                file_type=record.file_type,
                status=record.parser_status.status,
                rows=record.row_count,
                cols=len(record.columns),
                warnings="<br>".join(record.warnings) if record.warnings else "",
            )
        )

    lines.extend([
        "",
        "## LIANA 互作排序 Top 列表",
        "",
        "排序：有效数值 `pvalue` 升序，其次有效数值 `lr.mean` 降序，再使用稳定的文件/source/target/ligand/receptor 规则打破并列。",
        "缺失或无效排序值的行会从 Top 列表排除，并计入未排序分桶。",
        "",
        "| 排名 | 来源细胞 | 目标细胞 | 配体 | 受体 | pvalue | lr.mean | 溯源 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ])
    for rank, interaction in enumerate(summary.ranked_interactions, start=1):
        prov = interaction.provenance
        lines.append(
            f"| {rank} | {interaction.source} | {interaction.target} | {interaction.ligand} | "
            f"{interaction.receptor} | {interaction.pvalue:.6g} | {interaction.lr_mean:.6g} | "
            f"{prov['file']} 第 {prov['csv_row_number']} 行 |"
        )

    source_target_counts = Counter((item.source, item.target) for item in summary.ranked_interactions)
    lines.extend(["", "## 已排序 source/target 配对计数", ""])
    if source_target_counts:
        for (source, target), count in source_target_counts.most_common():
            lines.append(f"- `{source}` → `{target}`: {count}")
    else:
        lines.append("- 未发现可排序的 LIANA 互作。")

    lines.extend(["", "## 警告", ""])
    if summary.warnings:
        lines.extend(f"- {warning}" for warning in summary.warnings)
    else:
        lines.append("- 无")
    lines.append(f"- 因排序值缺失/无效而未排序的行数: {len(summary.unranked_rows)}")
    lines.append("")
    return "\n".join(lines)


def _read_csv_file(path: Path, data_root: Path) -> tuple[DataFileRecord, list[LianaInteraction], list[dict[str, Any]]]:
    warnings: list[str] = []
    interactions: list[LianaInteraction] = []
    unranked: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    profile = _profile_rows(columns, rows)
    missing_required = [column for column in LIANA_REQUIRED_COLUMNS if column not in columns]
    if missing_required:
        warnings.append("缺少 LIANA 必需列: " + ", ".join(missing_required))
    is_liana = not missing_required

    if is_liana:
        for data_row_index, row in enumerate(rows):
            csv_row_number = data_row_index + 2  # 表头是第 1 行
            pvalue = _coerce_float(row.get("pvalue"))
            lr_mean = _coerce_float(row.get("lr.mean"))
            provenance = {
                "file": str(path),
                "relative_file": str(path.relative_to(data_root)),
                "row_index": data_row_index,
                "csv_row_number": csv_row_number,
                "ranking_columns": list(RANKING_COLUMNS),
            }
            if pvalue is None or lr_mean is None:
                unranked.append(
                    {
                        "reason": "missing_or_invalid_ranking_value",
                        "pvalue": row.get("pvalue"),
                        "lr.mean": row.get("lr.mean"),
                        "provenance": provenance,
                    }
                )
                continue
            interactions.append(
                LianaInteraction(
                    source=(row.get("source") or "").strip(),
                    target=(row.get("target") or "").strip(),
                    ligand=(row.get("ligand") or "").strip(),
                    receptor=(row.get("receptor") or "").strip(),
                    ligand_complex=(row.get("ligand.complex") or "").strip(),
                    receptor_complex=(row.get("receptor.complex") or "").strip(),
                    pvalue=pvalue,
                    lr_mean=lr_mean,
                    metrics={
                        column: value
                        for column in LIANA_OPTIONAL_NUMERIC_COLUMNS
                        if (value := _coerce_float(row.get(column))) is not None
                    },
                    provenance=provenance,
                )
            )
    record = DataFileRecord(
        path=str(path),
        file_type="csv",
        parser_status=ParserStatus("parsed_liana_csv" if is_liana else "parsed_csv", "csv", "检测到 LIANA schema" if is_liana else "已解析通用 CSV"),
        row_count=len(rows),
        columns=columns,
        schema_profile=profile,
        warnings=warnings,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )
    return record, interactions, unranked


def _profile_rows(columns: list[str], rows: list[dict[str, str]]) -> dict[str, ColumnProfile]:
    profiles: dict[str, ColumnProfile] = {}
    for column in columns:
        present = 0
        missing = 0
        numeric_valid = 0
        numeric_invalid = 0
        numeric_min: float | None = None
        numeric_max: float | None = None
        examples: list[str] = []
        for row in rows:
            raw = row.get(column, "")
            value = raw.strip() if raw is not None else ""
            if value == "":
                missing += 1
                continue
            present += 1
            if len(examples) < 3 and value not in examples:
                examples.append(value)
            number = _coerce_float(value)
            if number is None:
                numeric_invalid += 1
            else:
                numeric_valid += 1
                numeric_min = number if numeric_min is None else min(numeric_min, number)
                numeric_max = number if numeric_max is None else max(numeric_max, number)
        profiles[column] = ColumnProfile(
            name=column,
            present_count=present,
            missing_count=missing,
            numeric_valid_count=numeric_valid,
            numeric_invalid_count=numeric_invalid,
            numeric_min=numeric_min,
            numeric_max=numeric_max,
            examples=examples,
        )
    return profiles


def _placeholder_record(path: Path, status: str, reason: str, *, file_type: str | None = None) -> DataFileRecord:
    return DataFileRecord(
        path=str(path),
        file_type=file_type or path.suffix.lower().lstrip(".") or "unknown",
        parser_status=ParserStatus(status, file_type or path.suffix.lower().lstrip("."), reason),
        warnings=[reason] if reason else [],
        sha256=_sha256(path) if path.exists() else None,
        size_bytes=path.stat().st_size if path.exists() else None,
    )


def _ranking_key(interaction: LianaInteraction) -> tuple[float, float, str, str, str, str, str, int]:
    prov = interaction.provenance
    return (
        interaction.pvalue,
        -interaction.lr_mean,
        prov["relative_file"],
        interaction.source,
        interaction.target,
        interaction.ligand,
        interaction.receptor,
        int(prov["csv_row_number"]),
    )


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(summary: LianaSummary, gate: SafetyGate | None = None) -> dict[str, Any]:
    safety_decisions = gate.decision_log() if gate is not None else summary.safety_decisions
    return {
        "generated_at": summary.generated_at,
        "input_dir": summary.input_dir,
        "files": [_to_jsonable(record) for record in summary.files],
        "warnings": summary.warnings,
        "safety_decisions": safety_decisions,
        "research_only": True,
    }


def _interactions_payload(summary: LianaSummary) -> dict[str, Any]:
    return {
        "ranking": {
            "pvalue": "ascending",
            "lr.mean": "descending",
            "tie_breakers": ["relative_file", "source", "target", "ligand", "receptor", "csv_row_number"],
        },
        "ranked_interactions": [_to_jsonable(interaction) for interaction in summary.ranked_interactions],
        "unranked_rows": summary.unranked_rows,
        "warnings": summary.warnings,
    }


def _write_json_unchecked(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_unchecked(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _assert_write_allowed(path: Path, gate: SafetyGate) -> None:
    decision = gate.allow_write_derived_output(path)
    if decision.status != SafetyStatus.ALLOWED:
        raise PermissionError(decision.rationale)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def summarize_liana(records: list[Any], safety_gate: SafetyGate, *, top_n: int = 50) -> dict[str, Any]:
    """面向协调器级 LIANA 契约的兼容包装器。

    早期团队通道使用 ``discover_data_files(...)->records``，随后调用返回 dict 的
    ``summarize_liana``。数据通道后来引入了信息更丰富的 ``summarize_liana_files``
    dataclass API。保留该包装器可以避免重复解析代码，同时为旧版协调器/测试保留确定性的排序与溯源。
    """

    paths = [Path(record.path) for record in records if getattr(record, "file_type", "") == "csv"]
    data_root = safety_gate.data_dir or (paths[0].parent if paths else Path("data"))
    summary = summarize_liana_files(data_root, top_n=top_n, safety_gate=safety_gate)
    status_by_path = {Path(record.path).resolve(): record.parser_status.status for record in summary.files}
    for record in records:
        parsed_status = status_by_path.get(Path(record.path).resolve())
        if parsed_status and hasattr(record, "parser_status"):
            record.parser_status = parsed_status

    top_interactions = [_legacy_interaction_dict(item) for item in summary.ranked_interactions]
    return {
        "ranking_method": "pvalue 升序、lr.mean 降序，并使用稳定的 file/source/target/ligand/receptor 规则打破并列",
        "top_interactions": top_interactions,
        "unranked_interactions": [
            {
                "reason": "invalid pvalue or lr.mean",
                "pvalue": row.get("pvalue"),
                "lr.mean": row.get("lr.mean"),
                "provenance": row.get("provenance", {}),
            }
            for row in summary.unranked_rows
        ],
        "warnings": summary.warnings,
        "files": [_to_jsonable(record) for record in summary.files],
        "safety_decisions": summary.safety_decisions,
    }


def _legacy_interaction_dict(interaction: LianaInteraction) -> dict[str, Any]:
    provenance = dict(interaction.provenance)
    return {
        "source_file": provenance["file"],
        "row_index": provenance["csv_row_number"],
        "source_cell": interaction.source,
        "target_cell": interaction.target,
        "ligand": interaction.ligand,
        "receptor": interaction.receptor,
        "pvalue": interaction.pvalue,
        "lr_mean": interaction.lr_mean,
        "provenance": provenance,
    }


def main(argv: Iterable[str] | None = None) -> int:
    """用于生成数据通道产物的小型 CLI。"""

    import argparse

    parser = argparse.ArgumentParser(description="将 CSV/LIANA 数据处理为生成的科研产物。")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifacts = process_data_dir(args.data_dir, args.output_dir, top_n=args.top_n)
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
