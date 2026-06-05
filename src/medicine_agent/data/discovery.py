from __future__ import annotations

from pathlib import Path

from medicine_agent.models import DataFileRecord, OperationClass
from medicine_agent.safety import SafetyGate
from medicine_agent.utils.io import sha256_file

SUPPORTED_SUFFIXES = {".csv": "csv", ".xlsx": "excel", ".xls": "excel", ".docx": "word", ".doc": "word"}


def discover_data_files(data_dir: Path, safety: SafetyGate) -> list[DataFileRecord]:
    records: list[DataFileRecord] = []
    if not data_dir.exists():
        return [DataFileRecord(path=str(data_dir), file_type="missing", parser_status="missing", warnings=["数据目录不存在"])]

    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        safety.assert_allowed(OperationClass.READ_FILE, path, "检查输入数据文件元数据")
        suffix = path.suffix.lower()
        file_type = SUPPORTED_SUFFIXES.get(suffix, "unsupported")
        status = "pending" if file_type == "csv" else "unsupported_dependency_gate" if file_type in {"excel", "word"} else "unsupported"
        warnings = [] if file_type == "csv" else [f"{file_type} 解析需要获批的可选解析依赖"]
        records.append(
            DataFileRecord(
                path=str(path),
                file_type=file_type,
                parser_status=status,
                warnings=warnings,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return records
