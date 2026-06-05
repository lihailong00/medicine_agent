from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medicine_agent.models import OperationClass
from medicine_agent.safety import SafetyGate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any, safety: SafetyGate) -> None:
    safety.assert_allowed(OperationClass.WRITE_DERIVED_OUTPUT, path, "写入派生 JSON 产物")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str, safety: SafetyGate) -> None:
    safety.assert_allowed(OperationClass.WRITE_DERIVED_OUTPUT, path, "写入派生文本产物")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
