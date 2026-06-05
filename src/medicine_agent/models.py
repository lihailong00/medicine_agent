from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class SafetyDecisionStatus(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    NEEDS_CONFIRMATION = "needs_confirmation"


class OperationClass(str, Enum):
    READ_FILE = "READ_FILE"
    WRITE_DERIVED_OUTPUT = "WRITE_DERIVED_OUTPUT"
    NETWORK_CALL = "NETWORK_CALL"
    RUN_SCRIPT = "RUN_SCRIPT"
    INSTALL_DEP = "INSTALL_DEP"
    USE_API_KEY = "USE_API_KEY"
    OVERWRITE_INPUT = "OVERWRITE_INPUT"
    LONG_JOB = "LONG_JOB"


ClaimStatus = Literal[
    "literature_supported",
    "data_supported",
    "literature_and_data_supported",
    "conflicting",
    "hypothesis",
    "out_of_scope_clinical",
]

SourceState = Literal["attempted", "succeeded", "skipped", "failed", "rate_limited", "needs_confirmation"]


@dataclass(frozen=True)
class SafetyDecision:
    operation: OperationClass
    target: str
    status: SafetyDecisionStatus
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "operation": self.operation.value,
            "target": self.target,
            "status": self.status.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ResearchRequest:
    question: str
    data_dir: Path = Path("data")
    output_dir: Path = Path("generated/medicine_agent")
    offline: bool = True
    include_preprints: bool = False
    live_api: bool = False
    full_text: bool = False

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ValueError("question is required")
        if self.full_text and (self.offline or not self.live_api):
            raise ValueError("full_text requires live_api=True and offline=False")
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass
class SourcePlan:
    source: str
    query: str
    rationale: str
    requires_live: bool = False
    requires_api_key: bool = False


@dataclass
class SourceStatus:
    provider: str
    endpoint_family: str
    query: str
    status: SourceState
    timestamp: str
    result_ids: list[str] = field(default_factory=list)
    error_class: str | None = None
    retry_backoff: str | None = None
    reason: str | None = None


@dataclass
class PaperRecord:
    title: str
    abstract: str
    source: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    source_url: str | None = None
    open_access_url: str | None = None

    @property
    def evidence_id(self) -> str:
        return self.pmid or self.pmcid or self.doi or self.arxiv_id or self.semantic_scholar_id or self.title


@dataclass
class DataFileRecord:
    path: str
    file_type: str
    parser_status: str
    schema_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass
class AnalysisArtifact:
    path: str
    method: str
    provenance: dict[str, Any]


@dataclass
class LianaInteraction:
    source_file: str
    row_index: int
    source_cell: str
    target_cell: str
    ligand: str
    receptor: str
    pvalue: float
    lr_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    claim: str
    status: ClaimStatus
    evidence_refs: list[str] = field(default_factory=list)
    confidence: str = "medium"
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("EvidenceItem claim is required")
        if self.status not in {"hypothesis", "out_of_scope_clinical"} and not self.evidence_refs:
            raise ValueError("supported/conflicting claims require evidence references")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value
