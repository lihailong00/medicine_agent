"""SafetyGate for research-only, non-destructive bioinformatics workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medicine_agent.models import OperationClass, SafetyDecision, SafetyDecisionStatus

# Compatibility alias used by the LIANA lane tests and artifacts.
SafetyStatus = SafetyDecisionStatus


class SafetyGate:
    """Central chokepoint for reads, generated-output writes, and risky actions.

    The constructor accepts both historical call styles used by the team lanes:
    ``SafetyGate(data_dir, output_dir)`` and
    ``SafetyGate(output_dir=..., data_dir=...)``.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        *,
        non_interactive: bool = True,
    ) -> None:
        if output_dir is None:
            output_dir = Path("generated/medicine_agent")
        self.data_dir = Path(data_dir).resolve() if data_dir is not None else None
        self.output_dir = Path(output_dir).resolve()
        self.non_interactive = non_interactive
        self.decisions: list[SafetyDecision] = []

    def decide(self, operation: OperationClass | str, target: str | Path, rationale: str) -> SafetyDecision:
        op = _coerce_operation(operation)
        target_path = Path(target) if isinstance(target, Path) or _looks_like_path(str(target)) else None
        target_text = str(target_path.resolve()) if target_path is not None else str(target)

        if op == OperationClass.READ_FILE:
            status = (
                SafetyDecisionStatus.ALLOWED
                if self.data_dir is None or (target_path is not None and _is_relative_to(target_path.resolve(), self.data_dir))
                else SafetyDecisionStatus.NEEDS_CONFIRMATION
            )
            decision_rationale = rationale if status == SafetyDecisionStatus.ALLOWED else "read target is outside configured data directory"
        elif op == OperationClass.WRITE_DERIVED_OUTPUT:
            status = (
                SafetyDecisionStatus.ALLOWED
                if target_path is not None and _is_relative_to(target_path.resolve(), self.output_dir)
                else SafetyDecisionStatus.BLOCKED
            )
            decision_rationale = (
                rationale
                if status == SafetyDecisionStatus.ALLOWED
                else "derived artifact writes must stay inside the generated output directory"
            )
        elif op == OperationClass.OVERWRITE_INPUT:
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = "overwriting input data requires explicit confirmation"
        elif op in {
            OperationClass.NETWORK_CALL,
            OperationClass.RUN_SCRIPT,
            OperationClass.INSTALL_DEP,
            OperationClass.USE_API_KEY,
            OperationClass.LONG_JOB,
        }:
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = f"{op.value} requires explicit confirmation in research-only non-interactive mode"
        else:  # pragma: no cover - defensive for future enum expansion.
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = "unknown operation requires explicit confirmation"

        decision = SafetyDecision(operation=op, target=target_text, status=status, rationale=decision_rationale)
        self.decisions.append(decision)
        return decision

    def assert_allowed(self, operation: OperationClass | str, target: str | Path, rationale: str) -> SafetyDecision:
        decision = self.decide(operation, target, rationale)
        if decision.status != SafetyDecisionStatus.ALLOWED:
            raise PermissionError(decision.rationale)
        return decision

    def allow_read_file(self, path: str | Path) -> SafetyDecision:
        return self.decide(OperationClass.READ_FILE, path, "read-only inspection is allowed")

    def allow_write_derived_output(self, path: str | Path) -> SafetyDecision:
        return self.decide(OperationClass.WRITE_DERIVED_OUTPUT, path, "write derived output artifact")

    def needs_confirmation(self, operation: OperationClass | str, target: str | Path, rationale: str) -> SafetyDecision:
        op = _coerce_operation(operation)
        decision = SafetyDecision(
            operation=op,
            target=str(target),
            status=SafetyDecisionStatus.NEEDS_CONFIRMATION,
            rationale=rationale,
        )
        self.decisions.append(decision)
        return decision

    def screen_question(self, question: str) -> SafetyDecision | None:
        """Reject clinical-care questions while allowing research framing."""

        normalized = " ".join(question.lower().split())
        clinical_markers = {
            "treat this patient",
            "diagnose",
            "diagnosis",
            "prescribe",
            "dosage",
            "patient management",
            "what medication",
            "how should i treat",
            "therapy for my",
        }
        if any(marker in normalized for marker in clinical_markers):
            decision = SafetyDecision(
                operation=OperationClass.RUN_SCRIPT,
                target=question,
                status=SafetyDecisionStatus.BLOCKED,
                rationale=(
                    "clinical decision support is out of scope; reframe as research-only "
                    "mechanism, literature, or exploratory data analysis"
                ),
            )
            self.decisions.append(decision)
            return decision
        return None

    def decision_log(self) -> list[dict[str, str]]:
        return [decision.to_dict() for decision in self.decisions]


def _coerce_operation(operation: OperationClass | str | Any) -> OperationClass:
    if isinstance(operation, OperationClass):
        return operation
    value = getattr(operation, "value", operation)
    return OperationClass(str(value))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith(".") or bool(Path(value).suffix)
