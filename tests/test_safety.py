from pathlib import Path

from medicine_agent.models import OperationClass, SafetyDecisionStatus
from medicine_agent.safety import SafetyGate


def test_safety_gate_allows_only_derived_output_writes(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    allowed = gate.decide(OperationClass.WRITE_DERIVED_OUTPUT, tmp_path / "out" / "report.md", "derived")
    blocked = gate.decide(OperationClass.WRITE_DERIVED_OUTPUT, tmp_path / "elsewhere.md", "outside")
    assert allowed.status == SafetyDecisionStatus.ALLOWED
    assert blocked.status == SafetyDecisionStatus.BLOCKED


def test_safety_gate_requires_confirmation_for_high_risk_ops(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    for op in [OperationClass.INSTALL_DEP, OperationClass.USE_API_KEY, OperationClass.OVERWRITE_INPUT, OperationClass.LONG_JOB, OperationClass.RUN_SCRIPT, OperationClass.NETWORK_CALL]:
        assert gate.decide(op, "target", "risk").status == SafetyDecisionStatus.NEEDS_CONFIRMATION


def test_clinical_question_is_blocked_as_out_of_scope(tmp_path):
    gate = SafetyGate(Path("data"), tmp_path / "out")
    decision = gate.screen_question("How should I treat this patient?")
    assert decision is not None
    assert decision.status == SafetyDecisionStatus.BLOCKED
    assert "out of scope" in decision.rationale
