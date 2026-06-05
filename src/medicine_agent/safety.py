"""面向仅科研、非破坏性生信工作流的 SafetyGate。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medicine_agent.models import OperationClass, SafetyDecision, SafetyDecisionStatus
from medicine_agent.network_policy import ALLOWED_LIVE_HOSTS as NETWORK_ALLOWED_LIVE_HOSTS
from medicine_agent.network_policy import classify_allowed_url

# LIANA 通道测试与产物使用的兼容别名。
SafetyStatus = SafetyDecisionStatus
ALLOWED_LIVE_HOSTS = NETWORK_ALLOWED_LIVE_HOSTS


class SafetyGate:
    """读取、派生输出写入与高风险动作的集中检查点。

    构造函数同时兼容团队通道历史上使用过的两种调用方式：
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
        raw_target = str(target)
        target_path = (
            Path(target)
            if not _looks_like_url(raw_target) and (isinstance(target, Path) or _looks_like_path(raw_target))
            else None
        )
        target_text = str(target_path.resolve()) if target_path is not None else raw_target

        if op == OperationClass.READ_FILE:
            status = (
                SafetyDecisionStatus.ALLOWED
                if self.data_dir is None or (target_path is not None and _is_relative_to(target_path.resolve(), self.data_dir))
                else SafetyDecisionStatus.NEEDS_CONFIRMATION
            )
            decision_rationale = rationale if status == SafetyDecisionStatus.ALLOWED else "读取目标位于配置的数据目录之外"
        elif op == OperationClass.WRITE_DERIVED_OUTPUT:
            status = (
                SafetyDecisionStatus.ALLOWED
                if target_path is not None and _is_relative_to(target_path.resolve(), self.output_dir)
                else SafetyDecisionStatus.BLOCKED
            )
            decision_rationale = (
                rationale
                if status == SafetyDecisionStatus.ALLOWED
                else "派生产物写入必须位于生成输出目录内"
            )
        elif op == OperationClass.NETWORK_CALL:
            url_decision = classify_allowed_url(str(target))
            status = SafetyDecisionStatus.ALLOWED if url_decision.allowed else SafetyDecisionStatus.BLOCKED
            decision_rationale = (
                rationale
                if status == SafetyDecisionStatus.ALLOWED
                else url_decision.reason
            )
        elif op == OperationClass.OVERWRITE_INPUT:
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = "覆盖输入数据需要显式确认"
        elif op in {
            OperationClass.RUN_SCRIPT,
            OperationClass.INSTALL_DEP,
            OperationClass.USE_API_KEY,
            OperationClass.LONG_JOB,
        }:
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = f"{op.value} 在仅科研的非交互模式下需要显式确认"
        else:  # pragma: no cover - 为未来枚举扩展保留的防御分支。
            status = SafetyDecisionStatus.NEEDS_CONFIRMATION
            decision_rationale = "未知操作需要显式确认"

        decision = SafetyDecision(operation=op, target=target_text, status=status, rationale=decision_rationale)
        self.decisions.append(decision)
        return decision

    def assert_allowed(self, operation: OperationClass | str, target: str | Path, rationale: str) -> SafetyDecision:
        decision = self.decide(operation, target, rationale)
        if decision.status != SafetyDecisionStatus.ALLOWED:
            raise PermissionError(decision.rationale)
        return decision

    def allow_read_file(self, path: str | Path) -> SafetyDecision:
        return self.decide(OperationClass.READ_FILE, path, "允许只读检查")

    def allow_write_derived_output(self, path: str | Path) -> SafetyDecision:
        return self.decide(OperationClass.WRITE_DERIVED_OUTPUT, path, "写入派生输出产物")

    def allow_network_call(self, url: str) -> SafetyDecision:
        return self.decide(OperationClass.NETWORK_CALL, url, "向 allowlist 中的提供器发起实时文献 API 请求")

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
        """拒绝临床照护问题，同时允许科研表述。"""

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
                    "临床决策支持超出范围；请改写为仅科研的机制、文献或探索性数据分析问题"
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


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))
