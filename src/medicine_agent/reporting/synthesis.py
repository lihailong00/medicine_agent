"""结构化综述与证据表的确定性契约。"""

from __future__ import annotations

from typing import Mapping, Sequence

from medicine_agent.models import EvidenceItem, PaperRecord

SYNTHESIS_SCHEMA_VERSION = "review_synthesis.v1"
ALLOWED_CLAIM_STATUSES = {
    "literature_supported",
    "data_supported",
    "literature_and_data_supported",
    "conflicting",
    "hypothesis",
    "out_of_scope_clinical",
}
SUPPORTED_STATUSES_REQUIRE_REFS = {
    "literature_supported",
    "data_supported",
    "literature_and_data_supported",
    "conflicting",
}


def build_rule_based_review_synthesis(
    question: str,
    papers: Sequence[PaperRecord],
    evidence: Sequence[EvidenceItem],
    *,
    source: str = "rule_based",
    reason: str = "未启用 LLM 综述生成或 LLM 失败；使用确定性模板。",
) -> dict[str, object]:
    """生成和 LLM 输出同构的规则降级综述。"""

    paper_refs = [paper.evidence_id for paper in papers[:5]]
    key_findings: list[dict[str, object]] = []
    if papers:
        key_findings.append(
            {
                "claim": f"已围绕“{question}”检索到 {len(papers)} 条可追踪文献记录。",
                "status": "literature_supported",
                "evidence_refs": paper_refs,
                "confidence": "medium" if source.startswith("llm") else "low",
                "limitations": ["规则模板只说明检索覆盖情况，不替代人工或 LLM 语义综合。"],
            }
        )
    key_findings.extend(item.to_dict() for item in evidence if item.status != "hypothesis")
    hypotheses = [item.claim for item in evidence if item.status == "hypothesis"] or [
        f"关于“{question}”的机制解释仍需直接文献与实验验证。"
    ]
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "source": source,
        "reason": reason,
        "executive_summary": _fallback_executive_summary(question, papers),
        "key_findings": key_findings[:6],
        "evidence_table": [item.to_dict() for item in evidence],
        "mechanism_review": [
            "规则降级模式不会从摘要中抽取复杂机制链；请查看证据表、文献记录与原始检索日志。",
        ],
        "hypotheses": hypotheses[:6],
        "limitations_conflicts": [
            "仅科研用途，不构成临床建议。",
            "没有 LLM 综述时，报告只做保守模板化综合。",
            "实时检索结果受各 API 可用性、限流和返回排序影响。",
        ],
        "reproducibility_notes": [
            "完整 query、来源状态、证据引用与产物路径写入 run_manifest.json。",
            "支持性引用必须来自本次检索或本地数据溯源，不能引用未检索到的论文。",
        ],
    }


def evidence_from_synthesis(
    synthesis: Mapping[str, object],
    fallback_evidence: Sequence[EvidenceItem],
    *,
    allowed_refs: set[str],
) -> list[EvidenceItem]:
    """从结构化综述中抽取可验证 EvidenceItem；失败时返回确定性证据。"""

    items = synthesis.get("evidence_table")
    if not isinstance(items, list):
        return list(fallback_evidence)
    extracted: list[EvidenceItem] = []
    for item in items[:12]:
        if not isinstance(item, Mapping):
            continue
        claim = _compact_text(item.get("claim"), max_chars=700)
        if not claim:
            continue
        status = _sanitize_status(item.get("status"))
        refs = _sanitize_refs(item.get("evidence_refs"), allowed_refs=allowed_refs)
        if status in SUPPORTED_STATUSES_REQUIRE_REFS and not refs:
            status = "hypothesis"
        confidence = _sanitize_confidence(item.get("confidence"))
        limitations = _sanitize_text_list(item.get("limitations"), max_items=5, max_chars=260)
        if not limitations and status == "hypothesis":
            limitations = ["LLM 未提供可验证引用，已降级为假设。"]
        try:
            extracted.append(
                EvidenceItem(
                    claim=claim,
                    status=status,  # type: ignore[arg-type]
                    evidence_refs=refs,
                    confidence=confidence,
                    limitations=limitations,
                )
            )
        except ValueError:
            continue
    return extracted or list(fallback_evidence)


def allowed_evidence_refs(papers: Sequence[PaperRecord], interactions: Sequence[Mapping[str, object]]) -> set[str]:
    """返回本次运行允许 LLM 引用的证据 ID 集合。"""

    refs = {paper.evidence_id for paper in papers if paper.evidence_id}
    for item in interactions:
        source_file = item.get("source_file")
        row_index = item.get("row_index")
        if source_file is not None and row_index is not None:
            refs.add(f"{source_file}#row-{row_index}")
    return refs


def sanitize_review_synthesis(raw: Mapping[str, object], *, allowed_refs: set[str]) -> dict[str, object] | None:
    """把 LLM JSON 清洗为稳定、可渲染、只含允许引用的结构化综述。"""

    evidence_table = _sanitize_evidence_rows(raw.get("evidence_table"), allowed_refs=allowed_refs)
    key_findings = _sanitize_evidence_rows(raw.get("key_findings"), allowed_refs=allowed_refs)
    executive_summary = _compact_text(raw.get("executive_summary"), max_chars=1200)
    mechanism_review = _sanitize_text_list(raw.get("mechanism_review"), max_items=8, max_chars=700)
    hypotheses = _sanitize_text_list(raw.get("hypotheses"), max_items=8, max_chars=500)
    limitations_conflicts = _sanitize_text_list(raw.get("limitations_conflicts"), max_items=10, max_chars=500)
    reproducibility_notes = _sanitize_text_list(raw.get("reproducibility_notes"), max_items=8, max_chars=500)
    if not any([executive_summary, evidence_table, key_findings, mechanism_review, hypotheses]):
        return None
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "source": _compact_text(raw.get("source"), max_chars=80) or "llm_deepseek",
        "reason": _compact_text(raw.get("reason"), max_chars=500) or "DeepSeek 生成结构化综述；已按本次运行允许的引用 ID 过滤。",
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "evidence_table": evidence_table,
        "mechanism_review": mechanism_review,
        "hypotheses": hypotheses,
        "limitations_conflicts": limitations_conflicts or ["LLM 综述仅限科研用途，不构成临床建议。"],
        "reproducibility_notes": reproducibility_notes or ["所有引用 ID 均来自本次运行的 manifest。"],
    }


def _fallback_executive_summary(question: str, papers: Sequence[PaperRecord]) -> str:
    if papers:
        return f"本次运行围绕“{question}”生成了 {len(papers)} 条文献记录，并保留可审计引用。"
    return f"本次运行围绕“{question}”未获得可用文献记录；报告只保留方法、限制与假设。"


def _sanitize_evidence_rows(value: object, *, allowed_refs: set[str]) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value[:12]:
        if not isinstance(item, Mapping):
            continue
        claim = _compact_text(item.get("claim"), max_chars=700)
        if not claim:
            continue
        status = _sanitize_status(item.get("status"))
        refs = _sanitize_refs(item.get("evidence_refs"), allowed_refs=allowed_refs)
        if status in SUPPORTED_STATUSES_REQUIRE_REFS and not refs:
            status = "hypothesis"
        rows.append(
            {
                "claim": claim,
                "status": status,
                "evidence_refs": refs,
                "confidence": _sanitize_confidence(item.get("confidence")),
                "limitations": _sanitize_text_list(item.get("limitations"), max_items=5, max_chars=260),
            }
        )
    return rows


def _sanitize_status(value: object) -> str:
    status = str(value) if isinstance(value, str) else "hypothesis"
    return status if status in ALLOWED_CLAIM_STATUSES else "hypothesis"


def _sanitize_confidence(value: object) -> str:
    confidence = str(value).lower() if isinstance(value, str) else "medium"
    return confidence if confidence in {"high", "medium", "low"} else "medium"


def _sanitize_refs(value: object, *, allowed_refs: set[str]) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    refs: list[str] = []
    for item in candidates:
        ref = item.strip()
        if ref in allowed_refs and ref not in refs:
            refs.append(ref)
    return refs[:8]


def _sanitize_text_list(value: object, *, max_items: int, max_chars: int) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    cleaned: list[str] = []
    for item in candidates:
        text = _compact_text(item, max_chars=max_chars)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:max_items]


def _compact_text(value: object, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_chars]
