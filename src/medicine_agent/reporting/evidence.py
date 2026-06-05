from __future__ import annotations

from typing import Mapping

from medicine_agent.models import EvidenceItem, PaperRecord


def build_evidence(
    question: str,
    papers: list[PaperRecord],
    interactions: list[dict],
    full_text_records: list[Mapping[str, object]] | None = None,
    full_text_enabled: bool = False,
    live_literature_enabled: bool = False,
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    if interactions:
        top = interactions[0]
        ref = f"{top['source_file']}#row-{top['row_index']}"
        evidence.append(EvidenceItem(
            claim=(
                f"给定数据中排序最靠前的 LIANA 互作是 "
                f"{top['source_cell']} 到 {top['target_cell']} 的 {top['ligand']} -> {top['receptor']}。"
            ),
            status="data_supported",
            evidence_refs=[ref],
            confidence="medium",
            limitations=["排序仅提供统计/溯源证据，不能证明机制因果关系。"],
        ))
    if papers:
        if full_text_enabled:
            scopes = _evidence_scopes(full_text_records or [])
            claim = "实时文献检索生成了可追踪记录，并标明了证据范围边界。"
            confidence = "medium" if "full_text_xml" in scopes else "low"
            limitations = [
                "证据范围逐条记录标注：摘要元数据、Semantic Scholar 片段、PMC XML 文本或 arXiv PDF 产物字节。",
                "arXiv PDF 产物会保存用于审计，但不会在无额外依赖的前提下声明已解析为全文。",
            ]
            if not scopes:
                limitations.append("没有获批全文路径检索成功；综合应仅依赖元数据/摘要证据。")
        elif live_literature_enabled:
            claim = "实时文献元数据/摘要检索从获批学术 API 生成了可追踪记录。"
            confidence = "medium"
            limitations = [
                "实时元数据/摘要记录不是全文证据；本次可能显式关闭了默认全文检索，或全文路径不可用。",
            ]
        else:
            claim = "文献记录缺少实时检索标记；综合时必须回查检索日志确认来源范围。"
            confidence = "low"
            limitations = ["当前项目只支持联网检索；若出现该状态，请检查调用方是否误传 live_literature_enabled=False。"]
        evidence.append(EvidenceItem(
            claim=claim,
            status="literature_supported",
            evidence_refs=[p.evidence_id for p in papers[:5]],
            confidence=confidence,
            limitations=limitations,
        ))
    evidence.append(EvidenceItem(
        claim=f"关于“{question}”的机制解读在获得直接文献与实验验证前仍属于可检验假设。",
        status="hypothesis",
        evidence_refs=[],
        confidence="low",
        limitations=["仅为科研规划生成；不是诊断或治疗建议。"],
    ))
    return evidence


def validate_evidence(items: list[EvidenceItem]) -> None:
    for item in items:
        EvidenceItem(item.claim, item.status, list(item.evidence_refs), item.confidence, list(item.limitations))


def _evidence_scopes(records: list[Mapping[str, object]]) -> set[str]:
    scopes: set[str] = set()
    for record in records:
        status = record.get("status")
        if not isinstance(status, Mapping):
            continue
        if isinstance(status.get("scope"), str):
            scopes.add(str(status["scope"]))
    return scopes
