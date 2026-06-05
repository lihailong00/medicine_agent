from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

from .data.discovery import discover_data_files
from .data.liana import summarize_liana, summarize_liana_files
from .literature.fulltext import retrieve_full_text_for_payloads
from .literature.providers import build_default_coordinator
from .literature.source_selector import decompose_question, select_sources
from .llm import require_deepseek_config, synthesize_review_with_llm
from .models import (
    EvidenceItem,
    OperationClass,
    PaperRecord,
    ResearchRequest,
    SourcePlan,
    SourceStatus,
    to_plain,
)
from .reporting.evidence import build_evidence, validate_evidence
from .reporting.markdown import render_report
from .reporting.synthesis import allowed_evidence_refs, build_rule_based_review_synthesis, evidence_from_synthesis
from .safety import SafetyGate
from .utils.io import write_json, write_text


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


DATA_INSPECTION_MARKERS = (
    "data/",
    "data\\",
    "data 目录",
    "data目录",
    "data directory",
    "data dir",
    "data folder",
    "数据目录",
    "数据文件",
    "本地数据",
    "我的数据",
    "当前目录下data",
    "当前目录 data",
    "读取data",
    "查看data",
    "分析data",
    "使用data",
    "看data",
    "csv",
    ".csv",
    "excel",
    ".xlsx",
    ".xls",
    "word",
    ".docx",
    ".doc",
    "liana",
)


def should_inspect_data_for_question(question: str) -> bool:
    """仅当 query 明确提到 data 目录或本地数据文件时才读取本地数据。"""

    normalized = " ".join(question.lower().split())
    compact = normalized.replace(" ", "")
    return any(marker in normalized or marker.replace(" ", "") in compact for marker in DATA_INSPECTION_MARKERS)


def run_research(request: ResearchRequest) -> dict[str, Any]:
    """运行默认联网的科研工作流，并写入生成产物。"""

    debug_steps: list[dict[str, str]] = []

    def step(message: str) -> None:
        entry = {"timestamp": utc_now(), "message": message}
        debug_steps.append(entry)
        if request.debug_steps:
            print(f"[medicine-agent] {message}", file=sys.stderr, flush=True)

    step("开始运行科研工作流")
    llm_config = require_deepseek_config()
    step(f"大模型配置检查通过：model={llm_config.model}")
    start = utc_now()
    output_dir = request.output_dir
    safety = SafetyGate(data_dir=request.data_dir, output_dir=output_dir, non_interactive=True)
    clinical_decision = safety.screen_question(request.question)
    if clinical_decision is None:
        step("安全筛查通过：未检测到临床诊断/治疗请求")
    else:
        step(f"安全筛查阻断：{clinical_decision.rationale}")

    # 目录创建之后的所有写入都受 SafetyGate 约束；任何输入路径都不会被覆盖。
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    step(f"已准备输出目录：{output_dir}")

    sources_override: tuple[str, ...] | None = None
    if request.include_preprints:
        sources_override = select_sources(request.question)
        if "arxiv" not in sources_override:
            sources_override = (*sources_override, "arxiv")
    planner_candidate = "DeepSeek LLM（必需）"
    step(f"开始查询分解：规划器候选={planner_candidate}")
    decomposition = decompose_question(
        request.question,
        sources=sources_override,
        allow_llm=True,
        require_llm=True,
        network_gate=safety,
    )
    sources = tuple(query.provider for query in decomposition.queries)
    source_plans = [
        SourcePlan(source=query.provider, query=query.query, rationale=query.rationale)
        for query in decomposition.queries
    ]
    step(
        f"已完成查询分解：{len(decomposition.subquestions)} 个子问题，"
        f"来源={', '.join(sources)}，规划器={decomposition.planner}"
    )

    if clinical_decision is None:
        step("开始文献检索")
        literature_payload = build_default_coordinator().search_question(
            request.question,
            allow_live=True,
            network_gate=safety,
            decomposition=decomposition,
        )
        paper_payloads = cast(list[Mapping[str, Any]], literature_payload["papers"])
        status_payloads = cast(list[Mapping[str, Any]], literature_payload["search_log"])
        paper_records = [_paper_from_provider_dict(item) for item in paper_payloads]
        source_statuses = [_status_from_provider_dict(item) for item in status_payloads]
        step(f"文献检索完成：{len(paper_records)} 篇记录，{len(source_statuses)} 条来源状态")
    else:
        step("因安全筛查阻断，跳过文献检索")
        literature_payload = {
            "decomposition": decomposition.to_dict(),
            "results": [],
            "papers": [],
            "search_log": [],
            "evidence_records": [],
        }
        paper_records = []
        source_statuses = [
            SourceStatus(
                provider=plan.source,
                endpoint_family="safety_policy",
                query=plan.query,
                status="skipped",
                timestamp=utc_now(),
                result_ids=[],
                reason=clinical_decision.rationale,
            )
            for plan in source_plans
        ]

    full_text_enabled = request.full_text and clinical_decision is None
    full_text_payload: dict[str, object] = {
        "requested": request.full_text,
        "enabled": full_text_enabled,
        "records": [],
        "statuses": [],
        "scope_counts": {},
        "reason": None if full_text_enabled else _full_text_disabled_reason(request, clinical_decision),
    }
    if full_text_enabled:
        step("开始全文/片段检索")
        retrieved = retrieve_full_text_for_payloads(
            paper_payloads,
            artifacts_dir=artifacts_dir / "full_text",
            safety_gate=safety,
            network_gate=safety,
        )
        full_text_payload.update(retrieved)
        retrieved_records = cast(list[Mapping[str, object]], full_text_payload["records"])
        step(f"全文/片段检索完成：{len(retrieved_records)} 条记录")
    else:
        step(f"跳过全文检索：{full_text_payload['reason']}")
    full_text_records = cast(list[Mapping[str, object]], full_text_payload["records"])

    data_requested = clinical_decision is None and should_inspect_data_for_question(request.question)
    if data_requested:
        step(f"检测到本地数据读取请求，开始扫描数据目录：{request.data_dir}")
        data_records = discover_data_files(request.data_dir, safety)
        liana_summary = summarize_liana(data_records, safety)
        rich_liana_summary: Any = summarize_liana_files(request.data_dir, safety_gate=safety)
        step(
            "数据处理完成："
            f"{len(data_records)} 个数据文件，{len(liana_summary.get('top_interactions', []))} 条 LIANA 排名互作"
        )
    else:
        reason = (
            "query 未明确要求查看 data 目录或本地数据文件；已跳过本地数据扫描。"
            if clinical_decision is None
            else "安全筛查已阻断请求；已跳过本地数据扫描。"
        )
        step(reason)
        data_records = []
        liana_summary = _empty_liana_summary(request, reason)
        rich_liana_summary = _empty_liana_detail_summary(request, reason)

    step("开始构建证据表")
    interactions_for_review = cast(list[Mapping[str, object]], liana_summary.get("top_interactions", []))
    evidence = build_evidence(
        request.question,
        paper_records,
        liana_summary.get("top_interactions", []),
        full_text_records if full_text_enabled else None,
        full_text_enabled=full_text_enabled,
        live_literature_enabled=clinical_decision is None,
    )
    if clinical_decision is not None:
        evidence.insert(
            0,
            EvidenceItem(
                claim="临床/诊断/治疗请求因超出范围而被拒绝，并被重述为仅科研问题。",
                status="out_of_scope_clinical",
                limitations=[clinical_decision.rationale],
            ),
        )
    validate_evidence(evidence)
    step(f"基础证据表构建完成：{len(evidence)} 条证据项")

    review_allowed_refs = allowed_evidence_refs(paper_records, interactions_for_review)
    review_synthesis: dict[str, object]
    if clinical_decision is None:
        step("开始 LLM 证据抽取与结构化综述生成")
        llm_synthesis = synthesize_review_with_llm(
            request.question,
            papers=paper_records,
            evidence=evidence,
            interactions=interactions_for_review,
            full_text_records=full_text_records if full_text_enabled else [],
            allowed_refs=review_allowed_refs,
            network_gate=safety,
        )
        if llm_synthesis is None:
            raise RuntimeError(
                "LLM 结构化综述失败：请检查 DEEPSEEK_API_KEY/MEDICINE_AGENT_DEEPSEEK_API_KEY、"
                "DEEPSEEK_MODEL、额度与网络连通性。若使用 deepseek-v4-pro，请确认 "
                "DEEPSEEK_REVIEW_MAX_TOKENS 与 DEEPSEEK_TIMEOUT_SECONDS 足够大。"
            )
        review_synthesis = llm_synthesis
        evidence = evidence_from_synthesis(review_synthesis, evidence, allowed_refs=review_allowed_refs)
        validate_evidence(evidence)
        step(f"LLM 结构化综述完成：证据表替换为 {len(evidence)} 条 LLM 清洗证据项")
    else:
        review_synthesis = build_rule_based_review_synthesis(
            request.question,
            paper_records,
            evidence,
            source="safety_blocked",
            reason="安全筛查阻断了文献与 LLM 综述生成。",
        )
    step(f"证据表构建完成：{len(evidence)} 条证据项；综述来源={review_synthesis.get('source')}")

    top_csv = artifacts_dir / "liana_top_interactions.csv"
    liana_json = artifacts_dir / "liana_summary.json"
    literature_json = artifacts_dir / "literature_results.json"
    full_text_json = artifacts_dir / "full_text_results.json"
    review_json = artifacts_dir / "review_synthesis.json"
    search_log = output_dir / "search_log.json"
    artifact_manifest = output_dir / "artifact_manifest.json"
    manifest_path = output_dir / "run_manifest.json"
    report_path = output_dir / "report.md"

    step("开始写入 JSON、CSV 与报告产物")
    _write_top_interactions_csv(top_csv, liana_summary.get("top_interactions", []), safety)
    write_json(liana_json, _to_jsonable_liana_summary(rich_liana_summary), safety)
    write_json(literature_json, literature_payload, safety)
    write_json(full_text_json, full_text_payload, safety)
    write_json(review_json, review_synthesis, safety)
    write_json(search_log, [to_plain(status) for status in source_statuses], safety)

    artifact_paths = [
        str(top_csv),
        str(liana_json),
        str(literature_json),
        str(full_text_json),
        str(review_json),
        str(search_log),
        str(artifact_manifest),
        str(report_path),
        str(manifest_path),
    ]
    write_json(artifact_manifest, {"artifacts": artifact_paths, "generated_at": utc_now()}, safety)

    report = render_report(
        request.question,
        list(decomposition.subquestions),
        source_plans,
        source_statuses,
        paper_records,
        data_records,
        liana_summary,
        evidence,
        artifact_paths,
        str(manifest_path),
        full_text_records if full_text_enabled else None,
        full_text_payload,
        review_synthesis,
    )
    write_text(report_path, report, safety)

    manifest = {
        "started_at": start,
        "completed_at": utc_now(),
        "request": {
            "question": request.question,
            "data_dir": str(request.data_dir),
            "output_dir": str(request.output_dir),
            "offline": request.offline,
            "live_api": request.live_api,
            "include_preprints": request.include_preprints,
            "full_text": request.full_text,
            "debug_steps": request.debug_steps,
        },
        "subquestions": list(decomposition.subquestions),
        "query_decomposition": decomposition.to_dict(),
        "source_plans": [asdict(plan) for plan in source_plans],
        "source_statuses": [to_plain(status) for status in source_statuses],
        "literature_evidence_records": literature_payload["evidence_records"],
        "full_text_results": full_text_payload,
        "review_synthesis": review_synthesis,
        "data_files": [asdict(record) for record in data_records],
        "safety_decisions": [decision.to_dict() for decision in safety.decisions],
        "evidence": [item.to_dict() for item in evidence],
        "warnings": sorted({warning for record in data_records for warning in record.warnings} | set(liana_summary.get("warnings", []))),
        "artifacts": artifact_paths,
        "debug_steps": debug_steps,
        "research_only": True,
    }
    step("写入运行清单并完成工作流")
    manifest["debug_steps"] = debug_steps
    write_json(manifest_path, manifest, safety)
    return {
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "artifact_manifest_path": str(artifact_manifest),
        "search_log_path": str(search_log),
        "full_text_results_path": str(full_text_json),
        "review_synthesis_path": str(review_json),
        "output_dir": str(output_dir),
    }


def _write_top_interactions_csv(path: Path, rows: list[dict[str, Any]], safety: SafetyGate) -> None:
    safety.assert_allowed(OperationClass.WRITE_DERIVED_OUTPUT, path, "写入派生 LIANA Top 互作表")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "row_index", "source_cell", "target_cell", "ligand", "receptor", "pvalue", "lr_mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _paper_from_provider_dict(item: Mapping[str, Any]) -> PaperRecord:
    data = dict(item)
    return PaperRecord(
        title=str(data.get("title") or ""),
        abstract=str(data.get("abstract") or ""),
        source=str(data.get("provider") or data.get("source") or "unknown"),
        year=data.get("year") if isinstance(data.get("year"), int) else None,
        authors=list(data.get("authors") or []),
        pmid=data.get("pmid") if isinstance(data.get("pmid"), str) else None,
        pmcid=data.get("pmcid") if isinstance(data.get("pmcid"), str) else None,
        doi=data.get("doi") if isinstance(data.get("doi"), str) else None,
        arxiv_id=data.get("arxiv_id") if isinstance(data.get("arxiv_id"), str) else None,
        semantic_scholar_id=data.get("semantic_scholar_id") if isinstance(data.get("semantic_scholar_id"), str) else None,
        venue=data.get("venue") if isinstance(data.get("venue"), str) else None,
        citation_count=data.get("citation_count") if isinstance(data.get("citation_count"), int) else None,
        source_url=data.get("source_url") if isinstance(data.get("source_url"), str) else None,
        open_access_url=data.get("open_access_url") if isinstance(data.get("open_access_url"), str) else None,
    )


def _status_from_provider_dict(item: Mapping[str, Any]) -> SourceStatus:
    data = dict(item)
    result_ids = data.get("result_ids")
    return SourceStatus(
        provider=str(data.get("provider") or "unknown"),
        endpoint_family=str(data.get("endpoint_family") or "unknown"),
        query=str(data.get("query") or ""),
        status=str(data.get("status") or "failed"),  # type: ignore[arg-type]
        timestamp=str(data.get("timestamp") or utc_now()),
        result_ids=list(result_ids) if isinstance(result_ids, list) else [],
        error_class=data.get("error_class") if isinstance(data.get("error_class"), str) else None,
        retry_backoff=str(data.get("retry_backoff_seconds")) if data.get("retry_backoff_seconds") is not None else None,
        reason=data.get("reason") if isinstance(data.get("reason"), str) else None,
    )


def _to_jsonable_liana_summary(summary: Any) -> dict[str, Any]:
    if isinstance(summary, Mapping):
        return dict(summary)
    return {
        "generated_at": summary.generated_at,
        "input_dir": summary.input_dir,
        "files": [to_plain(record) for record in summary.files],
        "ranked_interactions": [to_plain(item) for item in summary.ranked_interactions],
        "unranked_rows": summary.unranked_rows,
        "warnings": summary.warnings,
        "safety_decisions": summary.safety_decisions,
    }


def _empty_liana_summary(request: ResearchRequest, reason: str) -> dict[str, Any]:
    return {
        "ranking_method": "未请求本地数据分析；跳过 LIANA 排序。",
        "top_interactions": [],
        "unranked_interactions": [],
        "warnings": [reason],
        "files": [],
        "safety_decisions": [],
        "data_access_requested": False,
        "input_dir": str(request.data_dir),
    }


def _empty_liana_detail_summary(request: ResearchRequest, reason: str) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "input_dir": str(request.data_dir),
        "files": [],
        "ranked_interactions": [],
        "unranked_rows": [],
        "warnings": [reason],
        "safety_decisions": [],
        "data_access_requested": False,
    }


def _full_text_disabled_reason(request: ResearchRequest, clinical_decision: object | None) -> str:
    if clinical_decision is not None:
        return "临床安全筛查已跳过文献与全文检索。"
    if not request.full_text:
        return "未请求全文检索。"
    return "全文检索已禁用。"
