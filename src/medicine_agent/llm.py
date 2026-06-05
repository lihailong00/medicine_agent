"""可选的大模型查询规划器。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from medicine_agent.models import EvidenceItem, PaperRecord
from medicine_agent.network_policy import DEFAULT_TIMEOUT_SECONDS, post_json_bytes
from medicine_agent.reporting.synthesis import sanitize_review_synthesis
from medicine_agent.safety import SafetyGate

ALLOWED_LITERATURE_SOURCES = ("pubmed", "semantic_scholar", "arxiv")
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
MAX_SEARCH_TOPIC_CHARS = 320
MAX_RATIONALE_CHARS = 500
MAX_ABSTRACT_CHARS = 1800
MAX_FULL_TEXT_PREVIEW_CHARS = 1800


@dataclass(frozen=True)
class DeepSeekConfig:
    """从环境变量读取的 DeepSeek API 配置。"""

    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    @property
    def chat_completions_url(self) -> str:
        """返回 OpenAI 兼容 chat completions 端点。"""

        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


@dataclass(frozen=True)
class LLMQueryPlan:
    """大模型输出的检索规划结果。"""

    search_topic: str
    subquestions: tuple[str, ...]
    sources: tuple[str, ...]
    rationale: str
    planner: str = "llm_deepseek"

    def to_dict(self) -> dict[str, object]:
        return {
            "planner": self.planner,
            "search_topic": self.search_topic,
            "subquestions": list(self.subquestions),
            "sources": list(self.sources),
            "rationale": self.rationale,
        }


def load_deepseek_config() -> DeepSeekConfig | None:
    """从环境变量加载 DeepSeek 配置；缺少 key 时返回 None。"""

    api_key = _first_env_value("DEEPSEEK_API_KEY", "MEDICINE_AGENT_DEEPSEEK_API_KEY")
    if not api_key:
        return None
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip() or DEFAULT_DEEPSEEK_MODEL
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).strip() or DEFAULT_DEEPSEEK_BASE_URL
    timeout = _env_int("DEEPSEEK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    return DeepSeekConfig(api_key=api_key, model=model, base_url=base_url, timeout=timeout)


def plan_query_with_llm(
    question: str,
    *,
    allowed_sources: Sequence[str] = ALLOWED_LITERATURE_SOURCES,
    network_gate: SafetyGate | None = None,
) -> LLMQueryPlan | None:
    """使用 DeepSeek 生成英文检索主题；失败时返回 None 让调用方规则降级。"""

    config = load_deepseek_config()
    if config is None:
        return None
    sanitized_sources = _sanitize_sources(allowed_sources)
    if not sanitized_sources:
        sanitized_sources = ALLOWED_LITERATURE_SOURCES
    try:
        payload = _build_query_planning_payload(question, sanitized_sources, config.model)
        response = _post_deepseek_chat(
            config,
            payload,
            network_gate=network_gate,
            rationale="使用 DeepSeek API 进行科研检索 query 改写与来源规划",
        )
        content = _extract_message_content(response)
        plan_payload = _loads_json_object(content)
        return _sanitize_llm_plan(plan_payload, question=question, allowed_sources=sanitized_sources)
    except Exception:  # noqa: BLE001 - LLM 规划必须可安全降级。
        return None


def synthesize_review_with_llm(
    question: str,
    *,
    papers: Sequence[PaperRecord],
    evidence: Sequence[EvidenceItem],
    interactions: Sequence[Mapping[str, object]] = (),
    full_text_records: Sequence[Mapping[str, object]] = (),
    allowed_refs: set[str],
    network_gate: SafetyGate | None = None,
) -> dict[str, object] | None:
    """使用 DeepSeek 生成带引用的结构化综述；失败时返回 None。"""

    config = load_deepseek_config()
    if config is None:
        return None
    try:
        payload = _build_review_synthesis_payload(
            question,
            papers=papers,
            evidence=evidence,
            interactions=interactions,
            full_text_records=full_text_records,
            allowed_refs=allowed_refs,
            model=config.model,
        )
        response = _post_deepseek_chat(
            config,
            payload,
            network_gate=network_gate,
            rationale="使用 DeepSeek API 进行证据抽取与结构化综述生成",
        )
        content = _extract_message_content(response)
        raw = dict(_loads_json_object(content))
        raw.setdefault("source", "llm_deepseek")
        return sanitize_review_synthesis(raw, allowed_refs=allowed_refs)
    except Exception:  # noqa: BLE001 - 综述生成必须可安全降级。
        return None


def _build_query_planning_payload(question: str, allowed_sources: Sequence[str], model: str) -> dict[str, object]:
    system_prompt = (
        "你是生信科研文献检索规划器。你的任务是把用户科研问题改写为适合 PubMed/NCBI、"
        "Semantic Scholar 与 arXiv API 的英文检索主题，并拆成 2-4 个科研子问题。"
        "不要提供医学建议，不要臆造论文，不要引用全文内容。"
        "只能从 allowed_sources 中选择来源。"
        "只返回 JSON 对象，字段必须为 search_topic、subquestions、sources、rationale。"
    )
    user_payload = {
        "question": question,
        "allowed_sources": list(allowed_sources),
        "constraints": [
            "search_topic 使用英文，适合元数据/摘要检索",
            "如果问题只是最新综述/研究进展，优先 PubMed 与 Semantic Scholar；不要因为 latest/recent 自动加入 arXiv",
            "只有预印本、算法、模型、机器学习、计算生物学等问题才选择 arXiv",
            "不要输出 allowed_sources 之外的来源",
        ],
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }


def _build_review_synthesis_payload(
    question: str,
    *,
    papers: Sequence[PaperRecord],
    evidence: Sequence[EvidenceItem],
    interactions: Sequence[Mapping[str, object]],
    full_text_records: Sequence[Mapping[str, object]],
    allowed_refs: set[str],
    model: str,
) -> dict[str, object]:
    system_prompt = (
        "你是严谨的生信科研综述 agent。你只能基于用户提供的 papers、full_text_records、"
        "data_interactions 与 existing_evidence 生成中文结构化综述。"
        "所有支持性主张必须引用 allowed_refs 中的 evidence_refs；不要发明论文、作者、年份或 DOI。"
        "如果某个判断没有 allowed_refs 支持，必须标为 hypothesis。"
        "不要输出诊断、治疗、处方或患者管理建议。"
        "只返回 JSON 对象，字段为 executive_summary、key_findings、evidence_table、"
        "mechanism_review、hypotheses、limitations_conflicts、reproducibility_notes。"
    )
    user_payload = {
        "question": question,
        "allowed_refs": sorted(allowed_refs),
        "papers": [_paper_for_llm(paper) for paper in papers[:20]],
        "full_text_records": [_full_text_for_llm(record) for record in full_text_records[:12]],
        "data_interactions": [_interaction_for_llm(item) for item in interactions[:20]],
        "existing_evidence": [item.to_dict() for item in evidence[:12]],
        "required_schema": {
            "executive_summary": "中文摘要，必须说明证据范围",
            "key_findings": ["{claim,status,evidence_refs,confidence,limitations}"],
            "evidence_table": ["{claim,status,evidence_refs,confidence,limitations}"],
            "mechanism_review": ["中文机制综述要点"],
            "hypotheses": ["可检验假设"],
            "limitations_conflicts": ["冲突证据或局限"],
            "reproducibility_notes": ["复现说明"],
        },
        "allowed_statuses": [
            "literature_supported",
            "data_supported",
            "literature_and_data_supported",
            "conflicting",
            "hypothesis",
        ],
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "max_tokens": 2400,
        "response_format": {"type": "json_object"},
    }


def _paper_for_llm(paper: PaperRecord) -> dict[str, object]:
    return {
        "paper_id": paper.evidence_id,
        "title": paper.title,
        "abstract": _compact_text(paper.abstract, max_chars=MAX_ABSTRACT_CHARS),
        "source": paper.source,
        "year": paper.year,
        "authors": paper.authors[:8],
        "venue": paper.venue,
        "pmid": paper.pmid,
        "pmcid": paper.pmcid,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
        "source_url": paper.source_url,
    }


def _full_text_for_llm(record: Mapping[str, object]) -> dict[str, object]:
    candidate = record.get("candidate") if isinstance(record.get("candidate"), Mapping) else {}
    status = record.get("status") if isinstance(record.get("status"), Mapping) else {}
    candidate_map = candidate if isinstance(candidate, Mapping) else {}
    status_map = status if isinstance(status, Mapping) else {}
    return {
        "paper_id": str(candidate_map.get("paper_id") or status_map.get("paper_id") or ""),
        "title": str(candidate_map.get("title") or ""),
        "scope": str(status_map.get("scope") or ""),
        "status": str(status_map.get("status") or ""),
        "text_preview": _compact_text(record.get("text_preview"), max_chars=MAX_FULL_TEXT_PREVIEW_CHARS),
    }


def _interaction_for_llm(item: Mapping[str, object]) -> dict[str, object]:
    source_file = item.get("source_file")
    row_index = item.get("row_index")
    evidence_ref = f"{source_file}#row-{row_index}" if source_file is not None and row_index is not None else ""
    return {
        "evidence_ref": evidence_ref,
        "source_cell": item.get("source_cell"),
        "target_cell": item.get("target_cell"),
        "ligand": item.get("ligand"),
        "receptor": item.get("receptor"),
        "pvalue": item.get("pvalue"),
        "lr_mean": item.get("lr_mean"),
    }


def _post_deepseek_chat(
    config: DeepSeekConfig,
    payload: Mapping[str, object],
    *,
    network_gate: SafetyGate | None,
    rationale: str,
) -> Mapping[str, object]:
    headers = {"Authorization": f"Bearer {config.api_key}"}
    body = post_json_bytes(
        config.chat_completions_url,
        payload,
        headers=headers,
        network_gate=network_gate,
        timeout=config.timeout,
        rationale=rationale,
        max_bytes=512 * 1024,
    )
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("DeepSeek 响应不是 JSON 对象")
    return decoded


def _extract_message_content(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek 响应缺少 choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("DeepSeek choices[0] 不是对象")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("DeepSeek 响应缺少 message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek 响应缺少 content")
    return content.strip()


def _loads_json_object(content: str) -> Mapping[str, object]:
    cleaned = _strip_markdown_fence(content)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM query plan 不是 JSON 对象")
    return parsed


def _strip_markdown_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _sanitize_llm_plan(
    payload: Mapping[str, object],
    *,
    question: str,
    allowed_sources: Sequence[str],
) -> LLMQueryPlan | None:
    search_topic = _compact_text(payload.get("search_topic"), max_chars=MAX_SEARCH_TOPIC_CHARS)
    if len(search_topic) < 3:
        return None
    sources = _sanitize_sources(_string_sequence(payload.get("sources")))
    allowed_set = set(_sanitize_sources(allowed_sources))
    sources = tuple(source for source in sources if source in allowed_set)
    if not sources:
        sources = tuple(allowed_sources)
    subquestions = tuple(
        _compact_text(item, max_chars=220)
        for item in _string_sequence(payload.get("subquestions"))
        if _compact_text(item, max_chars=220)
    )[:4]
    if not subquestions:
        cleaned_question = " ".join(question.strip().split())
        subquestions = (
            f"哪些原始文献讨论了：{cleaned_question}？",
            "哪些发现由元数据/摘要支持，哪些仍属于假设？",
        )
    rationale = _compact_text(payload.get("rationale"), max_chars=MAX_RATIONALE_CHARS)
    if not rationale:
        rationale = "DeepSeek 返回了结构化检索规划；已按 allowlist 过滤来源。"
    return LLMQueryPlan(search_topic=search_topic, subquestions=subquestions, sources=sources, rationale=rationale)


def _sanitize_sources(values: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    allowed = set(ALLOWED_LITERATURE_SOURCES)
    aliases = {
        "semantic scholar": "semantic_scholar",
        "semanticscholar": "semantic_scholar",
        "s2": "semantic_scholar",
        "ncbi": "pubmed",
        "pmid": "pubmed",
        "pub med": "pubmed",
        "arxiv": "arxiv",
        "arxiv_api": "arxiv",
    }
    for value in values:
        source = aliases.get(value.strip().lower().replace("-", "_"), value.strip().lower().replace("-", "_"))
        if source in allowed and source not in normalized:
            normalized.append(source)
    return tuple(normalized)


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _compact_text(value: object, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    compact = " ".join(value.strip().split())
    return compact[:max_chars]


def _first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
