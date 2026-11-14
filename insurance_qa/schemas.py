"""Versioned boundary types. Structural validity does not prove factual support."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourcePolicy = Literal["terms_only", "official_materials"]
ExecutionStatus = Literal["ok", "degraded", "service_error", "validation_error"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(Contract):
    question: str = Field(min_length=1, max_length=12000)
    product_id: str = "sshy_s_hengying"
    source_policy: SourcePolicy = "terms_only"
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_revision: int | None = Field(default=None, ge=0)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value  # Keep the actual original wording, including whitespace.


class IntentHints(Contract):
    route: Literal["insurance", "assistant_info", "chitchat", "out_of_scope", "unknown"] = "unknown"
    tasks: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    raw_scores: dict[str, float] = Field(default_factory=dict)
    available: bool = False


class Aspect(Contract):
    aspect_id: str
    objective: str
    task: str = "unknown"
    topic_hints: list[str] = Field(default_factory=list)
    message_id: str = ""
    quote: str = ""
    start_char: int | None = None
    end_char: int | None = None


class Mention(Contract):
    mention_id: str
    message_id: str
    quote: str
    subject: str
    modality: Literal["asserted", "hypothetical", "reported", "ambiguous"]
    kind: str = "other"
    aspect_ids: list[str] = Field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None


class Query(Contract):
    query_id: str
    kind: Literal["original", "rewrite", "subquery"]
    text: str
    aspect_ids: list[str] = Field(default_factory=list)
    group: Literal["original", "generated"]
    mention_ids: list[str] = Field(default_factory=list)


class QueryPlan(Contract):
    original_question: str
    product_id: str
    source_policy: SourcePolicy
    aspects: list[Aspect]
    queries: list[Query]
    mentions: list[Mention] = Field(default_factory=list)
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    intent_hints: IntentHints = Field(default_factory=IntentHints)
    candidate_missing_information: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    illustration_requested: bool = False
    execution_status: Literal["ok", "degraded"] = "ok"
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    user_messages: list[dict[str, str]] = Field(default_factory=list)
    current_message_id: str = ""
    conversation_mode: Literal["standalone", "followup", "new_topic", "correction"] = "standalone"
    context_message_ids: list[str] = Field(default_factory=list)


class RecallHit(Contract):
    evidence_unit_id: str
    query_id: str
    channel: str
    score: float = Field(allow_inf_nan=False)
    index_version: str


class Gap(Contract):
    code: str
    message: str
    aspect_ids: list[str] = Field(default_factory=list)


class EvidenceItem(Contract):
    evidence_id: str
    unit_ids: list[str]
    source_span_ids: list[str]
    required_evidence_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    section: str = ""
    title: str = ""
    reference_for: list[str] = Field(default_factory=list)
    formula_layout_risk: bool = False
    text: str


class EvidencePack(Contract):
    pack_id: str
    corpus_version: str
    index_versions: dict[str, str]
    source_policy: SourcePolicy
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    aspect_coverage: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_user_conditions: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    execution_status: ExecutionStatus = "ok"
    versions: dict[str, str] = Field(default_factory=dict)
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    context: str = ""
    budget: dict[str, Any] = Field(default_factory=dict)
    selection: list[dict[str, Any]] = Field(default_factory=list)


class Claim(Contract):
    claim_id: str
    aspect_ids: list[str] = Field(min_length=1)
    kind: Literal["rule", "conditional_application", "calculation"]
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    premise_mention_ids: list[str] = Field(default_factory=list)
    depends_on_claim_ids: list[str] = Field(default_factory=list)
    calculation_id: None = None  # Exact calculation is disabled in this phase.


class AspectDraft(Contract):
    aspect_id: str
    disposition: Literal["addressed", "needs_user_info", "unavailable"]
    claim_ids: list[str] = Field(default_factory=list)
    gap_ids: list[str] = Field(default_factory=list)


class ClarificationDraft(Contract):
    candidate_id: str
    aspect_ids: list[str]
    question: str


class AnswerDraft(Contract):
    schema_version: Literal["answer-draft/1.0"] = "answer-draft/1.0"
    pack_id: str
    claims: list[Claim]
    aspect_results: list[AspectDraft] = Field(min_length=1)
    clarifications: list[ClarificationDraft] = Field(default_factory=list, max_length=2)


class AnswerBlock(Contract):
    text: str
    claim_id: str
    citation_ids: list[str] = Field(default_factory=list)


class Citation(Contract):
    citation_id: str
    document_id: str
    document_title: str
    section: str
    pdf_pages: list[int]
    source_url: str
    local_url: str
    quote: str


class AnswerResult(Contract):
    schema_version: Literal["answer-result/1.1"] = "answer-result/1.1"
    answer_status: Literal["answered", "needs_clarification", "insufficient_evidence"] | None = None
    execution_status: ExecutionStatus = "ok"
    coverage_status: Literal["complete", "partial", "none"] = "none"
    answer_blocks: list[AnswerBlock] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    aspect_results: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    validation: dict[str, str] = Field(default_factory=lambda: {
        "structural": "not_run", "source": "not_run", "dependency": "not_run",
        "domain": "not_run", "semantic": "not_run",
    })
    trace_id: str = ""
    session_id: str | None = None
    revision: int | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
