"""Structural citation guards; these functions do not establish entailment."""

from .errors import AppError
from .schemas import AnswerDraft, EvidencePack


def validate_citation_references(draft: AnswerDraft, pack: EvidencePack) -> None:
    if draft.pack_id != pack.pack_id:
        raise AppError("pack_mismatch", "答案与本次证据版本不一致。", 422)
    evidence = {e.evidence_id: e for e in pack.evidence_items}
    claims = {c.claim_id: c for c in draft.claims}
    if len(evidence) != len(pack.evidence_items) or len(claims) != len(draft.claims):
        raise AppError("duplicate_id", "证据或陈述标识重复。", 422)
    for claim in draft.claims:
        cited = set(claim.evidence_ids)
        if not cited <= evidence.keys():
            raise AppError("unknown_evidence", "答案引用了本轮未提供的证据。", 422)
        for eid in cited:
            if "reference" in evidence[eid].roles and not cited.intersection(evidence[eid].reference_for):
                raise AppError("reference_without_context", "借用的条款不能脱离本次主条款独立引用。", 422)
            if not set(evidence[eid].required_evidence_ids) <= cited:
                raise AppError("missing_dependency", "引用缺少必要条件或依赖。", 422)
        if not set(claim.depends_on_claim_ids) <= claims.keys():
            raise AppError("unknown_claim", "陈述引用了不存在的前提。", 422)
        if claim.kind == "calculation":
            raise AppError("calculation_disabled", "个人精确金额计算尚未启用。", 422)
    active: set[str] = set()
    done: set[str] = set()

    def visit(cid: str) -> None:
        if cid in active:
            raise AppError("claim_cycle", "陈述依赖不能成环。", 422)
        if cid in done:
            return
        active.add(cid)
        for child in claims[cid].depends_on_claim_ids:
            visit(child)
        active.remove(cid)
        done.add(cid)

    for cid in claims:
        visit(cid)

    # SourceSpan exact/hash, domain gates and semantic support still require the
    # future compiler. Passing this function alone must never mark them passed.
