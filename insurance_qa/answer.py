"""完整证据组预算、回答与最多一次条件修正、独立复核。出处由程序编译。"""

import hashlib
import json
import re
from copy import deepcopy
from time import perf_counter

from .answer_validation import compile_answer
from .errors import AppError
from .evidence import render_context
from .models.contracts import ModelServiceError
from .schemas import AnswerDraft, AnswerResult, Gap


REPAIRABLE_ERRORS = {
    "answer_aspect_mismatch", "invalid_aspect_disposition", "unknown_answer_gap",
    "unknown_claim", "claim_cycle", "expanded_exclusion_scope", "beneficiary_scope_ambiguous",
    "single_insured_scope_missing", "benefit_trigger_missing", "disclosure_limits_missing",
    "disclosure_causation_missing", "notice_exceptions_missing", "unsupported_surrender_fee",
    "surrender_source_mismatch", "eligibility_subject_mismatch", "policy_value_conditions_missing",
    "benefit_start_point_mismatch", "early_benefit_limits_missing",
}


def subset_pack(pack, root_ids):
    """保留主条款及全部依赖；同一原文的不同用途仍使用各自证据ID。"""
    by_id = {item.evidence_id: item for item in pack.evidence_items}
    selected, pending = set(), list(root_ids)
    while pending:
        eid = pending.pop()
        if eid not in selected:
            selected.add(eid)
            pending.extend(by_id[eid].required_evidence_ids)
    items = []
    for item in pack.evidence_items:
        if item.evidence_id not in selected:
            continue
        item = item.model_copy(deep=True)
        item.reference_for = [eid for eid in item.reference_for if eid in root_ids]
        if item.evidence_id not in root_ids:
            item.roles = [role for role in item.roles if role != "root"]
        items.append(item)
    span_ids = {sid for item in items for sid in item.source_span_ids}
    identity = pack.pack_id + ",".join(sorted(root_ids))
    context = render_context(items)
    spans = [span for span in pack.source_spans if span["span_id"] in span_ids]
    selection = []
    for row in pack.selection:
        if row["status"] == "included" and row["evidence_id"] not in root_ids:
            row = {**row, "status": "not_selected"}
        selection.append(row)
    coverage = []
    for row in pack.aspect_coverage:
        retained_ids = [eid for eid in row["retrieved_for_evidence_ids"] if eid in root_ids]
        coverage.append({**row, "retrieved_for_evidence_ids": retained_ids})
    return pack.model_copy(update={
        "pack_id": "pack-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "evidence_items": items, "source_spans": spans,
        "context": context, "gaps": [], "budget": {**pack.budget, "used_chars": len(context)},
        "selection": selection, "aspect_coverage": coverage,
    })


def apply_review(review, draft):
    """保留复核支持且依赖闭合的整条陈述；不改正文，也不把部分回应升级为完整。"""
    checked = []
    for name, key, field, expected in (
        ("claims", "claim_id", "supported", {c.claim_id for c in draft.claims}),
        ("aspects", "aspect_id", "complete", {a.aspect_id for a in draft.aspect_results}),
    ):
        try:
            rows = review[name]
            values = {row[key]: row[field] for row in rows}
            valid = (len(values) == len(rows) and set(values) == expected
                     and all(type(value) is bool for value in values.values()))
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise AppError("answer_support_failed", "复核须逐项覆盖且使用明确布尔判断，本次不交付草稿。", 422)
        checked.append(values)
    supported, complete = checked
    claims = {claim.claim_id: claim for claim in draft.claims}
    for row in review["claims"]:
        if not row["supported"]:
            continue
        pending, visited, allowed = [row["claim_id"]], set(), set()
        while pending:
            cid = pending.pop()
            if cid not in visited:
                visited.add(cid)
                allowed.update(claims[cid].evidence_ids)
                pending.extend(claims[cid].depends_on_claim_ids)
        cited = set(re.findall(r"(?<![A-Za-z0-9])E[0-9]+R?(?![A-Za-z0-9])", row.get("reason", "")))
        if not cited <= allowed:
            raise AppError("answer_support_failed", "复核理由借用了本句及其声明前提之外的证据，本次不交付草稿。", 422)
    rejected = {cid for cid, ok in supported.items() if not ok}
    while True:
        expanded = rejected | {c.claim_id for c in draft.claims if rejected.intersection(c.depends_on_claim_ids)}
        if expanded == rejected:
            break
        rejected = expanded
    if draft.claims and len(rejected) == len(draft.claims):
        raise AppError("answer_support_failed", "答案没有可独立保留的受支持陈述，本次不交付草稿。", 422)
    reduced = draft.model_copy(deep=True)
    reduced.claims = [c for c in reduced.claims if c.claim_id not in rejected]
    incomplete = []
    for aspect in reduced.aspect_results:
        removed = bool(rejected.intersection(aspect.claim_ids))
        aspect.claim_ids = [cid for cid in aspect.claim_ids if cid not in rejected]
        if removed or not complete[aspect.aspect_id]:
            aspect.disposition = "unavailable"
            aspect.gap_ids = ["gap_unverified_" + aspect.aspect_id]
            incomplete.append(aspect.aspect_id)
    return reduced, {"removed_claim_ids": sorted(rejected), "incomplete_aspect_ids": incomplete}


def complete_references(draft, pack):
    """只沿可信关系补必要引用，不改正文、不猜主条款；补完后仍须语义复核。"""
    evidence = {item.evidence_id: item for item in pack.evidence_items}
    additions = {}
    for claim in draft.claims:
        claim.evidence_ids = list(dict.fromkeys(claim.evidence_ids))
        selected, pending = set(claim.evidence_ids), list(claim.evidence_ids)
        while pending:
            eid = pending.pop()
            if eid not in evidence:
                raise AppError("unknown_evidence", "答案引用了本轮未提供的证据。", 422)
            for needed in evidence[eid].required_evidence_ids:
                if needed not in selected:
                    selected.add(needed)
                    pending.append(needed)
        additions[claim.claim_id] = sorted(selected - set(claim.evidence_ids))
        claim.evidence_ids.extend(additions[claim.claim_id])
    return additions


class Answerer:
    ready = True

    def __init__(self, client, project_root):
        self.client = client
        self.prompt = (project_root / "prompts/answer.txt").read_text(encoding="utf-8")
        self.review_prompt = (project_root / "prompts/answer_review.txt").read_text(encoding="utf-8")
        self.schema = json.loads((project_root / "schemas/answer_draft.schema.json").read_text())
        self.review_schema = json.loads((project_root / "schemas/answer_review.schema.json").read_text())

    def messages(self, plan, pack, *, review=False, draft=""):
        active = {plan.current_message_id, *plan.context_message_ids}
        data = {"question": plan.original_question, "source_policy": plan.source_policy, "pack_id": pack.pack_id,
                "conversation_mode": plan.conversation_mode,
                "user_messages": [row for row in plan.user_messages if row["message_id"] in active],
                "pending_conditions": pack.unresolved_user_conditions,
                "aspects": [{"aspect_id": a.aspect_id, "objective": a.objective, "task": a.task,
                             "quote": a.quote, "message_id": a.message_id} for a in plan.aspects],
                "mentions": [{"mention_id": m.mention_id, "quote": m.quote, "modality": m.modality,
                              "message_id": m.message_id, "kind": m.kind, "subject": m.subject,
                              "aspect_ids": m.aspect_ids} for m in plan.mentions]}
        data["cite_root_ids"] = [item.evidence_id for item in pack.evidence_items if "root" in item.roles]
        data["root_sections"] = {item.evidence_id: item.section + " " + item.title
                                 for item in pack.evidence_items if "root" in item.roles}
        context = pack.context if review else pack.context.replace("必须同引=", "程序自动补引用=")
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n证据：\n" + context
        if review:
            content += "\n待核对草稿：\n" + draft
        return [{"role": "system", "content": self.review_prompt if review else self.prompt},
                {"role": "user", "content": content}]

    def draft_schema(self, plan, pack):
        """本轮未提供用户前提时，只允许生成一般规则；已关闭能力不交给模型选择。"""
        schema = deepcopy(self.schema)
        fields = schema["properties"]
        claim_schema = fields["claims"]["items"]
        claim = claim_schema["properties"]
        mentions = [m.mention_id for m in plan.mentions if m.modality in {"asserted", "hypothetical"}]
        claim["kind"] = {"enum": ["rule", "conditional_application"] if mentions else ["rule"]}
        if mentions:
            claim["premise_mention_ids"]["items"]["enum"] = mentions
        else:
            claim["premise_mention_ids"]["maxItems"] = 0
        roots = [item.evidence_id for item in pack.evidence_items if "root" in item.roles]
        claim["evidence_ids"]["items"]["enum"] = roots
        claim["evidence_ids"]["maxItems"] = max(2, len(roots))
        claim["evidence_ids"].pop("uniqueItems", None)
        if mentions:
            # 解码时就绑定类型和前提，避免生成后擅自把一般规则改成个案结论。
            branches = []
            for kind, bound in (("rule", {"maxItems": 0}), ("conditional_application", {"minItems": 1})):
                branch = deepcopy(claim_schema)
                branch["properties"]["kind"] = {"const": kind}
                branch["properties"]["premise_mention_ids"].update(bound)
                branches.append(branch)
            fields["claims"]["items"] = {"anyOf": branches}
        fields["clarifications"]["maxItems"] = 0
        aspect = fields["aspect_results"]["items"]["properties"]
        aspect["disposition"]["enum"] = ["addressed", "unavailable"]
        return schema

    async def answer(self, plan, full_pack):
        config = self.client.config
        answer_limit = min(2048, config.max_tokens)
        review_limit = min(768, config.max_tokens)
        roots, omitted, excluded = [], [], []
        pack = subset_pack(full_pack, [])
        answer_tokens = 0
        # 按排名尝试增加完整组，真实tokenizer同时为生成和复核预留空间。
        for row in full_pack.selection:
            if row["status"] != "included":
                continue
            proposed = subset_pack(full_pack, roots + [row["evidence_id"]])
            if any(item.formula_layout_risk for item in proposed.evidence_items):
                excluded.append(row["evidence_id"])
                continue  # 版面尚未核准的公式整组不交给模型，不能只靠提示词约束。
            tokens = await self.client.count_tokens(self.messages(plan, proposed))
            review_tokens = await self.client.count_tokens(self.messages(plan, proposed, review=True))
            answer_budget = tokens + answer_limit
            review_budget = review_tokens + answer_limit + review_limit + 64
            if max(answer_budget, review_budget) <= config.context_window:
                roots.append(row["evidence_id"])
                pack, answer_tokens = proposed, tokens
            else:
                omitted.append(row["evidence_id"])
        trace = {"pack_id": pack.pack_id, "selected_root_ids": roots, "omitted_root_ids": omitted,
                 "formula_excluded_root_ids": excluded,
                 "context_window": config.context_window, "max_output_tokens": answer_limit,
                 "source_policy": plan.source_policy, "token_count_verified": bool(roots)}
        if not roots:
            if omitted:
                message = "本次未能在模型窗口内放入完整证据，请缩小问题范围。"
            elif excluded:
                message = "本次相关公式版面尚未核准，暂时无法据此解释或计算。"
            else:
                message = "本次未获取到可用于回答的完整证据。"
            return AnswerResult(answer_status="insufficient_evidence", execution_status="degraded",
                gaps=[Gap(code="answer_context_unavailable", message=message,
                          aspect_ids=[a.aspect_id for a in plan.aspects])], trace=trace, versions=pack.versions)
        messages = self.messages(plan, pack)
        trace["answer_input_tokens"] = answer_tokens  # 此完整请求已计数，无需再次请求同一tokenizer。
        trace["repair"] = {"attempts": 0, "status": "not_needed"}
        schema = self.draft_schema(plan, pack)
        try:
            for attempt in range(2):
                start = perf_counter()
                if attempt:
                    trace["repair"].update(attempts=1, status="failed")
                try:
                    generated = await self.client.generate("answer", messages, schema, max_tokens=answer_limit)
                finally:
                    if attempt:
                        trace["repair"]["latency_ms"] = round((perf_counter() - start) * 1000, 3)
                draft = AnswerDraft.model_validate_json(generated.content)
                try:
                    trace["added_dependency_ids"] = complete_references(draft, pack)
                    compile_answer(draft, plan, pack)
                    if attempt:
                        trace["repair"]["status"] = "draft_validated"
                    break
                except AppError as exc:
                    if attempt or exc.code not in REPAIRABLE_ERRORS:
                        raise
                    feedback = json.dumps({"code": exc.code, "message": exc.message}, ensure_ascii=False)
                    messages = messages + [{"role": "assistant", "content": generated.content},
                        {"role": "user", "content": "应用校验未通过：" + feedback
                         + "。请依据同一证据包修正，并重新输出完整草稿；不增加用户条件或证据。"}]
                    tokens = await self.client.count_tokens(messages)
                    trace["repair"].update(reason=exc.code, input_tokens=tokens, status="skipped_context_budget")
                    if tokens + answer_limit > config.context_window:
                        raise  # 不为修正截断原草稿或必要证据，保留第一次校验错误。
                    trace["initial_answer_input_tokens"] = trace["answer_input_tokens"]
                    trace["initial_answer_usage"] = generated.usage
                    trace["answer_input_tokens"] = tokens
        except (AppError, ModelServiceError) as exc:
            exc.answer_trace = trace
            raise
        cited = {eid for claim in draft.claims for eid in claim.evidence_ids}
        items = [item for item in pack.evidence_items if item.evidence_id in cited]
        review_pack = pack.model_copy(update={"evidence_items": items, "context": render_context(items)})
        messages = self.messages(plan, review_pack, review=True, draft=draft.model_dump_json())
        try:
            trace["review_input_tokens"] = await self.client.count_tokens(messages)
            if trace["review_input_tokens"] + review_limit > config.context_window:
                raise AppError("review_token_budget_exceeded", "完整草稿与证据超出复核窗口，本次不交付答案。", 422)
            checked = await self.client.generate("verify", messages, self.review_schema, max_tokens=review_limit)
            trace["review"] = json.loads(checked.content)
            reviewed, trace["review_filter"] = apply_review(trace["review"], draft)
            result = compile_answer(reviewed, plan, pack, semantic_checked=True)
        except (AppError, ModelServiceError) as exc:
            exc.answer_trace = trace
            raise
        result.trace.update(trace, answer_usage=generated.usage, review_usage=checked.usage)
        if reviewed != draft or omitted or excluded or full_pack.gaps:
            result.execution_status = "degraded"
        return result
