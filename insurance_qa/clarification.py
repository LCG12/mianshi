"""依据已有条款和缺失条件，选择当前确有必要的一条追问。"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClarificationCandidate:
    candidate_id: str
    question: str
    target_fields: tuple[str, ...]
    enabled_action: str
    basis_evidence_ids: tuple[str, ...]
    changes_current_goal: bool
    user_answerable: bool
    conflict_resolution: bool = False
    aspect_ids: tuple[str, ...] = ()


def select_candidates(candidates: list[ClarificationCandidate], *, available_actions: set[str],
                      known_fields: set[str], evidence_ids: set[str], limit: int = 1) -> list[ClarificationCandidate]:
    """按候选原顺序筛选，排除无依据、已知或重复的条件。"""
    if not 0 <= limit <= 2:
        raise ValueError("Clarification limit must be between zero and two")
    selected: list[ClarificationCandidate] = []
    asked_fields: set[str] = set()
    seen: set[str] = set()
    for candidate in candidates:
        if len(selected) >= limit:
            break
        fields = set(candidate.target_fields)
        if candidate.candidate_id in seen or not fields:
            continue
        if candidate.enabled_action not in available_actions:
            continue
        if not candidate.changes_current_goal or not candidate.user_answerable:
            continue
        if not candidate.basis_evidence_ids or not set(candidate.basis_evidence_ids) <= evidence_ids:
            continue
        if fields & known_fields and not candidate.conflict_resolution:
            continue
        if fields & asked_fields or len(fields | asked_fields) > 2:
            continue
        selected.append(candidate)
        asked_fields |= fields
        seen.add(candidate.candidate_id)
    return selected


def build_candidates(plan, pack, *, asked_questions=(), pending_questions=(), exclude_aspect_ids=()):
    """只对当前个人分支判断提出有原文依据的少量追问，不为金额计算收集参数。"""
    for aspect in plan.aspects:
        if aspect.aspect_id in exclude_aspect_ids:
            continue
        candidates = _for_aspect(plan, pack, aspect, asked_questions, pending_questions)
        if candidates:
            return candidates
    return []


def _for_aspect(plan, pack, aspect, asked_questions, pending_questions):
    goal = aspect.quote + " " + aspect.objective
    unclear_event = r"出(?:了)?(?:点|些|一点|一些)?事|出险"
    if aspect.task != "case_qa" and not (aspect.task == "unknown" and re.search(unclear_event, goal)):
        return []
    if re.search(r"精确|具体金额|多少钱|能拿多少|现金价值|未来.*红利", goal):
        return []
    facts = [m for m in plan.mentions if m.modality in {"asserted", "hypothetical"}
             and aspect.aspect_id in m.aspect_ids]
    known = set()
    for mention in facts:
        text = mention.quote
        if re.search(r"不清楚|不知道|不确定|是否|是不是|可能|听说", text):
            continue
        if (mention.kind == "insured_count" or re.search(
                r"单人|双人|被保险人(?:为|有|是|共)?[一二两12](?:人|位|名)|[一二两12](?:名|位|个)?被保险人", text)
                ) and not re.search(r"出险人数|身故人数|死亡人数", text):
            known.add("insured_count")
        if mention.kind == "cause" or re.search(r"因(?:疾病|病|意外)|非意外|交通事故", text):
            known.add("cause")
        duration = re.search(r"(?:\d+|[一二三四五六七八九十]+)(?:天|日|个月|月|年)|昨天|今天|前天", text)
        if duration and re.search(r"生效|复效", text):
            known.add("contract_timing")
        if duration and "签收" in text:
            known.add("receipt_elapsed")
        if duration and mention.kind == "time" and mention.message_id == plan.current_message_id:
            if any("合同生效" in question for question in pending_questions):
                known.add("contract_timing")
            if any("签收" in question for question in pending_questions):
                known.add("receipt_elapsed")
    # 历史目标原话可能含已更正条件，只采用当前原话和本轮仍有效的mentions。
    question = plan.original_question
    if len(plan.aspects) > 1:
        question = aspect.quote if aspect.message_id == plan.current_message_id else ""
    context = question + " " + " ".join(m.quote for m in facts)
    context = re.sub(r"(?:不是|并非|没有|没|未)(?:发生|因)?(?:身故|去世|死亡|全残|疾病|生病)", "", context)
    # 不让用户自行作保险意义上的全残认定。
    life_event = bool(re.search(r"身故|去世|死亡|全残", context))
    possible = []
    if re.search(unclear_event + "|事故", context) and not life_event and not re.search(r"住院|骨折|手术|伤残", context):
        possible = [("event_type", "你说的“出事”具体发生了什么？可以简单描述经过或已有诊断，不清楚也可以。",
                     [("2.5", "2.5.1.scope", "2.5.2.scope")])]
    elif life_event and aspect.task == "case_qa":
        possible = [
            ("insured_count", "这张保单是一位还是两位被保险人？如果中途减少过人数，也请说明；不清楚也可以。",
             [("2.5.1.scope", "2.5.2.scope", "2.5.3")]),
            ("cause", "这次身故或伤情是由什么原因导致的？可以描述已知经过，不清楚也可以。",
             [("2.5.1.scope", "2.5.1.early90", "2.5.2.scope", "2.5.2.ordered_early90", "2.5.2.unordered_early90")]),
        ]
        if re.search(r"疾病|生病|因病|非意外|猝死", context):
            possible.append(("contract_timing", "出事距离合同生效大约多久？如果复效过，请按最近一次复效起算；不清楚也可以。",
                             [("2.5.1.scope", "2.5.1.early90", "2.5.2.scope", "2.5.2.ordered_early90", "2.5.2.unordered_early90")]))
    elif aspect.task == "case_qa" and re.search(r"退保|退掉|退钱|全额退", context):
        possible = [("receipt_elapsed", "从签收保险合同到现在，是否还在15天内？不清楚也可以。", [("1.4",), ("9.1",)])]
    for field, question, groups in possible:
        if field in known:
            continue
        # 前置条件仍未知且问过，就回到有条件的一般说明，不能换着问后面的无关参数。
        if question in asked_questions:
            return []
        basis = []
        for group in groups:
            evidence_id = next((e.evidence_id for e in pack.evidence_items
                                if e.section in group and "reference" not in e.roles), None)
            basis.append(evidence_id)
        if not all(basis):
            return []
        candidate = ClarificationCandidate(
            candidate_id=field, question=question, target_fields=(field,),
            enabled_action="select_applicable_rule_branch", basis_evidence_ids=tuple(basis),
            changes_current_goal=True, user_answerable=True, aspect_ids=(aspect.aspect_id,),
        )
        return select_candidates([candidate], available_actions={"select_applicable_rule_branch"},
                                 known_fields=known, evidence_ids={e.evidence_id for e in pack.evidence_items})
    return []


def apply_clarification(result, candidates):
    """问题由应用模板给出；可保留其他方面的已验证答复，不接受模型自造追问。"""
    if not candidates:
        return result
    candidate = candidates[0]
    result.clarification_questions = [candidate.question]
    result.answer_status = "needs_clarification"
    result.coverage_status = "partial" if result.answer_blocks else "none"
    for aspect in result.aspect_results:
        if aspect["aspect_id"] in candidate.aspect_ids:
            aspect["disposition"] = "needs_user_info"
    result.trace["clarification"] = {"candidate_id": candidate.candidate_id,
        "aspect_ids": list(candidate.aspect_ids), "target_fields": list(candidate.target_fields),
        "basis_evidence_ids": list(candidate.basis_evidence_ids)}
    return result
