"""校验问题目标与原文条件；改写可降级，矛盾计划最多修复一次。"""

import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from jsonschema import Draft202012Validator, ValidationError

from .errors import AppError
from .models.contracts import ModelServiceError
from .schemas import Aspect, ChatRequest, Mention, Query, QueryPlan
from .session import SessionTicket

MAX_QUESTION_CHARS = 600  # 单条生成检索式的长度上限；模型输入另按真实token计数。
NON_BUSINESS_TASKS = {"assistant_info", "chitchat", "out_of_scope"}
NUMBERS = re.compile(r"单人|双人|\d+(?:\.\d+)?(?:\s*(?:[%％]|周岁|岁|天|日|年|月|元|万|人))?|[零一二两三四五六七八九十百千万]+(?:周岁|岁|天|日|年|月|元|人|个)")
CONDITIONS = re.compile(r"我爸|我妈|父亲|母亲|爸爸|妈妈|丈夫|妻子|儿子|女儿|单人|双人|复效|生效|前|后|内|外|如果|假如|万一|听说|据说")
QUESTION_START = re.compile(r"能不能|可不可以|会不会|能否|是否|还能|可以|怎么|如何|赔不赔|能买|能赔|会赔|有上限")
# 只保护几种容易改变责任分支的词，不把“出事”推断成某种事故。
EVENT_TERMS = (("意外",), ("疾病", "生病", "患病"), ("身故", "死亡", "去世", "人不在了"), ("全残",))
# 只识别完整的单目标规则问；这里提供检索词，保险答案仍须检索与复核。
SIMPLE_RULE_QUERIES = (
    (r"(?:能保多久|保多久|保障多久|(?:保险期间|保障期限|保障期间)(?:是多久|是多长|是什么|怎么规定|如何规定))", "保险期间如何规定？", "coverage"),
    (r"(?:交费|缴费)方式(?:有哪些|是什么|如何规定)", "交费方式有哪些？", "premium"),
    (r"(?:交费|缴费)期间(?:有哪些|是什么|如何规定)", "交费期间有哪些？", "premium"),
    (r"(?:被保险人的?)?投保年龄范围(?:是多少|是什么|如何规定)", "被保险人的投保年龄范围是多少？", "eligibility"),
)


def current_message_id(context: SessionTicket) -> str:
    return f"{context.session_id}:u{context.revision + 1}"


def _invalid(reason: str) -> AppError:
    return AppError("invalid_query_draft", f"查询草案校验失败：{reason}。", 422)


@lru_cache(maxsize=4)
def _schema(root: Path):
    return json.loads((root / "schemas/query_draft.schema.json").read_text(encoding="utf-8"))


def _messages(request, context):
    rows = [{"message_id": m.message_id, "text": m.text} for m in context.messages]
    rows.append({"message_id": current_message_id(context), "text": request.question})
    if (any(m.role != "user" for m in context.messages)
            or len({m["message_id"] for m in rows}) != len(rows)):
        raise _invalid("会话必须只含可区分的用户原话")
    return rows


def _location(row, messages):
    message_id = row["source_message_id"]
    question = messages.get(message_id, "")
    quote = row["quote"]
    start = question.find(quote)
    if (message_id not in messages or not quote.strip() or start < 0
            or question.find(quote, start + 1) >= 0):
        raise _invalid("引用不存在或位置不唯一")
    return {"message_id": message_id, "quote": quote, "start_char": start, "end_char": start + len(quote)}


def _check_mentions(mentions, messages, current_id, mode):
    for mention in mentions:
        # 同一句中的显式假设/传闻不能靠截短quote变成已陈述的事实。
        prefix = re.split(r"[。！？?；;\n]", messages[mention.message_id][:mention.end_char])[-1]
        if mention.modality == "asserted" and re.search(r"假如|假设|如果|万一|听说|据说|传闻", prefix):
            raise _invalid("假设或转述不能升级为事实")
    current = [m for m in mentions if m.message_id == current_id]
    if mode == "correction" and not current:
        raise _invalid("更正没有关联本轮原话")
    previous = [m for m in mentions if m.message_id != current_id]
    for old in previous:
        for new in current:
            if not set(old.aspect_ids).intersection(new.aspect_ids):
                continue
            if new.kind == "person" and old.kind != "person" and old.subject != new.subject:
                raise _invalid("切换主体后不能沿用另一人的条件")
            same_subject = old.subject == new.subject or "未明确" in {old.subject, new.subject}
            if old.kind == new.kind != "other" and same_subject and old.quote != new.quote:
                raise _invalid("同一方面仍含新旧冲突条件，请保留更正或标明歧义")


def _negatives(text):
    # “能不能/是不是”是问法，不据此断言肯定或否定。
    text = re.sub(r"能不能|会不会|可不可以|是不是|赔不赔|买不买", "是否", text)
    return set(re.findall(r"不|没|未|非|无", text))


def _has_condition(text):
    return bool(NUMBERS.search(text) or CONDITIONS.search(text) or _negatives(text))


def _conditions(question, aspects, mentions):
    """带数字/主体/时序等的原文条件；无法定位方面的条件视作共用。"""
    result = []
    for match in re.finditer(r"[^，,。！？?；;\n]+", question):
        clause = match.group().strip()
        if not _has_condition(clause):
            continue
        # 问法前后都可能有条件；无法安全拆开就保护整句。
        parts = QUESTION_START.split(clause, maxsplit=1)
        prefix = parts[0].strip()
        if not prefix or (len(parts) > 1 and _has_condition(parts[1])):
            prefix = clause
        owners = {a.aspect_id for a in aspects if a.start_char < match.end() and a.end_char > match.start()}
        for mention in mentions:
            if mention.start_char < match.end() and mention.end_char > match.start():
                owners.update(mention.aspect_ids)
        result.append((prefix, owners))
    return result


def _rewrite_error(row, question, aspects, mentions, conditions, current_id):
    text, scope = row["text"], set(row["aspect_ids"])
    relevant = [m for m in mentions if scope.intersection(m.aspect_ids)]
    if set(row["mention_ids"]) != {m.mention_id for m in relevant}:
        return "mention_scope"
    if any(m.quote not in text for m in relevant):
        return "mention_changed"
    required = [quote for quote, owners in conditions if not owners or scope.intersection(owners)]
    if any(quote not in text for quote in required):
        return "condition_changed"
    scoped = " ".join([a.quote for a in aspects if a.aspect_id in scope] + [m.quote for m in relevant] + required)
    active = " ".join([a.quote for a in aspects if a.aspect_id in scope and a.message_id == current_id]
                      + [m.quote for m in relevant] + required)
    if set(NUMBERS.findall(text)) - set(NUMBERS.findall(active)):
        return "new_number"
    for words in EVENT_TERMS:
        if any(word in text for word in words) and not any(word in scoped for word in words):
            return "new_condition"
    # 历史目标的引文可能仍含已替换条件；只强制保留本轮及有效mentions中的否定。
    if _negatives(text) - _negatives(scoped) or _negatives(active) - _negatives(text):
        return "negation_changed"
    if any(m.quote in text for m in mentions if m not in relevant and m.quote not in scoped):
        return "foreign_condition"
    if len(text) > MAX_QUESTION_CHARS or re.search(r"https?://|<\s*(?:script|system)|SELECT\s+.+\s+FROM", text, re.I):
        return "invalid_query_text"
    return None


def compile_draft(draft, request: ChatRequest, context: SessionTicket, *, schema=None) -> QueryPlan:
    """只证明结构、引用和少量字面约束；复杂语义仍需评测。"""
    schema = schema or _schema(Path(__file__).resolve().parents[1])
    try:
        # 检索增强不参与核心理解的结构校验；无效改写稍后单独丢弃。
        Draft202012Validator(schema).validate({**draft, "rewrites": []} if isinstance(draft, dict) else draft)
    except ValidationError:
        raise _invalid("结构不符") from None
    aspect_ids = [a["id"] for a in draft["aspects"]]
    mention_ids = [m["id"] for m in draft["mentions"]]
    aspect_id_set = set(aspect_ids)
    mention_id_set = set(mention_ids)
    if len(aspect_id_set) != len(aspect_ids) or len(mention_id_set) != len(mention_ids):
        raise _invalid("标识重复")
    for row in [*draft["mentions"], *draft["ambiguities"], *draft["missing_information"]]:
        if set(row["aspect_ids"]) - aspect_id_set:
            raise _invalid("提问方面引用无效")
    message_id, question = current_message_id(context), request.question
    user_messages = _messages(request, context)
    messages = {row["message_id"]: row["text"] for row in user_messages}
    mode = draft.get("conversation_mode", "independent")
    if mode == "independent":
        mode = "new_topic" if context.messages else "standalone"
    context_ids = draft.get("context_message_ids", [])
    if (len(set(context_ids)) != len(context_ids) or set(context_ids) - (messages.keys() - {message_id})
            or (context.messages and mode == "standalone")
            or (mode in {"followup", "correction"} and not context_ids)
            or (mode in {"standalone", "new_topic"} and context_ids)):
        raise _invalid("上下文选择与本轮关系不一致")
    allowed_messages = {message_id, *context_ids}
    source_rows = [*draft["aspects"], *draft["mentions"], *draft["ambiguities"]]
    if any(row["source_message_id"] not in allowed_messages for row in source_rows):
        raise _invalid("引用了本轮未选择的历史消息")
    used_history = {row["source_message_id"] for row in source_rows} - {message_id}
    if set(context_ids) != used_history:
        raise _invalid("所选历史没有对应的目标、条件或指代引用；独立问题应使用independent")
    aspects = [Aspect(aspect_id=a["id"], objective=a["objective"], task=a["task"], topic_hints=a["topic_hints"],
                      **_location(a, messages)) for a in draft["aspects"]]
    business_ids = {a.aspect_id for a in aspects if a.task not in NON_BUSINESS_TASKS}
    if any(a.message_id != message_id or a.topic_hints for a in aspects if a.task in NON_BUSINESS_TASKS):
        raise _invalid("非业务目标只引用本轮原话且不含保险主题")
    if not business_ids and (len(aspects) != 1 or context_ids or mode in {"followup", "correction"}):
        raise _invalid("纯非业务问题只保留一个本轮目标，不沿用保险上下文")
    if any(set(row["aspect_ids"]) - business_ids
           for row in [*draft["mentions"], *draft["ambiguities"], *draft["missing_information"]]):
        raise _invalid("条件、歧义和待补充信息只能关联保险目标")
    mentions = []
    for row in draft["mentions"]:
        if row["subject_text"] != "未明确" and row["subject_text"] not in messages.get(row["source_message_id"], ""):
            raise _invalid("主体不是用户原文")
        mentions.append(Mention(mention_id=row["id"], subject=row["subject_text"], kind=row["kind"],
                                modality=row["modality"], aspect_ids=row["aspect_ids"],
                                **_location(row, messages)))
    _check_mentions(mentions, messages, message_id, mode)
    ambiguities = [{**row, **_location(row, messages)} for row in draft["ambiguities"]]
    unresolved = {aid for row in ambiguities + [r for r in draft["missing_information"]
                  if r["needed_for"] == "referent_resolution"] for aid in row["aspect_ids"]}
    if any(a.task == "unknown" and a.aspect_id not in unresolved for a in aspects):
        raise _invalid("目标尚未明确；请还原必要历史目标，无法确定时记录具体的指代歧义或缺失信息")
    plan = QueryPlan(original_question=question, product_id=request.product_id, source_policy=request.source_policy,
                     aspects=aspects, mentions=mentions, ambiguities=ambiguities,
                     candidate_missing_information=draft["missing_information"],
                     illustration_requested=draft["illustration_requested"],
                     queries=[Query(query_id="q_original", kind="original", text=question,
                                    aspect_ids=aspect_ids, mention_ids=mention_ids, group="original")],
                     diagnostics=["lexical_checks_only"], user_messages=user_messages,
                     current_message_id=message_id, conversation_mode=mode, context_message_ids=context_ids)
    # 改写只作可选增强，不能因其结构或引用错误丢掉已校验的用户目标。
    rewrites = draft.get("rewrites", [])
    checker = Draft202012Validator({"$defs": schema["$defs"], **schema["properties"]["rewrites"]})
    if (not checker.is_valid(rewrites)
            or any(set(r["aspect_ids"]) - aspect_id_set or set(r["mention_ids"]) - mention_id_set for r in rewrites)):
        plan.diagnostics.append("rewrites_dropped:invalid_structure_or_reference")
        plan.execution_status, rewrites = "degraded", []
    search_ids = business_ids - unresolved  # 待澄清目标不要求先构造检索式。
    rewrites = [{**row, "aspect_ids": [a for a in row["aspect_ids"] if a in search_ids]}
                for row in rewrites if search_ids.intersection(row["aspect_ids"])]
    if ((len(search_ids) == 1 and len(rewrites) > 1)
            or (any(r["kind"] == "faithful" for r in rewrites)
                and (len(rewrites) != 1 or set(rewrites[0]["aspect_ids"]) != search_ids))):
        plan.diagnostics.append("rewrites_dropped:invalid_coverage_or_budget")
        plan.execution_status, rewrites = "degraded", []
    conditions = _conditions(question, [a for a in aspects if a.message_id == message_id],
                              [m for m in mentions if m.message_id == message_id])
    by_text = {question.strip().casefold(): plan.queries[0]}
    for index, row in enumerate(rewrites, 1):
        reason = _rewrite_error(row, question, aspects, mentions, conditions, message_id)
        if reason:
            plan.diagnostics.append(f"rewrite_dropped:{index}:{reason}")
            plan.execution_status = "degraded"
            continue
        key = row["text"].strip().casefold()
        if key in by_text:
            prior = by_text[key]
            prior.aspect_ids = list(dict.fromkeys(prior.aspect_ids + row["aspect_ids"]))
            prior.mention_ids = list(dict.fromkeys(prior.mention_ids + row["mention_ids"]))
            continue
        kind = "rewrite" if len(search_ids) == 1 or row["kind"] == "faithful" else "subquery"
        query = Query(query_id=f"q{index}", kind=kind,
                      text=row["text"], aspect_ids=row["aspect_ids"], mention_ids=row["mention_ids"], group="generated")
        plan.queries.append(query)
        by_text[key] = query
    covered = {a for q in plan.queries[1:] for a in q.aspect_ids}
    uncovered = search_ids - covered
    if uncovered:
        plan.diagnostics.append("generated_coverage_incomplete")
    if mode in {"followup", "correction"} and uncovered:
        # 历史整句可能含已替换的主体或原因；只拼无条件的目标片段，否则让模型修复引用。
        personal_update = mode == "correction" or any(
            m.message_id == message_id and (m.kind == "person" or m.subject != "未明确") for m in mentions)
        if personal_update or any(
                _has_condition(a.quote) or any(w in a.quote for words in EVENT_TERMS[:2] for w in words)
                for a in aspects if a.message_id != message_id and a.aspect_id in search_ids):
            raise AppError("context_resolution_failed", "主体或条件可能已变化，请引用纯目标片段并生成只含有效条件的查询。", 422)
        active_mentions = [m for m in mentions if search_ids.intersection(m.aspect_ids)]
        text = "；".join(dict.fromkeys([a.quote for a in aspects if a.aspect_id in search_ids]
                                     + [question] + [m.quote for m in active_mentions]))
        baseline = {"text": text, "aspect_ids": sorted(search_ids),
                    "mention_ids": [m.mention_id for m in active_mentions]}
        reason = _rewrite_error(baseline, question, aspects, mentions, conditions, message_id)
        if reason:
            raise AppError("context_resolution_failed", "无法用有效原话构造上下文查询：" + reason, 422)
        plan.queries = [plan.queries[0], Query(query_id="q_context", kind="rewrite", group="generated", **baseline)]
        plan.execution_status = "degraded"
        plan.diagnostics.append("context_query_from_quotes")
    validate_plan(plan, request, context)
    return plan


class ModelQueryPlanner:
    ready = True

    def __init__(self, client, project_root: Path):
        self.client = client
        self.schema = _schema(project_root)
        self.prompt = (project_root / "prompts/query_plan.txt").read_text(encoding="utf-8")
        sources = json.loads((project_root / "data/raw/sources.json").read_text(encoding="utf-8"))
        names = {row["product_name"] for row in sources["documents"]}
        if len(names) != 1:
            raise AppError("product_context_invalid", "来源清单必须明确绑定一个产品。", 422)
        self.product_name = names.pop()

    async def plan(self, request: ChatRequest, context: SessionTicket) -> QueryPlan:
        if not context.messages and not context.pending_questions:
            question = re.sub(r"\s+", "", request.question).rstrip("？?。!！")
            prefix = r"(?:(?:这个|这份|本|该)?(?:保险|产品|合同)的?)?"
            for pattern, text, topic in SIMPLE_RULE_QUERIES:
                if re.fullmatch(prefix + pattern, question):
                    plan = await OriginalQuestionPlanner().plan(request, context)
                    plan.execution_status, plan.diagnostics = "ok", ["simple_rule_plan"]
                    plan.aspects[0].task, plan.aspects[0].topic_hints = "rule_qa", [topic]
                    if text != request.question.strip():
                        plan.queries.append(Query(query_id="q1", kind="rewrite", group="generated", text=text,
                                                  aspect_ids=[plan.aspects[0].aspect_id]))
                    validate_plan(plan, request, context)
                    return plan
        payload = {"message_id": current_message_id(context), "question": request.question,
                   "product_context": {"product_id": request.product_id, "product_name": self.product_name},
                   "source_policy": request.source_policy,
                   "user_messages": _messages(request, context), "pending_questions": list(context.pending_questions),
                   "capabilities": {"exact_amount": False, "execute_action": False},
                   "budget": {"max_generated_queries": 3}}
        messages = [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        plan_limit = min(1536, self.client.config.max_tokens)
        schema = deepcopy(self.schema)
        if not context.messages:
            schema["properties"]["conversation_mode"]["enum"] = ["independent"]
        schema["required"] += ["conversation_mode", "context_message_ids"]
        trace = {"attempts": 0, "errors": []}
        start = perf_counter()
        try:
            for attempt in range(2):
                tokens = await self.client.count_tokens(messages)
                if tokens + plan_limit > self.client.config.context_window:
                    if not attempt:
                        raise AppError("context_budget_exceeded", "完整会话超出规划窗口，请新建会话并提供完整问题。", 422)
                    trace["errors"].append({"code": "repair_context_budget_exceeded"})
                    break  # 修复不能截断原话或已生成的草案。
                trace["attempts"] += 1
                result = await self.client.generate("query_plan", messages, schema, max_tokens=plan_limit)
                try:
                    draft = json.loads(result.content)
                    # 产品由元数据过滤，不让模型名称前缀稀释检索词。
                    rows = draft.get("rewrites", []) if isinstance(draft, dict) else []
                    for row in rows if isinstance(rows, list) else []:
                        if (isinstance(row, dict) and isinstance(row.get("text"), str)
                                and self.product_name not in request.question and row["text"].startswith(self.product_name)):
                            row["text"] = row["text"].removeprefix(self.product_name).removeprefix("的").strip()
                    plan = compile_draft(draft, request, context, schema=schema)
                    if attempt:
                        plan.diagnostics += ["query_plan_repaired"] + [
                            "planning_error:" + e["code"] + ":" + e.get("message", "") for e in trace["errors"]]
                    return plan
                except (AppError, json.JSONDecodeError) as exc:
                    error = {"code": getattr(exc, "code", "invalid_query_json"),
                             "message": exc.message if isinstance(exc, AppError) else "规划输出不是合法JSON。"}
                    trace["errors"].append(error)
                    messages += [{"role": "assistant", "content": result.content}, {"role": "user", "content":
                        "应用校验未通过：" + json.dumps(error, ensure_ascii=False)
                        + "。根据同一完整会话修复计划，保留本轮全部目标与有效条件。独立问题用independent；"
                        "真正追问须引用所需历史目标或条件。不要选无关历史、编造条件或更改产品范围。"
                        "改写无法忠实保留时返回空数组；指代不清请记录歧义。重新输出完整JSON。"}]
            raise AppError("query_planning_failed", "本次问题整理未完成，请重试；已有会话已保留。", 503, retryable=True)
        except (AppError, ModelServiceError, TimeoutError) as exc:
            trace["latency_ms"] = round((perf_counter() - start) * 1000, 3)
            exc.answer_trace = {"query_planning": trace}
            raise

    async def aclose(self):
        await self.client.aclose()


class OriginalQuestionPlanner:
    ready = False

    async def plan(self, request: ChatRequest, context: SessionTicket) -> QueryPlan:
        if context.messages:
            raise AppError("context_resolution_failed", "无法独立理解本轮补充，请提供完整问题。", 422)
        mid = current_message_id(context)
        return QueryPlan(original_question=request.question, product_id=request.product_id,
                         source_policy=request.source_policy,
                         aspects=[Aspect(aspect_id="a_original", objective=request.question, message_id=mid,
                                         quote=request.question, start_char=0, end_char=len(request.question))],
                         queries=[Query(query_id="q_original", kind="original", text=request.question,
                                        aspect_ids=["a_original"], group="original")], execution_status="degraded",
                         diagnostics=["original_only_fallback"], user_messages=_messages(request, context), current_message_id=mid)


def validate_plan(plan: QueryPlan, request: ChatRequest, context: SessionTicket | None = None) -> None:
    if (plan.original_question != request.question or not plan.queries
            or plan.queries[0].text != request.question
            or plan.queries[0].kind != "original" or plan.queries[0].group != "original"
            or len(plan.queries) > 4 or len(plan.aspects) > 6
            or plan.source_policy != request.source_policy or plan.product_id != request.product_id):
        raise AppError("invalid_query_plan", "查询计划改变了原问、来源范围或超出预算。", 422)
    ids, aspects, mentions = ([q.query_id for q in plan.queries], [a.aspect_id for a in plan.aspects],
                              [m.mention_id for m in plan.mentions])
    if (any(len(set(values)) != len(values) for values in (ids, aspects, mentions)) or not aspects
            or any(not q.text.strip() or not q.aspect_ids or set(q.aspect_ids) - set(aspects)
                   or set(q.mention_ids) - set(mentions) for q in plan.queries)
            or any(q.kind == "original" or q.group != "generated" for q in plan.queries[1:])):
        raise AppError("invalid_query_plan", "查询计划标识、分组或方面引用无效。", 422)
    if context is not None and (plan.user_messages or context.messages):
        if plan.user_messages != _messages(request, context) or plan.current_message_id != current_message_id(context):
            raise AppError("invalid_query_plan", "计划原话与当前会话不一致。", 422)
    if plan.user_messages:
        messages = {row.get("message_id"): row.get("text") for row in plan.user_messages}
        blocked = {aid for row in plan.ambiguities + [r for r in plan.candidate_missing_information
                   if r["needed_for"] == "referent_resolution"] for aid in row["aspect_ids"]}
        needs_query = any(a.task not in NON_BUSINESS_TASKS and a.aspect_id not in blocked for a in plan.aspects)
        if (len(messages) != len(plan.user_messages) or messages.get(plan.current_message_id) != request.question
                or not all(isinstance(k, str) and k and isinstance(v, str) for k, v in messages.items())
                or set(plan.context_message_ids) - (messages.keys() - {plan.current_message_id})
                or len(set(plan.context_message_ids)) != len(plan.context_message_ids)
                or (len(messages) > 1 and plan.conversation_mode == "standalone")
                or (plan.conversation_mode in {"standalone", "new_topic"} and plan.context_message_ids)
                or (plan.conversation_mode in {"followup", "correction"}
                    and (not plan.context_message_ids or (needs_query and len(plan.queries) == 1)))):
            raise AppError("invalid_query_plan", "上下文原话或引用范围无效。", 422)
        allowed = {plan.current_message_id, *plan.context_message_ids}
        for row in [a.model_dump() for a in [*plan.aspects, *plan.mentions]] + plan.ambiguities:
            mid, start, end = row.get("message_id"), row.get("start_char"), row.get("end_char")
            text = messages.get(mid, "")
            if (mid not in allowed or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text) or text[start:end] != row.get("quote")):
                raise AppError("invalid_query_plan", "计划引用与用户原话不一致。", 422)
