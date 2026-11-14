"""问答主流程：理解、检索、证据补齐、追问或生成，成功后保存用户原话。"""

import asyncio
import json
import time
from uuid import uuid4

from .answer import Answerer
from .clarification import apply_clarification, build_candidates
from .config import Settings, load_settings
from .errors import AppError
from .evidence import build_evidence
from .models.contracts import ModelServiceError
from .query_plan import NON_BUSINESS_TASKS, ModelQueryPlanner, OriginalQuestionPlanner, validate_plan
from .retrieval import HybridRetriever
from .schemas import AnswerBlock, AnswerResult, ChatRequest, Gap, IntentHints, QueryPlan
from .session import SessionStore, SessionTicket
from .store import PublishedStore


DIRECT_REPLIES = {
    "assistant_info": "我是这个项目的 AI 保险条款问答助手，可以帮你查询当前产品的保障、交费、退保等规则，并提供条款出处。",
    "chitchat": "你好！你可以继续向我咨询当前产品的保险条款。",
    "out_of_scope": "我主要帮助解答当前产品的保险条款，可以查询保障范围、保障期限、交费和退保等问题。",
}


def _small_talk(question):
    text = question.strip().rstrip("，,。.!！?？~～").strip().casefold()
    if text in {"你是谁", "你是什么", "你能做什么", "介绍一下自己", "介绍一下你自己"}:
        return DIRECT_REPLIES["assistant_info"]
    if text in {"你好", "你好呀", "您好", "嗨", "哈喽", "hello", "hi", "早上好", "下午好", "晚上好", "在吗"}:
        return "你好！你可以问我这款保险的保障范围、保障期限、交费或退保等条款问题。"
    if text in {"谢谢", "谢谢你", "多谢", "感谢"}:
        return "不客气！有其他条款问题，可以继续问我。"
    if text in {"再见", "拜拜", "bye"}:
        return "再见！有需要时可以继续咨询。"
    return None


def _direct_reply(tasks):
    return AnswerResult(answer_status="answered", coverage_status="complete", answer_blocks=[
        AnswerBlock(claim_id="direct_" + task, text=DIRECT_REPLIES[task])
        for task in dict.fromkeys(tasks)
    ])


def _subset_plan(plan, ids):
    """保留原问和历史，只筛选交给某阶段处理的目标及关联条件。"""
    def selected(rows):
        return [{**row, "aspect_ids": [a for a in row["aspect_ids"] if a in ids]}
                for row in rows if ids.intersection(row["aspect_ids"])]

    mentions = [m.model_copy(update={"aspect_ids": [a for a in m.aspect_ids if a in ids]})
                for m in plan.mentions if ids.intersection(m.aspect_ids)]
    mention_ids = {m.mention_id for m in mentions}
    return plan.model_copy(update={
        "aspects": [a for a in plan.aspects if a.aspect_id in ids], "mentions": mentions,
        "queries": [q.model_copy(update={"aspect_ids": [a for a in q.aspect_ids if a in ids],
                                         "mention_ids": [m for m in q.mention_ids if m in mention_ids]})
                    for q in plan.queries if ids.intersection(q.aspect_ids)],
        "ambiguities": selected(plan.ambiguities),
        "candidate_missing_information": selected(plan.candidate_missing_information),
    })


class Pipeline:
    def __init__(self, settings: Settings, *, store: PublishedStore,
                 planner=None, intent=None, retriever=None, answerer=None, sessions: SessionStore | None = None):
        self.settings = settings
        self.store = store
        self.planner = planner or OriginalQuestionPlanner()
        self.intent = intent
        self.retriever = retriever
        self.answerer = answerer
        policy = settings.policy
        self.sessions = sessions or SessionStore(
            ttl_seconds=policy.session_ttl_seconds,
            max_sessions=policy.max_sessions,
            max_messages=policy.max_session_messages,
        )

    def readiness(self) -> dict:
        storage = self.store.readiness()
        reasons = list(storage.get("reasons", []))
        stages = {"retrieval": self.retriever is not None and self.retriever.ready,
                  "evidence": storage["ready"], "answer_generation": self.answerer is not None,
                  "answer_compilation": True}
        for name, ready in stages.items():
            if not ready and name != "evidence":  # 证据未就绪的原因已由storage提供。
                reasons.append({"code": "stage_not_implemented", "stage": name,
                                "message": f"{name}阶段尚未接入业务实现。"})
        services = {name: {"enabled": bool(cfg.get("enabled", False)), "probed": False}
                    for name, cfg in self.settings.model_services.items()}
        ready = all(stages.values())
        if ready:
            phase = "conversational_qa"
        elif stages["evidence"] and stages["retrieval"]:
            phase = "evidence_ready"
        elif stages["retrieval"]:
            phase = "retrieval_ready"
        else:
            phase = "scaffold"
        return {"ready": ready,
                "build_id": storage.get("build_id"), "reasons": reasons,
                "stages": stages, "services": services,
                "query_understanding": {"planner": getattr(self.planner, "ready", False),
                                        "intent": self.intent is not None,
                                        "multi_turn": getattr(self.planner, "ready", False)},
                "application_phase": phase}

    def resolve_source(self, document_id: str, sha256: str):
        return self.store.resolve_source(document_id, sha256)

    async def aclose(self) -> None:
        clients = (self.intent, getattr(self.planner, "client", self.planner), self.retriever,
                   getattr(self.answerer, "client", None))
        services = {id(service): service for service in clients}
        for service in services.values():
            close = getattr(service, "aclose", None)
            if close is not None:
                await close()

    def _check_scope(self, request: ChatRequest) -> None:
        if request.product_id != self.settings.policy.product_id:
            raise AppError("unsupported_product", "该产品尚未接入当前知识库。", 422)
        if request.source_policy not in self.settings.policy.allowed_source_policies:
            raise AppError("source_policy_not_allowed", "当前应用未启用该来源范围。", 422)

    async def _classify(self, request, ticket):
        if self.intent is None:
            raise AppError("intent_unavailable", "意图服务未启用。")
        context = {"product_id": request.product_id,
                   "messages": [m.text for m in ticket.messages[-2:]],
                   "pending_questions": list(ticket.pending_questions)}
        raw = await self.intent.classify(request.question, context)
        if raw["execution_status"] != "ok":
            raise AppError("intent_degraded", "意图服务已降级。")
        return IntentHints(route=raw.get("route", "unknown"), tasks=[raw["task"]],
                           topics=raw["topics"], available=True)

    async def plan_query(self, request: ChatRequest, ticket: SessionTicket, *, intent_result=None) -> QueryPlan:
        """复用路由结果；CLI并行分类与规划，分类故障不阻断规划。"""
        self._check_scope(request)
        if (ticket.product_id, ticket.source_policy) != (request.product_id, request.source_policy):
            raise AppError("session_scope_conflict", "会话与问题的产品或来源范围不一致。", 409)
        latencies = {}

        async def measured(name, operation):
            start = time.perf_counter()
            try:
                return await asyncio.wait_for(operation, timeout=self.settings.policy.request_timeout_seconds)
            finally:
                latencies[name] = round((time.perf_counter() - start) * 1000, 3)

        if intent_result is None:
            planned, hints = await asyncio.gather(
                measured("query_plan", self.planner.plan(request, ticket)),
                measured("intent", self._classify(request, ticket)), return_exceptions=True)
        else:
            hints, latencies["intent"] = intent_result
            planned, = await asyncio.gather(
                measured("query_plan", self.planner.plan(request, ticket)), return_exceptions=True)
        if isinstance(hints, asyncio.CancelledError):
            raise hints
        try:
            if isinstance(planned, BaseException):
                raise planned
            validate_plan(planned, request, ticket)
        except Exception as exc:
            # 理解失败不能拿不完整原问继续检索；服务故障也不是用户信息不足。
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                error = AppError("request_timeout", "问题整理超时，请稍后重试。", retryable=True)
            elif isinstance(exc, ModelServiceError) or (
                    isinstance(exc, AppError) and exc.code in {"query_planning_failed", "context_budget_exceeded"}):
                error = exc
            else:
                error = AppError("query_planning_failed", "本次问题整理未完成，请重试；已有会话已保留。",
                                 retryable=True)
            trace = dict(getattr(exc, "answer_trace", {}))
            details = trace.setdefault("query_planning", {})
            details.update(stage_latencies=latencies, failure={"code": getattr(exc, "code", type(exc).__name__)})
            if isinstance(exc, AppError):
                details["failure"]["message"] = exc.message
            trace["intent"] = (hints.model_dump() if isinstance(hints, IntentHints) else
                               {"available": False, "error": getattr(hints, "code", type(hints).__name__)})
            error.answer_trace = trace
            if error is exc:
                raise
            raise error from exc
        if isinstance(hints, Exception):
            planned.diagnostics.append("intent_failed:" + getattr(hints, "code", type(hints).__name__))
            planned.execution_status = "degraded"
        else:
            planned.intent_hints = hints
            tasks = {a.task for a in planned.aspects} - {"unknown"}
            if tasks and set(hints.tasks) - tasks - {"unknown"}:
                planned.diagnostics.append("intent_task_conflict")
        planned.stage_latencies = latencies
        return planned

    async def _answer(self, request, ticket, trace):
        """业务主链路：理解 → 检索 → 证据补齐 → 追问或生成。"""
        question = request.question.strip().rstrip("。.!！")
        if ticket.pending_questions and question in {"不清楚", "不知道", "不确定", "还是不清楚", "还是不知道"}:
            trace["stages"].append("clarification_reply")
            return AnswerResult(answer_status="insufficient_evidence", gaps=[Gap(
                code="user_condition_unknown", message="关键个人条件仍不清楚，暂时无法确定适用分支；可以改问一般条款规则。")])

        start = time.perf_counter()
        try:
            hints = await self._classify(request, ticket)
        except Exception as exc:
            hints = exc  # 无法确定时继续规划，不能把服务故障或unknown当成无关问题。
        intent_ms = round((time.perf_counter() - start) * 1000, 3)
        if isinstance(hints, IntentHints) and hints.route in NON_BUSINESS_TASKS:
            trace.update(intent=hints.model_dump(), intent_latency_ms=intent_ms)
            trace["stages"].append("conversation_reply")
            return _direct_reply([hints.route])

        health = self.readiness()
        if not health["ready"]:
            return AnswerResult(execution_status="service_error",
                                gaps=[Gap(code=r["code"], message=r["message"]) for r in health["reasons"]])

        plan = await self.plan_query(request, ticket, intent_result=(hints, intent_ms))
        trace["query_plan"] = plan.model_dump()
        trace["stages"].append("query_plan")

        unresolved = plan.ambiguities + [row for row in plan.candidate_missing_information
                                         if row["needed_for"] == "referent_resolution"]
        blocked = {aid for row in unresolved for aid in row["aspect_ids"]}
        direct = [a for a in plan.aspects if a.task in NON_BUSINESS_TASKS]
        if not direct and not blocked:
            return await self._answer_business(plan, ticket, trace)
        reply = _direct_reply(a.task for a in direct)
        reply.aspect_results = [{"aspect_id": a.aspect_id, "disposition": "addressed",
                                 "claim_ids": ["direct_" + a.task], "gap_ids": []} for a in direct]
        ids = {a.aspect_id for a in plan.aspects if a.task not in NON_BUSINESS_TASKS} - blocked
        result = reply
        if ids:
            result = await self._answer_business(_subset_plan(plan, ids), ticket, trace,
                                                clarification_plan=plan, excluded_aspects=blocked)
            result.answer_blocks = reply.answer_blocks + result.answer_blocks
            result.aspect_results = reply.aspect_results + result.aspect_results
        elif not blocked:
            trace["stages"].append("conversation_reply")
        if blocked and result.execution_status in {"ok", "degraded"}:
            row = unresolved[0]
            quote = row.get("quote") or next(a.quote for a in plan.aspects if a.aspect_id in row["aspect_ids"])
            result.clarification_questions = [f"请说明“{quote}”具体指什么，以便继续回答。"]
            result.answer_status = "needs_clarification"
            result.aspect_results += [{"aspect_id": aid, "disposition": "needs_user_info",
                                       "claim_ids": [], "gap_ids": []} for aid in sorted(blocked)]
            result.trace["clarification"] = {"kind": "referent_resolution", "aspect_ids": sorted(blocked)}
            result.coverage_status = "partial" if result.answer_blocks else "none"
            trace["stages"].append("referent_clarification")
        elif reply.answer_blocks and result.coverage_status == "none":
            result.coverage_status = "partial"
        return result

    async def _answer_business(self, plan, ticket, trace, *, clarification_plan=None, excluded_aspects=()):
        snapshot = self.store.snapshot()
        recalled = await self.retriever.search_with_trace(plan, snapshot)
        trace["retrieval"] = recalled["diagnostics"]
        trace["stages"].append("retrieval")

        pack = await asyncio.to_thread(
            build_evidence, plan, recalled["hits"], snapshot,
            max_chars=100000, project_root=self.settings.project_root,
            candidates=recalled.get("candidates", []),
        )
        trace["evidence_selection"] = pack.selection
        trace["stages"].append("evidence")

        if plan.conversation_mode == "new_topic":
            pending, asked = (), ()
        else:
            pending, asked = ticket.pending_questions, ticket.asked_questions
        missing = build_candidates(clarification_plan or plan, pack, pending_questions=pending,
                                   exclude_aspect_ids=excluded_aspects)
        candidates = [c for c in missing if c.question not in asked]
        pack.unresolved_user_conditions = [
            {"aspect_ids": list(c.aspect_ids), "target_fields": list(c.target_fields)}
            for c in missing
        ]
        blocked = {aid for c in missing for aid in c.aspect_ids}
        if blocked == {a.aspect_id for a in plan.aspects}:
            result = AnswerResult(versions=pack.versions, aspect_results=[
                {"aspect_id": a.aspect_id, "disposition": "needs_user_info", "claim_ids": [], "gap_ids": []}
                for a in plan.aspects
            ])
            if not candidates:
                result.answer_status = "insufficient_evidence"
                result.gaps = [Gap(
                    code="user_condition_unknown",
                    message="关键个人条件仍不清楚，暂时无法确定适用分支；可以改问一般条款规则。",
                    aspect_ids=sorted(blocked),
                )]
        else:
            result = await self.answerer.answer(plan, pack)
            trace["stages"].extend(["answer_generation", "answer_compilation"])

        if result.execution_status in {"ok", "degraded"}:
            apply_clarification(result, candidates)
            if candidates:
                trace["stages"].append("clarification")
        upstream_degraded = "degraded" in (plan.execution_status, recalled["diagnostics"]["execution_status"])
        if upstream_degraded and result.execution_status == "ok":
            result.execution_status = "degraded"
        return result

    async def ask(self, request: ChatRequest) -> AnswerResult:
        """管理超时与会话提交；失败不更新会话，模型答案不存为用户事实。"""
        self._check_scope(request)
        start = time.perf_counter()
        reply = _small_talk(request.question)
        trace = {"stages": [], "source_policy": request.source_policy, "product_id": request.product_id}

        if reply:
            # 常见寒暄和身份询问不调用模型，不写业务历史或清掉待答追问。
            result = AnswerResult(answer_status="answered", coverage_status="complete",
                                  answer_blocks=[AnswerBlock(claim_id="small_talk", text=reply)])
            trace["stages"].append("small_talk")
            if request.session_id is not None or request.expected_revision is not None:
                ticket = self.sessions.begin(product_id=request.product_id, source_policy=request.source_policy,
                    session_id=request.session_id, expected_revision=request.expected_revision)
                result.session_id, result.revision = ticket.session_id, ticket.revision
        else:
            ticket = self.sessions.begin(product_id=request.product_id, source_policy=request.source_policy,
                session_id=request.session_id, expected_revision=request.expected_revision)
            try:
                result = await asyncio.wait_for(
                    self._answer(request, ticket, trace), self.settings.policy.request_timeout_seconds,
                )
            except (AppError, ModelServiceError, TimeoutError, asyncio.TimeoutError) as exc:
                trace.update(getattr(exc, "answer_trace", {}))
                if isinstance(exc, AppError) and exc.status_code == 409:
                    raise
                code = getattr(exc, "code", "request_timeout")
                message = exc.message if isinstance(exc, AppError) else "模型服务调用未完成，请稍后重试。"
                status = "validation_error" if isinstance(exc, AppError) and exc.status_code < 500 else "service_error"
                result = AnswerResult(execution_status=status, gaps=[Gap(code=code, message=message)])
            if (result.execution_status in {"ok", "degraded"} and result.answer_status is not None
                    and "conversation_reply" not in trace["stages"]):
                pending = result.clarification_questions
                if "clarification_reply" in trace["stages"]:
                    pending = list(ticket.pending_questions)
                if "referent_clarification" in trace["stages"]:
                    pending = list(dict.fromkeys([*ticket.pending_questions, *pending]))
                result.revision = self.sessions.commit(ticket, question=request.question, pending_questions=pending,
                    reset_questions=trace.get("query_plan", {}).get("conversation_mode") == "new_topic"
                    and "referent_clarification" not in trace["stages"])
                result.session_id = ticket.session_id
            if result.session_id is None and ticket.revision:
                result.session_id, result.revision = ticket.session_id, ticket.revision
        result.trace_id = uuid4().hex
        result.trace.update(trace, total_latency_ms=round((time.perf_counter() - start) * 1000, 3))
        if self.settings.policy.write_traces:
            self._write_trace(result)
        return result

    def _write_trace(self, result):
        try:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            record = {"trace_id": result.trace_id, "execution_status": result.execution_status,
                      "versions": result.versions, "trace": result.trace,
                      "gaps": [gap.model_dump() for gap in result.gaps]}
            with (self.settings.output_dir / "traces.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            result.trace["trace_persistence"] = "failed"


def create_pipeline(settings: Settings | None = None, *, rerank: bool | None = None) -> Pipeline:
    from .models.factory import create_auxiliary_client, create_generation_client

    settings = settings or load_settings()
    clients = {}

    def generation(role):
        name = settings.model_roles.get(role, {}).get("service", "generation")
        raw = settings.model_services.get(name, {})
        if not raw.get("enabled"):
            return None
        if name not in clients:
            clients[name] = create_generation_client(raw, name)
        return clients[name]

    planning_client, answer_client = generation("query_plan"), generation("answer")
    planner = ModelQueryPlanner(planning_client, settings.project_root) if planning_client else None
    answerer = Answerer(answer_client, settings.project_root) if answer_client else None
    intent = None
    name = settings.model_roles.get("intent", {}).get("service", "intent")
    raw = settings.model_services.get(name, {})
    if raw.get("enabled"):
        intent = create_auxiliary_client(raw, name)
    retriever = create_retriever(settings, rerank=rerank) if settings.policy.build_id else None
    return Pipeline(settings, store=PublishedStore(settings.project_root, settings.build_dir, settings.policy.build_id),
                    planner=planner, intent=intent, retriever=retriever, answerer=answerer)


def create_retriever(settings: Settings, *, rerank: bool | None = None) -> HybridRetriever:
    """CLI与在线流程共用检索器；模型客户端随检索器关闭。"""
    from .models.factory import create_auxiliary_client

    enabled = settings.policy.rerank_enabled if rerank is None else rerank
    embedding_name = settings.model_roles["embedding"]["service"]
    embedding = create_auxiliary_client(settings.model_services[embedding_name], embedding_name)
    reranker = None
    if enabled:
        name = settings.model_roles["rerank"]["service"]
        reranker = create_auxiliary_client(settings.model_services[name], name)
    return HybridRetriever(embedding, reranker, rerank=enabled)
