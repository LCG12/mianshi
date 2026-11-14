"""Bounded in-process conversation state with optimistic revision checks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from .errors import AppError


@dataclass(frozen=True)
class Message:
    message_id: str
    text: str
    role: str = "user"


@dataclass(frozen=True)
class SessionTicket:
    session_id: str
    revision: int
    product_id: str
    source_policy: str
    messages: tuple[Message, ...]
    pending_questions: tuple[str, ...]
    asked_questions: tuple[str, ...] = ()
    expires_at: float = 0.0


@dataclass
class _State:
    product_id: str
    source_policy: str
    touched_at: float
    revision: int = 0
    messages: list[Message] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    asked_questions: list[str] = field(default_factory=list)


class SessionStore:
    """Single-process store. No assistant answer is promoted into a user fact."""

    def __init__(self, *, ttl_seconds: float = 1800, max_sessions: int = 100,
                 max_messages: int = 20, clock: Callable[[], float] = time.monotonic):
        if ttl_seconds <= 0 or max_sessions <= 0 or max_messages <= 0:
            raise ValueError("Session limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self.clock = clock
        self._states: dict[str, _State] = {}

    def _purge(self, now):
        self._states = {sid: s for sid, s in self._states.items() if now - s.touched_at < self.ttl_seconds}

    def _check_capacity(self):
        if len(self._states) >= self.max_sessions:
            raise AppError("session_capacity", "当前会话容量已满，请稍后重试。", 503, retryable=True)

    def _check_history(self, state):
        if len(state.messages) >= self.max_messages:
            raise AppError("session_history_limit", "会话已达到轮数上限，请新建会话并补充完整问题。", 409)

    def begin(self, *, product_id: str, source_policy: str,
              session_id: str | None = None, expected_revision: int | None = None) -> SessionTicket:
        now = self.clock()
        self._purge(now)
        if session_id is None:
            if expected_revision is not None:
                raise AppError("unexpected_revision", "新会话不应包含旧修订号。", 409)
            self._check_capacity()
            # 未成功交付的请求不占会话槽位，也无需额外的abort流程。
            return SessionTicket(uuid4().hex, 0, product_id, source_policy, (), (), (), now + self.ttl_seconds)
        state = self._states.get(session_id)
        if state is None:
            raise AppError("session_expired", "会话已失效，请新建会话并补充完整问题。", 409)
        if (state.product_id, state.source_policy) != (product_id, source_policy):
            raise AppError("session_scope_conflict", "产品或来源范围已改变，请新建会话。", 409)
        if state.revision != (expected_revision if expected_revision is not None else 0):
            raise AppError("revision_conflict", "会话已更新，请使用最新修订号重试。", 409)
        self._check_history(state)
        return SessionTicket(session_id, state.revision, product_id, source_policy,
                             tuple(state.messages), tuple(state.pending_questions), tuple(state.asked_questions),
                             state.touched_at + self.ttl_seconds)

    def commit(self, ticket: SessionTicket, *, question: str, pending_questions: list[str],
               reset_questions: bool = False) -> int:
        now = self.clock()
        self._purge(now)
        state = self._states.get(ticket.session_id)
        if now >= ticket.expires_at or (state is None and ticket.revision != 0):
            raise AppError("session_expired", "会话已失效，请新建会话。", 409)
        if state is None:
            self._check_capacity()
            state = _State(ticket.product_id, ticket.source_policy, now)
        if (state.product_id, state.source_policy) != (ticket.product_id, ticket.source_policy):
            raise AppError("session_scope_conflict", "产品或来源范围已改变，请新建会话。", 409)
        if state.revision != ticket.revision:
            raise AppError("revision_conflict", "较新的请求已更新会话，本次结果未覆盖它。", 409)
        self._check_history(state)
        revision = state.revision + 1
        pending = list(pending_questions)
        asked = [] if reset_questions else state.asked_questions
        self._states[ticket.session_id] = _State(ticket.product_id, ticket.source_policy, now,
            revision=revision, messages=state.messages + [Message(f"{ticket.session_id}:u{revision}", question)],
            pending_questions=pending, asked_questions=list(dict.fromkeys(asked + pending)))
        return revision
