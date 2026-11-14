"""Stable application-side model contracts, independent of serving protocols."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping
from urllib.parse import urlsplit

GenerationRole = Literal["query_plan", "answer", "verify"]
Message = Mapping[str, str]


class ModelServiceError(RuntimeError):
    """A safe, machine-readable error: never carry response bodies or credentials."""

    def __init__(self, code: str, service_name: str = "model", *,
                 finish_reason: str | None = None, usage: dict[str, int] | None = None) -> None:
        self.code = code
        self.service_name = service_name
        self.finish_reason = finish_reason
        self.usage = dict(usage or {})
        super().__init__(f"{service_name}: {code}")


class ServiceUnavailable(ModelServiceError):
    def __init__(self, service_name: str = "model", code: str = "service_not_configured") -> None:
        super().__init__(code, service_name)


@dataclass(frozen=True)
class ServiceConfig:
    """One physical endpoint; addresses and protocol paths must be configured."""

    name: str
    base_url: str = field(repr=False)
    path: str = field(repr=False)
    model: str = field(repr=False)
    timeout_seconds: float = 30.0
    max_inflight: int = 1
    auth_token_env: str | None = None
    structured_output: bool = False
    max_tokens: int = 1024
    enable_thinking: bool | None = None
    schema_unique_items: bool = True
    context_window: int = 4096
    tokenize_path: str = field(default="/tokenize", repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", self.name):
            raise ValueError("service name must be a simple identifier")
        try:
            url = urlsplit(self.base_url)
            valid = (
                url.scheme in {"http", "https"}
                and bool(url.hostname)
                and not url.username
                and not url.password
                and not url.query
                and not url.fragment
            )
            _ = url.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("base_url must be HTTP(S), without credentials, query or fragment")
        for path in (self.path, self.tokenize_path):
            if not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or any(
                c in path for c in ("?", "#", "\\")
            ):
                raise ValueError("path must be an explicit service path without query or fragment")
        if not self.model.strip():
            raise ValueError("model must be explicitly configured")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(self.max_inflight) is not int or self.max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.context_window) is not int or self.context_window <= self.max_tokens:
            raise ValueError("context_window must be an integer larger than max_tokens")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean or null")
        if type(self.schema_unique_items) is not bool:
            raise ValueError("schema_unique_items must be a boolean")


@dataclass(frozen=True)
class GenerationResult:
    content: str
    model_id: str | None
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    service_name: str = "model"
    latency_ms: float = 0.0


