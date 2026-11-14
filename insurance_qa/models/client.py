"""共用HTTP连接、超时和并发控制；生成客户端直接继承它。"""

from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from typing import Any, Mapping, Sequence

import httpx
from jsonschema import Draft202012Validator, ValidationError

from .contracts import GenerationResult, GenerationRole, Message, ModelServiceError, ServiceConfig, ServiceUnavailable


class PhysicalServiceClient:
    """同一服务共用一个实例；超时包含排队和HTTP请求。"""

    def __init__(self, config: ServiceConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        headers = {}
        if config.auth_token_env:
            token = os.environ.get(config.auth_token_env)
            if not token:
                raise ServiceUnavailable(config.name, "auth_not_configured")
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=httpx.Limits(max_connections=config.max_inflight),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )
        self._slots = asyncio.Semaphore(config.max_inflight)
        self._closed = False

    async def post_json(self, payload: Mapping[str, Any], *, path: str | None = None) -> dict[str, Any]:
        if self._closed:
            raise ServiceUnavailable(self.config.name, "service_client_closed")
        try:
            return await asyncio.wait_for(self._post(payload, path), self.config.timeout_seconds)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise ModelServiceError("service_timeout", self.config.name) from None
        except httpx.HTTPError:
            # 不把地址、凭据或服务端内容放进应用错误。
            raise ModelServiceError("service_transport_error", self.config.name) from None

    async def _post(self, payload: Mapping[str, Any], path: str | None) -> dict[str, Any]:
        async with self._slots:
            if self._closed:
                raise ServiceUnavailable(self.config.name, "service_client_closed")
            url = self.config.base_url.rstrip("/") + (path or self.config.path)
            response = await self._http.post(url, json=dict(payload))
            if not 200 <= response.status_code < 300:
                raise ModelServiceError(f"service_http_{response.status_code}", self.config.name)
            try:
                value = response.json()
            except ValueError:
                raise ModelServiceError("service_invalid_json", self.config.name) from None
            if not isinstance(value, dict):
                raise ModelServiceError("service_invalid_response", self.config.name)
            return value

    async def aclose(self) -> None:
        self._closed = True
        await self._http.aclose()

    async def __aenter__(self) -> PhysicalServiceClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


def _decoding_schema(value):
    """当前解码器不支持uniqueItems；完整约束仍在本地校验。"""
    if isinstance(value, dict):
        return {key: _decoding_schema(item) for key, item in value.items()
                if key != "uniqueItems" or not isinstance(item, bool)}
    if isinstance(value, list):
        return [_decoding_schema(item) for item in value]
    return value


class GenerationClient(PhysicalServiceClient):
    """兼容chat-completions；规划和回答共用本实例。"""

    async def count_tokens(self, messages: Sequence[Message]) -> int:
        """让同一模型按生成时的聊天模板计数；接口不可用时不能改用字符估算。"""
        payload = {"model": self.config.model, "messages": [dict(m) for m in messages],
                   "add_generation_prompt": True, "add_special_tokens": False}
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
        raw = await self.post_json(payload, path=self.config.tokenize_path)
        count, tokens, limit = raw.get("count"), raw.get("tokens"), raw.get("max_model_len")
        if (type(count) is not int or count < 0 or not isinstance(tokens, list) or len(tokens) != count
                or any(type(t) is not int or t < 0 for t in tokens) or type(limit) is not int or limit < 1):
            raise ModelServiceError("tokenization_invalid_response", self.config.name)
        if limit < self.config.context_window:
            raise ModelServiceError("context_window_mismatch", self.config.name)
        return count

    async def generate(self, role: GenerationRole, messages: Sequence[Message],
                       output_schema: Mapping[str, Any] | None = None, *,
                       max_tokens: int | None = None) -> GenerationResult:
        if role not in {"query_plan", "answer", "verify"}:
            raise ValueError("unsupported generation role")
        config = self.config
        limit = config.max_tokens if max_tokens is None else max_tokens
        if type(limit) is not int or not 0 < limit <= config.max_tokens:
            raise ValueError("max_tokens must be positive and within the configured output budget")
        payload = {"model": config.model, "messages": [dict(m) for m in messages],
                   "stream": False, "max_tokens": limit, "temperature": 0}
        if config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}
        if output_schema is not None and config.structured_output:
            schema = output_schema if config.schema_unique_items else _decoding_schema(output_schema)
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": role, "schema": dict(schema)}}
        start = perf_counter()
        raw = await self.post_json(payload)
        try:
            choices = raw["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            finish_reason, content = choices[0]["finish_reason"], choices[0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError
            model_id = raw.get("model")
            if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
                raise ValueError
            usage = raw.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError
            usage = {key: value for key, value in usage.items()
                     if key in {"prompt_tokens", "completion_tokens", "total_tokens"}}
            if any(type(value) is not int or value < 0 for value in usage.values()):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ModelServiceError("generation_invalid_response", config.name) from None
        if finish_reason == "length":
            raise ModelServiceError("generation_truncated", config.name, finish_reason="length", usage=usage)
        if finish_reason != "stop":
            raise ModelServiceError("generation_incomplete", config.name)
        if output_schema is not None:
            try:
                Draft202012Validator(output_schema).validate(json.loads(content))
            except (ValueError, ValidationError):
                raise ModelServiceError("generation_schema_invalid", config.name) from None
        return GenerationResult(content=content, model_id=model_id, finish_reason=finish_reason,
                                usage=usage, service_name=config.name, latency_ms=(perf_counter() - start) * 1000)


class DisabledModelClient:
    """未启用的服务明确报错，不返回伪结果。"""

    def __init__(self, service_name="model"):
        self.service_name = service_name

    async def generate(self, *args, **kwargs):
        raise ServiceUnavailable(self.service_name)

    async def count_tokens(self, *args, **kwargs):
        raise ServiceUnavailable(self.service_name)

    async def classify(self, *args, **kwargs):
        raise ServiceUnavailable(self.service_name)

    async def embed(self, *args, **kwargs):
        raise ServiceUnavailable(self.service_name)

    async def rerank(self, *args, **kwargs):
        raise ServiceUnavailable(self.service_name)

    async def aclose(self):
        pass
