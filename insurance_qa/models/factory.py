"""从配置创建一个 LLM 客户端；查询规划和答案生成共用这个实例。"""

import os
from typing import Any, Mapping

from .client import DisabledModelClient, GenerationClient
from .auxiliary import LocalEmbeddingClient, LocalIntentClient, LocalRerankClient
from .contracts import ServiceConfig, ServiceUnavailable


def service_config(name: str, raw: Mapping[str, Any]) -> ServiceConfig:
    """地址、路径、模型名和可选凭据从环境变量读取。"""
    def read(key: str, required: bool = True) -> str | None:
        variable = raw.get(key)
        if variable is not None and not isinstance(variable, str):
            raise ServiceUnavailable(name, "service_configuration_invalid")
        value = os.environ.get(variable, "").strip() if variable else ""
        if required and not value:
            raise ServiceUnavailable(name, "service_configuration_missing")
        return value or None

    base_url, path, model = read("base_url_env"), read("path_env"), read("model_env")
    auth_required = raw.get("auth_required", False)
    structured_output = raw.get("structured_output", False)
    if not isinstance(auth_required, bool) or not isinstance(structured_output, bool):
        raise ServiceUnavailable(name, "service_configuration_invalid")
    token = read("auth_token_env", required=auth_required)
    try:
        return ServiceConfig(
            name=name, base_url=base_url, path=path, model=model,
            timeout_seconds=raw.get("timeout_seconds", 30),
            max_inflight=raw.get("max_inflight", 1),
            auth_token_env=raw.get("auth_token_env") if token else None,
            structured_output=structured_output,
            max_tokens=raw.get("max_tokens", 1024),
            enable_thinking=raw.get("enable_thinking"),
            schema_unique_items=raw.get("schema_unique_items", True),
            context_window=raw.get("context_window", 4096),
            tokenize_path=raw.get("tokenize_path", "/tokenize"),
        )
    except (TypeError, ValueError):
        raise ServiceUnavailable(name, "service_configuration_invalid") from None


def create_generation_client(raw: Mapping[str, Any], name: str = "generation"):
    """应用启动时调用一次，退出时调用客户端的 aclose()。"""
    if not raw.get("enabled", False):
        return DisabledModelClient(name)
    if raw.get("adapter") != "openai_compatible":
        raise ServiceUnavailable(name, "adapter_not_configured")
    return GenerationClient(service_config(name, raw))


def create_auxiliary_client(raw: Mapping[str, Any], name: str):
    """三个能力各一个薄适配器，不使用动态注册器。"""
    if not raw.get("enabled", False):
        return DisabledModelClient(name)
    adapters = {"local_intent": LocalIntentClient, "local_embedding": LocalEmbeddingClient,
                "local_rerank": LocalRerankClient}
    adapter = adapters.get(raw.get("adapter"))
    if adapter is None:
        raise ServiceUnavailable(name, "adapter_not_configured")
    return adapter(service_config(name, raw))
