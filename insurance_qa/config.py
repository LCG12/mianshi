"""Load explicit application configuration; never connect to model services here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelinePolicy:
    product_id: str = "sshy_s_hengying"
    allowed_source_policies: tuple[str, ...] = ("terms_only",)
    build_id: str | None = None
    request_timeout_seconds: float = 60.0
    session_ttl_seconds: float = 1800.0
    max_sessions: int = 100
    max_session_messages: int = 20
    write_traces: bool = False
    rerank_enabled: bool = False


@dataclass(frozen=True)
class Settings:
    project_root: Path
    policy: PipelinePolicy = field(default_factory=PipelinePolicy)
    model_services: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_roles: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def build_dir(self) -> Path:
        return self.project_root / "data" / "index" / (self.policy.build_id or "unpublished")

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置文件必须是映射：{path.name}")
    return value


def load_settings(project_root: str | Path | None = None) -> Settings:
    root = Path(project_root or os.environ.get("INSURANCE_QA_ROOT") or Path(__file__).resolve().parents[1]).resolve()
    load_dotenv(root / ".env", override=False)  # 启动环境优先，地址集中保存在项目 .env。
    policy_file = root / "configs" / "pipeline.yaml"
    models_file = root / "configs" / "models.yaml"
    raw = _mapping(policy_file)
    accepted = set(PipelinePolicy.__dataclass_fields__)
    if set(raw) - accepted:
        raise ValueError("pipeline.yaml包含未知配置项，请核对架构版本。")
    scopes = raw.get("allowed_source_policies", ["terms_only"])
    if (not isinstance(scopes, list) or not scopes or not all(isinstance(x, str) for x in scopes)
            or not set(scopes) <= {"terms_only", "official_materials"}):
        raise ValueError("来源策略必须是非空的允许列表。")
    raw["allowed_source_policies"] = tuple(scopes)
    build_id = raw.get("build_id")
    if build_id is not None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not isinstance(build_id, str) or not build_id or any(c not in allowed for c in build_id):
            raise ValueError("build_id只能包含字母、数字、下划线或短横线。")
    defaults = PipelinePolicy()
    for key in ("request_timeout_seconds", "session_ttl_seconds", "max_sessions", "max_session_messages"):
        value = raw.get(key, getattr(defaults, key))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"{key}必须是正数。")
        if key.startswith("max_") and not isinstance(value, int):
            raise ValueError(f"{key}必须是整数。")
    for key in ("write_traces", "rerank_enabled"):
        if not isinstance(raw.get(key, getattr(defaults, key)), bool):
            raise ValueError(f"{key}必须为布尔值。")
    product = raw.get("product_id", defaults.product_id)
    if not isinstance(product, str) or not product.strip():
        raise ValueError("product_id不能为空。")
    models = _mapping(models_file)
    if set(models) - {"services", "roles"}:
        raise ValueError("models.yaml包含未知配置项。")
    services, roles = models.get("services", {}), models.get("roles", {})
    if not isinstance(services, dict) or not isinstance(roles, dict):
        raise ValueError("模型services和roles必须是映射。")
    for name, service in services.items():
        if not isinstance(service, dict) or not isinstance(service.get("enabled", False), bool):
            raise ValueError("模型服务配置无效。")
        # Config stores environment variable NAMES, never the secret values.
        if any(k in service for k in ("password", "api_key", "auth_token")):
            raise ValueError("凭据只能通过环境变量引用配置。")
    for role, binding in roles.items():
        if not isinstance(binding, dict) or binding.get("service") not in services:
            raise ValueError("模型角色引用了不存在的服务。")
    return Settings(root, PipelinePolicy(**raw), services, roles)
