"""配置类型定义。

形状与 Claude Code 的 settings.json 高度兼容。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """模型配置：main 为主模型，其余类型未配置时回落 main。"""

    main: str = "claude-opus-4-8"
    subagent: str | None = None
    summarize: str | None = None


class PermissionConfig(BaseModel):
    mode: str | None = None
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class BudgetConfig(BaseModel):
    max_usd_per_session: float = 2.0
    max_iterations: int = 40


class ObservabilityConfig(BaseModel):
    log_format: str = "json"
    otel_endpoint: str | None = None


class HookEntry(BaseModel):
    matcher: str
    command: str


class HooksConfig(BaseModel):
    pre_tool_use: list[HookEntry] = Field(default_factory=list)
    post_tool_use: list[HookEntry] = Field(default_factory=list)


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
