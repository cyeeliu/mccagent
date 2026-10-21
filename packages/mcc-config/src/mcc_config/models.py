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


class LlmConfig(BaseModel):
    """LLM 网关配置：base_url 指向 OpenAI 兼容网关时连同 api_key 生效。

    settings.json 里持久化网关地址与凭证（本机配置， ~/.mcc 不入库），
    优先级：显式参数 > settings.json > 环境变量。
    """

    base_url: str | None = None
    api_key: str | None = None


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
    llm: LlmConfig = Field(default_factory=LlmConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
