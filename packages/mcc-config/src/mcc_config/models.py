"""配置类型定义。

形状与 Claude Code 的 settings.json 高度兼容。
"""

from __future__ import annotations

from typing import Literal

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


class SandboxConfig(BaseModel):
    """bash 命令沙箱配置：run_bash 的物理隔离层。

    mode 取值：

    - ``none``：不隔离（默认，行为与历史版本一致）
    - ``restrictions``：纯 Python 资源限制——rlimit（内存 / CPU /
      进程数 / 文件大小）+ 环境变量白名单 + umask + HOME 重定向；
      无法限制网络（如实标注）
    - ``bubblewrap``：bwrap 命名空间隔离（文件系统只读绑定 +
      独立 /tmp + 禁网）；不可用时显式报错而非静默降级
    - ``auto``：依次探测 bubblewrap → restrictions，自动降级并
      在 stderr 提示一次

    值为 0 表示该项不限制。
    """

    mode: Literal["none", "restrictions", "bubblewrap", "auto"] = "none"
    network: Literal["off", "on"] = "off"
    max_memory_mb: int = 512
    max_cpu_secs: int = 60
    max_processes: int = 64
    max_file_mb: int = 64
    env_allow: list[str] = Field(
        default_factory=lambda: [
            "PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL",
        ]
    )


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
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
