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
      独立 /tmp + 按需禁网）；不可用时显式报错而非静默降级。
      该档恒定做环境清洗与 HOME 重定向（真实 HOME 在沙箱内
      不可见，必须重定向）
    - ``auto``（默认）：依次探测 bubblewrap → restrictions，
      自动降级并在 stderr 提示一次

    环境策略 ``env_policy``：

    - ``inherit``（默认）：子进程继承完整环境（PATH / HOME /
      工具链变量原样）——harness 日常档，Agent 能正常使用项目
      venv、cargo 等用户工具链；隔离只剩 rlimit 防失控
    - ``allowlist``：仅白名单变量传入，PATH 重置为系统默认
      （不继承 harness 自身的 venv），HOME 可重定向，umask 077
      ——评测与高隔离档，防止被测 Agent 污染 harness 环境

    限额默认值按「防失控不挡正常」校准（编译类任务需要大内存与
    长时间 CPU）。值为 0 表示该项不限制。
    """

    mode: Literal["none", "restrictions", "bubblewrap", "auto"] = "auto"
    network: Literal["off", "on"] = "on"
    env_policy: Literal["inherit", "allowlist"] = "inherit"
    redirect_home: bool = False
    max_memory_mb: int = 2048
    max_cpu_secs: int = 600
    max_processes: int = 128
    max_file_mb: int = 1024
    env_allow: list[str] = Field(
        default_factory=lambda: [
            "PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL",
        ]
    )


class ContextConfig(BaseModel):
    """上下文压缩配置（四步管线，见 harness 的 compactor 模块）。

    ``auto_compact`` 关闭时完全不注入压缩器（行为与历史版本一致）；
    ``char_limit`` 及以下阈值与压缩模块常量同源默认。
    """

    auto_compact: bool = True
    char_limit: int = 120_000
    keep_head: int = 3
    keep_tail: int = 46
    keep_recent_results: int = 3
    large_result_chars: int = 30_000


class PermissionConfig(BaseModel):
    mode: str | None = None
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class BudgetConfig(BaseModel):
    max_usd_per_session: float = 2.0
    max_iterations: int = 40
    warn_threshold: float = 0.8
    max_usd_per_turn: float = 0.0


class ObservabilityConfig(BaseModel):
    log_format: str = "json"
    otel_endpoint: str | None = None


class HookEntry(BaseModel):
    matcher: str
    command: str


class HooksConfig(BaseModel):
    pre_tool_use: list[HookEntry] = Field(default_factory=list)
    post_tool_use: list[HookEntry] = Field(default_factory=list)


class SkillsConfig(BaseModel):
    """技能与 slash 命令加载配置。

    ``enabled`` 关闭时不扫描 skills/commands 目录、不注入 catalog、
    不注册 load_skill 工具（行为与历史版本一致）。
    """

    enabled: bool = True


class GoalConfig(BaseModel):
    """Goal Loop 配置：模型想停时由独立 evaluator 判断完成条件。

    ``enabled`` 关闭时完全不注入 GoalController（行为与历史版本一致）。
    ``evaluator_model`` 未配时回落 summarize_model → main model。
    """

    enabled: bool = True
    evaluator_model: str | None = None
    block_cap: int = 8
    max_goal_length: int = 4000


class McpServerConfig(BaseModel):
    """单个 MCP server 的启动配置（stdio transport）。

    ``command`` 为可执行文件名，``args`` 为命令行参数，``env`` 为
    额外环境变量（合入 os.environ 后传给子进程）。
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    goal: GoalConfig = Field(default_factory=GoalConfig)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
