"""Agent 主循环、权限、hooks、skills、subagent。

驱动 LLM 调用与工具执行的迭代循环，集成权限检查与事件分发。
"""

from __future__ import annotations

from mcc_harness.agent import (
    AgentLoop,
    AgentLoopBuilder,
    AgentRun,
    model_price_per_1m,
    permission_request_for,
    price_usd,
)
from mcc_harness.hooks import (
    Hook,
    HookContext,
    HookRegistry,
    HookResult,
    HookType,
    PythonHook,
    ShellHook,
)
from mcc_harness.permission import (
    Action,
    Allow,
    Ask,
    BashAction,
    Decision,
    Deny,
    PathAction,
    PermissionChecker,
    PermissionRequest,
    bash_word_match,
)
from mcc_harness.subagent import (
    SpawnSubagentTool,
    SubagentResult,
    SubagentRunner,
    SubagentSpec,
    default_presets,
    token_budget_to_usd,
)

__all__ = [
    "Action",
    "AgentLoop",
    "AgentLoopBuilder",
    "AgentRun",
    "Allow",
    "Ask",
    "BashAction",
    "Decision",
    "Deny",
    "Hook",
    "HookContext",
    "HookRegistry",
    "HookResult",
    "HookType",
    "PathAction",
    "PermissionChecker",
    "PermissionRequest",
    "PythonHook",
    "ShellHook",
    "SpawnSubagentTool",
    "SubagentResult",
    "SubagentRunner",
    "SubagentSpec",
    "bash_word_match",
    "default_presets",
    "model_price_per_1m",
    "permission_request_for",
    "price_usd",
    "token_budget_to_usd",
]
__version__ = "0.1.0"
