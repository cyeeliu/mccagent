"""AgentDefinition 与 Profile 类型（迁移方案 §3.2）。

AgentDefinition 是 Agent 的静态定义，不可变。
AgentFactory 根据 AgentDefinition 装配 AgentExecutor 和 Runtime。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mcc_core.runtime.enums import RuntimeKind
from mcc_core.runtime.identifiers import AgentId


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()
    api_base: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolProfile:
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    mcp_servers: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionProfile:
    auto_approve: frozenset[str] = field(default_factory=frozenset)
    require_approval: frozenset[str] = field(default_factory=frozenset)
    sandbox_enabled: bool = False
    network_enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionProfile:
    max_iterations: int = 10
    budget_tokens: int = 80_000
    timeout_seconds: float = 120.0
    require_plan: bool = False
    worktree_enabled: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: AgentId
    name: str
    role: str
    description: str
    model_profile: ModelProfile
    capabilities: frozenset[str] = field(default_factory=frozenset)
    tool_profile: ToolProfile = field(default_factory=ToolProfile)
    permission_profile: PermissionProfile = field(default_factory=PermissionProfile)
    execution_profile: ExecutionProfile = field(default_factory=ExecutionProfile)
    runtime_kind: RuntimeKind = RuntimeKind.INTERACTIVE
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "model_profile": {
                "model_id": self.model_profile.model_id,
                "max_tokens": self.model_profile.max_tokens,
                "temperature": self.model_profile.temperature,
            },
            "capabilities": sorted(self.capabilities),
            "runtime_kind": self.runtime_kind.value,
        }


__all__ = [
    "ModelProfile",
    "ToolProfile",
    "PermissionProfile",
    "ExecutionProfile",
    "AgentDefinition",
]
