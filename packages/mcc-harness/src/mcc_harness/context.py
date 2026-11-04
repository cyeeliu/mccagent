"""AgentContext 结构化上下文：统一传递 Agent 身份、父子关系、会话、
任务、工作区、权限、预算和追踪信息。

将散落在构造参数中的上下文收敛为不可变结构化对象，
提供子上下文派生方法，支持序列化用于日志和恢复。

AgentContext 是后续分布式 Agent 执行的基础协议。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

_INHERITED_FIELDS: frozenset[str] = frozenset({
    "tenant",
    "workspace_id",
    "budget_id",
    "trace_id",
    "tags",
})

_SENSITIVE_FIELDS: frozenset[str] = frozenset({
    "permission_profile_id",
})


class ContextValidationError(ValueError):
    """上下文校验失败：必填字段缺失或值非法。"""


@dataclass(frozen=True)
class AgentContext:
    """Agent 执行上下文：不可变，受控更新。

    统一传递 Agent 身份、父子关系、会话、任务、工作区、权限、
    预算和追踪信息。所有字段在创建后不可变，通过 replace/
    derive_child/with_* 方法生成新实例。

    字段语义：
    - agent_id: Agent 唯一标识（必填）
    - parent_agent_id: 父 Agent 标识（顶层 Agent 为 None）
    - task_id: 当前关联任务标识
    - session_id: 会话标识
    - workspace_id: 工作区标识（隔离文件系统作用域）
    - permission_profile_id: 权限配置标识（敏感字段）
    - budget_id: 预算账本标识
    - trace_id: 分布式追踪标识
    - tenant: 租户标识
    - tags: 自由标签集合
    """

    agent_id: str
    parent_agent_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    workspace_id: str | None = None
    permission_profile_id: str | None = None
    budget_id: str | None = None
    trace_id: str | None = None
    tenant: str = "default"
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ContextValidationError("agent_id must not be empty")
        if self.parent_agent_id == "":
            raise ContextValidationError(
                "parent_agent_id must be None or non-empty"
            )

    def derive_child(
        self,
        child_agent_id: str,
        *,
        inherit_permissions: bool = False,
        **overrides: Any,
    ) -> AgentContext:
        """派生子 Agent 上下文。

        自动设置 parent_agent_id 为当前 agent_id，保留父子关系。
        继承字段（tenant/workspace_id/budget_id/trace_id/tags）自动传递，
        可被 overrides 覆盖。

        敏感权限字段（permission_profile_id）默认不继承，
        需显式通过 overrides 传入或设置 inherit_permissions=True。
        session_id 和 task_id 默认不继承（子 Agent 有独立会话和任务）。

        Args:
            child_agent_id: 子 Agent 唯一标识
            inherit_permissions: 是否继承父 Agent 权限配置
            **overrides: 显式覆盖的字段值

        Returns:
            子 Agent 上下文

        Raises:
            ContextValidationError: child_agent_id 为空
        """
        if not child_agent_id:
            raise ContextValidationError("child_agent_id must not be empty")

        fields: dict[str, Any] = {
            "parent_agent_id": self.agent_id,
            "task_id": None,
            "session_id": None,
            "permission_profile_id": None,
        }

        for name in _INHERITED_FIELDS:
            fields[name] = getattr(self, name)

        if inherit_permissions:
            fields["permission_profile_id"] = self.permission_profile_id

        fields.update(overrides)
        return replace(
            self,
            agent_id=child_agent_id,
            **fields,
        )

    def replace(self, **overrides: Any) -> AgentContext:
        """受控更新：生成修改部分字段后的新实例。

        用于非派生场景的字段更新（如更新 task_id/trace_id）。
        不自动修改 parent_agent_id，不涉及继承逻辑。

        Args:
            **overrides: 要修改的字段值

        Returns:
            更新后的上下文
        """
        return replace(self, **overrides)

    def with_task_id(self, task_id: str) -> AgentContext:
        """关联任务标识。"""
        return replace(self, task_id=task_id)

    def with_trace_id(self, trace_id: str) -> AgentContext:
        """设置追踪标识。"""
        return replace(self, trace_id=trace_id)

    def with_session_id(self, session_id: str) -> AgentContext:
        """设置会话标识。"""
        return replace(self, session_id=session_id)

    def with_permission_profile(
        self,
        permission_profile_id: str,
    ) -> AgentContext:
        """显式设置权限配置标识。

        权限为敏感字段，不通过 derive_child 自动继承，
        需显式调用此方法或在 overrides 中传入。
        """
        return replace(self, permission_profile_id=permission_profile_id)

    def with_tags(self, tags: frozenset[str]) -> AgentContext:
        """替换标签集合。"""
        return replace(self, tags=tags)

    def add_tag(self, tag: str) -> AgentContext:
        """添加单个标签。"""
        return replace(self, tags=self.tags | frozenset({tag}))

    def to_dict(self) -> dict[str, str | None | list[str]]:
        """序列化为字典，用于日志和恢复。

        frozenset 转为排序后的 list 以保证确定性。
        """
        return {
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "permission_profile_id": self.permission_profile_id,
            "budget_id": self.budget_id,
            "trace_id": self.trace_id,
            "tenant": self.tenant,
            "tags": sorted(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentContext:
        """从字典反序列化。

        list 转回 frozenset，缺失字段使用默认值。
        """
        tags = data.get("tags")
        return cls(
            agent_id=data["agent_id"],
            parent_agent_id=data.get("parent_agent_id"),
            task_id=data.get("task_id"),
            session_id=data.get("session_id"),
            workspace_id=data.get("workspace_id"),
            permission_profile_id=data.get("permission_profile_id"),
            budget_id=data.get("budget_id"),
            trace_id=data.get("trace_id"),
            tenant=data.get("tenant", "default"),
            tags=frozenset(tags) if tags is not None else frozenset(),
        )

    @property
    def is_root(self) -> bool:
        """是否为顶层 Agent（无父 Agent）。"""
        return self.parent_agent_id is None

    @property
    def has_permissions(self) -> bool:
        """是否已配置权限。"""
        return self.permission_profile_id is not None

    @property
    def inherited_field_names(self) -> frozenset[str]:
        """派生时自动继承的字段名集合。"""
        return _INHERITED_FIELDS

    @property
    def sensitive_field_names(self) -> frozenset[str]:
        """敏感字段名集合（不自动继承）。"""
        return _SENSITIVE_FIELDS
