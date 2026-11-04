"""AgentContext 测试：创建、派生、隔离、序列化、敏感字段继承。

P2-04 验收标准：
- 所有 Agent 创建入口可生成合法 Context。
- 上下文可序列化用于日志和恢复。
- 测试派生、隔离、序列化和敏感字段继承。
"""

from __future__ import annotations

import pytest
from mcc_harness import AgentContext, ContextValidationError

# -- fixture --


@pytest.fixture
def root_context() -> AgentContext:
    return AgentContext(
        agent_id="lead-1",
        task_id="task-100",
        session_id="session-1",
        workspace_id="ws-main",
        permission_profile_id="admin",
        budget_id="budget-1",
        trace_id="trace-abc",
        tenant="team-alpha",
        tags=frozenset({"lead", "gpu"}),
    )


# -- 创建与校验 --


class TestCreation:
    """AgentContext 创建与校验。"""

    def test_create_minimal(self):
        ctx = AgentContext(agent_id="agent-1")
        assert ctx.agent_id == "agent-1"
        assert ctx.parent_agent_id is None
        assert ctx.task_id is None
        assert ctx.tenant == "default"
        assert ctx.tags == frozenset()

    def test_create_full(self, root_context):
        ctx = root_context
        assert ctx.agent_id == "lead-1"
        assert ctx.task_id == "task-100"
        assert ctx.session_id == "session-1"
        assert ctx.workspace_id == "ws-main"
        assert ctx.permission_profile_id == "admin"
        assert ctx.budget_id == "budget-1"
        assert ctx.trace_id == "trace-abc"
        assert ctx.tenant == "team-alpha"
        assert ctx.tags == frozenset({"lead", "gpu"})

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ContextValidationError, match="agent_id"):
            AgentContext(agent_id="")

    def test_empty_parent_agent_id_rejected(self):
        with pytest.raises(ContextValidationError, match="parent_agent_id"):
            AgentContext(agent_id="a", parent_agent_id="")

    def test_none_parent_agent_id_allowed(self):
        ctx = AgentContext(agent_id="a", parent_agent_id=None)
        assert ctx.parent_agent_id is None
        assert ctx.is_root


# -- 不可变与隔离 --


class TestImmutability:
    """AgentContext 不可变与隔离。"""

    def test_frozen(self, root_context):
        with pytest.raises(AttributeError):
            root_context.agent_id = "mutated"  # type: ignore[misc]

    def test_frozen_tags(self, root_context):
        with pytest.raises(AttributeError):
            root_context.tags.add("new")  # type: ignore[union-attr]

    def test_replace_creates_new_instance(self, root_context):
        new_ctx = root_context.replace(task_id="task-200")
        assert new_ctx is not root_context
        assert new_ctx.task_id == "task-200"
        assert root_context.task_id == "task-100"

    def test_replace_preserves_other_fields(self, root_context):
        new_ctx = root_context.replace(task_id="task-200")
        assert new_ctx.agent_id == root_context.agent_id
        assert new_ctx.tenant == root_context.tenant
        assert new_ctx.tags == root_context.tags

    def test_child_does_not_mutate_parent(self, root_context):
        child = root_context.derive_child("child-1")
        assert root_context.task_id == "task-100"
        assert child.task_id is None
        assert root_context.session_id == "session-1"
        assert child.session_id is None


# -- 派生 --


class TestDeriveChild:
    """子上下文派生。"""

    def test_sets_parent_agent_id(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.agent_id == "child-1"
        assert child.parent_agent_id == "lead-1"
        assert not child.is_root

    def test_inherits_workspace(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.workspace_id == "ws-main"

    def test_inherits_tenant(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.tenant == "team-alpha"

    def test_inherits_budget_id(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.budget_id == "budget-1"

    def test_inherits_trace_id(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.trace_id == "trace-abc"

    def test_inherits_tags(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.tags == frozenset({"lead", "gpu"})

    def test_does_not_inherit_session_id(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.session_id is None

    def test_does_not_inherit_task_id(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.task_id is None

    def test_permission_not_inherited_by_default(self, root_context):
        child = root_context.derive_child("child-1")
        assert child.permission_profile_id is None
        assert not child.has_permissions

    def test_permission_inherited_with_flag(self, root_context):
        child = root_context.derive_child("child-1", inherit_permissions=True)
        assert child.permission_profile_id == "admin"
        assert child.has_permissions

    def test_override_inherited_field(self, root_context):
        child = root_context.derive_child(
            "child-1", workspace_id="ws-isolated"
        )
        assert child.workspace_id == "ws-isolated"

    def test_override_permission(self, root_context):
        child = root_context.derive_child(
            "child-1", permission_profile_id="readonly"
        )
        assert child.permission_profile_id == "readonly"

    def test_override_takes_precedence_over_inherit_permissions(
        self, root_context
    ):
        child = root_context.derive_child(
            "child-1",
            inherit_permissions=True,
            permission_profile_id="custom",
        )
        assert child.permission_profile_id == "custom"

    def test_override_task_id(self, root_context):
        child = root_context.derive_child("child-1", task_id="task-200")
        assert child.task_id == "task-200"

    def test_override_session_id(self, root_context):
        child = root_context.derive_child(
            "child-1", session_id="session-child"
        )
        assert child.session_id == "session-child"

    def test_empty_child_id_rejected(self, root_context):
        with pytest.raises(ContextValidationError, match="child_agent_id"):
            root_context.derive_child("")

    def test_grandchild_chain(self, root_context):
        child = root_context.derive_child("child-1")
        grandchild = child.derive_child("grandchild-1")
        assert grandchild.agent_id == "grandchild-1"
        assert grandchild.parent_agent_id == "child-1"
        assert grandchild.tenant == "team-alpha"
        assert grandchild.workspace_id == "ws-main"

    def test_derive_from_minimal(self):
        ctx = AgentContext(agent_id="a")
        child = ctx.derive_child("b")
        assert child.agent_id == "b"
        assert child.parent_agent_id == "a"
        assert child.tenant == "default"
        assert child.workspace_id is None


# -- 敏感字段 --


class TestSensitiveFields:
    """敏感字段继承控制。"""

    def test_sensitive_field_names(self, root_context):
        assert "permission_profile_id" in root_context.sensitive_field_names

    def test_inherited_field_names(self, root_context):
        names = root_context.inherited_field_names
        assert "tenant" in names
        assert "workspace_id" in names
        assert "budget_id" in names
        assert "trace_id" in names
        assert "tags" in names
        assert "permission_profile_id" not in names

    def test_with_permission_profile(self, root_context):
        ctx = root_context.with_permission_profile("restricted")
        assert ctx.permission_profile_id == "restricted"
        assert root_context.permission_profile_id == "admin"

    def test_permission_not_auto_elevated(self, root_context):
        child = root_context.derive_child("child-1")
        grandchild = child.derive_child("grandchild-1")
        assert grandchild.permission_profile_id is None

    def test_explicit_permission_inheritance_chain(self, root_context):
        child = root_context.derive_child(
            "child-1", inherit_permissions=True
        )
        grandchild = child.derive_child(
            "grandchild-1", inherit_permissions=True
        )
        assert grandchild.permission_profile_id == "admin"


# -- 链式方法 --


class TestChainMethods:
    """链式更新方法。"""

    def test_with_task_id(self, root_context):
        ctx = root_context.with_task_id("task-new")
        assert ctx.task_id == "task-new"

    def test_with_trace_id(self, root_context):
        ctx = root_context.with_trace_id("trace-new")
        assert ctx.trace_id == "trace-new"

    def test_with_session_id(self, root_context):
        ctx = root_context.with_session_id("session-new")
        assert ctx.session_id == "session-new"

    def test_with_tags(self, root_context):
        ctx = root_context.with_tags(frozenset({"a", "b"}))
        assert ctx.tags == frozenset({"a", "b"})

    def test_add_tag(self, root_context):
        ctx = root_context.add_tag("high-priority")
        assert "high-priority" in ctx.tags
        assert "lead" in ctx.tags
        assert root_context.tags == frozenset({"lead", "gpu"})

    def test_add_tag_idempotent(self, root_context):
        ctx = root_context.add_tag("lead")
        assert ctx.tags == frozenset({"lead", "gpu"})

    def test_chain_multiple(self, root_context):
        ctx = (
            root_context
            .with_task_id("task-2")
            .with_trace_id("trace-2")
            .add_tag("urgent")
        )
        assert ctx.task_id == "task-2"
        assert ctx.trace_id == "trace-2"
        assert "urgent" in ctx.tags


# -- 序列化 --


class TestSerialization:
    """序列化与反序列化。"""

    def test_to_dict_basic(self, root_context):
        d = root_context.to_dict()
        assert d["agent_id"] == "lead-1"
        assert d["parent_agent_id"] is None
        assert d["task_id"] == "task-100"
        assert d["tenant"] == "team-alpha"

    def test_to_dict_tags_sorted(self, root_context):
        d = root_context.to_dict()
        assert d["tags"] == ["gpu", "lead"]

    def test_from_dict_roundtrip(self, root_context):
        d = root_context.to_dict()
        restored = AgentContext.from_dict(d)
        assert restored == root_context

    def test_from_dict_minimal(self):
        d = {"agent_id": "agent-1"}
        ctx = AgentContext.from_dict(d)
        assert ctx.agent_id == "agent-1"
        assert ctx.tenant == "default"
        assert ctx.tags == frozenset()

    def test_from_dict_with_tags(self):
        d = {"agent_id": "a", "tags": ["x", "y"]}
        ctx = AgentContext.from_dict(d)
        assert ctx.tags == frozenset({"x", "y"})

    def test_from_dict_empty_tags(self):
        d = {"agent_id": "a", "tags": []}
        ctx = AgentContext.from_dict(d)
        assert ctx.tags == frozenset()

    def test_serialization_preserves_parent(self, root_context):
        child = root_context.derive_child("child-1")
        d = child.to_dict()
        assert d["parent_agent_id"] == "lead-1"
        restored = AgentContext.from_dict(d)
        assert restored.parent_agent_id == "lead-1"

    def test_serialization_preserves_permissions(self, root_context):
        d = root_context.to_dict()
        assert d["permission_profile_id"] == "admin"
        restored = AgentContext.from_dict(d)
        assert restored.permission_profile_id == "admin"

    def test_serialization_child_no_permissions(self, root_context):
        child = root_context.derive_child("child-1")
        d = child.to_dict()
        assert d["permission_profile_id"] is None


# -- 属性 --


class TestProperties:
    """属性查询。"""

    def test_is_root(self, root_context):
        assert root_context.is_root
        child = root_context.derive_child("child-1")
        assert not child.is_root

    def test_has_permissions(self, root_context):
        assert root_context.has_permissions
        child = root_context.derive_child("child-1")
        assert not child.has_permissions

    def test_has_permissions_no_profile(self):
        ctx = AgentContext(agent_id="a")
        assert not ctx.has_permissions
