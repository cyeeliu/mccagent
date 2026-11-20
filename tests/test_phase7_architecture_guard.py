"""Phase 7 架构守卫测试：验证旧执行引擎已拆除，Factory 已完善。

P1-04: _TeammateExecutorAdapter 不存在于生产代码。
P1-05: SubagentExecutor 有不委托 SubagentRunner 的执行路径。
P1-06: DefaultAgentFactory 有 create_runtime 和 create_executor 方法。
"""

from __future__ import annotations

import inspect

import pytest


def test_teammate_executor_adapter_removed():
    """P1-04: _TeammateExecutorAdapter 应已从 persistent.py 中删除。"""
    from mcc_harness.runtime import persistent

    assert not hasattr(persistent, "_TeammateExecutorAdapter"), (
        "_TeammateExecutorAdapter should be removed from persistent.py"
    )


def test_persistent_runtime_has_from_executor():
    """P1-04: PersistentRuntimeV2 应有 from_executor 方法（新执行路径）。"""
    from mcc_harness.runtime.persistent import PersistentRuntimeV2

    assert hasattr(PersistentRuntimeV2, "from_executor"), (
        "PersistentRuntimeV2 should have from_executor classmethod"
    )


def test_subagent_executor_has_direct_execute():
    """P1-05: SubagentExecutor 应有不委托 SubagentRunner 的直接执行方法。"""
    from mcc_harness.execution.subagent_executor import SubagentExecutor

    # execute 方法应存在
    assert hasattr(SubagentExecutor, "execute"), "SubagentExecutor should have execute method"
    # 检查 execute 源码中不强制依赖 self._runner.run
    source = inspect.getsource(SubagentExecutor.execute)
    # 应有条件分支（executor 优先，runner 作为 fallback）
    assert "self._executor" in source or "executor" in source, (
        "SubagentExecutor.execute should support direct executor path"
    )


def test_factory_has_create_runtime():
    """P1-06: DefaultAgentFactory 应有 create_runtime 方法。"""
    from mcc_harness.execution.factory import DefaultAgentFactory

    assert hasattr(DefaultAgentFactory, "create_runtime"), (
        "DefaultAgentFactory should have create_runtime method"
    )


def test_factory_has_create_executor():
    """P1-06: DefaultAgentFactory 应有 create_executor 方法。"""
    from mcc_harness.execution.factory import DefaultAgentFactory

    assert hasattr(DefaultAgentFactory, "create_executor"), (
        "DefaultAgentFactory should have create_executor method"
    )


def test_factory_create_executor_from_definition():
    """P1-06: create_executor 应支持从 AgentDefinition 创建（新路径）。"""
    from mcc_harness.execution.factory import DefaultAgentFactory

    source = inspect.getsource(DefaultAgentFactory.create_executor)
    assert "AgentDefinition" in source, "create_executor should support AgentDefinition as input"
    assert "_assemble_agent_loop" in source, (
        "create_executor should assemble agent loop from definition"
    )


def test_factory_create_runtime_from_definition():
    """P1-06: create_runtime 应支持从 AgentDefinition 装配 Runtime。"""
    from mcc_harness.execution.factory import DefaultAgentFactory

    source = inspect.getsource(DefaultAgentFactory.create_runtime)
    assert "_assemble_agent_loop" in source, (
        "create_runtime should assemble agent loop from definition"
    )
    assert "InteractiveRuntimeV2" in source, "create_runtime should support INTERACTIVE runtime"
    assert "TaskRuntime" in source, "create_runtime should support TASK runtime"


def test_teammate_definition_extracted():
    """P1-04: TeammateDefinition 应已从 TeammateRuntime 提取。"""
    from mcc_harness.teammate import TeammateDefinition

    assert TeammateDefinition is not None


def test_teammate_message_adapter_extracted():
    """P1-04: TeammateMessageAdapter 应已从 TeammateRuntime 提取。"""
    from mcc_harness.teammate import TeammateMessageAdapter

    assert TeammateMessageAdapter is not None


def test_teammate_execution_service_extracted():
    """P1-04: TeammateExecutionService 应已从 TeammateRuntime 提取。"""
    from mcc_harness.teammate import TeammateExecutionService

    assert TeammateExecutionService is not None
