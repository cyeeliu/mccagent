"""AgentRegistry 测试：注册、发现、心跳、过期、幂等、并发。

P2-03 验收标准：
- 可按能力筛选可用 Agent。
- 失联 Agent 不再被调度。
- 注册与注销操作幂等。
"""

from __future__ import annotations

import threading
import time

import pytest
from mcc_harness import (
    AgentAlreadyRegisteredError,
    AgentDescriptor,
    AgentNotFoundError,
    AgentRegistration,
    AgentRegistry,
    RegistrationStatus,
    RuntimeMode,
)

# -- fixture --


def make_descriptor(
    agent_id: str = "agent-1",
    agent_name: str = "worker-1",
    *,
    capabilities: frozenset[str] | None = None,
    mode: RuntimeMode = RuntimeMode.EPHEMERAL,
    tenant: str = "default",
    tags: frozenset[str] | None = None,
) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        agent_name=agent_name,
        capabilities=capabilities or frozenset({"code_generation"}),
        mode=mode,
        tenant=tenant,
        tags=tags or frozenset(),
    )


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry(heartbeat_timeout=30.0)


# -- 注册 --


class TestRegister:
    """Agent 注册。"""

    def test_register_basic(self, registry):
        desc = make_descriptor()
        reg = registry.register(desc)
        assert reg.agent_id == "agent-1"
        assert reg.agent_name == "worker-1"
        assert reg.status == RegistrationStatus.AVAILABLE
        assert reg.registered_at > 0
        assert reg.last_heartbeat_at > 0

    def test_register_idempotent_same_descriptor(self, registry):
        desc = make_descriptor()
        reg1 = registry.register(desc)
        reg2 = registry.register(desc)
        assert reg1.agent_id == reg2.agent_id
        assert reg1.registered_at == reg2.registered_at

    def test_register_conflict_different_descriptor(self, registry):
        desc1 = make_descriptor(agent_name="worker-1")
        registry.register(desc1)
        desc2 = make_descriptor(agent_name="worker-2")
        with pytest.raises(AgentAlreadyRegisteredError, match="already registered"):
            registry.register(desc2)

    def test_register_conflict_different_capabilities(self, registry):
        registry.register(make_descriptor(capabilities=frozenset({"a"})))
        with pytest.raises(AgentAlreadyRegisteredError):
            registry.register(make_descriptor(capabilities=frozenset({"b"})))

    def test_register_different_agents(self, registry):
        registry.register(make_descriptor("agent-1"))
        registry.register(make_descriptor("agent-2", "worker-2"))
        assert registry.count() == 2

    def test_register_persistent_mode(self, registry):
        desc = make_descriptor(mode=RuntimeMode.PERSISTENT)
        reg = registry.register(desc)
        assert reg.mode == RuntimeMode.PERSISTENT

    def test_register_with_tenant(self, registry):
        desc = make_descriptor(tenant="team-alpha")
        reg = registry.register(desc)
        assert reg.tenant == "team-alpha"

    def test_register_with_tags(self, registry):
        desc = make_descriptor(tags=frozenset({"gpu", "high-memory"}))
        reg = registry.register(desc)
        assert reg.descriptor.tags == frozenset({"gpu", "high-memory"})


# -- 注销 --


class TestUnregister:
    """Agent 注销（幂等）。"""

    def test_unregister_existing(self, registry):
        registry.register(make_descriptor())
        assert registry.unregister("agent-1") is True
        with pytest.raises(AgentNotFoundError):
            registry.get("agent-1")

    def test_unregister_nonexistent_idempotent(self, registry):
        assert registry.unregister("nonexistent") is False

    def test_unregister_then_re_register(self, registry):
        registry.register(make_descriptor())
        registry.unregister("agent-1")
        reg = registry.register(make_descriptor())
        assert reg.agent_id == "agent-1"

    def test_unregister_does_not_affect_others(self, registry):
        registry.register(make_descriptor("agent-1"))
        registry.register(make_descriptor("agent-2", "worker-2"))
        registry.unregister("agent-1")
        assert registry.count() == 1
        assert registry.get("agent-2").agent_id == "agent-2"


# -- 心跳 --


class TestHeartbeat:
    """Agent 心跳。"""

    def test_heartbeat_updates_timestamp(self, registry):
        registry.register(make_descriptor())
        original = registry.get("agent-1")
        time.sleep(0.01)
        reg = registry.heartbeat("agent-1")
        assert reg.last_heartbeat_at > original.last_heartbeat_at
        assert reg.status == RegistrationStatus.AVAILABLE

    def test_heartbeat_nonexistent(self, registry):
        with pytest.raises(AgentNotFoundError):
            registry.heartbeat("nonexistent")

    def test_heartbeat_restores_expired(self, registry):
        short_timeout = AgentRegistry(heartbeat_timeout=0.05)
        short_timeout.register(make_descriptor())
        time.sleep(0.1)
        short_timeout.prune_expired()
        assert short_timeout.get("agent-1").status == RegistrationStatus.EXPIRED
        reg = short_timeout.heartbeat("agent-1")
        assert reg.status == RegistrationStatus.AVAILABLE

    def test_heartbeat_restores_unavailable(self, registry):
        registry.register(make_descriptor())
        registry.mark_unavailable("agent-1")
        reg = registry.heartbeat("agent-1")
        assert reg.status == RegistrationStatus.AVAILABLE


# -- 获取 --


class TestGet:
    """Agent 获取。"""

    def test_get_existing(self, registry):
        registry.register(make_descriptor())
        reg = registry.get("agent-1")
        assert reg.agent_id == "agent-1"

    def test_get_nonexistent(self, registry):
        with pytest.raises(AgentNotFoundError):
            registry.get("nonexistent")


# -- 列表 --


class TestList:
    """Agent 列表查询。"""

    def test_list_empty(self, registry):
        assert registry.list() == []

    def test_list_all(self, registry):
        registry.register(make_descriptor("a", "A"))
        registry.register(make_descriptor("b", "B"))
        result = registry.list()
        assert len(result) == 2

    def test_list_by_tenant(self, registry):
        registry.register(make_descriptor("a", "A", tenant="team-1"))
        registry.register(make_descriptor("b", "B", tenant="team-2"))
        result = registry.list(tenant="team-1")
        assert len(result) == 1
        assert result[0].agent_id == "a"

    def test_list_by_status(self, registry):
        registry.register(make_descriptor("a", "A"))
        registry.register(make_descriptor("b", "B"))
        registry.mark_unavailable("b")
        available = registry.list(status=RegistrationStatus.AVAILABLE)
        unavailable = registry.list(status=RegistrationStatus.UNAVAILABLE)
        assert len(available) == 1
        assert len(unavailable) == 1

    def test_list_by_mode(self, registry):
        registry.register(make_descriptor("a", "A", mode=RuntimeMode.EPHEMERAL))
        registry.register(make_descriptor("b", "B", mode=RuntimeMode.PERSISTENT))
        ephemeral = registry.list(mode=RuntimeMode.EPHEMERAL)
        persistent = registry.list(mode=RuntimeMode.PERSISTENT)
        assert len(ephemeral) == 1
        assert len(persistent) == 1

    def test_list_combined_filters(self, registry):
        registry.register(
            make_descriptor("a", "A", tenant="t1", mode=RuntimeMode.PERSISTENT)
        )
        registry.register(
            make_descriptor("b", "B", tenant="t1", mode=RuntimeMode.EPHEMERAL)
        )
        registry.register(
            make_descriptor("c", "C", tenant="t2", mode=RuntimeMode.PERSISTENT)
        )
        result = registry.list(tenant="t1", mode=RuntimeMode.PERSISTENT)
        assert len(result) == 1
        assert result[0].agent_id == "a"


# -- 能力筛选 --


class TestFindByCapability:
    """按能力筛选 Agent。"""

    def test_find_single_capability(self, registry):
        registry.register(
            make_descriptor("a", "A", capabilities=frozenset({"code_generation"}))
        )
        registry.register(
            make_descriptor("b", "B", capabilities=frozenset({"code_review"}))
        )
        result = registry.find_by_capability("code_generation")
        assert len(result) == 1
        assert result[0].agent_id == "a"

    def test_find_capability_no_match(self, registry):
        registry.register(
            make_descriptor(capabilities=frozenset({"code_generation"}))
        )
        assert registry.find_by_capability("translation") == []

    def test_find_capability_multiple_matches(self, registry):
        registry.register(
            make_descriptor("a", "A", capabilities=frozenset({"code_generation"}))
        )
        registry.register(
            make_descriptor("b", "B", capabilities=frozenset({"code_generation"}))
        )
        result = registry.find_by_capability("code_generation")
        assert len(result) == 2

    def test_find_capability_with_tenant(self, registry):
        registry.register(
            make_descriptor(
                "a", "A", capabilities=frozenset({"code_generation"}), tenant="t1"
            )
        )
        registry.register(
            make_descriptor(
                "b", "B", capabilities=frozenset({"code_generation"}), tenant="t2"
            )
        )
        result = registry.find_by_capability("code_generation", tenant="t1")
        assert len(result) == 1
        assert result[0].agent_id == "a"

    def test_find_capability_excludes_unavailable(self, registry):
        registry.register(
            make_descriptor("a", "A", capabilities=frozenset({"code_generation"}))
        )
        registry.register(
            make_descriptor("b", "B", capabilities=frozenset({"code_generation"}))
        )
        registry.mark_unavailable("a")
        result = registry.find_by_capability("code_generation")
        assert len(result) == 1
        assert result[0].agent_id == "b"

    def test_find_capability_include_unavailable(self, registry):
        registry.register(
            make_descriptor("a", "A", capabilities=frozenset({"code_generation"}))
        )
        registry.mark_unavailable("a")
        result = registry.find_by_capability(
            "code_generation", only_available=False
        )
        assert len(result) == 1

    def test_find_by_capabilities_intersection(self, registry):
        registry.register(
            make_descriptor(
                "a", "A", capabilities=frozenset({"code_generation", "code_review"})
            )
        )
        registry.register(
            make_descriptor(
                "b", "B", capabilities=frozenset({"code_generation"})
            )
        )
        result = registry.find_by_capabilities({"code_generation", "code_review"})
        assert len(result) == 1
        assert result[0].agent_id == "a"

    def test_find_by_capabilities_empty(self, registry):
        registry.register(make_descriptor())
        assert registry.find_by_capabilities(set()) == []


# -- 过期清理 --


class TestPruneExpired:
    """心跳过期清理。"""

    def test_prune_marks_expired(self):
        reg = AgentRegistry(heartbeat_timeout=0.05)
        reg.register(make_descriptor("a", "A"))
        reg.register(make_descriptor("b", "B"))
        time.sleep(0.1)
        expired = reg.prune_expired()
        assert set(expired) == {"a", "b"}
        assert reg.get("a").status == RegistrationStatus.EXPIRED
        assert reg.get("b").status == RegistrationStatus.EXPIRED

    def test_prune_keeps_fresh(self):
        reg = AgentRegistry(heartbeat_timeout=10.0)
        reg.register(make_descriptor("a", "A"))
        expired = reg.prune_expired()
        assert expired == []
        assert reg.get("a").status == RegistrationStatus.AVAILABLE

    def test_prune_keeps_heartbeated(self):
        reg = AgentRegistry(heartbeat_timeout=0.1)
        reg.register(make_descriptor("a", "A"))
        reg.register(make_descriptor("b", "B"))
        time.sleep(0.08)
        reg.heartbeat("a")
        time.sleep(0.05)
        expired = reg.prune_expired()
        assert "a" not in expired
        assert "b" in expired

    def test_prune_skips_unavailable(self):
        reg = AgentRegistry(heartbeat_timeout=0.05)
        reg.register(make_descriptor("a", "A"))
        reg.mark_unavailable("a")
        time.sleep(0.1)
        expired = reg.prune_expired()
        assert expired == []
        assert reg.get("a").status == RegistrationStatus.UNAVAILABLE

    def test_prune_expired_not_scheduled(self):
        reg = AgentRegistry(heartbeat_timeout=0.05)
        reg.register(make_descriptor("a", "A"))
        time.sleep(0.1)
        reg.prune_expired()
        result = reg.find_by_capability("code_generation")
        assert result == []

    def test_is_expired(self):
        reg = AgentRegistry(heartbeat_timeout=0.05)
        reg.register(make_descriptor("a", "A"))
        assert not reg.is_expired("a")
        time.sleep(0.1)
        assert reg.is_expired("a")

    def test_is_expired_nonexistent(self, registry):
        assert not registry.is_expired("nonexistent")


# -- 手动标记不可用 --


class TestMarkUnavailable:
    """手动标记 Agent 不可用。"""

    def test_mark_unavailable(self, registry):
        registry.register(make_descriptor())
        reg = registry.mark_unavailable("agent-1")
        assert reg.status == RegistrationStatus.UNAVAILABLE
        assert not reg.is_available()

    def test_mark_unavailable_nonexistent(self, registry):
        with pytest.raises(AgentNotFoundError):
            registry.mark_unavailable("nonexistent")

    def test_mark_unavailable_excludes_from_find(self, registry):
        registry.register(
            make_descriptor(capabilities=frozenset({"code_generation"}))
        )
        registry.mark_unavailable("agent-1")
        assert registry.find_by_capability("code_generation") == []


# -- 并发 --


class TestConcurrency:
    """并发安全。"""

    def test_concurrent_register_different_agents(self):
        reg = AgentRegistry()
        errors: list[Exception] = []

        def register_agent(idx: int) -> None:
            try:
                reg.register(make_descriptor(f"agent-{idx}", f"worker-{idx}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_agent, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert reg.count() == 20

    def test_concurrent_register_same_agent_idempotent(self):
        reg = AgentRegistry()
        desc = make_descriptor()
        success_count = 0
        lock = threading.Lock()

        def register() -> None:
            nonlocal success_count
            try:
                reg.register(desc)
                with lock:
                    success_count += 1
            except AgentAlreadyRegisteredError:
                pass

        threads = [threading.Thread(target=register) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 10
        assert reg.count() == 1

    def test_concurrent_unregister_idempotent(self):
        reg = AgentRegistry()
        reg.register(make_descriptor())
        deleted_count = 0
        lock = threading.Lock()

        def unregister() -> None:
            nonlocal deleted_count
            if reg.unregister("agent-1"):
                with lock:
                    deleted_count += 1

        threads = [threading.Thread(target=unregister) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert deleted_count == 1

    def test_concurrent_heartbeat_and_prune(self):
        reg = AgentRegistry(heartbeat_timeout=0.1)
        reg.register(make_descriptor())
        errors: list[Exception] = []

        def heartbeat() -> None:
            try:
                for _ in range(5):
                    reg.heartbeat("agent-1")
                    time.sleep(0.02)
            except Exception as e:
                errors.append(e)

        def prune() -> None:
            try:
                for _ in range(5):
                    reg.prune_expired()
                    time.sleep(0.02)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=heartbeat)
        t2 = threading.Thread(target=prune)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


# -- 自定义 Repository --


class TestCustomRepository:
    """自定义 Repository 注入。"""

    def test_custom_repository(self):
        from mcc_harness import InMemoryRepository

        repo = InMemoryRepository()
        reg1 = AgentRegistry(repository=repo)
        reg2 = AgentRegistry(repository=repo)
        reg1.register(make_descriptor())
        assert reg2.get("agent-1").agent_id == "agent-1"
        assert reg2.count() == 1


# -- 属性 --


class TestProperties:
    """属性查询。"""

    def test_heartbeat_timeout(self):
        reg = AgentRegistry(heartbeat_timeout=60.0)
        assert reg.heartbeat_timeout == 60.0

    def test_count(self, registry):
        assert registry.count() == 0
        registry.register(make_descriptor("a", "A"))
        assert registry.count() == 1
        registry.register(make_descriptor("b", "B"))
        assert registry.count() == 2

    def test_clear(self, registry):
        registry.register(make_descriptor("a", "A"))
        registry.register(make_descriptor("b", "B"))
        registry.clear()
        assert registry.count() == 0

    def test_registration_properties(self, registry):
        desc = make_descriptor(
            capabilities=frozenset({"a", "b"}),
            tenant="team-1",
            mode=RuntimeMode.PERSISTENT,
        )
        reg = registry.register(desc)
        assert reg.agent_id == "agent-1"
        assert reg.agent_name == "worker-1"
        assert reg.capabilities == frozenset({"a", "b"})
        assert reg.tenant == "team-1"
        assert reg.mode == RuntimeMode.PERSISTENT
        assert reg.is_available()

    def test_descriptor_is_frozen(self):
        desc = make_descriptor()
        with pytest.raises(AttributeError):
            desc.agent_id = "mutated"  # type: ignore[misc]
