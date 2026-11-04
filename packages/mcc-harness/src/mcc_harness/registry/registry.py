"""AgentRegistry 内存实现：注册、发现、心跳与过期清理。

提供 Agent 注册、能力发现、状态查询和生命周期索引。
通过心跳过期策略自动标记失联 Agent 为不可用。
注册与注销操作幂等。
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .base import (
    AgentAlreadyRegisteredError,
    AgentDescriptor,
    AgentNotFoundError,
    AgentRegistration,
    InMemoryRepository,
    RegistrationStatus,
    RegistryRepository,
    _now,
)


class AgentRegistry:
    """Agent 注册表：注册、发现、心跳与过期管理。

    职责：
    - 维护 Agent 注册信息（描述 + 心跳 + 状态）
    - 按能力/租户/标签筛选可用 Agent
    - 心跳过期自动标记不可用
    - 注册与注销幂等

    不负责调度决策。调度器通过 find_by_capability 获取候选列表
    后自行决定调度策略。
    """

    def __init__(
        self,
        *,
        repository: RegistryRepository | None = None,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        self._repo: RegistryRepository = repository or InMemoryRepository()
        self._heartbeat_timeout = heartbeat_timeout
        self._lock = threading.Lock()

    @property
    def heartbeat_timeout(self) -> float:
        """心跳超时时间（秒）。"""
        return self._heartbeat_timeout

    def register(self, descriptor: AgentDescriptor) -> AgentRegistration:
        """注册 Agent（幂等）。

        若 agent_id 已存在且描述信息相同，返回已有注册记录。
        若描述信息不同，抛出 AgentAlreadyRegisteredError。

        Args:
            descriptor: Agent 描述信息

        Returns:
            注册记录

        Raises:
            AgentAlreadyRegisteredError: agent_id 已被占用且描述不同
        """
        now = _now()
        with self._lock:
            existing = self._repo.load(descriptor.agent_id)
            if existing is not None:
                if existing.descriptor != descriptor:
                    raise AgentAlreadyRegisteredError(
                        f"agent {descriptor.agent_id} already registered "
                        f"with different descriptor"
                    )
                return existing
            registration = AgentRegistration(
                descriptor=descriptor,
                registered_at=now,
                last_heartbeat_at=now,
                status=RegistrationStatus.AVAILABLE,
            )
            self._repo.save(registration)
            return registration

    def unregister(self, agent_id: str) -> bool:
        """注销 Agent（幂等）。

        若 agent_id 不存在，静默返回 False。
        若存在，删除并返回 True。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            是否实际删除
        """
        with self._lock:
            return self._repo.delete(agent_id)

    def heartbeat(self, agent_id: str) -> AgentRegistration:
        """更新 Agent 心跳时间。

        若 Agent 已过期（EXPIRED），心跳恢复其状态为 AVAILABLE。
        若 Agent 不存在，抛出 AgentNotFoundError。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            更新后的注册记录

        Raises:
            AgentNotFoundError: Agent 不在注册表中
        """
        now = _now()
        with self._lock:
            reg = self._repo.load(agent_id)
            if reg is None:
                raise AgentNotFoundError(agent_id)
            updated = AgentRegistration(
                descriptor=reg.descriptor,
                registered_at=reg.registered_at,
                last_heartbeat_at=now,
                status=RegistrationStatus.AVAILABLE,
            )
            self._repo.save(updated)
            return updated

    def mark_unavailable(self, agent_id: str) -> AgentRegistration:
        """手动标记 Agent 不可用。

        用于 Agent 主动下线或调度器探测到不可达时调用。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            更新后的注册记录

        Raises:
            AgentNotFoundError: Agent 不在注册表中
        """
        with self._lock:
            reg = self._repo.load(agent_id)
            if reg is None:
                raise AgentNotFoundError(agent_id)
            updated = AgentRegistration(
                descriptor=reg.descriptor,
                registered_at=reg.registered_at,
                last_heartbeat_at=reg.last_heartbeat_at,
                status=RegistrationStatus.UNAVAILABLE,
            )
            self._repo.save(updated)
            return updated

    def get(self, agent_id: str) -> AgentRegistration:
        """获取 Agent 注册记录。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            注册记录

        Raises:
            AgentNotFoundError: Agent 不在注册表中
        """
        with self._lock:
            reg = self._repo.load(agent_id)
            if reg is None:
                raise AgentNotFoundError(agent_id)
            return reg

    def list(
        self,
        *,
        tenant: str | None = None,
        status: RegistrationStatus | None = None,
        mode: str | None = None,
    ) -> list[AgentRegistration]:
        """列出注册记录，支持按租户/状态/模式筛选。

        Args:
            tenant: 租户标识（None 表示不筛选）
            status: 注册状态（None 表示不筛选）
            mode: 运行模式（None 表示不筛选）

        Returns:
            注册记录列表
        """
        with self._lock:
            records = self._repo.list_all()
        result: list[AgentRegistration] = []
        for reg in records:
            if tenant is not None and reg.tenant != tenant:
                continue
            if status is not None and reg.status != status:
                continue
            if mode is not None and reg.mode != mode:
                continue
            result.append(reg)
        return result

    def find_by_capability(
        self,
        capability: str,
        *,
        tenant: str | None = None,
        only_available: bool = True,
    ) -> list[AgentRegistration]:
        """按能力筛选 Agent。

        Args:
            capability: 能力标识
            tenant: 租户标识（None 表示不筛选）
            only_available: 是否仅返回可用 Agent

        Returns:
            匹配的注册记录列表
        """
        with self._lock:
            records = self._repo.list_all()
        result: list[AgentRegistration] = []
        for reg in records:
            if capability not in reg.capabilities:
                continue
            if tenant is not None and reg.tenant != tenant:
                continue
            if only_available and not reg.is_available():
                continue
            result.append(reg)
        return result

    def find_by_capabilities(
        self,
        capabilities: Iterable[str],
        *,
        tenant: str | None = None,
        only_available: bool = True,
    ) -> list[AgentRegistration]:
        """按多个能力筛选 Agent（需具备全部指定能力）。

        Args:
            capabilities: 能力标识集合
            tenant: 租户标识（None 表示不筛选）
            only_available: 是否仅返回可用 Agent

        Returns:
            匹配的注册记录列表
        """
        required = frozenset(capabilities)
        if not required:
            return []
        with self._lock:
            records = self._repo.list_all()
        result: list[AgentRegistration] = []
        for reg in records:
            if not required.issubset(reg.capabilities):
                continue
            if tenant is not None and reg.tenant != tenant:
                continue
            if only_available and not reg.is_available():
                continue
            result.append(reg)
        return result

    def prune_expired(self) -> list[str]:
        """清理心跳超时的 Agent，返回被清理的 agent_id 列表。

        将超时 Agent 标记为 EXPIRED（不删除记录），
        使其不再被 find_by_capability 返回。
        已在终态（UNAVAILABLE）的 Agent 不受影响。

        Returns:
            被标记为 EXPIRED 的 agent_id 列表
        """
        now = _now()
        expired_ids: list[str] = []
        with self._lock:
            records = self._repo.list_all()
            for reg in records:
                if reg.status != RegistrationStatus.AVAILABLE:
                    continue
                if now - reg.last_heartbeat_at > self._heartbeat_timeout:
                    updated = AgentRegistration(
                        descriptor=reg.descriptor,
                        registered_at=reg.registered_at,
                        last_heartbeat_at=reg.last_heartbeat_at,
                        status=RegistrationStatus.EXPIRED,
                    )
                    self._repo.save(updated)
                    expired_ids.append(reg.agent_id)
        return expired_ids

    def is_expired(self, agent_id: str) -> bool:
        """检查 Agent 是否已心跳超时。

        Args:
            agent_id: Agent 唯一标识

        Returns:
            是否超时（不存在也返回 False）
        """
        now = _now()
        with self._lock:
            reg = self._repo.load(agent_id)
            if reg is None:
                return False
            return now - reg.last_heartbeat_at > self._heartbeat_timeout

    def count(self) -> int:
        """注册总数。"""
        with self._lock:
            return len(self._repo.list_all())

    def clear(self) -> None:
        """清空所有注册记录。"""
        with self._lock:
            for reg in list(self._repo.list_all()):
                self._repo.delete(reg.agent_id)
