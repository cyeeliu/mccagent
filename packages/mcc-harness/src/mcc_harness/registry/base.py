"""AgentRegistry 类型定义与 Repository 协议。

为 Agent 提供注册、能力发现、状态查询和生命周期索引。
Registry 不负责调度决策，仅维护注册信息和心跳状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from mcc_harness.runtime import RuntimeMode


class RegistrationStatus(StrEnum):
    """注册状态：反映 Agent 当前可用性。"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"


class AgentAlreadyRegisteredError(RuntimeError):
    """注册冲突：agent_id 已被占用且描述信息不同。"""


class AgentNotFoundError(KeyError):
    """Agent 不在注册表中。"""


class AgentUnavailableError(RuntimeError):
    """Agent 已注册但不可用（失联或过期）。"""


@dataclass(frozen=True)
class AgentDescriptor:
    """Agent 描述信息：注册时提供，不可变。

    区分 agent_id（全局唯一标识）与 agent_name（人类可读名称）。
    capabilities 为能力集合，用于能力发现和筛选。
    tenant 为租户标识，支持租户内唯一性。
    tags 为自由标签，供业务层筛选。
    mode 为运行模式，复用 RuntimeMode。
    """

    agent_id: str
    agent_name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    mode: RuntimeMode = RuntimeMode.EPHEMERAL
    tenant: str = "default"
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentRegistration:
    """Agent 注册记录：描述信息 + 生命周期时间戳 + 状态。

    registered_at 为注册时间，last_heartbeat_at 为最近心跳时间。
    status 反映当前可用性，由心跳过期策略驱动。
    """

    descriptor: AgentDescriptor
    registered_at: float
    last_heartbeat_at: float
    status: RegistrationStatus = RegistrationStatus.AVAILABLE

    @property
    def agent_id(self) -> str:
        return self.descriptor.agent_id

    @property
    def agent_name(self) -> str:
        return self.descriptor.agent_name

    @property
    def capabilities(self) -> frozenset[str]:
        return self.descriptor.capabilities

    @property
    def tenant(self) -> str:
        return self.descriptor.tenant

    @property
    def mode(self) -> RuntimeMode:
        return self.descriptor.mode

    def is_available(self) -> bool:
        """是否可被调度。"""
        return self.status == RegistrationStatus.AVAILABLE


class RegistryRepository(Protocol):
    """Registry 存储协议：预留接口供分布式实现替换。

    单机版本用内存 dict 实现；分布式版本可替换为 Redis/PG。
    所有方法由调用方保证线程安全。
    """

    def save(self, registration: AgentRegistration) -> None:
        """写入或更新注册记录。"""
        ...

    def load(self, agent_id: str) -> AgentRegistration | None:
        """按 agent_id 读取注册记录，不存在返回 None。"""
        ...

    def delete(self, agent_id: str) -> bool:
        """删除注册记录，返回是否实际删除。"""
        ...

    def list_all(self) -> list[AgentRegistration]:
        """列出全部注册记录。"""
        ...


class InMemoryRepository:
    """内存存储实现：dict + 直接读写。"""

    def __init__(self) -> None:
        self._store: dict[str, AgentRegistration] = {}

    def save(self, registration: AgentRegistration) -> None:
        self._store[registration.agent_id] = registration

    def load(self, agent_id: str) -> AgentRegistration | None:
        return self._store.get(agent_id)

    def delete(self, agent_id: str) -> bool:
        return self._store.pop(agent_id, None) is not None

    def list_all(self) -> list[AgentRegistration]:
        return list(self._store.values())


def _now() -> float:
    """单调时钟，用于时间戳。"""
    return time.monotonic()
