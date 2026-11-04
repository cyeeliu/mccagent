"""AgentRegistry：Agent 注册、能力发现、状态查询和生命周期索引。

为 Agent 提供统一注册入口，支持按能力筛选、心跳过期检测和
租户隔离。单机版本用内存实现，预留 RegistryRepository 协议
供分布式实现替换。
"""

from __future__ import annotations

from .base import (
    AgentAlreadyRegisteredError,
    AgentDescriptor,
    AgentNotFoundError,
    AgentRegistration,
    AgentUnavailableError,
    InMemoryRepository,
    RegistrationStatus,
    RegistryRepository,
)
from .registry import AgentRegistry

__all__ = [
    "AgentAlreadyRegisteredError",
    "AgentDescriptor",
    "AgentNotFoundError",
    "AgentRegistration",
    "AgentRegistry",
    "AgentUnavailableError",
    "InMemoryRepository",
    "RegistrationStatus",
    "RegistryRepository",
]
