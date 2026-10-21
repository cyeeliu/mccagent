"""配置加载层。

home (~/.mcc/settings.json) + project (.mcc/settings.json) 合并，
形状与 Claude Code 高度兼容。模型解析与 provider 构建的统一入口
也在本包（resolve.py），CLI 与评测共用。
"""

from mcc_config.loader import ensure_default_config, load, merge
from mcc_config.models import (
    BudgetConfig,
    Config,
    ContextConfig,
    HookEntry,
    HooksConfig,
    LlmConfig,
    ModelConfig,
    ObservabilityConfig,
    PermissionConfig,
    SandboxConfig,
)
from mcc_config.resolve import DEFAULT_MODEL, build_llm_provider, resolve_model

__all__ = [
    "ensure_default_config",
    "load",
    "merge",
    "build_llm_provider",
    "resolve_model",
    "DEFAULT_MODEL",
    "BudgetConfig",
    "Config",
    "ContextConfig",
    "HookEntry",
    "HooksConfig",
    "LlmConfig",
    "ModelConfig",
    "SandboxConfig",
    "ObservabilityConfig",
    "PermissionConfig",
]

__version__ = "0.1.0"
