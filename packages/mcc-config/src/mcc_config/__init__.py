"""配置加载层。

home (~/.mcc/settings.json) + project (.mcc/settings.json) 合并，
形状与 Claude Code 高度兼容。
"""

from mcc_config.loader import load, merge
from mcc_config.models import (
    BudgetConfig,
    Config,
    HookEntry,
    HooksConfig,
    ModelConfig,
    ObservabilityConfig,
    PermissionConfig,
)

__all__ = [
    "load",
    "merge",
    "BudgetConfig",
    "Config",
    "HookEntry",
    "HooksConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "PermissionConfig",
]

__version__ = "0.1.0"
