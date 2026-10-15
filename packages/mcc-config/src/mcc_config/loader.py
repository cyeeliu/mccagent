"""配置加载与合并逻辑。

home (~/.mcc/settings.json) 与 project (cwd/.mcc/settings.json) 合并：
- permissions.deny / allow 追加
- permissions.mode project 覆盖（若设置）
- budget / model 仅在 project 显式设置（非默认值）时覆盖
"""

from __future__ import annotations

import json
from pathlib import Path

from mcc_config.models import (
    BudgetConfig,
    Config,
    ModelConfig,
)


def _read_config(path: Path) -> Config | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return Config.model_validate_json(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def merge(home: Config | None, project: Config | None) -> Config:
    base = home if home is not None else Config()

    if project is None:
        return base

    base.permissions.deny.extend(project.permissions.deny)
    base.permissions.allow.extend(project.permissions.allow)
    if project.permissions.mode is not None:
        base.permissions.mode = project.permissions.mode

    if project.budget != BudgetConfig():
        base.budget = project.budget
    if project.model != ModelConfig():
        base.model = project.model

    return base


async def load(cwd: Path) -> Config:
    from pathlib import Path as _P

    home_dir = _P.home()
    home = _read_config(home_dir / ".mcc" / "settings.json")
    project = _read_config(cwd / ".mcc" / "settings.json")
    return merge(home, project)
