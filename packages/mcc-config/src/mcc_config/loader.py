"""配置加载与合并逻辑。

home (~/.mcc/settings.json) 与 project (cwd/.mcc/settings.json) 合并：
- permissions.deny / allow 追加
- permissions.mode project 覆盖（若设置）
- budget / model.main 仅在 project 显式设置（非默认值）时覆盖
- model.subagent / model.summarize 在 project 显式设置（非 null）时覆盖
"""

from __future__ import annotations

import json
from pathlib import Path

from mcc_config.models import (
    BudgetConfig,
    Config,
    ModelConfig,
    PermissionConfig,
)

_DEFAULT_CONFIG = Config(
    model=ModelConfig(),
    permissions=PermissionConfig(
        mode="default",
        allow=[],
        deny=[],
    ),
    budget=BudgetConfig(max_usd_per_session=2.0, max_iterations=40),
)


def _default_json() -> str:
    return json.dumps(_DEFAULT_CONFIG.model_dump(), indent=2, ensure_ascii=False)


def ensure_default_config(cwd: Path) -> Config:
    """确保配置目录与文件存在，返回合并后的配置。

    首次运行时在 ~/.mcc/ 和 cwd/.mcc/ 下创建默认 settings.json。
    """
    home_dir = Path.home()
    home_path = home_dir / ".mcc" / "settings.json"
    project_path = cwd / ".mcc" / "settings.json"

    if not home_path.exists():
        home_path.parent.mkdir(parents=True, exist_ok=True)
        home_path.write_text(_default_json(), encoding="utf-8")
    if not project_path.exists():
        project_path.parent.mkdir(parents=True, exist_ok=True)
        project_path.write_text(_default_json(), encoding="utf-8")

    return _read_and_merge(home_path, project_path)


def _read_and_merge(home_path: Path, project_path: Path) -> Config:
    home = _read_config(home_path)
    project = _read_config(project_path)
    return merge(home, project)


def _read_config(path: Path) -> Config | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return Config.model_validate_json(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def merge(home: Config | None, project: Config | None) -> Config:
    """home 为基线，project 逐字段覆盖。

    语义规则（新增子配置零代码并入）：
    - ``permissions``：deny/allow 追加，mode 显式覆盖（非 None）
    - ``model``/``llm``：逐字段 None 不覆盖（model.main 需偏离默认）
    - ``mcp_servers``：非空 dict 才覆盖
    - 其余子配置：project 值偏离默认时整体覆盖（避免逐字段拼接出
      语义含混的组合）
    """
    base = home if home is not None else Config()
    if project is None:
        return base

    base.permissions.deny.extend(project.permissions.deny)
    base.permissions.allow.extend(project.permissions.allow)
    if project.permissions.mode is not None:
        base.permissions.mode = project.permissions.mode

    if project.model.main != ModelConfig().main:
        base.model.main = project.model.main
    for name in ("subagent", "summarize"):
        value = getattr(project.model, name)
        if value is not None:
            setattr(base.model, name, value)
    for name in ("base_url", "api_key"):
        value = getattr(project.llm, name)
        if value is not None:
            setattr(base.llm, name, value)
    if project.mcp_servers:
        base.mcp_servers = project.mcp_servers

    defaults = Config()
    for name in Config.model_fields:
        if name in ("permissions", "model", "llm", "mcp_servers"):
            continue
        value = getattr(project, name)
        if value != getattr(defaults, name):
            setattr(base, name, value)

    return base


async def load(cwd: Path) -> Config:
    home_dir = Path.home()
    home_path = home_dir / ".mcc" / "settings.json"
    project_path = cwd / ".mcc" / "settings.json"
    return _read_and_merge(home_path, project_path)


# ---------- settings 写回 ----------


def _home_settings_path() -> Path:
    return Path.home() / ".mcc" / "settings.json"


def read_home_settings() -> dict:
    """读取 home settings.json 的原始 dict；不存在返回 {}。"""
    path = _home_settings_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update_home_setting(keys: list[str], value: object) -> None:
    """按嵌套 key 路径更新 home settings.json 的单个字段。

    dict 级操作（不经 Config 往返）：只改目标 key，不固化
    “回落 main”的派生值，保持用户文件最小。原子写 + 0600。
    示例：``update_home_setting(["model", "main"], "claude-x")``
    """
    if not keys:
        raise ValueError("keys must not be empty")
    path = _home_settings_path()
    data = read_home_settings()
    node = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.chmod(0o600)
    import os

    os.replace(tmp, path)
