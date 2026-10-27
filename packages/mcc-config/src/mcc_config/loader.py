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
    BackgroundConfig,
    BudgetConfig,
    Config,
    ContextConfig,
    CronConfig,
    GoalConfig,
    ModelConfig,
    ObservabilityConfig,
    PermissionConfig,
    ReliabilityConfig,
    SandboxConfig,
    SecurityConfig,
    SkillsConfig,
    TaskConfig,
    TeamConfig,
    TodoConfig,
    WorkflowConfig,
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
    base = home if home is not None else Config()

    if project is None:
        return base

    base.permissions.deny.extend(project.permissions.deny)
    base.permissions.allow.extend(project.permissions.allow)
    if project.permissions.mode is not None:
        base.permissions.mode = project.permissions.mode

    if project.budget != BudgetConfig():
        base.budget = project.budget
    if project.model.main != ModelConfig().main:
        base.model.main = project.model.main
    if project.model.subagent is not None:
        base.model.subagent = project.model.subagent
    if project.model.summarize is not None:
        base.model.summarize = project.model.summarize
    if project.llm.base_url is not None:
        base.llm.base_url = project.llm.base_url
    if project.llm.api_key is not None:
        base.llm.api_key = project.llm.api_key
    # 与 budget 同语义：project 显式配置过沙箱（任何字段偏离默认）
    # 时整体覆盖，避免逐字段拼接出语义含混的组合
    if project.sandbox != SandboxConfig():
        base.sandbox = project.sandbox
    if project.context != ContextConfig():
        base.context = project.context
    if project.skills != SkillsConfig():
        base.skills = project.skills
    if project.goal != GoalConfig():
        base.goal = project.goal
    if project.todo != TodoConfig():
        base.todo = project.todo
    if project.tasks != TaskConfig():
        base.tasks = project.tasks
    if project.mcp_servers:
        base.mcp_servers = project.mcp_servers
    if project.team != TeamConfig():
        base.team = project.team
    if project.background != BackgroundConfig():
        base.background = project.background
    if project.cron != CronConfig():
        base.cron = project.cron
    if project.workflow != WorkflowConfig():
        base.workflow = project.workflow
    if project.observability != ObservabilityConfig():
        base.observability = project.observability
    if project.security != SecurityConfig():
        base.security = project.security
    if project.reliability != ReliabilityConfig():
        base.reliability = project.reliability

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
