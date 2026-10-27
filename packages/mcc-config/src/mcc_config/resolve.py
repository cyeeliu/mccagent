"""统一模型解析。

CLI（mcc）与评测（mcc-eval）共用同一份解析顺序，杜绝两处实现
各自漂移——模型配置只有这一个入口：

  resolve_model:    显式参数 > MODEL 环境变量 > 配置该类型（kind）
                    > 配置 model.main > default

provider 构建已下沉至 mcc_llm.factory.build_llm_provider（配置层
保持纯数据，不依赖 LLM 客户端）。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcc_config.loader import load
from mcc_config.models import Config, ModelConfig

# 兜底默认与 ModelConfig.main 的默认值保持同源：无论「无配置文件」
# 还是「配置加载失败」，解析结果都是同一个模型，调用方不感知差异
DEFAULT_MODEL = ModelConfig().main


def resolve_model(
    kind: str = "main",
    explicit: str | None = None,
    *,
    config: Config | None = None,
    work_dir: Path | None = None,
    default: str = DEFAULT_MODEL,
) -> str:
    """按用途解析模型名。

    优先级：``explicit`` 显式参数 > ``MODEL`` 环境变量 > 配置该类型
    （main / subagent / summarize，配置只写了 main 时其余类型回落
    main）> ``default``。``config`` 已加载则直接用；否则从
    ``work_dir``（默认当前目录）按 home+project 合并加载，加载失败
    回落 ``default``——评测等无人值守场景不因配置损坏而中断。

    仅在无事件循环的同步上下文调用（CLI 参数处理、评测批次提交）；
    循环内的调用方应传入已加载的 ``config``。
    """
    if explicit:
        return str(explicit)
    env = os.environ.get("MODEL")
    if env:
        return env

    if config is None:
        try:
            config = asyncio.run(load(work_dir or Path.cwd()))
        except Exception:
            return default

    configured = getattr(config.model, kind, None)
    if configured:
        return str(configured)
    if config.model.main:
        return str(config.model.main)
    return default
