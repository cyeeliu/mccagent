"""统一模型解析与 LLM provider 构建。

CLI（mcc）与评测（mcc-eval）共用同一份解析顺序，杜绝两处实现
各自漂移——模型配置只有这一个入口：

  resolve_model:    显式参数 > MODEL 环境变量 > 配置该类型（kind）
                    > 配置 model.main > default
  build_llm_provider: 显式 / MCC_LLM_BASE_URL 网关 > 按环境自动探测

provider 的 import 保持惰性：只做模型名解析的调用方不会连带加载
LLM 客户端（httpx 等）。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

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


def build_llm_provider(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Config | None = None,
) -> Any:
    """构建 LLM provider，settings.json 与环境变量共用一套约定。

    优先级（base_url 与 api_key 各自独立生效）：

    - ``base_url``：显式参数 > settings.json 的 ``llm.base_url`` >
      ``MCC_LLM_BASE_URL`` 环境变量
    - ``api_key``：显式参数 > settings.json 的 ``llm.api_key`` >
      各厂商环境变量（``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``）

    分支语义：

    - ``base_url`` 非空 → OpenAI 兼容网关（vLLM / SGLang / Ollama
      等），api_key 仍缺省时用占位值——本地网关通常不鉴权；
    - 无 ``base_url`` → 按环境变量自动探测厂商（Anthropic 优先）；
      环境变量全空但 settings 配了 ``llm.api_key`` 时按 Anthropic
      构建（mcc 默认模型系列）；
    - 两手皆空 → RuntimeError。
    """
    from mcc_llm import auto_provider_from_env
    from mcc_llm.anthropic import AnthropicClient
    from mcc_llm.openai import OpenAiClient

    url = base_url
    if url is None and config is not None and config.llm.base_url:
        url = config.llm.base_url
    if url is None:
        url = os.environ.get("MCC_LLM_BASE_URL")

    key = api_key
    if key is None and config is not None and config.llm.api_key:
        key = config.llm.api_key

    if url:
        return OpenAiClient(
            api_key=key or os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=url,
        )
    try:
        return auto_provider_from_env()
    except RuntimeError:
        if key:
            return AnthropicClient(api_key=key)
        raise
