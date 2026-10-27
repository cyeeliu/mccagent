"""LLM provider 工厂：settings.json 与环境变量共用一套约定。

从配置层下沉至此：配置包保持纯数据（无 LLM 客户端依赖），
provider 构建归 LLM 层。``config`` 参数 duck-typing 读取
``config.llm.base_url`` / ``config.llm.api_key``，不反向依赖
配置包。模型名解析（resolve_model）仍在配置层。
"""

from __future__ import annotations

import os
from typing import Any


def build_llm_provider(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Any | None = None,
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
