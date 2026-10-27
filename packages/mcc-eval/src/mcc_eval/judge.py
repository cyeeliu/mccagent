"""LLM-as-Judge 评分：rubric 驱动的 1-5 打分，预评分后融入断言权重。

judge 断言需要真实 LLM；provider 缺省经 mcc-config 统一链路惰性构建
（与 CLI 同一 settings.json/env 约定，judge 类型未配置时回落 main）。
调用失败不 crash 评测：返回 (0, "judge unavailable: ...")；无 provider
时预评分标注 skipped，断言权重回退（与未知类型语义一致），保住
mock/CI 冒烟链路。
"""

from __future__ import annotations

from typing import Any

from mcc_core import Message


def _parse_score(text: str) -> int | None:
    """从 judge 输出解析 1-5 整数：取首个落在区间的数字字符。"""
    for ch in text.strip():
        if ch.isdigit() and 1 <= int(ch) <= 5:
            return int(ch)
    return None


class JudgeScorer:
    """LLM judge：rubric + answer → 1-5 整数评分。

    解析失败时追加指令重试一次；provider 异常降级为
    (0, "judge unavailable")。
    """

    def __init__(
        self,
        provider: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model

    @property
    def available(self) -> bool:
        """provider 可用（已注入或可构建）。"""
        return self._ensure() is not None

    def _ensure(self) -> tuple[Any, str] | None:
        if self._provider is not None:
            return self._provider, self._model or ""
        try:
            from mcc_config import build_llm_provider, resolve_model

            provider = build_llm_provider()
            model = resolve_model("judge")
        except Exception:
            return None
        self._provider = provider
        self._model = model
        return provider, model

    async def score(self, rubric: str, answer: str) -> tuple[int, str]:
        """按 rubric 给 answer 打 1-5 分，返回 (分数, 原因)。"""
        built = self._ensure()
        if built is None:
            return 0, "judge unavailable: no LLM provider"
        provider, model = built
        from mcc_llm import CompleteRequest

        prompt = (
            f"Rubric:\n{rubric}\n\n"
            f"Candidate answer:\n<answer>\n{answer}\n</answer>\n\n"
            "Rate the answer against the rubric on a 1-5 scale "
            "(5 = perfect). Output ONLY the integer."
        )
        for attempt in (1, 2):
            try:
                resp = await provider.complete(
                    CompleteRequest(
                        model=model,
                        max_tokens=64,
                        messages=[Message.user(prompt)],
                        system="You are a strict eval judge.",
                        temperature=0.0,
                    )
                )
                text = "".join(
                    getattr(b, "text", "") for b in resp.content
                )
                parsed = _parse_score(text)
                if parsed is not None:
                    return parsed, f"judge scored {parsed}/5"
            except Exception as exc:
                if attempt == 2:
                    return 0, f"judge unavailable: {exc}"
            prompt += "\n\nOutput ONLY the number."
        return 0, "judge unavailable: unparseable score"

    async def prescore(
        self, final_text: str, expectations: list[dict[str, Any]]
    ) -> dict[int, tuple[int, str]]:
        """批量预评分所有 judge 断言：返回 expectation 索引 → (分数, 原因)。

        provider 不可用时仅标注 skipped，不真正调用——评分层据此
        回退权重，保住无 key 的冒烟链路。
        """
        results: dict[int, tuple[int, str]] = {}
        has_judge = any(
            exp.get("type") == "judge" for exp in expectations
        )
        if not has_judge:
            return results
        if self._ensure() is None:
            for idx, exp in enumerate(expectations):
                if exp.get("type") == "judge":
                    results[idx] = (0, "judge skipped (no LLM)")
            return results
        for idx, exp in enumerate(expectations):
            if exp.get("type") != "judge":
                continue
            rubric = str(exp.get("prompt") or exp.get("rubric") or "")
            if not rubric.strip():
                results[idx] = (0, "judge skipped: empty rubric")
                continue
            results[idx] = await self.score(rubric, final_text)
        return results
