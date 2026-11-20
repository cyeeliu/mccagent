"""事件 payload 构建共享 helper（P0-03 结果契约闭环）。

所有 Runtime（TaskRuntime / InteractiveRuntimeV2 / WorkerPool）在任务完成时
构建统一格式的事件 payload，携带 result_ref 和完整结果字段，确保
TaskDispatcher._record_result 能从 payload 读取完整数据写入 TaskStore。

设计原则（方案 §P0-03）：
- result_ref 指向持久化结果，格式 "result://{task_id}/{attempt_id}"
- payload 携带 output / usage / cost_usd / artifacts 完整字段
- 禁止通过 RuntimeEvent payload 重新拼装 RuntimeResult（payload 只用于传递）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcc_core.runtime import RuntimeEventType, TaskStatus

if TYPE_CHECKING:
    from mcc_core.runtime.result import RuntimeResult

# TaskStatus → RuntimeEventType 映射
_STATUS_TO_EVENT: dict[TaskStatus, RuntimeEventType] = {
    TaskStatus.SUCCEEDED: RuntimeEventType.TASK_COMPLETED,
    TaskStatus.FAILED: RuntimeEventType.TASK_FAILED,
    TaskStatus.CANCELLED: RuntimeEventType.TASK_CANCELLED,
    TaskStatus.TIMED_OUT: RuntimeEventType.TASK_TIMED_OUT,
}


def result_event_type(result: RuntimeResult) -> RuntimeEventType:
    """根据 RuntimeResult.status 返回对应的 RuntimeEventType。"""
    event_type = _STATUS_TO_EVENT.get(result.status)
    if event_type is None:
        # 非终态结果默认为 TASK_FAILED（防御性）
        return RuntimeEventType.TASK_FAILED
    return event_type


def build_result_payload(
    task_id: str,
    attempt_id: str,
    result: RuntimeResult,
) -> dict[str, object]:
    """构建完整的事件 payload（P0-03）。

    携带：
    - result_ref: 持久化结果引用 "result://{task_id}/{attempt_id}"
    - status: 结果状态字符串
    - summary: 结果摘要（如果有）
    - output: 结果输出（如果有）
    - usage: token 用量（如果有）
    - cost_usd: 花费（如果有）
    - artifacts: 产物列表（如果有）
    - error: 错误信息（如果有）
    """
    payload: dict[str, object] = {
        "result_ref": f"result://{task_id}/{attempt_id}",
        "status": str(result.status),
    }

    if result.summary:
        payload["summary"] = result.summary

    if result.output:
        payload["output"] = dict(result.output)

    if result.usage is not None:
        payload["usage"] = result.usage.to_dict()

    if result.cost_usd is not None:
        payload["cost_usd"] = float(result.cost_usd)

    if result.artifacts:
        payload["artifacts"] = [a.to_dict() for a in result.artifacts]

    if result.error is not None:
        payload["error"] = str(result.error)

    return payload


__all__ = [
    "result_event_type",
    "build_result_payload",
]
