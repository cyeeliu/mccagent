"""eval-runner：评测平台执行端。

向 API 服务轮询领取任务，在适配器内执行被测 Agent，
以 TraceEvent 流的形式批量上报轨迹并汇报结果。
"""
