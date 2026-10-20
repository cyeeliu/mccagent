# mcc-eval

轻量评测 harness：拉取业界基准数据集，用 mcc Agent 执行，轨迹与产物
落本地文件，评分后出 JSON + markdown 报告。设计对标 Inspect AI
（CLI 一次跑完、log 落目录、viewer 可选）与 SWE-bench（infer / report
分离、产物自包含可重评）——**没有 server、没有数据库、没有常驻服务**。

## 快速开始

```bash
# 冒烟（无需 API key，走 Mock 适配器）
uv run --package mcc-eval mcc-eval run --cases examples/mock-cases.json --mock

# HumanEval 前 5 题（用 mcc 已配置的模型）
uv run --package mcc-eval mcc-eval run --dataset humaneval --limit 5

# 查看支持的数据集 / 离线导出
uv run --package mcc-eval mcc-eval datasets
uv run --package mcc-eval mcc-eval datasets --dataset mbpp --limit 20 --out mbpp.json
uv run --package mcc-eval mcc-eval run --cases mbpp.json

# 对已存轨迹与产物重新评分（不重新执行 Agent）
uv run --package mcc-eval mcc-eval report eval-runs/<批次目录>
```

模型解析复用 mcc 的约定：`--model` > `MODEL` 环境变量 >
`~/.mcc/settings.json` 的 `model.main` > 默认值。

## 数据集与评分语义

| 数据集 | 规模 | 评分 |
| --- | --- | --- |
| humaneval | 164 | 执行式：Agent 产物 + 官方测试子进程运行（对齐官方单样本判定） |
| mbpp | 500 | 执行式：同上 |
| swebench-lite | 300 | 行为断言（官方评测需 Docker 任务容器，此处不等价官方解决率） |

## 产物布局

```
eval-runs/<时间戳>-<数据集>-<数量>/
  cases.json        # 本批 Case 定义（重评分依据）
  results.json      # 机器读汇总
  report.md         # 人读报告
  runs/<case>.jsonl # 每 case 完整轨迹（六类事件）
  work/<case>/      # Agent 工作目录（保留现场，执行式评分在其上运行）
```

轨迹事件契约见 `schemas/trace-event.schema.json`（schema_version = 1）。

## 注意

- 执行式评分的子进程不是安全沙箱（与 HumanEval 官方 harness 取舍
  一致）：仅运行可信来源的测试代码。
- Mock 适配器只发模拟轨迹、不写 solution.py，因此执行式评分会判
  failed——这是真实语义，mock 用于链路冒烟而非得分演示。
