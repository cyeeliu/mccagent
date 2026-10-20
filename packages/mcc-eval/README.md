# mcc-eval

轻量评测 harness：拉取业界基准数据集，用 mcc Agent 执行，轨迹与产物
落本地文件，评分后出 JSON + markdown 报告。设计对标 Inspect AI
（CLI 一次跑完、log 落目录、viewer 可选）与 SWE-bench（infer / report
分离、产物自包含可重评）——**没有 server、没有数据库、没有常驻服务**。

评测方法（三层断言、失败回流、多次采样、判定器校准）见仓库
`docs/evals.md`。

## 快速开始

```bash
# 冒烟（无需 API key，走 Mock 适配器；含多次采样路径）
uv run --package mcc-eval mcc-eval run --cases examples/mock-cases.json \
    --mock --runs-per-case 2

# HumanEval 前 5 题（用 mcc 已配置的模型），每 case 采样 3 次
uv run --package mcc-eval mcc-eval run --dataset humaneval --limit 5 \
    --runs-per-case 3

# 把失败生产会话回流为评测 case 草稿（失败回流是评测集的主力来源）
uv run --package mcc-eval mcc-eval distill --limit 20

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

业界基准测模型能力上限；harness 在自己任务分布上的表现靠失败回流
的 case（`distill` + `evals/cases/`）。

## 产物布局

```
eval-runs/<时间戳>-<数据集>-<数量>/
  cases.json                # 本批 Case 定义（重评分依据，含 setup）
  progress.json             # 实时进度（每轮 loop / 每 case 状态迁移即原子重写）
  results.json              # 机器读汇总（含 case 级 k/N 与 flaky 统计）
  report.md                 # 人读报告（明细 + case 级聚合 + 失败理由）
  runs/<case>-a<N>.jsonl    # 每次采样的完整轨迹（六类事件，逐事件落盘）
  work/<case>-a<N>/         # 每次采样的 Agent 工作目录（保留现场）
```

`--runs-per-case N`（默认 1）控制每 case 采样次数：Agent 输出非确定，
单次 pass/fail 没有信息量，报告按 case 聚合为 k/N（稳定通过 / 不稳定
/ 稳定失败）。

## Case 结构

```json
{
  "case_id": "fix-port",
  "task": "把配置端口改为 8081 并验证",
  "setup": {"config/app.yaml": "port: 8080\n"},
  "check": null,
  "expectations": [
    {"type": "tool_called", "tool": "read_file", "weight": 0.4},
    {"type": "no_loops", "weight": 0.3},
    {"type": "turns_budget", "max_turns": 10, "weight": 0.3}
  ]
}
```

- `setup`：声明式环境预置（`{相对路径: 文件内容}`，路径限定在
  工作目录内），Agent 开跑前写入
- `check`：官方测试代码，存在时走执行式评分
- `expectations`：加权行为断言（七类，见 `docs/evals.md` 速查表）

## 注意

- 执行式评分的子进程不是安全沙箱（与 HumanEval 官方 harness 取舍
  一致）：仅运行可信来源的测试代码。
- Mock 适配器只发模拟轨迹、不写 solution.py，因此执行式评分会判
  failed——这是真实语义，mock 用于链路冒烟而非得分演示。
