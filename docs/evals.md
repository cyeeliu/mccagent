# 评测

mcc 的评测由 `mcc-eval` 包承担：轻量本地 harness，拉业界基准或自建
case，用 mcc Agent 执行，轨迹与产物落文件，评分后出 JSON + markdown
报告。没有 server、没有数据库、没有常驻服务——状态只存在于一个进程
的内存与文件系统里。

对应包文档见 `packages/mcc-eval/README.md`；本文讲方法：评测集怎么建、
判定器怎么写、什么时候跑。

## 为什么评测要单独做

Agent 输出是非确定的：同一任务跑两次结果可能不同。因此单次
pass/fail 没有信息量，结论要看多次采样的 k/N 分布；最终结果对了
不代表过程健康，结果错了也不代表全盘皆输——评测必须分层断言。

| 层 | 断言对象 | mcc-eval 对应机制 |
| --- | --- | --- |
| 结果级 | 任务终态（产物 / 官方测试） | `check` 执行式评分（骨架，每个 case 必备其一） |
| 轨迹级 | 过程健康（无循环、预算内） | `no_loops` / `cost_budget` / `turns_budget` 等断言 |
| 单元级 | 单次工具调用纪律 | `tool_called` / `tool_call_count`（含参数正则） |

能程序化的判定永远程序化：`report.md 存在且包含正确行数` 不需要
LLM 来判断。开放式产出（总结质量、代码可读性）目前不在 mcc-eval
范围内。

## 评测集：失败回流为主，业界基准为参照

业界基准（HumanEval / MBPP / SWE-bench Lite）测的是模型能力上限，
测不出 harness 在你的任务分布上的表现。评测集的主力应该是**从真实
失败里长出来的 case**：

```bash
# 把失败生产会话回流为 case 草稿（默认扫 ~/.mcc/sessions，收 errored）
uv run --package mcc-eval mcc-eval distill --limit 20
```

草稿落在 `evals/cases/<case_id>.json`，`expectations` / `setup` 留空，
需要人工补齐判定器并用已知结论校准后再入正式评测集。20 条真实
case 起步即可暴露 harness 的明显问题，比 500 条从来跑不动的 case
有价值得多。

回流是幂等的：已存在的草稿不会被覆盖，人工修订过的判定器安全。

## 多次采样：k/N 才是结论

```bash
# 每个 case 跑 3 次，报告按 case 聚合为 k/N
uv run --package mcc-eval mcc-eval run --cases evals/cases/x.json \
    --runs-per-case 3
```

产物按采样隔离：`runs/<case>-a<N>.jsonl` 与 `work/<case>-a<N>/`。
报告的「Case 级聚合」表给出三态判定：

- **稳定通过**（k == N）
- **不稳定**（0 < k < N）——harness 迭代最值得排查的对象
- **稳定失败**（k == 0）

`results.json` 的 summary 同步携带 `stable_cases` / `flaky_cases`，
供脚本消费。

批次运行中还有 `progress.json` 实时进度镜像：Agent 每推进一轮
loop（轨迹事件 turn 变化）或发生 case 状态迁移（开始 / 完成）即
原子重写，记录当前各采样的 loop 数、事件数与最新事件。运行长批次
时直接读它监控，不必等 `results.json`：

```bash
watch -n2 cat eval-runs/<批次目录>/progress.json
```

## 环境预置：setup

Case 的 `setup` 是声明式文件映射，Agent 开跑前写入其工作目录：

```json
{
  "case_id": "fix-port",
  "task": "把配置端口改为 8081 并验证",
  "setup": {"config/app.yaml": "port: 8080\n"},
  "expectations": [
    {"type": "tool_called", "tool": "read_file", "weight": 0.4},
    {"type": "no_tool_errors", "weight": 0.3},
    {"type": "turns_budget", "max_turns": 10, "weight": 0.3}
  ]
}
```

路径限定在工作目录内（拒绝 `..` 逃逸），父目录按需创建。回流
case 补齐 setup 后即可表达「给定环境快照，改出正确终态」的任务。

## 断言类型速查

| type | 参数 | 检查什么 |
| --- | --- | --- |
| `tool_called` | `tool` / `args_regex` / `after_tool` | 工具是否被调用（可约束参数与顺序） |
| `tool_call_count` | `tool` / `min` / `max` | 调用次数区间 |
| `no_tool_errors` | — | 无工具执行错误 |
| `no_loops` | `max_repeat`（默认 3） | 无连续相同调用（循环检测） |
| `cost_budget` | `max_usd` | 单 run 成本上限 |
| `turns_budget` | `max_turns` | LLM 轮次上限 |
| `output_contains` | `pattern` / `regex` | 最终输出文本断言 |

`check` 字段（官方测试代码）存在时走执行式：写入工作目录、独立
子进程运行、退出码 0 即 passed——从产物层面杜绝「Agent 声称通过
但代码错误」的假阳性。执行式与断言式可并存，前者决定 status/score，
后者进 details 供诊断。

## 重评分与判定器校准

批次目录自包含（cases.json + 轨迹 + 产物），随时可离线重评：

```bash
uv run --package mcc-eval mcc-eval report eval-runs/<批次目录>
```

新 case 入库前，先用已知结论校准判定器：构造一份「已知会通过」与
「已知会失败」的轨迹（可由 mock 或历史 run 导出），跑 `report`
确认判定方向正确。没校准过的判定器混进评测集，测的是判定器的
bug 而不是 Agent。

## CI 集成

`ci.yml` 的 `eval-smoke` job 用 mock 适配器跑示例 case（零 API
成本、含多次采样路径），产物上传为 artifact 供失败归因。harness
的行为改动（工具描述、权限规则、上下文组装）都应有评测佐证——
评测是和 harness 同步生长的另一半代码，不是建成后的验收环节。

## 成本权衡

- 先降采样次数，再砍 case 数：case 覆盖分布，采样只影响置信度
- 轨迹级断言（循环 / 预算）是程序化检查，跑多少次都免费
- 业界基准动辄 164-500 题，`--limit` 控制规模；`datasets --out`
  可离线导出后再跑
