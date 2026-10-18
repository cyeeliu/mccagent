# 踩坑与排障

项目开发中实际踩过并修复的坑，按主题归类。新问题修复后请追加条目（现象 / 根因 / 修复）。

## LLM 与 Provider

### 推理模型生成会话标题恒为空

**现象**：使用 DeepSeek 推理模型时，会话标题永远不生成。API 响应中 `finish_reason='length'`、`content` 为空、`reasoning_tokens=32`。

**根因**：标题生成请求 `max_tokens=32`。推理模型的思考过程（`reasoning_content`）**同样计入输出配额**，32 个 token 全部被思考消耗，正文尚未开始输出即被截断。

**修复**：输出配额提升为 `TITLE_MAX_TOKENS = 512`，为思考留足余量；同时后处理取响应首行并截断到 30 字符（提示词的"10 个中文字"约束对模型不具强制力）。（commit `9f6ecc2`）

### OpenAI 兼容端点的流式细节

实现 OpenAI 兼容客户端时需同时处理：

- **空 choices chunk**：部分端点会推送 `choices: []` 的心跳 chunk，直接跳过，否则 IndexError
- **finish_reason 映射**：`stop → end_turn`、`tool_calls → tool_use`、`length → max_tokens`，映射错会导致主循环把正常结束当异常
- **tool_calls 按 index 去重**：流式中函数名会重复推送，须按 `index` 追踪已发出的 `ToolUseStart`，否则工具块重复
- **ToolResult 必须转 role:tool 消息**：Anthropic 的 tool_result 是 user 消息内的块，OpenAI 侧必须拆成独立的 `role: "tool"` 消息，否则请求被拒

## 权限与安全

### bash 前缀匹配的词边界

**现象**：`Bash(rm:*)` 规则会连带放行/拒绝 `rmdir`、`rmrerun` 等命令。

**根因**：朴素 `startswith` 匹配把 `rm` 当成了 `rmdir` 的前缀。

**修复**：`bash_word_match` 要求前缀之后是结尾、空白或 `/`——在词边界处匹配。这是权限系统的安全关键路径。

### 子代理的 Ask 权限即拒绝

子代理没有交互确认通道（权限响应队列只在主代理 TUI 接线），`Ask` 决策默认拒绝。这是刻意设计：无人值守的子代理不应静默等待放行；需要子代理使用某工具时，在配置中显式加 allow 规则。

## Python / asyncio

### `Path.home()` 模块级求值导致测试污染

**现象**：monkeypatch `HOME` 环境变量后，测试仍读写真实用户的 `~/.mcc/`。

**根因**：`Path.home()` 在模块导入时求值存为常量，运行期环境变量变化不影响它。

**修复**：所有默认路径改为**函数内动态求值**（`default_sessions_base()` / `default_memory_base()` 每次调用时取 `Path.home()`）。

### `pytest asyncio_mode=auto` 仍需 `async def`

**根因**：auto 模式省的是 `@pytest.mark.asyncio` 装饰器，不是 `async` 声明——测试体内用了 `await` 就必须 `async def`，否则函数返回 coroutine 而非结果。

### 异步事件测试用哨兵排空，不用 cancel

测试事件分发时，consumer 任务用哨兵值（`None`）排空退出、断言从 sink 列表取事件；`task.cancel()` 与队列排空之间存在时序竞态，会导致偶发失败。

## CLI 与工具链

### click `nargs=-1` 参数收不到未知选项

**现象**：`mcc sessions purge --older-than 30d` 报"no such option"，因为 `args` 是 `nargs=-1` 的可变参数，click 不允许在可变参数后跟未注册的选项。

**修复**：在主命令上显式注册 `hidden=True` 的 `--older-than` 选项。

### MockLlm 的标题轮次预留

**坑**：首轮快照录制后触发标题生成，`MockLlm.complete` 与 `stream` 共享轮次计数器，标题生成会**额外消耗一轮脚本**。mock 脚本要在主流程轮次之间预留"标题轮"，否则后续轮次错位（例如最终答案变成标题文本）。

### write_file 拒绝覆盖已存在文件

工具语义：`write_file` 只创建新文件，重复写同一路径返回错误——这是防意外覆盖的刻意设计。测试"权限放行后二次写入"时，失败可能来自工具语义而非权限系统，排查时先分清。

### 一次 run_streaming 只产生一个 turn 快照

`end_turn` 结束的那轮才落一个快照；多轮成本累积的测试要用 tool_use 调用链（多轮迭代）或多次 `run`，单轮纯文本对话只有一个快照。

## 子代理

### 并行子代理的 MockLlm 轮次分配顺序不定

多个子代理并行运行共享 MockLlm 时，`_next_turn` 虽无 await（事件循环单线程内原子），但子代理启动顺序取决于调度，**轮次分配顺序不定**。断言用输出集合（`set`）而非顺序列表。

### 超时用 `asyncio.wait_for`，Python 3.12 抛内置 TimeoutError

`asyncio.wait_for` 超时在 Python 3.12+ 抛内置 `TimeoutError`（不再是 `asyncio.TimeoutError`，后者已成别名）。子代理超时返回 `reason=timeout` 的结果对象而非异常，工具层据此标记错误。
