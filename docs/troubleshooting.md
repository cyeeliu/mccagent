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

### resume 后 TUI 不显示历史轨迹

**现象**：`mcc resume <id>` 恢复会话后，对话能正常延续（模型记得之前内容），但 TUI 界面空白，看不到任何历史消息。

**根因**：恢复的 history 只传给了 `AgentLoop`（作为 LLM 请求上下文前缀），而 TUI 渲染层只消费事件队列——历史从未以事件形式下发，渲染层自然无内容可画。"上下文存在"与"轨迹可见"是两条独立通路。

**修复**：`_history_events(history)` 把重放的消息序列转换为事件（user → `UserEcho`，assistant 文本 → `TextDelta`，tool_use → `ToolCallStart`，tool 结果 → `ToolCallEnd`，末尾 `TurnEnd` 携带会话累计成本收尾），在 TUI 启动前注入事件队列。工具输出重放时同样截断 500 字符，与实时渲染一致。

### TUI 面板的 session id 与恢复的会话不一致

**根因**：TUI 显示的短 id 在 `_run_tui_mode` 内部随机生成，resume 场景下与实际恢复的会话 id 脱节（用户无法据此再次 resume/delete）。

**修复**：`_run_tui_mode` 接受 `session_id` 参数，resume 时传入 `meta.id`；新建会话仍随机生成。

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

### click 8.5 的补全脚本 API 变化

**现象**：`comp.get_completion_script()` 抛 `AttributeError`——click 8.1 的 `ShellComplete` 方法在 8.5 中不存在。

**根因**：click 8.5 把补全脚本生成改为 `source()` 方法，且 `get_completion_class("powershell")` 返回的类没有脚本生成能力（bash/zsh/fish 才有）。

**修复**：运行时探测 `getattr(comp, "get_completion_script", None)`，旧 API 存在则调用，否则用 `comp.source()`；shell 参数在白名单（`bash/zsh/fish`）内校验后才生成。

### 版本号解析的 rc 段陷阱

`"0.1.0rc1"` 的第三段若提取段内**所有**数字会得到 `1`（`rc1` 的 1），导致 `(0, 1, 1)` 与 `(0, 1, 0)` 误判为不同版本。每段只取**前导数字**（`re.match(r"\d+", seg)`），`rc` 后缀自然归零。

## 评测平台

### 评测平台 compose 拉不到 MinIO 镜像

**现象**：`docker compose up` 报 `failed to resolve reference "docker.io/minio/minio:..."`，无论 `RELEASE.2025-09-07T16-13-09Z` 还是 `latest` 均 404。

**根因**：MinIO 社区版已停止发布预编译产物——官方仓库 README 声明 *"The MinIO community edition is now distributed as source code only. We will no longer provide pre-compiled binary releases"*，Docker Hub 上不再有可拉取的新镜像（容器场景官方指引是自行 clone 源码构建）。此前"tag 号虚构"的诊断不成立，问题是任何 tag 都拉不到。

**修复**：本地栈移除 minio / minio-init 服务，轨迹后端改用 `TRACE_BACKEND=local`（存容器卷，`/workspace/mccagent/eval-platform/docker-compose.yaml`）；S3 后端留给生产环境接真实对象存储端点。教训：编排选型时确认镜像的发布策略（是否持续发布、渠道是否变更），而不只是核对 tag 存在性。

### 评测平台 compose 端口被占用

**现象**：`docker compose up` 报 `Bind for 127.0.0.1:3000 failed: port is already allocated`（3000 为例，其他端口同理）。

**根因**：宿主机端口发布冲突。报 `port is already allocated` 通常是**另一个 Docker 容器**已发布该端口（如本机已有的 Grafana、其他 compose 项目残留）；报 `address already in use` 则是宿主机上的非容器进程占用。

**修复**：先 `docker ps --filter publish=3000` 定位占用容器（本项目残留先 `docker compose down`），宿主机进程用 `ss -ltnp | grep :3000` 查。compose 的四个宿主机端口均支持环境变量覆盖（`SERVER_PORT` / `WEB_PORT` / `PROMETHEUS_PORT` / `GRAFANA_PORT`，默认 8081 / 5173 / 9090 / 3000），如 `GRAFANA_PORT=3001 docker compose up`；改 `WEB_PORT` 后须同步设置 `CORS_ALLOWED_ORIGINS`。

### 评测平台 server 容器 unhealthy

**现象**：`docker compose up` 卡在 `dependency failed to start: container eval-platform-server-1 is unhealthy`，`docker compose logs server` 可见 `/healthz/ready` 报 500 或 `PermissionError`。

**根因**：非 root 运行 + 命名卷属主的组合问题。server 镜像以 uid 10001 运行；Docker 命名卷挂载到镜像中**不存在**的路径时，卷根目录由 root 拥有（只有挂载到镜像中已存在的目录时，才按 copy-up 语义继承目录属主）。ready 探针会向轨迹目录写 `_healthcheck` 探测文件，root 拥有的卷上直接 `PermissionError`。

**修复**：镜像 runtime 阶段预创建 `/traces` 并 `chown app:app`（`/workspace/mccagent/eval-platform/server/Dockerfile`），首次挂载的卷继承属主即可写。**注意：已经创建过的卷不会随镜像重建而重新初始化属主**——修复前已存在的 `traces` 卷仍是 root 拥有，须 `docker compose down -v` 删卷重建（本地开发数据可丢弃），或 `docker compose run --rm -u root server chown -R 10001:10001 /traces` 保留数据修复。

### 评测平台 runner 镜像不含 mcc

**现象**：compose 启动后提交 `mini-claude-code` 批次，run 立即报错 `ModuleNotFoundError: No module named 'mcc_core'`（或 mcc_config / mcc_harness 等）。

**根因**：runner 镜像构建上下文曾为 `eval-platform/`，不含仓库根的 `packages/` 源码，mcc 系列包无法打进镜像；须在宿主机手动 editable 安装后本地 `uv run` 跑 runner，容器形态长期不可用。

**修复**：runner 服务的构建上下文改为仓库根（compose 里 `context: ..` + `dockerfile: eval-platform/runner/Dockerfile`），Dockerfile 内追加 mcc 六包的 COPY 与 editable 安装；仓库根新增 `.dockerignore`（排除 .git/.venv/node_modules 等）控制上下文体积。**注意上下文变更不会自动触发重建**，须 `docker compose up --build` 强制重建 runner 镜像。

### 评测集拉取失败：容器无法访问外网

**现象**：`python -m runner.benchmarks --submit` 报 `httpx.ConnectError` 或长时间无响应后超时。

**根因**：基准数据经 HuggingFace datasets-server 公开 API（`datasets-server.huggingface.co`）在线拉取，runner 容器无外网访问（公司代理、K8s prod overlay 的 NetworkPolicy 断网等场景）时必然失败。

**修复**：本地 compose 栈容器默认可联网；受限网络环境先在有网机器上 `--out batch.json` 导出批次 JSON，再经 `POST /api/batches` 手工提交。另注意 LLM 凭证（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`）未设置时任务会在首次调用模型时报错——compose 已透传宿主机同名变量，启动前 export 即可；用 `MCC_LLM_BASE_URL` 自托管网关时地址里的 `localhost` / `127.0.0.1` 在容器内指向容器自身，须改 `http://host.docker.internal:端口/v1`（compose 已配置 host-gateway 映射）。

### 评测批次实际模型与提交模型不一致

**现象**：benchmarks 提交时打印「模型: deepseek-...」，但 web 轨迹事件的 run_started 里 `model` 仍是旧值（如 claude-sonnet-4-6）。

**根因**：批次挂在 profile 上执行，run 的模型取自 profile 表（lease 时 `p.model`）。`submit_batch` 的幂等逻辑是「profile 名已存在即跳过创建」——此前用别的模型建过同名 profile 时，本次解析出的模型只是打印，批次仍静默挂到旧 profile 上按旧模型执行。

**修复**：profile 名缺省按模型派生 `mcc-<model>`（模型不同即不同 profile，语义上 profile 本就是被测 Agent 配置快照）；显式 `--profile` 指定已有名称时校验模型一致性，不一致直接报错退出并提示换名，不再静默沿用。

### 评测 run 跑完却没有分数

**现象**：批次执行完成，web 上 run 无 score（status 为 error），轨迹里能看到工具调用与模型输出。

**根因**：runner 的评分分支条件是 `final.status in ("passed", "failed")`——mcc 适配器的 RunFinished.status 只有两态（出现 Error 事件即 error，如 git clone 超时、run_bash 非零退出、max_iterations 超限），error 状态直接跳过 expectations 评分，`score=None` 上报。SWE-bench 任务大量 clone/测试操作极易触发 Error，导致轨迹明明可评却无分。

**修复**：评分条件改为「有 expectations 且轨迹有实质内容（工具调用或非空模型输出）」——带错轨迹照样评分，`no_tool_errors` 断言（权重 0.3）会正确惩罚它；零产出 error（凭证未配、LLM 首调即挂）保留 error 状态不评分，避免把基础设施故障误判成 Agent 低分。教训：**「执行状态」与「可评性」是两个正交维度**，用执行状态做评分开关会漏掉「跑出错但轨迹完整」的大类。

### 轨迹已 passed 但 run 状态是 error

**现象**：web 事件详情里 `run_finished` 显示 `passed` / `score=1`，但 run 列表状态却是 error，同一 run 的两条信息自相矛盾。

**根因**：轨迹事件（JSONL buf 逐段上报）与 run 表状态（complete 端点 UPDATE）是**两条独立链路**，收尾阶段的异常会让二者矛盾。两个触发路径：① runner 的心跳循环对单次 HTTP 失败零容错——一次网络抖动即判死心跳任务，`ensure_heartbeat_alive()` 抛异常进入 except 收尾分支，把 buf 里完整的 `run_finished` 照常 flush 进轨迹（事件链路正常），却以 `complete(status="error")` 收尾（状态链路被覆盖）；② 心跳断供超过 `LEASE_TTL_S`（120s）后 server reaper 直接把 run 置 error，runner 随后的 complete 打在 `WHERE status IN ('leased','running')` 上不中，状态永远停在 error。

**修复**：两层。① 心跳容错：单次失败改为短间隔（5s）重试计数，连续 3 次才判死心跳；server 主动宣告 `alive=false` 仍立即失败。② 评分保全：except 收尾分支里若已收到 `RunFinished`（`final is not None`），按正常评分逻辑取 status/score 上报，error 字段附「completed with finish error」说明；仅未产出 final 的 run 才报纯 error。教训：**基础设施故障不应覆盖已得出的业务结论**——收尾路径的容错（链路层）与结果的保全（业务层）要分开处理。

### run 结束了前端 LIVE 常亮

**现象**：轨迹页显示完整的 17 个事件、结尾 `run_finished=passed`，但头部 LIVE 徽标不灭；刷新页面依旧。

**根因**：run 状态确实又回到了 running——**它正在被重跑**。链路：心跳任务死亡后，execute 的租约检查点是事件驱动的（`ensure_heartbeat_alive` 只在收到下一个事件时检查），而工具执行 / 慢 LLM 期间可能长时间没有事件——静默窗口超过 `LEASE_TTL_S`（120s）后 reaper 回收租约，`retries < 3` 时把 run **置回 queued 重排**（而非置 error）。执行流随后跑完：`run_finished` 照常落轨迹（前端显示「结束」），但 complete 打在 `WHERE status IN ('leased','running')` 上不中（状态已是 queued），409 被旧代码 `except Exception: pass` 静默吞掉；runner 主循环下一轮 poll 又 lease 到这个 run 重新执行——status 回到 running，LIVE 重新亮起。上次「error」表象是同一根因的另一半：retries 耗尽时 reaper 走的是置 error 分支。

**修复**：三层。① **判死即止损**：heartbeat_loop 判死（连续 3 次失败或 alive=false）时持有执行任务引用，立即 `cancel()` 执行流——不再等下一个事件，静默窗口里也不再空跑烧 LLM 成本；execute 主体包成子任务，区分「心跳取消的 body」与「execute 自身被取消」（后者继续向上传播）。② **complete 409 可观测**：收尾路径的 complete 失败不再静默吞——409 特异性提示「lease already reclaimed by reaper, run will be retried」，其余异常同样打 stderr。③ **事件回灌**：flush 上报失败时把批次回灌缓冲由收尾路径重试，瞬时故障不再丢事件。教训：**检查点是事件驱动的设计，天然存在「无事件」盲区**——心跳这类独立监督任务发现异常后必须能主动打断被监督者，否则止损永远比故障晚一个事件（在静默窗口里就是晚到永远）。

### 评测 API 传非 UUID 直接 500

**现象**：`GET /api/runs/not-a-uuid`（或 `POST /internal/runs/xxx/complete`、`?batch=xxx` 过滤等）返回 500，日志里是 asyncpg 的 `invalid input syntax for type uuid`；创建重名 profile 同样 500（`duplicate key value violates unique constraint`）。

**根因**：路径参数与查询参数未校验直接进 SQL——UUID 列上的非法字面量由数据库驱动抛异常，FastAPI 兜底成 500。profile 创建是先查后插，两个并发请求都能通过「不存在」检查，随后 INSERT 撞 UNIQUE 约束，异常同样未被捕获。批次 `idempotency_key` 存在同样的并发窗口，且语义上应当幂等重放而非报错。

**修复**：入口统一 `_uuid_param()` 校验（10 个端点），非法输入 400 并回显具体值；鉴权头改 `secrets.compare_digest` 常数时间比较。profile 重名先查返回 409，`UniqueViolationError` 捕获兜底同样 409；批次 `idempotency_key` 并发重放按约束名判定后重查返回已有批次（`deduplicated: true`）。教训：**先查后插永远存在竞态窗口，唯一约束是最后防线——捕获它做幂等或 409，而不是让它漏成 500**。

### mcc 适配器的 ToolCall.args 恒为 None

**现象**：SWE-bench / HumanEval / MBPP 批次里带 `args_regex` 的 `tool_called` 断言全部失败，理由是「args did not match」——但轨迹里工具明明以正确参数被调用。

**根因**：mcc 事件流只携带 `args_preview`（参数预览字符串），适配器转译时 `ToolCall.args` 硬编码 None；评分侧 `pattern.search(str(tc.args or ""))` 对 None 得到空串，任何正则都不命中。

**修复**：转译改为 `args=ev.args_preview`，评分正则在预览上匹配。教训：**跨层转译时「拿不到完整数据」不等于「传空值安全」**——下游存在针对该字段的断言时，空值会让断言静默恒假，比抛异常更难发现。

### after_tool 断言把多次调用误判为顺序不符

**现象**：Agent 先 bash、再 edit、又一次 bash（验证改动），`tool=bash after_tool=edit` 断言失败——但确实存在 edit 之后的 bash 调用。

**根因**：旧实现只取「第一次匹配调用」的索引与 `after_tool` 的调用位置比较；工具被多次调用时，第一次在前的位置会掩盖后续在后的事实。

**修复**：改为「任意一次匹配调用出现在 after_tool 的某次调用之后即满足」——对全部匹配索引与 after_tool 索引做笛卡尔比较。

### run 流断线后事件永久缺失

**现象**：浏览器 SSE 断线重连后，断线窗口内的轨迹事件在 web 上永久缺失，刷新页面也不出现。

**根因**：web client 的 `onerror` 处理器注释声称「服务端按 Last-Event-ID 续发」，但 server 的 SSE 实现并无续发逻辑——断线窗口内的事件既不在重连补发范围内，也没有触发任何 REST 补齐，注释掩盖了真实缺口。

**修复**：断线与「落后被踢」走同一条补偿路径——触发 REST 全量重读 trace 分页后幂等合并，不再依赖不存在的续发语义。

### run 卡 running 无评分（状态长时间不翻转）

**现象**：demo 批次的 `fix-div-by-zero` run，轨迹 17 个事件完整（结尾 `run_finished=passed`、$0.0142、3 轮、grep/edit/bash 工具），但 run 状态长时间停在 running、LIVE 不灭、无 score。

**根因**：沙箱端到端复现（真 uvicorn + FakeDb + runner 子进程 + mock adapter 全链路，当前版本与上一版均正常 passed/1.0）证明代码无回归；页面数据特征（三工具集、`1200+turn*800` tokens、`uniform(0.01,0.06)` 成本）全部命中 demo-batch.sh 的 mock adapter 输出，而 mock run 约 5 秒即完成，心跳/租约超时不成立——最可能是容器跑旧代码（源码 COPY 进镜像，普通 `up`/`restart` 不生效）。排查同时暴露两个观测缺口：run 被 reaper 重排重跑时完全静默（重跑事件 seq 与旧轨迹重复，前端 `mergeBySeq` 去重后 UI 无任何变化），且 `/api/runs/{id}` 不暴露 `retries` 与 `lease_expires_at`——「active 状态 + NULL 租约」是 reaper 永不回收的唯一持久卡死形态，缺这两个字段无法定位。

**修复**：观测性增强——`/api/runs` 列表与详情暴露 `retries`，详情额外暴露 `lease_expires_at`；前端 RunSchema 同步新字段（全部可缺失，兼容旧后端），轨迹页在 LIVE 且 `retries > 0` 时显示「↻ 重跑中」徽标（悬停提示当前第几次尝试）。处置：容器环境务必 `docker compose up --build`（runner 与 server 都要）重建镜像；若 run 已卡死，`retries`/`lease_expires_at` 字段可直接区分「正在重跑」与「NULL 租约永久卡死」两种形态。
