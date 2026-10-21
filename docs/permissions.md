# 权限与 Hooks

## 权限规则语法

`settings.json` 的 `permissions` 段：

```json
{
  "permissions": {
    "mode": "default",
    "allow": ["Read(**)", "Bash(git status:*)"],
    "deny": ["Bash(rm:*)", "Read(.env)"]
  }
}
```

规则格式 `Category(匹配)`：

| 规则 | 类别 | 匹配语义 |
|---|---|---|
| `Bash(git:*)` | Bash | 命令前缀匹配，**词边界**约束（见下） |
| `Read(**/*.py)` | Read | 文件路径 glob（fnmatch） |
| `Write(src/**)` | Write | 文件路径 glob |
| `Edit(**)` | Edit | 文件路径 glob |

### bash 词边界匹配

前缀匹配在词边界处截断，防止 `Bash(rm:*)` 误伤 `rmdir`：

- `rm -rf /` → 匹配 `rm`（rm 后是空格）
- `rmdir /tmp` → 不匹配 `rm`（不同命令）
- `rm` → 匹配（完整相等）
- `rm/rerun.sh` → 匹配（`/` 视为边界）

## 决策顺序

```
deny 匹配 ──► Deny（最高优先）
allow 匹配 ──► Allow
无匹配    ──► mode 默认策略
```

mode 默认策略（`permissions.mode`）：

| mode | 行为 |
|---|---|
| `default` | Read / Memory / Subagent 放行；Write / Edit / Bash 交互确认 |
| `acceptEdits` | 同上，但 Write / Edit 直接放行 |
| `bypassPermissions` | 全部放行（危险，仅沙箱使用） |

### 各工具的权限类别

| 工具 | 类别 | 匹配对象 | 默认 |
|---|---|---|---|
| `read_file` / `list_dir` / `grep` / `glob` | Read | path | 放行 |
| `write_file` | Write | path | 确认 |
| `edit_file` | Edit | path | 确认 |
| `run_bash` | Bash | command | 确认 |
| `save_memory` / `read_memory` | Memory | name | 放行 |
| `spawn_subagent` | Subagent | `spawn_subagent <preset>` | 放行 |

`spawn_subagent` 映射为 bash 风格命令 `spawn_subagent <preset>`，因此 `Bash(spawn_subagent:*)` 可禁用全部派生、`Bash(spawn_subagent test_runner:*)` 只禁某个预设。子代理默认放行的安全性由两层保证：子代理内部的**每个**工具调用仍经过同一权限检查器；且子代理没有交互确认通道，`Ask` 即拒绝。

## 交互式确认（TUI）

默认策略下需要确认的工具会发出 `PermissionRequest` 事件，TUI 显示权限面板：

| 按键 | 决策 | 效果 |
|---|---|---|
| `y` | 允许一次 | 本次放行 |
| `a` | 总是允许 | 追加运行时 allow 规则（精确匹配本次命令/路径） |
| `n` | 拒绝 | 工具收到 "permission denied by user" |
| `e` | 拒绝+反馈 | 输入原因，随拒绝结果回传给 Agent |

headless / 子代理上下文中无响应通道，等待超时（默认 120s）后默认拒绝。

## Hooks

### 配置

```json
{
  "hooks": {
    "pre_tool_use": [
      {"matcher": "run_bash", "command": "scripts/guard.sh"}
    ],
    "post_tool_use": []
  }
}
```

| 字段 | 说明 |
|---|---|
| `matcher` | 工具名 glob（`*` 匹配全部），仅对 pre/post-tool 生效 |
| `command` | 脚本路径，相对路径基于工作目录 |

### 执行契约

外部脚本通过环境变量接收上下文：

| 变量 | 内容 |
|---|---|
| `MCC_HOOK_TYPE` | `pre_tool` / `post_tool` |
| `MCC_TOOL_NAME` / `MCC_TOOL_INPUT` / `MCC_TOOL_OUTPUT` | 工具名 / 输入 JSON / 输出 |
| `MCC_IS_ERROR` | 工具是否失败 |
| `MCC_SESSION_ID` / `MCC_ITERATION` / `MCC_COST_USD` | 会话上下文 |

退出码语义：

- `0` → allow，stdout 作为 Notice 消息展示
- 非 `0` → **deny**（pre-tool 阻止工具执行），stderr 作为拒绝原因回传 Agent

脚本超时（默认 10s）或执行失败视为 allow（附错误消息），钩子故障不阻断主流程。

## 沙箱

权限是**逻辑层**（工具「应该做」什么），沙箱是**物理层**（工具「能做」什么）——即使 prompt injection 骗过模型与权限检查，破坏性操作在沙箱内物理不可达。两层正交：deny 规则永远先行，沙箱在其后兜底。沙箱只作用于 `run_bash`。

**默认开启**（`mode: auto`）：无 bwrap 环境降级为资源限制档并在 stderr 提示一次；限额默认按「防失控不挡正常」校准（内存 2GB / CPU 600s / 进程 128 / 单文件 1GB），编译类任务不受影响。完全关闭需显式 `mode: none`。

### 档位

| mode | 隔离手段 | 禁网 | 说明 |
|---|---|---|---|
| `none` | 无 | — | 完全关闭（显式声明） |
| `restrictions` | rlimit + 环境策略（见下） | ❌ 做不到 | 纯 Python 实现，任何环境可用 |
| `bubblewrap` | bwrap 命名空间：系统目录只读绑定 + 独立 tmpfs /tmp + 按需 `--unshare-net` + `--die-with-parent` | `network: off` 时 | 恒定环境清洗 + HOME 重定向（真实 HOME 在沙箱内不可见）；不可用时**报错**而非静默降级 |
| `auto`（默认） | 探测 bubblewrap → restrictions | 随档 | 降级在 stderr 提示一次 |

### 环境策略 env_policy

| 策略 | 子进程环境 | 适用 |
|---|---|---|
| `inherit`（默认） | 完整继承（PATH / HOME / 工具链变量原样），仅 rlimit 防失控 | harness 日常——Agent 能用项目 venv、cargo 等用户工具链 |
| `allowlist` | 白名单变量；PATH 重置系统默认（不继承 harness venv）；umask 077；`redirect_home` 时 HOME 落工作目录内 | 评测与高隔离——防被测 Agent 污染 harness 环境 |

### 配置

```json
{
  "sandbox": {
    "mode": "auto",
    "network": "on",
    "env_policy": "inherit",
    "redirect_home": false,
    "max_memory_mb": 2048,
    "max_cpu_secs": 600,
    "max_processes": 128,
    "max_file_mb": 1024,
    "env_allow": ["PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL"]
  }
}
```

- 数值为 `0` 表示该项不限制
- `network` 默认 `on`（clone / pip 是 harness 刚需）；防数据泄露场景显式 `off`
- `env_allow` 仅在 `env_policy: allowlist` 与 bubblewrap 档生效
- `RLIMIT_NPROC` 语义是 per-uid 进程总数：单用户工作站/容器即等价限制本沙箱，共享多用户机器请评估该 uid 既有进程数
- project 级配置显式设置过 sandbox（任何字段偏离默认）时整体覆盖 home 级

评测（mcc-eval）固定注入「零限额 + allowlist + HOME 重定向」清洗档：被测 Agent 的 shell 不继承 harness 的 venv 与环境变量，其 `pip install` 落在 case 工作目录内；资源不限（评测衡量 Agent 能力，失控防护由 runner 超时兜底）。

### 选型建议

- 默认（auto + inherit + network on）：日常开发——rlimit 防失控，工具链无感
- 要禁网防泄露：`network: off`（bwrap 可用时生效）
- 强隔离（不受信任务）：`env_policy: allowlist` + `redirect_home: true`
- CI / 受限容器（无 user namespace 权能）：显式 `restrictions` 免去每次探测
- 更强隔离（gVisor / microVM）属于服务器集群部署层，超出单机 CLI 场景
