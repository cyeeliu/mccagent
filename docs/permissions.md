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

### 档位

| mode | 隔离手段 | 禁网 | 说明 |
|---|---|---|---|
| `none`（默认） | 无 | — | 行为与历史版本一致 |
| `restrictions` | rlimit（内存/CPU/进程数/文件大小）+ 环境变量白名单 + umask 077 + HOME 重定向 | ❌ 做不到 | 纯 Python 实现，任何环境可用 |
| `bubblewrap` | bwrap 命名空间：系统目录只读绑定 + 独立 tmpfs /tmp + `--unshare-net` + `--die-with-parent` | ✅ | 不可用时**报错**而非静默降级 |
| `auto` | 依次探测 bubblewrap → restrictions | 随档 | 降级在 stderr 提示一次 |

### 配置

```json
{
  "sandbox": {
    "mode": "auto",
    "network": "off",
    "max_memory_mb": 512,
    "max_cpu_secs": 60,
    "max_processes": 64,
    "max_file_mb": 64,
    "env_allow": ["PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL"]
  }
}
```

- 数值为 `0` 表示该项不限制
- `restrictions` 档下 `HOME` 被重定向到工作目录内 `.sandbox-home/`，环境变量仅白名单可见（API key 等天然不出现在子进程里）
- `RLIMIT_NPROC` 语义是 per-uid 进程总数：单用户工作站/容器即等价限制本沙箱，共享多用户机器请评估该 uid 既有进程数
- `env_allow` 同时作用于 restrictions 与 bubblewrap 档
- project 级配置显式设置过 sandbox（任何字段偏离默认）时整体覆盖 home 级

### 选型建议

- 桌面 / 容器开发机：`auto`（有 bwrap 用 bwrap，没有降 restrictions）
- CI / 受限容器（无 user namespace 权能）：`restrictions`
- 需要禁网（数据防泄露）且可装包：`bubblewrap`
- 更强隔离（gVisor / microVM）属于服务器集群部署层，超出单机 CLI 场景
