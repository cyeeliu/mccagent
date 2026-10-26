# Skills 与 Slash Commands

mcc 支持可复用的技能包（Skill）与一行别名（Slash Command），采用懒加载设计：system prompt 只注入轻量目录（name + description），模型按需调用 `load_skill` 工具读取完整指令。

## 概念

| 概念 | 定位 | 格式 |
|---|---|---|
| SlashCommand | `/xxx` 一行别名 | 单 `.md` 文件 |
| Skill | 可复用能力包 | 目录（`SKILL.md`） |

Slash command 往往触发 Skill；Skill 可包含多个 slash command。

## 目录布局

两层合并，project 同名覆盖 home：

```
~/.mcc/skills/<name>/SKILL.md     # home 级（用户全局，不入库）
skills/<name>/SKILL.md            # project 级（团队共享，入 git）
~/.mcc/commands/<name>.md         # home 级 slash
commands/<name>.md                # project 级 slash
```

## SKILL.md 格式

```markdown
---
name: pr-reviewer
description: 在用户要求 code review 时使用
triggers:
  - /review
  - review this PR
model: claude-opus-4-7
requires_tools:
  - read_file
  - run_bash
  - grep
---

# PR Reviewer

当触发时：
1. git diff origin/main...
2. 读相关文件理解上下文
3. 按 critical / major / minor 三档给意见
```

### Frontmatter 字段

| 字段 | 必需 | 说明 |
|---|---|---|
| name | 否（缺省用目录名） | 技能名 |
| description | 否（缺省用正文首行） | 一句话摘要，进 catalog |
| triggers | 否 | 字面触发词列表 |
| model | 否 | 激活时覆盖当前模型 |
| requires_tools | 否 | 声明所需工具（写入供模型自律） |

## SlashCommand 格式

```markdown
---
command: /testCm
description: Run tests and summarize
---

执行 pytest，列出失败摘要。
```

`command` 缺省时用 `/<文件名>`。用户输入 `/test -xvs` 时，`-xvs` 作为「用户补充参数」追加到 body 末尾。

## 工作原理：懒加载

1. 启动时扫描 `skills/` 与 `commands/` 目录
2. system prompt 注入轻量 catalog（仅 name + description）
3. 用户输入 `/test` 时，SlashCommand 替换为 body（进 AgentLoop 前，零 LLM 调用）
4. 模型自主调用 `load_skill(name)` 工具，按需读取完整 SKILL.md

未激活的技能不占上下文 token。

## 配置

### settings.json

```json
{
  "skills": {
    "enabled": false
  }
}
```

`enabled: false` 禁用技能加载（不扫描目录、不注入 catalog、不注册 load_skill）。

### CLI

```sh
mcc --no-skills          # 禁用技能与 slash 命令加载
```

## 示例

本项目自带：

- `skills/test-runner/` — 运行项目测试并汇总失败
- `commands/commit.md` — `/commit` 提交代码
- `commands/analyze-repo.md` — `/analyze-repo` 并行探索仓库架构
