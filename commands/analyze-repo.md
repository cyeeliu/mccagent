---
command: /analyze-repo
description: 并行探索当前仓库的架构
---

请并行启动 4 个子 Agent 分析当前仓库：

1. spawn_subagent(code_explorer, "分析入口点（main/bin）")
2. spawn_subagent(code_explorer, "分析核心业务模块（core/domain）")
3. spawn_subagent(code_explorer, "分析 I/O 层（db/api/rpc）")
4. spawn_subagent(code_explorer, "分析测试组织")

收到 4 份摘要后，综合成一个整体架构说明：模块图、关键抽象、风险点。
