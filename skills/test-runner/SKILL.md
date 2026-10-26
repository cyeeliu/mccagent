---
name: test-runner
description: 运行项目测试并汇总失败
triggers:
  - "/test"
  - "run tests"
  - "跑测试"
requires_tools:
  - run_bash
  - read_file
---

# Test Runner

当用户要求运行测试时：

1. 执行项目的测试命令（pytest / cargo nextest / npm test 等）
2. 如有失败，列出每个失败的测试名与错误摘要
3. 建议下一步修复动作
