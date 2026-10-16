#!/usr/bin/env bash
# 使用 MockLlm JSON 脚本通过 CLI 运行各示例。
# 用法：bash examples/run_examples.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. 纯文本对话 ==="
uv run mcc --mock-script examples/simple_text.json -p "What is the capital of France?"

echo ""
echo "=== 2. 写文件 → 读文件 ==="
uv run mcc --mock-script examples/write_then_read.json -p "Write and read a file"

echo ""
echo "=== 3. Bash 命令 ==="
uv run mcc --mock-script examples/bash_echo.json -p "Run echo"

echo ""
echo "=== 4. 并行工具调用 ==="
uv run mcc --mock-script examples/parallel_writes.json -p "Write three files"

echo ""
echo "=== 5. 探索代码库 ==="
uv run mcc --mock-script examples/explore_codebase.json -p "Explore the codebase"

echo ""
echo "=== 6. 编程式 MockLlm 示例 ==="
uv run python examples/run_mock.py

echo ""
echo "✅ All examples completed."
