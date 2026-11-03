# 计算器模块设计文档

## 1. 需求概述

实现一个简单的 Python 计算器模块，提供四则运算功能（加、减、乘、除），支持整数和浮点数运算，具备完善的异常处理。

## 2. 模块结构

| 函数名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `add(a, b)` | `float, float` | `float` | 加法运算 a + b |
| `subtract(a, b)` | `float, float` | `float` | 减法运算 a - b |
| `multiply(a, b)` | `float, float` | `float` | 乘法运算 a * b |
| `divide(a, b)` | `float, float` | `float` | 除法运算 a / b，b=0 时抛出 ValueError |

## 3. 异常处理设计

- **除零异常**：当 `divide` 的除数 b 为 0 时，抛出 `ValueError("除数不能为零")`
- **类型检查**：所有函数使用 `isinstance` 校验参数为 `int` 或 `float`，否则抛出 `TypeError("参数必须是数字类型")`

## 4. 目录结构

```
calculator_project/
├── src/
│   └── calculator.py      # 核心模块
├── tests/
│   └── test_calculator.py  # 单元测试
└── docs/
    └── design.md           # 设计文档
```

## 5. 测试策略

- 使用 `pytest` 框架
- 测试覆盖：
  - 正常用例（正数、负数、浮点数）
  - 边界用例（除零、零值运算）
  - 异常用例（非数字类型参数）
- 目标：100% 函数覆盖，所有测试通过
