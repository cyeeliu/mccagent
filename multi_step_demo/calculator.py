"""一个简单的计算器模块，支持基本运算和统计功能。"""


class Calculator:
    """基础计算器，支持加减乘除。"""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b


class Statistics:
    """统计计算器，支持均值、方差、最大最小值。"""

    def mean(self, numbers):
        if not numbers:
            raise ValueError("列表不能为空")
        return sum(numbers) / len(numbers)

    def variance(self, numbers):
        if not numbers:
            raise ValueError("列表不能为空")
        m = self.mean(numbers)
        return sum((x - m) ** 2 for x in numbers) / len(numbers)

    def max(self, numbers):
        if not numbers:
            raise ValueError("列表不能为空")
        return max(numbers)

    def min(self, numbers):
        if not numbers:
            raise ValueError("列表不能为空")
        return min(numbers)


if __name__ == "__main__":
    calc = Calculator()
    stats = Statistics()

    print("=== 计算器测试 ===")
    print(f"2 + 3 = {calc.add(2, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"5 * 6 = {calc.multiply(5, 6)}")
    print(f"20 / 4 = {calc.divide(20, 4)}")

    print("\n=== 统计测试 ===")
    data = [1, 2, 3, 4, 5]
    print(f"数据: {data}")
    print(f"均值: {stats.mean(data)}")
    print(f"方差: {stats.variance(data)}")
    print(f"最大值: {stats.max(data)}")
    print(f"最小值: {stats.min(data)}")
