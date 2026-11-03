"""数字工具模块 — 提供质数判断、斐波那契数列、最大公约数、最小公倍数等常用数论函数。"""


def is_prime(n: int) -> bool:
    """判断 n 是否为质数。

    Args:
        n: 待判断的整数。

    Returns:
        True 如果 n 是质数，否则 False。n < 2 一律返回 False。
    """
    if n < 2:
        return False
    if n < 4:
        return True  # 2, 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    # 6k±1 优化
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def fibonacci(n: int) -> list[int]:
    """返回前 n 个斐波那契数。

    序列定义: F(0)=0, F(1)=1, F(k)=F(k-1)+F(k-2)。

    Args:
        n: 要返回的斐波那契数个数。

    Returns:
        包含前 n 个斐波那契数的列表。n <= 0 返回空列表。
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for _ in range(2, n):
        result.append(result[-1] + result[-2])
    return result


def gcd(a: int, b: int) -> int:
    """计算 a 和 b 的最大公约数（欧几里得算法）。

    结果始终为非负整数。gcd(0, 0) 返回 0。

    Args:
        a: 第一个整数。
        b: 第二个整数。

    Returns:
        |a| 和 |b| 的最大公约数。
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """计算 a 和 b 的最小公倍数。

    任一参数为 0 时返回 0。

    Args:
        a: 第一个整数。
        b: 第二个整数。

    Returns:
        |a| 和 |b| 的最小公倍数；若 a 或 b 为 0 则返回 0。
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
