"""素数查找工具 - 优化版本（试除到√n + 埃拉托斯特尼筛法）。"""

import math


def is_prime_v2(n):
    """判断 n 是否为素数 - 优化：只试除到 √n，跳过偶数。"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # 只检查奇数，到 √n 为止
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def find_primes_v2(limit):
    """找出 limit 以内的所有素数 - 埃拉托斯特尼筛法。"""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            # 从 i*i 开始标记，步长为 i
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


if __name__ == "__main__":
    import time

    print("=== 素数查找 v2（优化版：√n试除 + 筛法）===")

    # 测试 is_prime_v2
    start = time.perf_counter()
    primes_check = [n for n in range(2, 10001) if is_prime_v2(n)]
    t_check = time.perf_counter() - start
    print(f"[is_prime_v2] 10000 以内素数个数: {len(primes_check)}, 耗时: {t_check:.4f}s")

    # 测试 find_primes_v2（筛法）
    start = time.perf_counter()
    primes_sieve = find_primes_v2(10000)
    t_sieve = time.perf_counter() - start
    print(f"[find_primes_v2] 10000 以内素数个数: {len(primes_sieve)}, 耗时: {t_sieve:.4f}s")

    print(f"前 10 个素数: {primes_sieve[:10]}")
    print(f"结果一致: {primes_check == primes_sieve}")
