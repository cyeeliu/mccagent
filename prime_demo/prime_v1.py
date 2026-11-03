"""素数查找工具 - 初始版本（朴素试除法）。"""


def is_prime_v1(n):
    """判断 n 是否为素数 - 朴素方法：试除 2 到 n-1。"""
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def find_primes_v1(limit):
    """找出 limit 以内的所有素数 - 逐个判断。"""
    return [n for n in range(2, limit + 1) if is_prime_v1(n)]


if __name__ == "__main__":
    import time

    print("=== 素数查找 v1（朴素试除法）===")
    start = time.perf_counter()
    primes = find_primes_v1(10000)
    elapsed = time.perf_counter() - start
    print(f"10000 以内的素数个数: {len(primes)}")
    print(f"前 10 个素数: {primes[:10]}")
    print(f"耗时: {elapsed:.4f}s")
