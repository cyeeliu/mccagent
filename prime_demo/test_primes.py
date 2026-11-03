"""素数查找工具的单元测试 - 验证 v1 和 v2 结果一致性与正确性。"""

import unittest
from prime_v1 import is_prime_v1, find_primes_v1
from prime_v2 import is_prime_v2, find_primes_v2


class TestIsPrime(unittest.TestCase):
    """测试素数判断函数。"""

    # 已知素数
    KNOWN_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101, 9973]

    # 已知非素数
    KNOWN_NON_PRIMES = [0, 1, 4, 6, 8, 9, 10, 15, 21, 25, 100, 9999]

    def test_v1_known_primes(self):
        for p in self.KNOWN_PRIMES:
            with self.subTest(n=p):
                self.assertTrue(is_prime_v1(p), f"{p} 应该是素数")

    def test_v1_known_non_primes(self):
        for n in self.KNOWN_NON_PRIMES:
            with self.subTest(n=n):
                self.assertFalse(is_prime_v1(n), f"{n} 不应该是素数")

    def test_v2_known_primes(self):
        for p in self.KNOWN_PRIMES:
            with self.subTest(n=p):
                self.assertTrue(is_prime_v2(p), f"{p} 应该是素数")

    def test_v2_known_non_primes(self):
        for n in self.KNOWN_NON_PRIMES:
            with self.subTest(n=n):
                self.assertFalse(is_prime_v2(n), f"{n} 不应该是素数")

    def test_v1_v2_consistency(self):
        """v1 和 v2 对同一批数判断结果应一致。"""
        for n in range(0, 1000):
            with self.subTest(n=n):
                self.assertEqual(is_prime_v1(n), is_prime_v2(n))


class TestFindPrimes(unittest.TestCase):
    """测试批量素数查找函数。"""

    def test_small_limits(self):
        self.assertEqual(find_primes_v1(1), [])
        self.assertEqual(find_primes_v1(2), [2])
        self.assertEqual(find_primes_v1(10), [2, 3, 5, 7])
        self.assertEqual(find_primes_v2(1), [])
        self.assertEqual(find_primes_v2(2), [2])
        self.assertEqual(find_primes_v2(10), [2, 3, 5, 7])

    def test_v1_v2_same_results(self):
        """v1 和 v2 找出的素数列表应完全一致。"""
        for limit in [10, 100, 1000, 10000]:
            with self.subTest(limit=limit):
                self.assertEqual(find_primes_v1(limit), find_primes_v2(limit))

    def test_prime_count_10000(self):
        """10000 以内应有 1229 个素数（已知数学结论）。"""
        self.assertEqual(len(find_primes_v2(10000)), 1229)

    def test_all_results_are_prime(self):
        """筛法返回的每个数都应该是素数。"""
        primes = find_primes_v2(1000)
        for p in primes:
            with self.subTest(p=p):
                self.assertTrue(is_prime_v2(p))


class TestPerformance(unittest.TestCase):
    """性能对比测试。"""

    def test_v2_faster_than_v1(self):
        """v2 筛法应比 v1 朴素法快。"""
        import time

        start = time.perf_counter()
        find_primes_v1(5000)
        t_v1 = time.perf_counter() - start

        start = time.perf_counter()
        find_primes_v2(5000)
        t_v2 = time.perf_counter() - start

        self.assertLess(t_v2, t_v1, f"v2({t_v2:.4f}s) 应比 v1({t_v1:.4f}s) 快")


if __name__ == "__main__":
    unittest.main()
