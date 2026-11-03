"""number_utils.py 的单元测试。"""

from number_utils import is_prime, fibonacci, gcd, lcm


# ── is_prime ──────────────────────────────────────────────
class TestIsPrime:
    def test_primes(self):
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 97):
            assert is_prime(p) is True, f"{p} should be prime"

    def test_composites(self):
        for c in (4, 6, 8, 9, 10, 12, 15, 21, 100):
            assert is_prime(c) is False, f"{c} should not be prime"

    def test_zero_and_one(self):
        assert is_prime(0) is False
        assert is_prime(1) is False

    def test_negative(self):
        assert is_prime(-1) is False
        assert is_prime(-7) is False
        assert is_prime(-13) is False


# ── fibonacci ─────────────────────────────────────────────
class TestFibonacci:
    def test_zero(self):
        assert fibonacci(0) == []

    def test_one(self):
        assert fibonacci(1) == [0]

    def test_five(self):
        assert fibonacci(5) == [0, 1, 1, 2, 3]

    def test_ten(self):
        assert fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_negative(self):
        assert fibonacci(-1) == []


# ── gcd ───────────────────────────────────────────────────
class TestGcd:
    def test_normal(self):
        assert gcd(12, 18) == 6

    def test_coprime(self):
        assert gcd(7, 13) == 1

    def test_with_zero(self):
        assert gcd(0, 5) == 5
        assert gcd(5, 0) == 5

    def test_both_zero(self):
        assert gcd(0, 0) == 0

    def test_negative(self):
        assert gcd(-12, 18) == 6
        assert gcd(-12, -18) == 6


# ── lcm ───────────────────────────────────────────────────
class TestLcm:
    def test_normal(self):
        assert lcm(4, 6) == 12

    def test_coprime(self):
        assert lcm(3, 7) == 21

    def test_with_zero(self):
        assert lcm(0, 5) == 0
        assert lcm(5, 0) == 0

    def test_negative(self):
        assert lcm(-4, 6) == 12
        assert lcm(-4, -6) == 12
