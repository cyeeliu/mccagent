"""计算器模块单元测试"""
import os
import sys

import pytest

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calculator import add, divide, multiply, subtract


class TestAdd:
    """加法运算测试"""

    def test_positive_numbers(self):
        assert add(3, 5) == 8.0

    def test_negative_numbers(self):
        assert add(-3, -5) == -8.0

    def test_mixed_sign(self):
        assert add(-3, 5) == 2.0

    def test_float_numbers(self):
        assert add(1.5, 2.5) == 4.0

    def test_zero(self):
        assert add(0, 0) == 0.0

    def test_type_error(self):
        with pytest.raises(TypeError):
            add("3", 5)


class TestSubtract:
    """减法运算测试"""

    def test_positive_numbers(self):
        assert subtract(10, 3) == 7.0

    def test_negative_numbers(self):
        assert subtract(-3, -5) == 2.0

    def test_float_numbers(self):
        assert subtract(5.5, 2.5) == 3.0

    def test_zero_result(self):
        assert subtract(5, 5) == 0.0

    def test_type_error(self):
        with pytest.raises(TypeError):
            subtract(3, None)


class TestMultiply:
    """乘法运算测试"""

    def test_positive_numbers(self):
        assert multiply(3, 4) == 12.0

    def test_negative_numbers(self):
        assert multiply(-3, -4) == 12.0

    def test_mixed_sign(self):
        assert multiply(-3, 4) == -12.0

    def test_float_numbers(self):
        assert multiply(2.5, 4) == 10.0

    def test_multiply_by_zero(self):
        assert multiply(100, 0) == 0.0

    def test_type_error(self):
        with pytest.raises(TypeError):
            multiply([1], 2)


class TestDivide:
    """除法运算测试"""

    def test_positive_numbers(self):
        assert divide(10, 2) == 5.0

    def test_negative_numbers(self):
        assert divide(-10, -2) == 5.0

    def test_mixed_sign(self):
        assert divide(-10, 2) == -5.0

    def test_float_result(self):
        assert divide(10, 3) == pytest.approx(3.3333, rel=1e-3)

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="除数不能为零"):
            divide(10, 0)

    def test_zero_divided(self):
        assert divide(0, 5) == 0.0

    def test_type_error(self):
        with pytest.raises(TypeError):
            divide(10, "2")
