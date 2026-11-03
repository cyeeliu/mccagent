"""calculator 模块的单元测试。"""

import unittest
from calculator import Calculator, Statistics


class TestCalculator(unittest.TestCase):
    """Calculator 类的测试用例。"""

    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertAlmostEqual(self.calc.add(0.1, 0.2), 0.3, places=7)

    def test_subtract(self):
        self.assertEqual(self.calc.subtract(10, 4), 6)
        self.assertEqual(self.calc.subtract(0, 0), 0)
        self.assertEqual(self.calc.subtract(-5, -3), -2)

    def test_multiply(self):
        self.assertEqual(self.calc.multiply(5, 6), 30)
        self.assertEqual(self.calc.multiply(0, 100), 0)
        self.assertEqual(self.calc.multiply(-2, 3), -6)

    def test_divide(self):
        self.assertEqual(self.calc.divide(20, 4), 5)
        self.assertAlmostEqual(self.calc.divide(1, 3), 0.3333, places=4)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)


class TestStatistics(unittest.TestCase):
    """Statistics 类的测试用例。"""

    def setUp(self):
        self.stats = Statistics()

    def test_mean(self):
        self.assertEqual(self.stats.mean([1, 2, 3, 4, 5]), 3)
        self.assertAlmostEqual(self.stats.mean([1, 2]), 1.5)

    def test_variance(self):
        self.assertEqual(self.stats.variance([1, 2, 3, 4, 5]), 2.0)

    def test_max(self):
        self.assertEqual(self.stats.max([3, 7, 1, 9, 2]), 9)

    def test_min(self):
        self.assertEqual(self.stats.min([3, 7, 1, 9, 2]), 1)

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            self.stats.mean([])
        with self.assertRaises(ValueError):
            self.stats.variance([])
        with self.assertRaises(ValueError):
            self.stats.max([])
        with self.assertRaises(ValueError):
            self.stats.min([])


if __name__ == "__main__":
    unittest.main()
