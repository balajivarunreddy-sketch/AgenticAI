"""Focused tests for the calculator's supported and rejected expressions."""

import unittest

from tools.calculator_tool import calculate_expression


class CalculatorTests(unittest.TestCase):
    def test_calculates_nested_arithmetic(self) -> None:
        self.assertEqual(calculate_expression("(12 + 8) * 3 / 2"), 30)

    def test_rejects_python_execution(self) -> None:
        with self.assertRaises(ValueError):
            calculate_expression("__import__('os').getcwd()")

    def test_bounds_exponents(self) -> None:
        with self.assertRaises(ValueError):
            calculate_expression("2 ** 101")


if __name__ == "__main__":
    unittest.main()
