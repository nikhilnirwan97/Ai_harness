"""Tests for calculator — these SHOULD pass once the bugs are fixed."""

import pytest
from calculator import add, subtract, multiply, divide


class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -1) == -2

    def test_zero(self):
        assert add(0, 0) == 0


class TestSubtract:
    def test_basic(self):
        assert subtract(10, 4) == 6

    def test_negative_result(self):
        assert subtract(3, 7) == -4


class TestMultiply:
    def test_basic(self):
        assert multiply(3, 4) == 12

    def test_by_zero(self):
        assert multiply(5, 0) == 0


class TestDivide:
    def test_integer_division(self):
        """10 / 3 should give 3.333..., NOT 3 (integer division)."""
        result = divide(10, 3)
        assert result == pytest.approx(3.3333, rel=1e-3)

    def test_exact_division(self):
        assert divide(10, 2) == 5.0

    def test_divide_by_zero_raises_value_error(self):
        """The docstring says ValueError, so it must be ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(1, 0)
