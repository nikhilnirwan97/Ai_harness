"""Simple calculator module with a deliberate bug in divide."""


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of a and b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return a divided by b.

    Raises:
        ValueError: If b is zero.
    """
    # BUG: integer division instead of true division, and wrong exception type
    if b == 0:
        raise TypeError("Cannot divide by zero")  # BUG: should be ValueError
    return a // b  # BUG: should be a / b
