"""Utility helpers used by the calculator."""


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between low and high (inclusive)."""
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))


def is_positive(n: float) -> bool:
    """Return True if n is strictly positive."""
    return n > 0
