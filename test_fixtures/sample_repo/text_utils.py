"""Text processing utilities."""

from helpers import clean_whitespace


def format_greeting(name: str) -> str:
    """Return a polite greeting for the given name.

    Example:
        >>> format_greeting("Alice")
        'Hello, Alice!'
    """
    clean_name = clean_whitespace(name)
    # BUG: Missing comma after 'Hello'
    return f"Hello {clean_name}!"


def truncate(text: str, max_length: int = 10) -> str:
    """Truncate text to max_length with an ellipsis if longer."""
    cleaned = clean_whitespace(text)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length] + "..."
