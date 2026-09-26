"""Helper functions for string processing."""


def clean_whitespace(text: str) -> str:
    """Normalize and strip redundant whitespaces."""
    if not text:
        return ""
    return " ".join(text.strip().split())
