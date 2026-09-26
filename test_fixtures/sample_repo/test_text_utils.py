"""Unit tests for text processing utilities."""

import pytest
from text_utils import format_greeting, truncate
from helpers import clean_whitespace


def test_clean_whitespace():
    assert clean_whitespace("   hello    world  \n ") == "hello world"


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello world this is long", 5) == "hello..."


def test_format_greeting():
    """format_greeting should return 'Hello, <name>!' with a comma."""
    assert format_greeting("Alice") == "Hello, Alice!"
