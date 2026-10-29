"""A simple standard for sharing entity annotations."""

from .api import hello, square

# being explicit about exports is important!
__all__ = [
    "hello",
    "square",
]
