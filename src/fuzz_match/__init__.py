"""High-performance fuzzy string matching."""

from .core import matrix_cosine, rapid_fuzz_wratio

__all__ = ["matrix_cosine", "rapid_fuzz_wratio"]
