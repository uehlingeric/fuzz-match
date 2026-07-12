"""Tests for fuzzy matching core functions."""

import pytest

from fuzz_match.core import (
    matrix_cosine,
    normalize_text,
    rapid_fuzz_wratio,
)


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        assert normalize_text("APPLE") == "apple"

    def test_punctuation_removal(self):
        assert normalize_text("Apple, Inc.") == "apple inc"

    def test_unicode(self):
        assert normalize_text("Café") == "cafe"

    def test_empty(self):
        assert normalize_text("") == ""


class TestMatrixCosine:
    """Tests for TF-IDF cosine similarity matching."""

    def test_exact_match(self):
        """Exact matches should score high."""
        to_match = ["Apple"]
        to_match_to = ["Apple"]
        result = matrix_cosine(to_match, to_match_to)
        assert len(result) == 1
        assert result.iloc[0]["score"] == 100.0

    def test_similar_match(self):
        """Similar strings should score above baseline."""
        to_match = ["Apple Inc"]
        to_match_to = ["Apple Computer"]
        result = matrix_cosine(to_match, to_match_to)
        assert len(result) == 1
        assert result.iloc[0]["score"] > 20

    def test_skip_100(self):
        """skip_100=True should exclude exact matches."""
        to_match = ["Apple"]
        to_match_to = ["Apple", "Apples"]
        result = matrix_cosine(to_match, to_match_to, skip_100=True)
        assert len(result) == 1
        assert result.iloc[0]["matched_name"] != "Apple"

    def test_multiple_items(self):
        """Should handle multiple items."""
        to_match = ["Apple", "Microsoft"]
        to_match_to = ["Apple Inc", "Microsoft Corp"]
        result = matrix_cosine(to_match, to_match_to)
        assert len(result) == 2


class TestRapidFuzzWRatio:
    """Tests for Levenshtein WRatio matching."""

    def test_exact_match(self):
        """Exact matches should score 100."""
        to_match = ["Apple"]
        to_match_to = ["Apple"]
        result = rapid_fuzz_wratio(to_match, to_match_to)
        assert len(result) == 1
        assert result.iloc[0]["score"] == 100

    def test_typo_match(self):
        """Typos should score > 70."""
        to_match = ["Apple"]
        to_match_to = ["Aple"]
        result = rapid_fuzz_wratio(to_match, to_match_to)
        assert len(result) == 1
        assert result.iloc[0]["score"] > 70

    def test_skip_100(self):
        """skip_100=True should exclude exact matches."""
        to_match = ["Apple"]
        to_match_to = ["Apple", "Orange"]
        result = rapid_fuzz_wratio(to_match, to_match_to, skip_100=True)
        assert len(result) == 1
        assert result.iloc[0]["matched_name"] == "Orange"

    def test_multiple_items(self):
        """Should handle multiple items."""
        to_match = ["Apple", "Banana"]
        to_match_to = ["Apple", "Banana"]
        result = rapid_fuzz_wratio(to_match, to_match_to)
        assert len(result) == 2
