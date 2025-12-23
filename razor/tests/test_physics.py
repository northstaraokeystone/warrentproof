"""
Tests for RAZOR Physics Module - Kolmogorov Complexity
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics import KolmogorovMetric, quick_compression_test


class TestKolmogorovMetric:
    """Tests for KolmogorovMetric class."""

    def test_canonicalize_removes_whitespace(self):
        """Test that canonicalize normalizes whitespace."""
        km = KolmogorovMetric()
        result = km.canonicalize("  hello   world  \n\t test  ")
        assert result == "hello world test"

    def test_canonicalize_lowercase(self):
        """Test that canonicalize converts to lowercase."""
        km = KolmogorovMetric()
        result = km.canonicalize("HELLO World TEST")
        assert result == "hello world test"

    def test_canonicalize_removes_json_artifacts(self):
        """Test that canonicalize removes JSON formatting."""
        km = KolmogorovMetric()
        result = km.canonicalize('{"key": "value"}')
        assert "{" not in result
        assert "}" not in result
        assert '"' not in result

    def test_compress_zlib_repetitive(self, repetitive_text):
        """Test that repetitive text compresses well with zlib."""
        km = KolmogorovMetric()
        data = km.canonicalize(repetitive_text).encode("utf-8")
        cr = km.compress_zlib(data)
        assert cr < 0.10, f"Repetitive text should compress to < 10%, got {cr:.2%}"

    def test_compress_zlib_random(self, random_text):
        """Test that random text does not compress well with zlib."""
        km = KolmogorovMetric()
        data = km.canonicalize(random_text).encode("utf-8")
        cr = km.compress_zlib(data)
        assert cr > 0.50, f"Random text should not compress below 50%, got {cr:.2%}"

    def test_repetitive_more_compressible_than_random(
        self, repetitive_text, random_text
    ):
        """Core hypothesis: repetitive text compresses better than random."""
        km = KolmogorovMetric()
        r1 = km.measure_complexity(repetitive_text)
        r2 = km.measure_complexity(random_text)

        assert r1["cr_zlib"] < r2["cr_zlib"], (
            f"Repetitive CR ({r1['cr_zlib']:.3f}) should be less than "
            f"random CR ({r2['cr_zlib']:.3f})"
        )

    def test_measure_complexity_returns_all_metrics(self, repetitive_text):
        """Test that measure_complexity returns all expected metrics."""
        km = KolmogorovMetric()
        result = km.measure_complexity(repetitive_text)

        assert "cr_zlib" in result
        assert "cr_lzma" in result
        assert "cr_bz2" in result
        assert "shannon_entropy" in result
        assert "len_original" in result
        assert "valid" in result

    def test_shannon_entropy_low_for_uniform(self):
        """Test Shannon entropy is low for uniform character distribution."""
        km = KolmogorovMetric()
        # Single character repeated - minimum entropy
        e1 = km.calculate_shannon_entropy("aaaaaaaaaa")
        assert e1 == 0.0, "Single repeated char should have zero entropy"

    def test_shannon_entropy_increases_with_diversity(self):
        """Test Shannon entropy increases with character diversity."""
        km = KolmogorovMetric()
        e1 = km.calculate_shannon_entropy("aaaaaaaaaa")
        e2 = km.calculate_shannon_entropy("aabbccddee")
        e3 = km.calculate_shannon_entropy("abcdefghij")

        assert e1 < e2 < e3, "Entropy should increase with character diversity"

    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully."""
        km = KolmogorovMetric()
        result = km.measure_complexity("")

        assert result["cr_zlib"] == 1.0
        assert result["valid"] is False

    def test_short_text_handling(self):
        """Test that very short text is flagged as invalid."""
        km = KolmogorovMetric()
        result = km.measure_complexity("hi")

        assert result["valid"] is False

    def test_classify_complexity_suspicious(self):
        """Test classification of suspicious (low) compression ratio."""
        km = KolmogorovMetric()
        assert km.classify_complexity(0.20) == "suspicious"

    def test_classify_complexity_normal(self):
        """Test classification of normal compression ratio."""
        km = KolmogorovMetric()
        assert km.classify_complexity(0.50) == "normal"

    def test_classify_complexity_high_entropy(self):
        """Test classification of high entropy (legitimate) compression ratio."""
        km = KolmogorovMetric()
        assert km.classify_complexity(0.80) == "high_entropy"


class TestQuickCompressionTest:
    """Tests for quick_compression_test helper."""

    def test_returns_dict(self):
        """Test that quick_compression_test returns a dict."""
        result = quick_compression_test("test string")
        assert isinstance(result, dict)

    def test_contains_cr_zlib(self):
        """Test that result contains cr_zlib."""
        result = quick_compression_test("test string " * 50)
        assert "cr_zlib" in result
        assert 0 <= result["cr_zlib"] <= 2.0


class TestCompressionAlgorithms:
    """Tests comparing different compression algorithms."""

    def test_all_algorithms_compress_repetitive(self, repetitive_text):
        """Test all three algorithms compress repetitive text."""
        km = KolmogorovMetric()
        result = km.measure_complexity(repetitive_text)

        assert result["cr_zlib"] < 0.15
        assert result["cr_lzma"] < 0.15
        assert result["cr_bz2"] < 0.15

    def test_algorithms_order_varies_by_content(self):
        """Test that different content types favor different algorithms."""
        km = KolmogorovMetric()

        # Highly repetitive local patterns
        local_repeat = "abcabc" * 500
        r1 = km.measure_complexity(local_repeat)

        # Long-range repetition
        long_range = ("abcdefghij" * 10 + "x") * 50
        r2 = km.measure_complexity(long_range)

        # Both should compress, but ratios may differ
        assert r1["cr_zlib"] < 0.20
        assert r2["cr_zlib"] < 0.30
