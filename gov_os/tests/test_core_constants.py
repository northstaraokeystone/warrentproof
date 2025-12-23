"""
Tests for Gov-OS Core Constants

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    VOLATILITY_ALPHA,
    RAF_MIN_CYCLE_LENGTH,
    CASCADE_DERIVATIVE_THRESHOLD,
    HOLOGRAPHIC_DETECTION_PROB,
    THOMPSON_FP_TARGET,
    COMPLETENESS_THRESHOLD,
    DISCLAIMER,
)


class TestConstants(unittest.TestCase):
    """Test universal physics constants."""

    def test_compression_bounds(self):
        """Legitimate floor must be above fraud ceiling."""
        self.assertGreater(COMPRESSION_LEGITIMATE_FLOOR, COMPRESSION_FRAUD_CEILING)
        self.assertEqual(COMPRESSION_LEGITIMATE_FLOOR, 0.85)
        self.assertEqual(COMPRESSION_FRAUD_CEILING, 0.60)

    def test_volatility_alpha(self):
        """Alpha should be small positive value."""
        self.assertGreater(VOLATILITY_ALPHA, 0)
        self.assertLess(VOLATILITY_ALPHA, 1)
        self.assertEqual(VOLATILITY_ALPHA, 0.1)

    def test_raf_min_cycle(self):
        """Minimum cycle length should be 3 (triangle)."""
        self.assertEqual(RAF_MIN_CYCLE_LENGTH, 3)

    def test_cascade_threshold(self):
        """Cascade derivative threshold should be positive."""
        self.assertGreater(CASCADE_DERIVATIVE_THRESHOLD, 0)
        self.assertEqual(CASCADE_DERIVATIVE_THRESHOLD, 0.05)

    def test_holographic_probability(self):
        """Holographic detection should be near-certain."""
        self.assertGreater(HOLOGRAPHIC_DETECTION_PROB, 0.99)
        self.assertEqual(HOLOGRAPHIC_DETECTION_PROB, 0.9999)

    def test_thompson_fp_target(self):
        """False positive target should be low."""
        self.assertLess(THOMPSON_FP_TARGET, 0.05)
        self.assertEqual(THOMPSON_FP_TARGET, 0.02)

    def test_completeness_threshold(self):
        """Completeness should be near 1."""
        self.assertGreater(COMPLETENESS_THRESHOLD, 0.99)
        self.assertEqual(COMPLETENESS_THRESHOLD, 0.999)

    def test_disclaimer_present(self):
        """Disclaimer must be present and correct."""
        self.assertIn("SIMULATION", DISCLAIMER)
        self.assertIn("ACADEMIC RESEARCH", DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
