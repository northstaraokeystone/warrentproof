"""
Tests for Gov-OS Core Detect

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.detect import (
    adaptive_threshold,
    thompson_sample,
    detect_anomaly,
    ThresholdDistribution,
)
from src.core.constants import VOLATILITY_ALPHA


class TestAdaptiveThreshold(unittest.TestCase):
    """Test adaptive threshold computation."""

    def test_adaptive_baseline(self):
        """No volatility should return base threshold."""
        result = adaptive_threshold(0.85, 1.0, alpha=0.1)
        self.assertAlmostEqual(result, 0.85, places=2)

    def test_adaptive_high_volatility(self):
        """High volatility should raise threshold."""
        base = 0.85
        high_vol = 1.5
        result = adaptive_threshold(base, high_vol, alpha=0.1)
        self.assertGreater(result, base)

    def test_adaptive_low_volatility(self):
        """Low volatility should lower threshold."""
        base = 0.85
        low_vol = 0.5
        result = adaptive_threshold(base, low_vol, alpha=0.1)
        self.assertLess(result, base)

    def test_adaptive_bounds(self):
        """Threshold should stay between 0 and 1."""
        result = adaptive_threshold(0.85, 10.0, alpha=0.5)
        self.assertLessEqual(result, 1.0)

        result = adaptive_threshold(0.85, 0.1, alpha=0.5)
        self.assertGreaterEqual(result, 0.0)


class TestThompsonSampling(unittest.TestCase):
    """Test Thompson sampling."""

    def test_thompson_sample_range(self):
        """Sample should be positive."""
        dist = ThresholdDistribution()
        sample = thompson_sample(dist)
        self.assertGreater(sample, 0)

    def test_thompson_sample_varies(self):
        """Samples should vary (probabilistic)."""
        dist = ThresholdDistribution()
        samples = [thompson_sample(dist) for _ in range(100)]
        # At least some variation expected
        self.assertGreater(max(samples) - min(samples), 0)


class TestThresholdDistribution(unittest.TestCase):
    """Test threshold distribution."""

    def test_distribution_init(self):
        """Distribution should initialize with defaults."""
        dist = ThresholdDistribution()
        self.assertIsNotNone(dist)
        self.assertEqual(dist.alpha, 1.0)
        self.assertEqual(dist.beta, 1.0)

    def test_distribution_update_success(self):
        """Update with success should increase alpha."""
        dist = ThresholdDistribution()
        initial_alpha = dist.alpha
        dist.update(True)
        self.assertGreater(dist.alpha, initial_alpha)

    def test_distribution_update_failure(self):
        """Update with failure should increase beta."""
        dist = ThresholdDistribution()
        initial_beta = dist.beta
        dist.update(False)
        self.assertGreater(dist.beta, initial_beta)

    def test_distribution_mean(self):
        """Mean should be alpha / (alpha + beta)."""
        dist = ThresholdDistribution(alpha=3.0, beta=1.0)
        expected = 3.0 / 4.0
        self.assertAlmostEqual(dist.mean(), expected, places=5)


class TestDetectAnomaly(unittest.TestCase):
    """Test anomaly detection."""

    def test_detect_anomaly_returns_bool(self):
        """Detection should return boolean."""
        receipt = {"amount": 1000, "vendor": "test"}
        history = [{"amount": 900}, {"amount": 1100}]
        result = detect_anomaly(receipt, history, volatility_factor=1.0)
        self.assertIsInstance(result, bool)

    def test_detect_anomaly_empty_history(self):
        """Empty history should not crash."""
        receipt = {"amount": 1000}
        result = detect_anomaly(receipt, [], volatility_factor=1.0)
        self.assertIsInstance(result, bool)

    def test_detect_anomaly_with_distribution(self):
        """Should work with Thompson distribution."""
        receipt = {"amount": 1000}
        history = [{"amount": i * 100} for i in range(10)]
        dist = ThresholdDistribution()
        result = detect_anomaly(receipt, history, volatility_factor=1.0, distribution=dist)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
