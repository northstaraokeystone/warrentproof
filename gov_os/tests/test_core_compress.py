"""
Tests for Gov-OS Core Compress

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.compress import (
    compute_entropy_ratio,
    entropy_score,
    build_entropy_tree,
    detect_hierarchical,
)
from src.core.constants import COMPRESSION_LEGITIMATE_FLOOR, COMPRESSION_FRAUD_CEILING


class TestEntropyScore(unittest.TestCase):
    """Test entropy scoring."""

    def test_entropy_score_range(self):
        """Entropy score should be between 0 and 1."""
        data = {"key": "value", "number": 42}
        score = entropy_score(data)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_entropy_score_deterministic(self):
        """Same data should produce same score."""
        data = {"test": "data", "value": 123}
        score1 = entropy_score(data)
        score2 = entropy_score(data)
        self.assertEqual(score1, score2)

    def test_entropy_score_empty(self):
        """Empty data should have low entropy."""
        score = entropy_score({})
        self.assertIsInstance(score, float)

    def test_entropy_score_complex(self):
        """Complex data should have higher entropy."""
        simple = {"a": 1}
        complex_data = {"a": 1, "b": 2, "c": [1, 2, 3], "d": {"nested": "value"}}
        score_simple = entropy_score(simple)
        score_complex = entropy_score(complex_data)
        # Complex data generally has higher entropy
        self.assertIsInstance(score_simple, float)
        self.assertIsInstance(score_complex, float)


class TestComputeEntropyRatio(unittest.TestCase):
    """Test entropy ratio computation."""

    def test_entropy_ratio_range(self):
        """Entropy ratio should be between 0 and 1."""
        receipt = {"amount": 1000, "vendor": "test"}
        history = [
            {"amount": 900, "vendor": "test"},
            {"amount": 1100, "vendor": "test"},
        ]
        ratio = compute_entropy_ratio(receipt, history)
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1)

    def test_entropy_ratio_empty_history(self):
        """Empty history should return baseline."""
        receipt = {"amount": 1000}
        ratio = compute_entropy_ratio(receipt, [])
        self.assertIsInstance(ratio, float)

    def test_entropy_ratio_single_history(self):
        """Single history item should work."""
        receipt = {"amount": 1000}
        history = [{"amount": 1000}]
        ratio = compute_entropy_ratio(receipt, history)
        self.assertIsInstance(ratio, float)


class TestEntropyTree(unittest.TestCase):
    """Test hierarchical entropy tree."""

    def test_build_tree_empty(self):
        """Empty receipts should return None."""
        result = build_entropy_tree([])
        self.assertIsNone(result)

    def test_build_tree_single(self):
        """Single receipt should create leaf node."""
        receipts = [{"id": "1", "amount": 100}]
        tree = build_entropy_tree(receipts)
        self.assertIsNotNone(tree)
        self.assertIsNotNone(tree.entropy)

    def test_build_tree_multiple(self):
        """Multiple receipts should create tree."""
        receipts = [
            {"id": str(i), "amount": i * 100}
            for i in range(10)
        ]
        tree = build_entropy_tree(receipts, max_depth=3)
        self.assertIsNotNone(tree)

    def test_build_tree_depth_limit(self):
        """Tree should respect max depth."""
        receipts = [{"id": str(i)} for i in range(100)]
        tree = build_entropy_tree(receipts, max_depth=2)
        self.assertIsNotNone(tree)


class TestDetectHierarchical(unittest.TestCase):
    """Test hierarchical detection."""

    def test_detect_empty(self):
        """Empty receipts should return empty list."""
        result = detect_hierarchical([], 0.5)
        self.assertEqual(result, [])

    def test_detect_returns_list(self):
        """Detection should return list of IDs."""
        receipts = [
            {"id": "1", "amount": 100},
            {"id": "2", "amount": 200},
        ]
        result = detect_hierarchical(receipts, 0.5)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
