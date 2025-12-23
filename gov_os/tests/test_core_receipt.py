"""
Tests for Gov-OS Core Receipt

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.receipt import (
    emit_receipt,
    emit_L0,
    emit_L1,
    emit_L2,
    emit_L3,
    emit_L4,
    completeness_check,
    StopRule,
)
from src.core.constants import DISCLAIMER


class TestEmitReceipt(unittest.TestCase):
    """Test receipt emission."""

    def test_emit_receipt_basic(self):
        """Basic receipt emission."""
        receipt = emit_receipt("test_receipt", {"key": "value"})
        self.assertIn("receipt_id", receipt)
        self.assertIn("receipt_type", receipt)
        self.assertIn("timestamp", receipt)
        self.assertEqual(receipt["receipt_type"], "test_receipt")

    def test_emit_receipt_level(self):
        """Receipt should include level."""
        receipt = emit_receipt("test_receipt", {}, level=2)
        self.assertEqual(receipt["level"], 2)

    def test_emit_receipt_simulation_flag(self):
        """Receipt should include simulation flag."""
        receipt = emit_receipt("test_receipt", {})
        self.assertEqual(receipt["simulation_flag"], DISCLAIMER)

    def test_emit_L0(self):
        """L0 receipt should have level 0."""
        receipt = emit_L0("ingest", {"data": "test"})
        self.assertEqual(receipt["level"], 0)

    def test_emit_L1(self):
        """L1 receipt should have level 1."""
        receipt = emit_L1("raf_cycle", {"cycle": ["a", "b"]})
        self.assertEqual(receipt["level"], 1)

    def test_emit_L2(self):
        """L2 receipt should have level 2."""
        receipt = emit_L2("cascade", {"derivative": 0.1})
        self.assertEqual(receipt["level"], 2)

    def test_emit_L3(self):
        """L3 receipt should have level 3."""
        receipt = emit_L3("holographic", {"boundary": "hash"})
        self.assertEqual(receipt["level"], 3)

    def test_emit_L4(self):
        """L4 receipt should have level 4."""
        receipt = emit_L4("meta", {"children": []})
        self.assertEqual(receipt["level"], 4)


class TestCompletenessCheck(unittest.TestCase):
    """Test completeness checking."""

    def test_completeness_check(self):
        """Completeness check should return dict."""
        result = completeness_check()
        self.assertIsInstance(result, dict)
        self.assertIn("complete", result)
        self.assertIn("ratio", result)


class TestStopRule(unittest.TestCase):
    """Test stop rule exception."""

    def test_stop_rule_is_exception(self):
        """StopRule should be an exception."""
        self.assertTrue(issubclass(StopRule, Exception))

    def test_stop_rule_raise(self):
        """Should be able to raise StopRule."""
        with self.assertRaises(StopRule):
            raise StopRule("Test stop")


if __name__ == "__main__":
    unittest.main()
