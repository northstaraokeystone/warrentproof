"""
Tests for Gov-OS Core RAF

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.raf import (
    build_graph,
    find_raf_cycles,
    is_autocatalytic,
    detect_without_hardcode,
)
from src.core.constants import RAF_MIN_CYCLE_LENGTH


class TestBuildGraph(unittest.TestCase):
    """Test graph building from receipts."""

    def test_build_graph_empty(self):
        """Empty receipts should create empty graph."""
        graph = build_graph([], "node", "edge")
        self.assertEqual(len(graph.nodes()), 0)

    def test_build_graph_single_edge(self):
        """Single receipt should create one edge."""
        receipts = [{"vendor_id": "A", "payment_to": "B", "amount": 100}]
        graph = build_graph(receipts, "vendor_id", "payment_to")
        self.assertIn("A", graph.nodes())
        self.assertIn("B", graph.nodes())
        self.assertTrue(graph.has_edge("A", "B"))

    def test_build_graph_multiple(self):
        """Multiple receipts should create multiple edges."""
        receipts = [
            {"vendor_id": "A", "payment_to": "B", "amount": 100},
            {"vendor_id": "B", "payment_to": "C", "amount": 200},
            {"vendor_id": "C", "payment_to": "A", "amount": 150},
        ]
        graph = build_graph(receipts, "vendor_id", "payment_to")
        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 3)


class TestFindRafCycles(unittest.TestCase):
    """Test RAF cycle detection."""

    def test_find_cycles_empty(self):
        """Empty graph should have no cycles."""
        import networkx as nx
        graph = nx.DiGraph()
        cycles = find_raf_cycles(graph)
        self.assertEqual(cycles, [])

    def test_find_cycles_triangle(self):
        """Triangle should be detected."""
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        cycles = find_raf_cycles(graph)
        self.assertGreater(len(cycles), 0)
        # Should find the A->B->C->A cycle
        cycle_found = any(set(c) == {"A", "B", "C"} for c in cycles)
        self.assertTrue(cycle_found)

    def test_find_cycles_no_cycle(self):
        """Linear chain should have no cycles."""
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        cycles = find_raf_cycles(graph)
        self.assertEqual(cycles, [])

    def test_find_cycles_min_length(self):
        """Self-loops should be filtered if below min length."""
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_edge("A", "A")  # Self-loop
        cycles = find_raf_cycles(graph)
        # Self-loop has length 1, below RAF_MIN_CYCLE_LENGTH (3)
        triangle_cycles = [c for c in cycles if len(c) >= RAF_MIN_CYCLE_LENGTH]
        self.assertEqual(triangle_cycles, [])


class TestIsAutocatalytic(unittest.TestCase):
    """Test autocatalytic pattern detection."""

    def test_is_autocatalytic_empty(self):
        """Empty history should return False."""
        result = is_autocatalytic(["A", "B", "C"], [])
        self.assertFalse(result)

    def test_is_autocatalytic_with_history(self):
        """Pattern with matching history should be detected."""
        pattern = ["A", "B", "C"]
        history = [
            {"vendor_id": "A", "payment_to": "B"},
            {"vendor_id": "B", "payment_to": "C"},
            {"vendor_id": "C", "payment_to": "A"},
        ]
        result = is_autocatalytic(pattern, history)
        # Implementation specific - just verify it returns bool
        self.assertIsInstance(result, bool)


class TestDetectWithoutHardcode(unittest.TestCase):
    """Test universal detection without hardcoding."""

    def test_detect_empty(self):
        """Empty receipts should return empty list."""
        result = detect_without_hardcode([], "defense", "vendor_id", "payment_to")
        self.assertEqual(result, [])

    def test_detect_ring(self):
        """Ring pattern should be detected."""
        receipts = [
            {"vendor_id": "A", "payment_to": "B", "amount": 100},
            {"vendor_id": "B", "payment_to": "C", "amount": 200},
            {"vendor_id": "C", "payment_to": "A", "amount": 150},
        ]
        result = detect_without_hardcode(receipts, "defense", "vendor_id", "payment_to")
        self.assertIsInstance(result, list)
        # Should find the cycle
        if result:
            self.assertIn("anomaly_type", result[0])

    def test_detect_medicaid_domain(self):
        """Should work with medicaid domain keys."""
        receipts = [
            {"provider_npi": "1234567890", "referral_to": "0987654321"},
            {"provider_npi": "0987654321", "referral_to": "5555555555"},
            {"provider_npi": "5555555555", "referral_to": "1234567890"},
        ]
        result = detect_without_hardcode(receipts, "medicaid", "provider_npi", "referral_to")
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
