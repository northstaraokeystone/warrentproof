"""
Tests for Gov-OS Core Harness

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.harness import (
    run_simulation,
    SimState,
    ScenarioResult,
    run_all_scenarios,
)


class TestSimState(unittest.TestCase):
    """Test simulation state."""

    def test_sim_state_init(self):
        """SimState should initialize with defaults."""
        state = SimState(domain="defense")
        self.assertEqual(state.domain, "defense")
        self.assertEqual(state.cycle, 0)
        self.assertEqual(state.receipts, [])
        self.assertEqual(state.violations, [])

    def test_sim_state_add_receipt(self):
        """Should be able to add receipts."""
        state = SimState(domain="test")
        state.receipts.append({"id": "1"})
        self.assertEqual(len(state.receipts), 1)


class TestScenarioResult(unittest.TestCase):
    """Test scenario result."""

    def test_scenario_result_init(self):
        """ScenarioResult should initialize."""
        result = ScenarioResult(
            name="TEST",
            passed=True,
            message="Test passed"
        )
        self.assertEqual(result.name, "TEST")
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Test passed")

    def test_scenario_result_with_metrics(self):
        """ScenarioResult should accept metrics."""
        result = ScenarioResult(
            name="TEST",
            passed=False,
            message="Test failed",
            metrics={"detection_rate": 0.5}
        )
        self.assertEqual(result.metrics["detection_rate"], 0.5)


class TestRunSimulation(unittest.TestCase):
    """Test simulation runner."""

    def test_run_simulation_defense(self):
        """Should run defense simulation."""
        state = run_simulation("defense", n_cycles=10, seed=42)
        self.assertIsInstance(state, SimState)
        self.assertEqual(state.domain, "defense")
        self.assertGreaterEqual(state.cycle, 0)

    def test_run_simulation_medicaid(self):
        """Should run medicaid simulation."""
        state = run_simulation("medicaid", n_cycles=10, seed=42)
        self.assertIsInstance(state, SimState)
        self.assertEqual(state.domain, "medicaid")

    def test_run_simulation_deterministic(self):
        """Same seed should produce same results."""
        state1 = run_simulation("defense", n_cycles=10, seed=42)
        state2 = run_simulation("defense", n_cycles=10, seed=42)
        self.assertEqual(len(state1.receipts), len(state2.receipts))


class TestRunAllScenarios(unittest.TestCase):
    """Test scenario runner."""

    def test_run_all_defense(self):
        """Should run all defense scenarios."""
        results = run_all_scenarios("defense")
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        for name, result in results.items():
            self.assertIsInstance(result, ScenarioResult)

    def test_run_all_medicaid(self):
        """Should run all medicaid scenarios."""
        results = run_all_scenarios("medicaid")
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
