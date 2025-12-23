"""
Tests for Gov-OS Defense Module

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.defense.data import DefenseReceipts, ingest_contract
from src.modules.defense.schema import validate_contract, validate_vendor
from src.modules.defense.volatility import SteelVolatility
from src.modules.defense.scenarios import run_defense_scenario, DEFENSE_SCENARIOS


class TestDefenseData(unittest.TestCase):
    """Test defense data ingestion."""

    def test_defense_receipts_init(self):
        """DefenseReceipts should initialize empty."""
        receipts = DefenseReceipts()
        self.assertEqual(receipts.receipts, [])

    def test_defense_inject_fraud(self):
        """Should inject fraud receipts."""
        receipts = DefenseReceipts()
        fraudulent = receipts.inject_fraud("subcontractor_ring", count=5)
        self.assertEqual(len(fraudulent), 5)
        for r in fraudulent:
            self.assertTrue(r.get("_is_fraud"))


class TestDefenseSchema(unittest.TestCase):
    """Test defense schema validation."""

    def test_validate_contract_valid(self):
        """Valid contract should pass."""
        data = {
            "contract_id": "W900001",
            "vendor_id": "V12345",
            "amount": 10000,
            "date": "2024-01-15",
        }
        self.assertTrue(validate_contract(data))

    def test_validate_contract_missing_field(self):
        """Missing field should fail."""
        data = {
            "contract_id": "W900001",
            "amount": 10000,
        }
        self.assertFalse(validate_contract(data))

    def test_validate_vendor_valid(self):
        """Valid vendor should pass."""
        data = {
            "vendor_id": "V12345",
            "name": "Test Vendor",
        }
        self.assertTrue(validate_vendor(data))


class TestDefenseVolatility(unittest.TestCase):
    """Test defense volatility index."""

    def test_steel_volatility_current(self):
        """Should return current value."""
        vol = SteelVolatility()
        current = vol.current()
        self.assertIsInstance(current, float)
        self.assertGreater(current, 0)

    def test_steel_volatility_historical(self):
        """Should return historical value."""
        vol = SteelVolatility()
        hist = vol.historical("2024-01-01")
        self.assertIsInstance(hist, float)


class TestDefenseScenarios(unittest.TestCase):
    """Test defense scenarios."""

    def test_scenarios_list(self):
        """Should have defined scenarios."""
        self.assertIn("SUBCONTRACTOR_RING", DEFENSE_SCENARIOS)
        self.assertIn("SHIPYARD_DISRUPTION", DEFENSE_SCENARIOS)

    def test_run_scenario_subcontractor(self):
        """Should run subcontractor ring scenario."""
        result = run_defense_scenario("SUBCONTRACTOR_RING")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "SUBCONTRACTOR_RING")

    def test_run_unknown_scenario(self):
        """Unknown scenario should return failed result."""
        result = run_defense_scenario("UNKNOWN")
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
