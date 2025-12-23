"""
Tests for Gov-OS Medicaid Module

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.medicaid.data import MedicaidReceipts, ingest_claim
from src.modules.medicaid.schema import validate_claim, validate_provider, validate_cpt
from src.modules.medicaid.volatility import MedicalCPI
from src.modules.medicaid.scenarios import run_medicaid_scenario, MEDICAID_SCENARIOS


class TestMedicaidData(unittest.TestCase):
    """Test medicaid data ingestion."""

    def test_medicaid_receipts_init(self):
        """MedicaidReceipts should initialize empty."""
        receipts = MedicaidReceipts()
        self.assertEqual(receipts.receipts, [])

    def test_medicaid_inject_fraud(self):
        """Should inject fraud receipts."""
        receipts = MedicaidReceipts()
        fraudulent = receipts.inject_fraud("upcoding", count=5)
        self.assertEqual(len(fraudulent), 5)
        for r in fraudulent:
            self.assertTrue(r.get("_is_fraud"))

    def test_medicaid_inject_phantom(self):
        """Should inject phantom billing fraud."""
        receipts = MedicaidReceipts()
        fraudulent = receipts.inject_fraud("phantom_billing", count=3)
        self.assertEqual(len(fraudulent), 3)
        for r in fraudulent:
            self.assertIn("DECEASED", r.get("beneficiary_id", ""))


class TestMedicaidSchema(unittest.TestCase):
    """Test medicaid schema validation."""

    def test_validate_claim_valid(self):
        """Valid claim should pass."""
        data = {
            "claim_id": "CLAIM001",
            "provider_npi": "1234567890",
            "beneficiary_id": "BEN001",
            "cpt_codes": ["99213"],
            "amount": 150.0,
            "date": "2024-01-15",
        }
        self.assertTrue(validate_claim(data))

    def test_validate_claim_invalid_npi(self):
        """Invalid NPI should fail."""
        data = {
            "claim_id": "CLAIM001",
            "provider_npi": "123",  # Too short
            "beneficiary_id": "BEN001",
            "amount": 150.0,
            "date": "2024-01-15",
        }
        self.assertFalse(validate_claim(data))

    def test_validate_provider_valid(self):
        """Valid provider should pass."""
        data = {
            "npi": "1234567890",
            "name": "Dr. Test",
        }
        self.assertTrue(validate_provider(data))

    def test_validate_cpt_valid(self):
        """Valid CPT code should pass."""
        data = {"code": "99213"}
        self.assertTrue(validate_cpt(data))

    def test_validate_cpt_invalid(self):
        """Invalid CPT code should fail."""
        data = {"code": "123"}  # Too short
        self.assertFalse(validate_cpt(data))


class TestMedicaidVolatility(unittest.TestCase):
    """Test medicaid volatility index."""

    def test_medical_cpi_current(self):
        """Should return current value."""
        cpi = MedicalCPI()
        current = cpi.current()
        self.assertIsInstance(current, float)
        self.assertGreater(current, 0)

    def test_medical_cpi_historical(self):
        """Should return historical value."""
        cpi = MedicalCPI()
        hist = cpi.historical("2024-01-01")
        self.assertIsInstance(hist, float)


class TestMedicaidScenarios(unittest.TestCase):
    """Test medicaid scenarios."""

    def test_scenarios_list(self):
        """Should have defined scenarios."""
        self.assertIn("PROVIDER_RING", MEDICAID_SCENARIOS)
        self.assertIn("UPCODING_DETECTION", MEDICAID_SCENARIOS)
        self.assertIn("PHANTOM_BILLING", MEDICAID_SCENARIOS)

    def test_run_scenario_provider_ring(self):
        """Should run provider ring scenario."""
        result = run_medicaid_scenario("PROVIDER_RING")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "PROVIDER_RING")

    def test_run_scenario_upcoding(self):
        """Should run upcoding scenario."""
        result = run_medicaid_scenario("UPCODING_DETECTION")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "UPCODING_DETECTION")

    def test_run_unknown_scenario(self):
        """Unknown scenario should return failed result."""
        result = run_medicaid_scenario("UNKNOWN")
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
