"""
ShieldProof v2.0 Reconcile Module Tests

Tests for automated reconciliation.
"""

import pytest

from src.shieldproof.core import clear_ledger
from src.shieldproof.contract import register_contract
from src.shieldproof.milestone import submit_deliverable, verify_milestone
from src.shieldproof.payment import release_payment
from src.shieldproof.reconcile import (
    reconcile_contract,
    reconcile_all,
    flag_anomaly,
    get_waste_summary,
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Clear ledger before and after each test."""
    clear_ledger()
    yield
    clear_ledger()


@pytest.fixture
def sample_contract():
    """Create a sample contract for testing."""
    return register_contract(
        contractor="Test Corp",
        amount=1000000.00,
        milestones=[
            {"id": "M1", "description": "Phase 1", "amount": 250000.00},
            {"id": "M2", "description": "Phase 2", "amount": 750000.00},
        ],
        terms={"payment_terms": "net30"},
    )


@pytest.fixture
def paid_milestone(sample_contract):
    """Create a contract with a paid milestone."""
    contract_id = sample_contract["contract_id"]
    submit_deliverable(contract_id, "M1", b"Deliverable content")
    verify_milestone(contract_id, "M1", "INSPECTOR-001", passed=True)
    release_payment(contract_id, "M1")
    return sample_contract


class TestReconcileContract:
    """Tests for reconcile_contract function."""

    def test_reconcile_contract_on_track(self, paid_milestone):
        """reconcile_contract should return ON_TRACK for healthy contract."""
        report = reconcile_contract(paid_milestone["contract_id"])
        assert report["status"] == "ON_TRACK"
        assert report["discrepancy"] == 0

    def test_reconcile_contract_not_found(self):
        """reconcile_contract should return error for unknown contract."""
        report = reconcile_contract("NON-EXISTENT")
        assert report["status"] == "ERROR"
        assert "error" in report

    def test_reconcile_contract_fields(self, paid_milestone):
        """reconcile_contract should include all required fields."""
        report = reconcile_contract(paid_milestone["contract_id"])
        assert "contract_id" in report
        assert "contractor" in report
        assert "amount_fixed" in report
        assert "amount_paid" in report
        assert "milestones_total" in report
        assert "status" in report

    def test_reconcile_contract_counts(self, paid_milestone):
        """reconcile_contract should count milestone states correctly."""
        report = reconcile_contract(paid_milestone["contract_id"])
        assert report["milestones_paid"] >= 1
        assert report["milestones_pending"] >= 1


class TestReconcileAll:
    """Tests for reconcile_all function."""

    def test_reconcile_all_empty(self):
        """reconcile_all should return empty list when no contracts."""
        reports = reconcile_all()
        assert reports == []

    def test_reconcile_all_multiple(self, paid_milestone):
        """reconcile_all should reconcile all contracts."""
        # Add another contract
        register_contract(
            contractor="Second Corp",
            amount=50000.00,
            milestones=[{"id": "M1", "amount": 50000.00}],
            terms={},
        )

        reports = reconcile_all()
        assert len(reports) >= 2


class TestFlagAnomaly:
    """Tests for flag_anomaly function."""

    def test_flag_anomaly_creates_receipt(self, sample_contract):
        """flag_anomaly should create anomaly receipt."""
        r = flag_anomaly(sample_contract["contract_id"], "Manual review requested")
        assert r["receipt_type"] == "anomaly"
        assert r["metric"] == "manual_flag"
        assert r["reason"] == "Manual review requested"


class TestGetWasteSummary:
    """Tests for get_waste_summary function."""

    def test_get_waste_summary_empty(self):
        """get_waste_summary should return zeros when no contracts."""
        summary = get_waste_summary()
        assert summary["total_contracts"] == 0
        assert summary["waste_identified"] == 0

    def test_get_waste_summary_healthy(self, paid_milestone):
        """get_waste_summary should show no waste for healthy contracts."""
        summary = get_waste_summary()
        assert summary["total_contracts"] >= 1
        assert summary["waste_identified"] == 0

    def test_get_waste_summary_fields(self, paid_milestone):
        """get_waste_summary should include all required fields."""
        summary = get_waste_summary()
        assert "total_contracts" in summary
        assert "total_committed" in summary
        assert "total_paid" in summary
        assert "total_verified" in summary
        assert "waste_identified" in summary
        assert "milestones_pending" in summary
        assert "milestones_disputed" in summary
        assert "contracts_on_track" in summary
        assert "contracts_overpaid" in summary
