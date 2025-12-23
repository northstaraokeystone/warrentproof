"""
ShieldProof v2.0 Payment Module Tests

Tests for payment release on verification.
"""

import pytest

from src.shieldproof.core import StopRule, clear_ledger
from src.shieldproof.contract import register_contract
from src.shieldproof.milestone import submit_deliverable, verify_milestone
from src.shieldproof.payment import (
    release_payment,
    get_payment,
    total_paid,
    total_outstanding,
    list_payments,
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
def verified_milestone(sample_contract):
    """Create a contract with a verified milestone."""
    contract_id = sample_contract["contract_id"]
    submit_deliverable(contract_id, "M1", b"Deliverable content")
    verify_milestone(contract_id, "M1", "INSPECTOR-001", passed=True)
    return sample_contract


@pytest.fixture
def paid_milestone(verified_milestone):
    """Create a contract with a paid milestone."""
    contract_id = verified_milestone["contract_id"]
    release_payment(contract_id, "M1")
    return verified_milestone


class TestReleasePayment:
    """Tests for release_payment function."""

    def test_release_payment_basic(self, verified_milestone):
        """release_payment should create payment receipt for verified milestone."""
        r = release_payment(verified_milestone["contract_id"], "M1")
        assert r["receipt_type"] == "payment"
        assert r["amount"] == 250000.00
        assert "payment_hash" in r
        assert "released_at" in r

    def test_release_payment_hash(self, verified_milestone):
        """release_payment should include payment hash."""
        r = release_payment(verified_milestone["contract_id"], "M1")
        assert ":" in r["payment_hash"]


class TestReleasePaymentStoprules:
    """Tests for release_payment stoprules."""

    def test_stoprule_unverified_milestone_pending(self, sample_contract):
        """release_payment should HALT for PENDING milestone."""
        with pytest.raises(StopRule) as exc_info:
            release_payment(sample_contract["contract_id"], "M1")
        assert "HALT" in str(exc_info.value)
        assert "unverified" in str(exc_info.value).lower()

    def test_stoprule_unverified_milestone_delivered(self, sample_contract):
        """release_payment should HALT for DELIVERED milestone."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")

        with pytest.raises(StopRule) as exc_info:
            release_payment(sample_contract["contract_id"], "M1")
        assert "HALT" in str(exc_info.value)

    def test_stoprule_already_paid(self, paid_milestone):
        """release_payment should reject already paid milestone."""
        with pytest.raises(StopRule) as exc_info:
            release_payment(paid_milestone["contract_id"], "M1")
        assert "already paid" in str(exc_info.value).lower()

    def test_stoprule_milestone_not_found(self, sample_contract):
        """release_payment should reject non-existent milestone."""
        with pytest.raises(StopRule) as exc_info:
            release_payment(sample_contract["contract_id"], "NON-EXISTENT")
        assert "not found" in str(exc_info.value).lower()


class TestGetPayment:
    """Tests for get_payment function."""

    def test_get_payment_exists(self, paid_milestone):
        """get_payment should return existing payment."""
        p = get_payment(paid_milestone["contract_id"], "M1")
        assert p is not None
        assert p["amount"] == 250000.00

    def test_get_payment_not_paid(self, verified_milestone):
        """get_payment should return None for unpaid milestone."""
        p = get_payment(verified_milestone["contract_id"], "M1")
        assert p is None


class TestTotalPaid:
    """Tests for total_paid function."""

    def test_total_paid_none(self, sample_contract):
        """total_paid should return 0 when no payments."""
        paid = total_paid(sample_contract["contract_id"])
        assert paid == 0

    def test_total_paid_single(self, paid_milestone):
        """total_paid should return amount of single payment."""
        paid = total_paid(paid_milestone["contract_id"])
        assert paid == 250000.00

    def test_total_paid_multiple(self, paid_milestone):
        """total_paid should sum multiple payments."""
        # Pay second milestone
        submit_deliverable(paid_milestone["contract_id"], "M2", b"content")
        verify_milestone(paid_milestone["contract_id"], "M2", "INSPECTOR", passed=True)
        release_payment(paid_milestone["contract_id"], "M2")

        paid = total_paid(paid_milestone["contract_id"])
        assert paid == 1000000.00  # 250000 + 750000


class TestTotalOutstanding:
    """Tests for total_outstanding function."""

    def test_total_outstanding_full(self, sample_contract):
        """total_outstanding should return full amount when nothing paid."""
        outstanding = total_outstanding(sample_contract["contract_id"])
        assert outstanding == 1000000.00

    def test_total_outstanding_partial(self, paid_milestone):
        """total_outstanding should return remaining after payment."""
        outstanding = total_outstanding(paid_milestone["contract_id"])
        assert outstanding == 750000.00

    def test_total_outstanding_none(self, paid_milestone):
        """total_outstanding should return 0 when fully paid."""
        submit_deliverable(paid_milestone["contract_id"], "M2", b"content")
        verify_milestone(paid_milestone["contract_id"], "M2", "INSPECTOR", passed=True)
        release_payment(paid_milestone["contract_id"], "M2")

        outstanding = total_outstanding(paid_milestone["contract_id"])
        assert outstanding == 0


class TestListPayments:
    """Tests for list_payments function."""

    def test_list_payments_empty(self, sample_contract):
        """list_payments should return empty for unpaid contract."""
        payments = list_payments(sample_contract["contract_id"])
        assert payments == []

    def test_list_payments_exists(self, paid_milestone):
        """list_payments should return payments for contract."""
        payments = list_payments(paid_milestone["contract_id"])
        assert len(payments) >= 1

    def test_list_payments_all(self, paid_milestone):
        """list_payments should return all payments without filter."""
        payments = list_payments()
        assert len(payments) >= 1
