"""
ShieldProof v2.0 Milestone Module Tests

Tests for milestone tracking and verification.
"""

import pytest

from src.shieldproof.core import StopRule, clear_ledger
from src.shieldproof.contract import register_contract
from src.shieldproof.milestone import (
    submit_deliverable,
    verify_milestone,
    get_milestone,
    list_pending,
    list_verified,
    list_disputed,
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


class TestSubmitDeliverable:
    """Tests for submit_deliverable function."""

    def test_submit_deliverable_basic(self, sample_contract):
        """submit_deliverable should create DELIVERED milestone receipt."""
        r = submit_deliverable(
            sample_contract["contract_id"],
            "M1",
            b"Deliverable content",
        )
        assert r["receipt_type"] == "milestone"
        assert r["status"] == "DELIVERED"
        assert "deliverable_hash" in r

    def test_submit_deliverable_string(self, sample_contract):
        """submit_deliverable should accept string content."""
        r = submit_deliverable(
            sample_contract["contract_id"],
            "M1",
            "String deliverable content",
        )
        assert r["status"] == "DELIVERED"

    def test_submit_deliverable_hash(self, sample_contract):
        """submit_deliverable should hash deliverable content."""
        r = submit_deliverable(
            sample_contract["contract_id"],
            "M1",
            b"Content to hash",
        )
        assert ":" in r["deliverable_hash"]


class TestSubmitDeliverableStoprules:
    """Tests for submit_deliverable stoprules."""

    def test_stoprule_unknown_contract(self):
        """submit_deliverable should reject unknown contract."""
        with pytest.raises(StopRule) as exc_info:
            submit_deliverable("UNKNOWN-CONTRACT", "M1", b"content")
        assert "Unknown contract" in str(exc_info.value)

    def test_stoprule_unknown_milestone(self, sample_contract):
        """submit_deliverable should reject unknown milestone."""
        with pytest.raises(StopRule) as exc_info:
            submit_deliverable(
                sample_contract["contract_id"],
                "NON-EXISTENT-MILESTONE",
                b"content",
            )
        assert "Unknown milestone" in str(exc_info.value)


class TestVerifyMilestone:
    """Tests for verify_milestone function."""

    def test_verify_milestone_approved(self, sample_contract):
        """verify_milestone should set status to VERIFIED when approved."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")
        r = verify_milestone(
            sample_contract["contract_id"],
            "M1",
            "INSPECTOR-001",
            passed=True,
        )
        assert r["receipt_type"] == "milestone"
        assert r["status"] == "VERIFIED"
        assert r["verifier_id"] == "INSPECTOR-001"

    def test_verify_milestone_rejected(self, sample_contract):
        """verify_milestone should set status to DISPUTED when rejected."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")
        r = verify_milestone(
            sample_contract["contract_id"],
            "M1",
            "INSPECTOR-001",
            passed=False,
        )
        assert r["status"] == "DISPUTED"

    def test_verify_milestone_timestamp(self, sample_contract):
        """verify_milestone should include verification timestamp."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")
        r = verify_milestone(
            sample_contract["contract_id"],
            "M1",
            "INSPECTOR-001",
            passed=True,
        )
        assert "verification_ts" in r
        assert r["verification_ts"] is not None


class TestVerifyMilestoneStoprules:
    """Tests for verify_milestone stoprules."""

    def test_stoprule_unknown_contract(self):
        """verify_milestone should reject unknown contract."""
        with pytest.raises(StopRule) as exc_info:
            verify_milestone("UNKNOWN-CONTRACT", "M1", "INSPECTOR", passed=True)
        assert "Unknown contract" in str(exc_info.value)

    def test_stoprule_unknown_milestone(self, sample_contract):
        """verify_milestone should reject unknown milestone."""
        with pytest.raises(StopRule) as exc_info:
            verify_milestone(
                sample_contract["contract_id"],
                "NON-EXISTENT-MILESTONE",
                "INSPECTOR",
                passed=True,
            )
        assert "Unknown milestone" in str(exc_info.value)

    def test_stoprule_already_verified(self, verified_milestone):
        """verify_milestone should reject already verified milestone."""
        with pytest.raises(StopRule) as exc_info:
            verify_milestone(
                verified_milestone["contract_id"],
                "M1",
                "INSPECTOR-002",
                passed=True,
            )
        assert "already verified" in str(exc_info.value)


class TestGetMilestone:
    """Tests for get_milestone function."""

    def test_get_milestone_exists(self, sample_contract):
        """get_milestone should return existing milestone."""
        m = get_milestone(sample_contract["contract_id"], "M1")
        assert m is not None
        assert m["id"] == "M1"

    def test_get_milestone_not_found(self, sample_contract):
        """get_milestone should return None for non-existent milestone."""
        m = get_milestone(sample_contract["contract_id"], "NON-EXISTENT")
        assert m is None

    def test_get_milestone_updated_status(self, verified_milestone):
        """get_milestone should reflect updated status."""
        m = get_milestone(verified_milestone["contract_id"], "M1")
        assert m["status"] == "VERIFIED"


class TestListFunctions:
    """Tests for list functions."""

    def test_list_pending(self, sample_contract):
        """list_pending should return DELIVERED milestones."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")
        pending = list_pending()
        assert len(pending) >= 1

    def test_list_verified(self, verified_milestone):
        """list_verified should return VERIFIED milestones."""
        verified = list_verified()
        assert len(verified) >= 1

    def test_list_disputed(self, sample_contract):
        """list_disputed should return DISPUTED milestones."""
        submit_deliverable(sample_contract["contract_id"], "M1", b"content")
        verify_milestone(sample_contract["contract_id"], "M1", "INSPECTOR", passed=False)
        disputed = list_disputed()
        assert len(disputed) >= 1
