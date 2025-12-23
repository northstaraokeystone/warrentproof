"""
ShieldProof v2.0 Contract Module Tests

Tests for contract registration and management.
"""

import pytest

from src.shieldproof.core import StopRule, clear_ledger
from src.shieldproof.contract import (
    register_contract,
    get_contract,
    list_contracts,
    get_contract_milestones,
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


class TestRegisterContract:
    """Tests for register_contract function."""

    def test_register_contract_basic(self):
        """register_contract should create a valid contract receipt."""
        r = register_contract(
            contractor="ACME Corp",
            amount=100000.00,
            milestones=[{"id": "M1", "amount": 100000.00}],
            terms={},
        )
        assert r["receipt_type"] == "contract"
        assert r["contractor"] == "ACME Corp"
        assert r["amount_fixed"] == 100000.00
        assert "contract_id" in r

    def test_register_contract_with_id(self):
        """register_contract should use provided contract_id."""
        r = register_contract(
            contractor="Test Corp",
            amount=50000.00,
            milestones=[{"id": "M1", "amount": 50000.00}],
            terms={},
            contract_id="CUSTOM-ID-123",
        )
        assert r["contract_id"] == "CUSTOM-ID-123"

    def test_register_contract_terms_hash(self):
        """register_contract should hash terms."""
        r = register_contract(
            contractor="Test Corp",
            amount=50000.00,
            milestones=[{"id": "M1", "amount": 50000.00}],
            terms={"payment_terms": "net30"},
        )
        assert "terms_hash" in r
        assert ":" in r["terms_hash"]

    def test_register_contract_milestone_normalization(self):
        """register_contract should normalize milestones with PENDING status."""
        r = register_contract(
            contractor="Test Corp",
            amount=100000.00,
            milestones=[
                {"id": "M1", "amount": 50000.00},
                {"id": "M2", "amount": 50000.00},
            ],
            terms={},
        )
        for m in r["milestones"]:
            assert m["status"] == "PENDING"


class TestContractStoprules:
    """Tests for contract stoprules."""

    def test_stoprule_duplicate_contract(self):
        """register_contract should reject duplicate contract_id."""
        register_contract(
            contractor="First Corp",
            amount=100000.00,
            milestones=[{"id": "M1", "amount": 100000.00}],
            terms={},
            contract_id="DUPLICATE-ID",
        )

        with pytest.raises(StopRule) as exc_info:
            register_contract(
                contractor="Second Corp",
                amount=50000.00,
                milestones=[{"id": "M1", "amount": 50000.00}],
                terms={},
                contract_id="DUPLICATE-ID",
            )
        assert "Duplicate contract" in str(exc_info.value)

    def test_stoprule_invalid_amount_negative(self):
        """register_contract should reject negative amount."""
        with pytest.raises(StopRule) as exc_info:
            register_contract(
                contractor="Test Corp",
                amount=-100.00,
                milestones=[{"id": "M1", "amount": 100.00}],
                terms={},
            )
        assert "Invalid amount" in str(exc_info.value)

    def test_stoprule_invalid_amount_zero(self):
        """register_contract should reject zero amount."""
        with pytest.raises(StopRule) as exc_info:
            register_contract(
                contractor="Test Corp",
                amount=0,
                milestones=[{"id": "M1", "amount": 0}],
                terms={},
            )
        assert "Invalid amount" in str(exc_info.value)

    def test_stoprule_milestone_sum_mismatch(self):
        """register_contract should reject if milestones don't sum to amount."""
        with pytest.raises(StopRule) as exc_info:
            register_contract(
                contractor="Test Corp",
                amount=100000.00,
                milestones=[
                    {"id": "M1", "amount": 30000.00},
                    {"id": "M2", "amount": 30000.00},
                ],  # Sum is 60000, not 100000
                terms={},
            )
        assert "Milestone sum" in str(exc_info.value)


class TestGetContract:
    """Tests for get_contract function."""

    def test_get_contract_exists(self, sample_contract):
        """get_contract should return existing contract."""
        c = get_contract(sample_contract["contract_id"])
        assert c is not None
        assert c["contract_id"] == sample_contract["contract_id"]

    def test_get_contract_not_found(self):
        """get_contract should return None for non-existent contract."""
        c = get_contract("NON-EXISTENT-ID")
        assert c is None


class TestListContracts:
    """Tests for list_contracts function."""

    def test_list_contracts_empty(self):
        """list_contracts should return empty list when no contracts."""
        contracts = list_contracts()
        assert contracts == []

    def test_list_contracts_all(self, sample_contract):
        """list_contracts should return all contracts."""
        contracts = list_contracts()
        assert len(contracts) >= 1

    def test_list_contracts_filter_status(self, sample_contract):
        """list_contracts should filter by milestone status."""
        contracts = list_contracts(status="PENDING")
        assert len(contracts) >= 1


class TestGetContractMilestones:
    """Tests for get_contract_milestones function."""

    def test_get_contract_milestones_initial(self, sample_contract):
        """get_contract_milestones should return initial milestones."""
        milestones = get_contract_milestones(sample_contract["contract_id"])
        assert len(milestones) == 2
        assert all(m["status"] == "PENDING" for m in milestones)

    def test_get_contract_milestones_not_found(self):
        """get_contract_milestones should return empty for non-existent contract."""
        milestones = get_contract_milestones("NON-EXISTENT-ID")
        assert milestones == []
