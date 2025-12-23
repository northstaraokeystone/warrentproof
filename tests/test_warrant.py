"""
WarrantProof Warrant Module Tests

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️
"""

import pytest
import time

from src.warrant import (
    generate_warrant,
    quality_attestation,
    milestone_warrant,
    cost_variance_warrant,
    contract_award_warrant,
    delivery_warrant,
)
from src.core import DISCLAIMER, StopRuleException


class TestGenerateWarrant:
    """Tests for generate_warrant function."""

    def test_basic_warrant_generation(self):
        """Test basic warrant creation."""
        receipt = generate_warrant(
            transaction={"type": "contract", "amount": 1000000, "description": "Test"},
            approver="TEST_OFFICIAL",
            branch="Navy"
        )

        assert receipt["receipt_type"] == "warrant"
        assert receipt["branch"] == "Navy"
        assert receipt["approver"] == "TEST_OFFICIAL"
        assert "simulation_flag" in receipt

    def test_warrant_latency_slo(self):
        """SLO: Warrant generation <= 50ms."""
        t0 = time.time()
        generate_warrant(
            transaction={"type": "contract", "amount": 1000000},
            approver="TEST_OFFICIAL",
            branch="Navy"
        )
        latency_ms = (time.time() - t0) * 1000

        assert latency_ms <= 50, f"Latency {latency_ms}ms > 50ms SLO"

    def test_warrant_with_lineage(self):
        """Test warrant with parent lineage."""
        parent = generate_warrant(
            transaction={"type": "contract", "amount": 1000000},
            approver="PARENT_OFFICIAL",
            branch="Navy"
        )

        child = generate_warrant(
            transaction={"type": "modification", "amount": 500000},
            approver="CHILD_OFFICIAL",
            branch="Navy",
            parent_receipt_id=parent["payload_hash"]
        )

        assert parent["payload_hash"] in child["decision_lineage"]

    def test_warrant_missing_approver_stoprule(self):
        """Test that missing approver triggers stoprule."""
        with pytest.raises(StopRuleException):
            generate_warrant(
                transaction={"type": "contract", "amount": 1000000},
                approver="",  # Empty approver
                branch="Navy"
            )

    def test_warrant_invalid_branch(self):
        """Test that invalid branch raises error."""
        with pytest.raises(ValueError):
            generate_warrant(
                transaction={"type": "contract", "amount": 1000000},
                approver="TEST_OFFICIAL",
                branch="InvalidBranch"
            )


class TestQualityAttestation:
    """Tests for quality_attestation function."""

    def test_basic_attestation(self):
        """Test basic quality attestation."""
        receipt = quality_attestation(
            item="hull_weld_section_1",
            inspector="INSPECTOR_001",
            certification={"passed": True, "grade": "A"}
        )

        assert receipt["receipt_type"] == "quality_attestation"
        assert receipt["item"] == "hull_weld_section_1"
        assert receipt["certification"]["passed"] == True
        assert "citation" in receipt

    def test_attestation_with_citation(self):
        """Test attestation includes citation."""
        receipt = quality_attestation(
            item="test_item",
            inspector="INSPECTOR_001",
            certification={"passed": True}
        )

        assert "citation" in receipt
        assert "url" in receipt["citation"]


class TestMilestoneWarrant:
    """Tests for milestone_warrant function."""

    def test_basic_milestone(self):
        """Test basic milestone creation."""
        receipt = milestone_warrant(
            program="Test Program",
            milestone="MS-B",
            status={"complete": True, "on_schedule": True}
        )

        assert receipt["receipt_type"] == "milestone"
        assert receipt["program"] == "Test Program"
        assert receipt["milestone"] == "MS-B"

    def test_milestone_with_delay(self):
        """Test milestone with schedule variance."""
        receipt = milestone_warrant(
            program="Delayed Program",
            milestone="CDR",
            status={"complete": True, "on_schedule": False, "schedule_variance_days": 45}
        )

        assert receipt["status"]["schedule_variance_days"] == 45


class TestCostVarianceWarrant:
    """Tests for cost_variance_warrant function."""

    def test_low_variance(self):
        """Test low severity cost variance."""
        receipt = cost_variance_warrant(
            program="Test Program",
            baseline=1000000,
            actual=1080000,
            variance_pct=8.0
        )

        assert receipt["severity"] == "low"

    def test_medium_variance(self):
        """Test medium severity cost variance."""
        receipt = cost_variance_warrant(
            program="Test Program",
            baseline=1000000,
            actual=1150000,
            variance_pct=15.0
        )

        assert receipt["severity"] == "medium"

    def test_critical_variance(self):
        """Test critical severity cost variance."""
        receipt = cost_variance_warrant(
            program="Test Program",
            baseline=1000000,
            actual=1800000,
            variance_pct=80.0
        )

        assert receipt["severity"] == "critical"


class TestContractAndDelivery:
    """Tests for contract and delivery warrants."""

    def test_contract_award(self):
        """Test contract award warrant."""
        receipt = contract_award_warrant(
            contract_number="N00024-24-C-1234",
            vendor="TestVendor",
            amount=5000000,
            branch="Navy",
            approver="CONTRACTING_OFFICER"
        )

        assert receipt["receipt_type"] == "warrant"
        assert receipt["transaction_type"] == "contract_award"

    def test_delivery_warrant(self):
        """Test delivery warrant."""
        receipt = delivery_warrant(
            item="Test Item",
            quantity=100,
            unit_cost=1000,
            branch="Army",
            approver="RECEIVING_OFFICER"
        )

        assert receipt["receipt_type"] == "warrant"
        assert receipt["transaction_type"] == "delivery"
        assert receipt["amount_usd"] == 100000  # 100 * 1000
