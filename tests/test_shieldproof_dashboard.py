"""
ShieldProof v2.0 Dashboard Module Tests

Tests for public audit trail dashboard.
"""

import json
import os
import pytest
import tempfile

from src.shieldproof.core import clear_ledger
from src.shieldproof.contract import register_contract
from src.shieldproof.milestone import submit_deliverable, verify_milestone
from src.shieldproof.payment import release_payment
from src.shieldproof.dashboard import (
    generate_summary,
    calculate_health_score,
    contract_status,
    export_csv,
    export_json,
    format_currency,
    check,
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


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_generate_summary_empty(self):
        """generate_summary should work with no contracts."""
        summary = generate_summary()
        assert "generated_at" in summary
        assert summary["total_contracts"] == 0

    def test_generate_summary_with_data(self, paid_milestone):
        """generate_summary should include contract data."""
        summary = generate_summary()
        assert summary["total_contracts"] >= 1
        assert summary["total_paid"] >= 250000.00

    def test_generate_summary_fields(self, paid_milestone):
        """generate_summary should include all required fields."""
        summary = generate_summary()
        assert "generated_at" in summary
        assert "version" in summary
        assert "tenant_id" in summary
        assert "total_contracts" in summary
        assert "total_committed" in summary
        assert "total_paid" in summary
        assert "total_verified" in summary
        assert "milestones_pending" in summary
        assert "milestones_disputed" in summary
        assert "waste_identified" in summary
        assert "health_score" in summary


class TestCalculateHealthScore:
    """Tests for calculate_health_score function."""

    def test_health_score_empty(self):
        """health_score should be 100 for empty portfolio."""
        summary = {"total_contracts": 0}
        score = calculate_health_score(summary)
        assert score == 100.0

    def test_health_score_healthy(self):
        """health_score should be high for healthy portfolio."""
        summary = {
            "total_contracts": 10,
            "contracts_on_track": 10,
            "contracts_disputed": 0,
            "total_paid": 1000000,
            "total_verified": 1000000,
        }
        score = calculate_health_score(summary)
        assert score >= 90.0

    def test_health_score_issues(self):
        """health_score should be lower for portfolio with issues."""
        summary = {
            "total_contracts": 10,
            "contracts_on_track": 5,
            "contracts_disputed": 3,
            "total_paid": 1000000,
            "total_verified": 500000,
        }
        score = calculate_health_score(summary)
        assert score < 70.0


class TestContractStatus:
    """Tests for contract_status function."""

    def test_contract_status_found(self, paid_milestone):
        """contract_status should return contract details."""
        status = contract_status(paid_milestone["contract_id"])
        assert status["contract_id"] == paid_milestone["contract_id"]
        assert "contractor" in status
        assert "amount_fixed" in status
        assert "amount_paid" in status
        assert "milestones" in status

    def test_contract_status_not_found(self):
        """contract_status should return error for unknown contract."""
        status = contract_status("NON-EXISTENT")
        assert "error" in status


class TestExportCsv:
    """Tests for export_csv function."""

    def test_export_csv_empty(self):
        """export_csv should handle empty data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            export_csv(filepath)
            with open(filepath, "r") as f:
                content = f.read()
            assert len(content) > 0
        finally:
            os.unlink(filepath)

    def test_export_csv_with_data(self, paid_milestone):
        """export_csv should export contract data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            export_csv(filepath)
            with open(filepath, "r") as f:
                content = f.read()
            assert "contract_id" in content
            assert paid_milestone["contractor"] in content
        finally:
            os.unlink(filepath)


class TestExportJson:
    """Tests for export_json function."""

    def test_export_json_structure(self, paid_milestone):
        """export_json should create valid JSON with summary and contracts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            export_json(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "summary" in data
            assert "contracts" in data
        finally:
            os.unlink(filepath)


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_format_currency_dollars(self):
        """format_currency should format small amounts."""
        assert format_currency(500) == "$500.00"

    def test_format_currency_thousands(self):
        """format_currency should format thousands."""
        assert format_currency(5000) == "$5.00K"

    def test_format_currency_millions(self):
        """format_currency should format millions."""
        assert format_currency(5000000) == "$5.00M"

    def test_format_currency_billions(self):
        """format_currency should format billions."""
        assert format_currency(5000000000) == "$5.00B"


class TestCheck:
    """Tests for check function."""

    def test_check_returns_true(self):
        """check should return True when dashboard works."""
        assert check() is True
