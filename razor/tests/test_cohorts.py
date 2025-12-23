"""
Tests for RAZOR Cohorts Module - Historical Fraud Cohort Definitions
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cohorts import (
    list_cohorts,
    get_cohort_config,
    validate_cohort,
    get_cohort_description,
    get_fraud_type,
    get_expected_signal,
    emit_cohort_summary_receipt,
    COHORTS,
    FAT_LEONARD_CONFIG,
    TRANSDIGM_CONFIG,
    BOEING_CONFIG,
)
from src.core import StopRule


class TestCohortDefinitions:
    """Tests for cohort definitions."""

    def test_all_cohorts_defined(self):
        """Test that all expected cohorts are defined."""
        cohorts = list_cohorts()
        assert "fat_leonard" in cohorts
        assert "transdigm" in cohorts
        assert "boeing" in cohorts
        assert len(cohorts) == 3

    def test_all_cohorts_valid(self):
        """Test that all cohorts have valid configuration."""
        for cohort_name in list_cohorts():
            assert validate_cohort(cohort_name), f"{cohort_name} is invalid"

    def test_invalid_cohort_returns_false(self):
        """Test that non-existent cohort returns False."""
        assert validate_cohort("nonexistent") is False


class TestGetCohortConfig:
    """Tests for get_cohort_config function."""

    def test_get_fat_leonard_config(self):
        """Test getting Fat Leonard config."""
        config = get_cohort_config("fat_leonard")

        assert config["name"] == "Fat Leonard (GDMA)"
        assert "fraud_config" in config
        assert "control_config" in config
        assert "hypothesis" in config
        assert "husbanding" in config["fraud_config"]["keywords"]

    def test_get_transdigm_config(self):
        """Test getting TransDigm config."""
        config = get_cohort_config("transdigm")

        assert config["name"] == "TransDigm Monopoly Pricing"
        assert "fraud_config" in config
        assert config["fraud_type"] == "value_decoupling"

    def test_get_boeing_config(self):
        """Test getting Boeing config."""
        config = get_cohort_config("boeing")

        assert config["name"] == "Boeing KC-767 Tanker Conflict"
        assert "KC-767" in config["fraud_config"]["keywords"]

    def test_get_invalid_cohort_raises(self):
        """Test that getting invalid cohort raises StopRule."""
        with pytest.raises(StopRule):
            get_cohort_config("nonexistent")


class TestCohortDescriptions:
    """Tests for cohort description functions."""

    def test_get_description_fat_leonard(self):
        """Test getting Fat Leonard description."""
        desc = get_cohort_description("fat_leonard")
        assert "Leonard" in desc or "GDMA" in desc or "bribed" in desc

    def test_get_description_transdigm(self):
        """Test getting TransDigm description."""
        desc = get_cohort_description("transdigm")
        assert "TransDigm" in desc or "price" in desc

    def test_get_description_invalid(self):
        """Test description for invalid cohort."""
        desc = get_cohort_description("nonexistent")
        assert "Unknown" in desc


class TestFraudTypes:
    """Tests for fraud type classification."""

    def test_fat_leonard_fraud_type(self):
        """Test Fat Leonard is classified as copy_paste."""
        assert get_fraud_type("fat_leonard") == "copy_paste"

    def test_transdigm_fraud_type(self):
        """Test TransDigm is classified as value_decoupling."""
        assert get_fraud_type("transdigm") == "value_decoupling"

    def test_boeing_fraud_type(self):
        """Test Boeing is classified as template_fraud."""
        assert get_fraud_type("boeing") == "template_fraud"

    def test_invalid_fraud_type(self):
        """Test invalid cohort returns unknown."""
        assert get_fraud_type("nonexistent") == "unknown"


class TestExpectedSignals:
    """Tests for expected signal retrieval."""

    def test_expected_signal_fat_leonard(self):
        """Test expected signal for Fat Leonard."""
        signal = get_expected_signal("fat_leonard")

        assert signal["expected_z_score"] == "< -2.5"
        assert signal["fraud_type"] == "copy_paste"
        assert "husbanding" in signal["hypothesis"].lower()

    def test_expected_signal_transdigm(self):
        """Test expected signal for TransDigm."""
        signal = get_expected_signal("transdigm")

        assert signal["expected_z_score"] == "< -1.5"

    def test_expected_signal_invalid(self):
        """Test expected signal for invalid cohort."""
        signal = get_expected_signal("nonexistent")
        assert signal == {}


class TestCohortConfigs:
    """Tests for raw cohort configurations."""

    def test_fat_leonard_config_structure(self):
        """Test Fat Leonard config has required fields."""
        assert "keywords" in FAT_LEONARD_CONFIG
        assert "recipient_search_text" in FAT_LEONARD_CONFIG
        assert "time_period" in FAT_LEONARD_CONFIG
        assert "agencies" in FAT_LEONARD_CONFIG
        assert "award_type_codes" in FAT_LEONARD_CONFIG

    def test_transdigm_config_structure(self):
        """Test TransDigm config has required fields."""
        assert "recipient_search_text" in TRANSDIGM_CONFIG
        assert "time_period" in TRANSDIGM_CONFIG

    def test_boeing_config_structure(self):
        """Test Boeing config has required fields."""
        assert "keywords" in BOEING_CONFIG
        assert "recipient_search_text" in BOEING_CONFIG
        assert "KC-767" in BOEING_CONFIG["keywords"]


class TestEmitCohortSummary:
    """Tests for cohort summary receipt."""

    def test_emit_summary_receipt(self):
        """Test emitting cohort summary receipt."""
        receipt = emit_cohort_summary_receipt()

        assert receipt["receipt_type"] == "cohort_summary"
        assert receipt["cohort_count"] == 3
        assert "cohorts" in receipt
        assert "fat_leonard" in receipt["cohorts"]
        assert "transdigm" in receipt["cohorts"]
        assert "boeing" in receipt["cohorts"]

    def test_summary_receipt_structure(self):
        """Test structure of cohort summary."""
        receipt = emit_cohort_summary_receipt()

        for cohort_name, info in receipt["cohorts"].items():
            assert "display_name" in info
            assert "fraud_type" in info
            assert "expected_z_score" in info
            assert "time_period" in info


class TestCohortIntegrity:
    """Integration tests for cohort definitions."""

    def test_all_cohorts_have_control_configs(self):
        """Test all cohorts have control configurations."""
        for cohort_name in list_cohorts():
            config = get_cohort_config(cohort_name)
            assert "control_config" in config
            assert "naics_codes" in config["control_config"] or \
                   "psc_codes" in config["control_config"]

    def test_all_cohorts_have_time_periods(self):
        """Test all cohorts have time periods."""
        for cohort_name in list_cohorts():
            config = get_cohort_config(cohort_name)
            assert "time_period" in config["fraud_config"]
            assert len(config["fraud_config"]["time_period"]) > 0

    def test_control_excludes_fraud_entities(self):
        """Test control configs exclude fraud entities."""
        for cohort_name in list_cohorts():
            config = get_cohort_config(cohort_name)
            assert "exclude_keywords" in config["control_config"]
            assert len(config["control_config"]["exclude_keywords"]) > 0
