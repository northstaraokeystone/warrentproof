"""
Tests for RAZOR Ingest Module - USASpending API Client
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.core import API_BASE_URL, API_MAX_RETRIES


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests required")
class TestUSASpendingIngestor:
    """Tests for USASpendingIngestor class."""

    def test_init_default_values(self):
        """Test ingestor initialization with defaults."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()

        assert ingestor.api_base_url == API_BASE_URL
        assert ingestor.max_retries == API_MAX_RETRIES

    def test_init_custom_values(self):
        """Test ingestor initialization with custom values."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor(
            api_base_url="https://custom.api.gov",
            rate_limit_delay=2.0,
            max_retries=3,
        )

        assert ingestor.api_base_url == "https://custom.api.gov"
        assert ingestor.rate_limit_delay == 2.0
        assert ingestor.max_retries == 3

    def test_build_payload_basic(self):
        """Test payload building with basic config."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        config = {
            "time_period": [{"start_date": "2020-01-01", "end_date": "2020-12-31"}],
            "keywords": ["test"],
        }

        payload = ingestor.build_payload(config, page=1, limit=100)

        assert "filters" in payload
        assert "time_period" in payload["filters"]
        assert "keywords" in payload["filters"]
        assert payload["page"] == 1
        assert payload["limit"] == 100

    def test_build_payload_with_agencies(self):
        """Test payload building with agency filters."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        config = {
            "time_period": [{"start_date": "2020-01-01", "end_date": "2020-12-31"}],
            "agencies": [{"type": "awarding", "tier": "toptier", "name": "Navy"}],
        }

        payload = ingestor.build_payload(config)

        assert "agencies" in payload["filters"]
        assert payload["filters"]["agencies"][0]["name"] == "Navy"

    def test_build_payload_pagination(self):
        """Test payload building with pagination."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        config = {"time_period": [{"start_date": "2020-01-01", "end_date": "2020-12-31"}]}

        p1 = ingestor.build_payload(config, page=1)
        p2 = ingestor.build_payload(config, page=5)

        assert p1["page"] == 1
        assert p2["page"] == 5


@pytest.mark.skipif(not HAS_REQUESTS or not HAS_PANDAS, reason="requests and pandas required")
class TestResultsToDataFrame:
    """Tests for results to DataFrame conversion."""

    def test_empty_results(self):
        """Test conversion of empty results."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        df = ingestor._results_to_dataframe([])

        assert len(df) == 0
        assert "award_id" in df.columns

    def test_results_conversion(self):
        """Test conversion of sample results."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        results = [
            {
                "Award ID": "ABC123",
                "Recipient Name": "Test Corp",
                "Description": "Test description",
                "Award Amount": 50000,
                "Start Date": "2020-06-15",
                "NAICS Code": "123456",
                "PSC Code": "A1",
            }
        ]

        df = ingestor._results_to_dataframe(results)

        assert len(df) == 1
        assert df.iloc[0]["award_id"] == "ABC123"
        assert df.iloc[0]["recipient_name"] == "Test Corp"
        assert df.iloc[0]["total_obligation"] == 50000

    def test_handles_missing_fields(self):
        """Test handling of results with missing fields."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        results = [
            {
                "Award ID": "ABC123",
                # Missing most fields
            }
        ]

        df = ingestor._results_to_dataframe(results)

        assert len(df) == 1
        assert df.iloc[0]["award_id"] == "ABC123"
        assert df.iloc[0]["description"] == ""


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests required")
class TestConnectivityTest:
    """Tests for API connectivity testing."""

    def test_connectivity_test_structure(self):
        """Test that connectivity test returns expected structure."""
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        # Note: This may fail in CI without network access
        result = ingestor.test_connectivity()

        assert "status" in result
        assert result["status"] in ["ok", "error"]
        assert "endpoint" in result


class TestImportWithoutDependencies:
    """Tests for graceful handling of missing dependencies."""

    def test_module_import(self):
        """Test that module can be imported."""
        # This should not raise even if requests is missing
        try:
            from src import ingest
            assert True
        except ImportError as e:
            # requests missing is expected in some environments
            assert "requests" in str(e).lower()
