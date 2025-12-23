"""
RAZOR Ingest Module - USASpending.gov API Client

Robust ingestion with pagination and rate limiting for federal
procurement data. Implements exponential backoff and stoprule
compliance per CLAUDEME.

API Endpoint: https://api.usaspending.gov/api/v2/search/spending_by_award/

THE PHYSICS:
  We fetch REAL procurement data - not synthetic simulations.
  Real-world logistics = high-entropy gas (chaotic, incompressible)
  Fraudulent billing = low-entropy crystal (ordered, compressible)
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime

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

from .core import (
    API_BASE_URL,
    API_RATE_LIMIT_DELAY,
    API_MAX_RETRIES,
    API_BACKOFF_FACTOR,
    MIN_COHORT_SIZE,
    emit_receipt,
    StopRule,
    stoprule_api_failure,
    stoprule_insufficient_data,
)

# ============================================================================
# USASPENDING INGESTOR CLASS
# ============================================================================

class USASpendingIngestor:
    """
    USASpending.gov API client with robust pagination and rate limiting.

    Implements:
      - Exponential backoff on rate limits
      - Pagination loop (has_next_page)
      - Receipt emission per API call and per cohort
      - Stoprule compliance

    DataFrame Columns:
      - award_id: Unique identifier
      - recipient_name: Contractor name
      - description: Award description (primary compression target)
      - total_obligation: Dollar amount
      - action_date: Transaction date
      - naics_code: Industry classification
      - psc_code: Product/Service code
    """

    def __init__(
        self,
        api_base_url: str = API_BASE_URL,
        rate_limit_delay: float = API_RATE_LIMIT_DELAY,
        max_retries: int = API_MAX_RETRIES,
        backoff_factor: float = API_BACKOFF_FACTOR,
    ):
        """
        Initialize the ingestor.

        Args:
            api_base_url: Base URL for USASpending API
            rate_limit_delay: Seconds to wait between requests
            max_retries: Maximum retry attempts on failure
            backoff_factor: Multiplier for exponential backoff
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required for API access")

        self.api_base_url = api_base_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.search_endpoint = f"{api_base_url}/search/spending_by_award/"

    def build_payload(
        self,
        cohort_config: dict,
        page: int = 1,
        limit: int = 100,
    ) -> dict:
        """
        Construct POST body with filters.

        Args:
            cohort_config: Cohort configuration with filters
            page: Page number for pagination
            limit: Records per page

        Returns:
            POST payload dict
        """
        filters = {}

        # Time period filter
        if "time_period" in cohort_config:
            filters["time_period"] = cohort_config["time_period"]

        # Agency filter
        if "agencies" in cohort_config:
            filters["agencies"] = cohort_config["agencies"]

        # Keywords filter
        if "keywords" in cohort_config:
            filters["keywords"] = cohort_config["keywords"]

        # Award type codes
        if "award_type_codes" in cohort_config:
            filters["award_type_codes"] = cohort_config["award_type_codes"]

        # Recipient search text
        if "recipient_search_text" in cohort_config:
            # USASpending uses different filter for recipient
            filters["recipient_search_text"] = cohort_config["recipient_search_text"]

        # NAICS codes for control cohorts
        if "naics_codes" in cohort_config:
            filters["naics_codes"] = cohort_config["naics_codes"]

        # PSC codes for control cohorts
        if "psc_codes" in cohort_config:
            filters["psc_codes"] = cohort_config["psc_codes"]

        # Fields to return
        fields = [
            "Award ID",
            "Recipient Name",
            "Description",
            "Award Amount",
            "Start Date",
            "NAICS Code",
            "PSC Code",
            "Awarding Agency",
            "Award Type",
        ]

        return {
            "filters": filters,
            "fields": fields,
            "page": page,
            "limit": limit,
            "sort": "Award Amount",
            "order": "desc",
        }

    def fetch_page(
        self,
        payload: dict,
        page: int,
        cohort_name: str = "unknown",
    ) -> dict:
        """
        Execute POST request with pagination and exponential backoff.

        Args:
            payload: POST body with filters
            page: Page number
            cohort_name: Name of cohort for receipts

        Returns:
            API response dict with results and pagination info

        Raises:
            StopRule: If max retries exceeded
        """
        payload["page"] = page
        url = self.search_endpoint

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                if attempt > 0 or page > 1:
                    delay = self.rate_limit_delay * (self.backoff_factor ** attempt)
                    time.sleep(delay)

                response = requests.post(
                    url,
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"},
                )

                # Emit receipt for this API call
                receipt_data = {
                    "url": url,
                    "status_code": response.status_code,
                    "page": page,
                    "cohort_name": cohort_name,
                    "attempt": attempt + 1,
                }

                if response.status_code == 200:
                    data = response.json()
                    receipt_data["records_fetched"] = len(data.get("results", []))
                    receipt_data["has_next_page"] = data.get("page_metadata", {}).get(
                        "hasNext", False
                    )
                    emit_receipt("ingest", receipt_data, to_stdout=False)
                    return data

                elif response.status_code == 429:
                    # Rate limited - will retry with backoff
                    receipt_data["action"] = "retry_rate_limit"
                    emit_receipt("ingest", receipt_data, to_stdout=False)
                    continue

                else:
                    # Non-200, non-429 status
                    receipt_data["action"] = "retry_error"
                    emit_receipt("ingest", receipt_data, to_stdout=False)

            except requests.exceptions.Timeout:
                emit_receipt("ingest", {
                    "url": url,
                    "status_code": 0,
                    "page": page,
                    "cohort_name": cohort_name,
                    "attempt": attempt + 1,
                    "action": "retry_timeout",
                }, to_stdout=False)

            except requests.exceptions.RequestException as e:
                emit_receipt("ingest", {
                    "url": url,
                    "status_code": 0,
                    "page": page,
                    "cohort_name": cohort_name,
                    "attempt": attempt + 1,
                    "action": "retry_network_error",
                    "error": str(e)[:100],
                }, to_stdout=False)

        # Max retries exceeded
        stoprule_api_failure(
            status_code=response.status_code if 'response' in dir() else 0,
            retries=self.max_retries,
            url=url,
        )

    def ingest_cohort(
        self,
        cohort_name: str,
        cohort_config: dict,
        target_records: int = MIN_COHORT_SIZE,
    ) -> "pd.DataFrame":
        """
        Ingest all pages for a cohort until target reached or exhausted.

        Args:
            cohort_name: Name of the cohort
            cohort_config: Configuration with filters
            target_records: Minimum records to fetch

        Returns:
            DataFrame with all fetched records

        Raises:
            StopRule: If insufficient data or API failure
        """
        if not HAS_PANDAS:
            raise ImportError("pandas library required for cohort ingestion")

        all_records = []
        page = 1
        api_calls = 0
        start_time = datetime.utcnow()

        while True:
            payload = self.build_payload(cohort_config, page=page)
            data = self.fetch_page(payload, page, cohort_name)
            api_calls += 1

            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)

            # Check pagination
            page_meta = data.get("page_metadata", {})
            has_next = page_meta.get("hasNext", False)

            if not has_next or len(all_records) >= target_records:
                break

            page += 1

            # Safety limit
            if page > 100:
                break

        # Convert to DataFrame
        df = self._results_to_dataframe(all_records)

        # Emit cohort receipt
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        emit_receipt("cohort", {
            "cohort_name": cohort_name,
            "total_records": len(df),
            "api_calls": api_calls,
            "duration_seconds": duration,
            "time_span": cohort_config.get("time_period", []),
        }, to_stdout=False)

        return df

    def _results_to_dataframe(self, results: List[dict]) -> "pd.DataFrame":
        """
        Convert API results to normalized DataFrame.

        Args:
            results: List of result dicts from API

        Returns:
            Normalized DataFrame with standard columns
        """
        if not results:
            return pd.DataFrame(columns=[
                "award_id",
                "recipient_name",
                "description",
                "total_obligation",
                "action_date",
                "naics_code",
                "psc_code",
            ])

        records = []
        for r in results:
            records.append({
                "award_id": r.get("Award ID", ""),
                "recipient_name": r.get("Recipient Name", ""),
                "description": r.get("Description", "") or "",
                "total_obligation": float(r.get("Award Amount", 0) or 0),
                "action_date": r.get("Start Date", ""),
                "naics_code": r.get("NAICS Code", ""),
                "psc_code": r.get("PSC Code", ""),
                "awarding_agency": r.get("Awarding Agency", ""),
                "award_type": r.get("Award Type", ""),
            })

        return pd.DataFrame(records)

    def build_control_cohort(
        self,
        control_config: dict,
        fraud_cohort_name: str,
        target_records: int = MIN_COHORT_SIZE,
    ) -> "pd.DataFrame":
        """
        Fetch control cohort peers excluding fraud entities.

        Args:
            control_config: Control cohort configuration
            fraud_cohort_name: Name of fraud cohort for reference
            target_records: Minimum records to fetch

        Returns:
            DataFrame with control cohort records
        """
        # Build config for control cohort ingestion
        config = {
            "time_period": [control_config["time_period"]],
            "award_type_codes": ["A", "B", "C", "D"],
        }

        if "naics_codes" in control_config:
            config["naics_codes"] = control_config["naics_codes"]

        if "psc_codes" in control_config:
            config["psc_codes"] = control_config["psc_codes"]

        if "agencies" in control_config:
            config["agencies"] = control_config["agencies"]

        control_name = f"{fraud_cohort_name}_control"
        df = self.ingest_cohort(control_name, config, target_records)

        # Exclude fraud entities if specified
        if "exclude_keywords" in control_config:
            exclude = control_config["exclude_keywords"]
            mask = ~df["recipient_name"].str.contains(
                "|".join(exclude),
                case=False,
                na=False,
            )
            df = df[mask].reset_index(drop=True)

        return df

    def test_connectivity(self) -> dict:
        """
        Test API connectivity with minimal request.

        Returns:
            Dict with status and latency
        """
        payload = {
            "filters": {
                "keywords": ["test"],
                "time_period": [{"start_date": "2023-01-01", "end_date": "2023-01-31"}],
            },
            "page": 1,
            "limit": 1,
        }

        start = time.time()
        try:
            response = requests.post(
                self.search_endpoint,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            latency = time.time() - start

            result = {
                "status": "ok" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": int(latency * 1000),
                "endpoint": self.search_endpoint,
            }

            emit_receipt("api_test", result, to_stdout=False)
            return result

        except Exception as e:
            result = {
                "status": "error",
                "error": str(e)[:100],
                "endpoint": self.search_endpoint,
            }
            emit_receipt("api_test", result, to_stdout=False)
            return result


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("# RAZOR Ingest Module", file=sys.stderr)

    if not HAS_REQUESTS:
        print("# SKIP: requests library not installed", file=sys.stderr)
        sys.exit(0)

    ingestor = USASpendingIngestor()
    print(f"# Endpoint: {ingestor.search_endpoint}", file=sys.stderr)

    # Test connectivity
    result = ingestor.test_connectivity()
    print(f"# Connectivity test: {result['status']}", file=sys.stderr)

    if result["status"] == "ok":
        print(f"# Latency: {result['latency_ms']}ms", file=sys.stderr)
        print("# PASS: RAZOR ingest module self-test", file=sys.stderr)
    else:
        print(f"# FAIL: {result.get('error', 'Unknown error')}", file=sys.stderr)
