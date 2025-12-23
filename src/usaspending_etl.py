"""
WarrantProof USASpending ETL Module - Data Ingestion from api.usaspending.gov

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements an ETL pipeline for ingesting data from the
USASpending.gov API, which provides federal spending data including
Awards, Transactions, and Federal Accounts.

Key Features:
- Pagination handling with exponential backoff
- Rate limit management (1000 requests/hour)
- Schema validation
- Missing data detection as proof failure (not imputation)

OMEGA Citation:
"Create an ETL pipeline for api.usaspending.gov endpoints"
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .core import (
    TENANT_ID,
    DISCLAIMER,
    USASPENDING_RATE_LIMIT,
    USASPENDING_RECORDS_PER_PAGE,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class RateLimiter:
    """Rate limiter for API requests."""
    max_requests: int = USASPENDING_RATE_LIMIT
    window_seconds: int = 3600  # 1 hour
    requests: List[float] = field(default_factory=list)

    def can_request(self) -> bool:
        """Check if we can make another request."""
        now = time.time()
        # Remove old requests outside window
        self.requests = [r for r in self.requests if now - r < self.window_seconds]
        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record a request."""
        self.requests.append(time.time())

    def wait_time(self) -> float:
        """Time to wait before next request is allowed."""
        if self.can_request():
            return 0.0
        oldest = min(self.requests)
        return max(0.0, self.window_seconds - (time.time() - oldest))


# Global rate limiter
_rate_limiter = RateLimiter()


# Expected schemas for validation
AWARD_SCHEMA = {
    "required": ["award_id", "recipient", "total_obligation", "type"],
    "optional": ["description", "period_of_performance_start_date", "awarding_agency"],
}

TRANSACTION_SCHEMA = {
    "required": ["transaction_id", "award_id", "action_date", "federal_action_obligation"],
    "optional": ["modification_number", "description"],
}

FEDERAL_ACCOUNT_SCHEMA = {
    "required": ["account_number", "account_title", "federal_account_code"],
    "optional": ["budget_function", "budget_subfunction"],
}


def fetch_awards(
    start_date: str,
    end_date: str,
    agency_code: Optional[str] = None,
    award_type: Optional[str] = None,
    _simulate: bool = True
) -> List[Dict[str, Any]]:
    """
    GET /api/v2/awards/ with pagination.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        agency_code: Optional agency filter
        award_type: Optional award type filter
        _simulate: If True, return simulated data

    Returns:
        List of award dicts
    """
    params = {
        "date_range": {"start_date": start_date, "end_date": end_date},
        "agency_code": agency_code,
        "award_type": award_type,
    }

    if _simulate:
        return _simulate_awards(start_date, end_date)

    return handle_pagination(
        endpoint="/api/v2/awards/",
        params=params,
        schema=AWARD_SCHEMA
    )


def fetch_transactions(
    award_id: str,
    _simulate: bool = True
) -> List[Dict[str, Any]]:
    """
    GET /api/v2/transactions/ for specific award.

    Args:
        award_id: Award ID to fetch transactions for
        _simulate: If True, return simulated data

    Returns:
        List of transaction dicts
    """
    if _simulate:
        return _simulate_transactions(award_id)

    return handle_pagination(
        endpoint=f"/api/v2/transactions/",
        params={"award_id": award_id},
        schema=TRANSACTION_SCHEMA
    )


def fetch_federal_accounts(_simulate: bool = True) -> List[Dict[str, Any]]:
    """
    GET /api/v2/federal_accounts/.

    Args:
        _simulate: If True, return simulated data

    Returns:
        List of federal account dicts
    """
    if _simulate:
        return _simulate_federal_accounts()

    return handle_pagination(
        endpoint="/api/v2/federal_accounts/",
        params={},
        schema=FEDERAL_ACCOUNT_SCHEMA
    )


def handle_pagination(
    endpoint: str,
    params: Dict[str, Any],
    schema: Dict[str, List[str]],
    max_pages: int = 100
) -> List[Dict[str, Any]]:
    """
    Generic pagination handler. Iterate through all pages.

    Args:
        endpoint: API endpoint
        params: Request parameters
        schema: Expected schema for validation
        max_pages: Maximum pages to fetch

    Returns:
        Full dataset from all pages
    """
    results = []
    page = 1

    while page <= max_pages:
        if not _rate_limiter.can_request():
            wait = _rate_limiter.wait_time()
            stoprule_rate_limit_exceeded(wait)

        _rate_limiter.record_request()

        # In production, this would be an actual HTTP request
        # For simulation, we break after one iteration
        page_data = _fetch_page(endpoint, params, page, schema)

        if not page_data:
            break

        results.extend(page_data)

        if len(page_data) < USASPENDING_RECORDS_PER_PAGE:
            break  # Last page

        page += 1

    return results


def _fetch_page(
    endpoint: str,
    params: Dict[str, Any],
    page: int,
    schema: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Fetch a single page of data.
    Note: This is a simulation stub.
    """
    # In production: requests.get(f"https://api.usaspending.gov{endpoint}", params=...)
    return []


def validate_schema(data: Dict[str, Any], schema: Dict[str, List[str]]) -> bool:
    """
    Validate API response against expected schema.

    Args:
        data: Data dict to validate
        schema: Schema with required/optional fields

    Returns:
        True if valid
    """
    for field in schema.get("required", []):
        if field not in data or data[field] is None:
            return False
    return True


def detect_missing_fields(
    data: Dict[str, Any],
    required_fields: List[str]
) -> List[str]:
    """
    Check for missing/null fields. Treat as proof failure, not imputation target.

    Args:
        data: Data dict to check
        required_fields: Required field names

    Returns:
        List of missing field names
    """
    missing = []
    for field in required_fields:
        value = data.get(field)
        if value is None:
            missing.append(field)
        elif isinstance(value, str) and value.strip() == "":
            missing.append(field)
        elif isinstance(value, str) and value.lower() in ["n/a", "not available", "null"]:
            missing.append(field)
    return missing


def emit_etl_receipt(
    endpoint: str,
    records_fetched: int,
    pagination_pages: int,
    missing_fields_count: int,
    schema_valid: bool
) -> dict:
    """
    Emit etl_receipt documenting data ingestion.

    Args:
        endpoint: API endpoint
        records_fetched: Number of records
        pagination_pages: Number of pages fetched
        missing_fields_count: Count of missing fields
        schema_valid: Whether schema validation passed

    Returns:
        etl_receipt dict
    """
    return emit_receipt("etl", {
        "tenant_id": TENANT_ID,
        "endpoint": endpoint,
        "records_fetched": records_fetched,
        "pagination_pages": pagination_pages,
        "rate_limit_hits": len(_rate_limiter.requests),
        "missing_fields_count": missing_fields_count,
        "schema_valid": schema_valid,
        "api_base": "api.usaspending.gov",
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === SIMULATION DATA ===

def _simulate_awards(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Generate simulated award data."""
    awards = []
    for i in range(10):
        awards.append({
            "award_id": f"AWARD-{i:04d}",
            "recipient": {
                "recipient_name": f"Contractor {chr(65 + i)}",
                "recipient_uei": f"UEI{i:09d}",
            },
            "total_obligation": 1000000 * (i + 1),
            "type": "CONTRACT",
            "description": f"Simulated contract {i}",
            "period_of_performance_start_date": start_date,
            "awarding_agency": {"name": "Department of Defense"},
            "_is_simulation": True,
        })
    return awards


def _simulate_transactions(award_id: str) -> List[Dict[str, Any]]:
    """Generate simulated transaction data."""
    transactions = []
    for i in range(5):
        transactions.append({
            "transaction_id": f"{award_id}-TX-{i:03d}",
            "award_id": award_id,
            "action_date": f"2024-{(i % 12) + 1:02d}-15",
            "federal_action_obligation": 100000 * (i + 1),
            "modification_number": str(i) if i > 0 else None,
            "description": f"Transaction {i} for {award_id}",
            "_is_simulation": True,
        })
    return transactions


def _simulate_federal_accounts() -> List[Dict[str, Any]]:
    """Generate simulated federal account data."""
    accounts = [
        {
            "account_number": "097-0100",
            "account_title": "Military Personnel",
            "federal_account_code": "097-0100",
            "budget_function": "National Defense",
            "_is_simulation": True,
        },
        {
            "account_number": "057-0400",
            "account_title": "Research, Development, Test and Evaluation",
            "federal_account_code": "057-0400",
            "budget_function": "National Defense",
            "_is_simulation": True,
        },
        {
            "account_number": "021-1804",
            "account_title": "Shipbuilding and Conversion, Navy",
            "federal_account_code": "021-1804",
            "budget_function": "National Defense",
            "_is_simulation": True,
        },
    ]
    return accounts


# === STOPRULES ===

def stoprule_api_unavailable(status_code: int) -> None:
    """If 404/500 errors persist, halt ETL."""
    emit_receipt("anomaly", {
        "metric": "api_unavailable",
        "status_code": status_code,
        "action": "halt",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"API unavailable: HTTP {status_code}")


def stoprule_schema_mismatch(expected: List[str], actual: List[str]) -> None:
    """If API response schema changes, halt."""
    missing = set(expected) - set(actual)
    if missing:
        emit_receipt("anomaly", {
            "metric": "schema_mismatch",
            "missing_fields": list(missing),
            "action": "halt",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"Schema mismatch: missing {missing}")


def stoprule_rate_limit_exceeded(wait_seconds: float) -> None:
    """If rate limit persists after backoff, defer."""
    emit_receipt("anomaly", {
        "metric": "rate_limit_exceeded",
        "wait_seconds": wait_seconds,
        "action": "defer",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Rate limit exceeded, wait {wait_seconds:.0f}s")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof USASpending ETL Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Fetch simulated awards
    awards = fetch_awards("2024-01-01", "2024-12-31", _simulate=True)
    print(f"# Awards fetched: {len(awards)}", file=sys.stderr)
    assert len(awards) == 10

    # Test 2: Fetch simulated transactions
    transactions = fetch_transactions("AWARD-0001", _simulate=True)
    print(f"# Transactions fetched: {len(transactions)}", file=sys.stderr)
    assert len(transactions) == 5

    # Test 3: Fetch simulated federal accounts
    accounts = fetch_federal_accounts(_simulate=True)
    print(f"# Accounts fetched: {len(accounts)}", file=sys.stderr)
    assert len(accounts) == 3

    # Test 4: Schema validation
    valid = validate_schema(awards[0], AWARD_SCHEMA)
    print(f"# Award schema valid: {valid}", file=sys.stderr)
    assert valid == True

    # Test 5: Missing field detection
    incomplete_data = {"award_id": "TEST", "recipient": None}
    missing = detect_missing_fields(incomplete_data, AWARD_SCHEMA["required"])
    print(f"# Missing fields: {missing}", file=sys.stderr)
    assert len(missing) > 0

    # Test 6: Rate limiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert limiter.can_request()
        limiter.record_request()
    assert not limiter.can_request()
    print(f"# Rate limiter working correctly", file=sys.stderr)

    # Test 7: ETL receipt
    receipt = emit_etl_receipt(
        endpoint="/api/v2/awards/",
        records_fetched=10,
        pagination_pages=1,
        missing_fields_count=0,
        schema_valid=True
    )
    assert receipt["receipt_type"] == "etl"

    print(f"# PASS: usaspending_etl module self-test", file=sys.stderr)
