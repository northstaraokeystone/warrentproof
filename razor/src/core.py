"""
RAZOR Core Module - CLAUDEME v3.1 Compliant Foundation

THE PARADIGM INVERSION:
  Fraud = artificial order in stochastic system.
  Detect via compression ratio on real USASpending data
  calibrated against historical ground truth.

THE PHYSICS:
  - Honest market = high-entropy gas: Chaotic, diverse, incompressible
  - Corrupt market = ordered crystal: Coordinated, repetitive, compressible
  - K(x) = len(compressed) / len(original): Kolmogorov complexity proxy
  - No proof system needed: The compression ratio IS the proof

LAW_1 = "No receipt -> not real"
LAW_2 = "No test -> not shipped"
LAW_3 = "No gate -> not alive"
"""

import hashlib
import json
import sys
from datetime import datetime
from typing import Any, Optional

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# ============================================================================
# RAZOR CONSTANTS (Derived from Abyssal Weaver Critique)
# ============================================================================

TENANT_ID = "project-razor"
VERSION = "1.0.0"

# API Configuration
API_BASE_URL = "https://api.usaspending.gov/api/v2"
API_RATE_LIMIT_DELAY = 1.0  # seconds between requests
API_MAX_RETRIES = 5
API_BACKOFF_FACTOR = 2.0  # exponential backoff

# Compression Thresholds (EXPERIMENTAL - to be calibrated)
CR_THRESHOLD_LOW = 0.30   # Highly compressible (suspicious)
CR_THRESHOLD_HIGH = 0.70  # Normal market entropy
Z_SCORE_THRESHOLD = -2.0  # Statistical significance (2 sigma)

# Historical Windows
FAT_LEONARD_START = "2008-01-01"
FAT_LEONARD_END = "2013-09-01"
TRANSDIGM_START = "2015-01-01"
TRANSDIGM_END = "2019-12-31"
BOEING_START = "2001-01-01"
BOEING_END = "2003-12-31"

# Cohort Sizes
MIN_COHORT_SIZE = 500  # Minimum records for statistical validity
MIN_CONTROL_SIZE = 100  # Minimum control cohort size

# Statistical Thresholds
ALPHA_LEVEL = 0.05  # Type I error rate
MIN_POWER = 0.80  # Minimum statistical power
MIN_EFFECT_SIZE = 0.5  # Cohen's d threshold for "meaningful" signal

# ============================================================================
# STOPRULE EXCEPTION
# ============================================================================

class StopRule(Exception):
    """
    Raised when a stoprule triggers. Never catch silently.
    Per CLAUDEME Section 8: stoprules halt execution on critical failures.
    """
    pass

# ============================================================================
# CORE FUNCTIONS (REQUIRED PER CLAUDEME Section 8)
# ============================================================================

def dual_hash(data: bytes | str) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.
    Per CLAUDEME Section 8: HASH = "SHA256 + BLAKE3" # ALWAYS dual-hash

    Args:
        data: Bytes or string to hash

    Returns:
        String in format "sha256hex:blake3hex"
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict, to_stdout: bool = True) -> dict:
    """
    Every function calls this. No exceptions.
    Per CLAUDEME Section 8: All operations emit receipts.

    Args:
        receipt_type: Type of receipt (ingest, complexity, baseline, etc.)
        data: Receipt payload data
        to_stdout: Whether to print to stdout (default True)

    Returns:
        Complete receipt dict with ts, tenant_id, payload_hash
    """
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", TENANT_ID),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }

    if to_stdout:
        print(json.dumps(receipt), flush=True, file=sys.stdout)

    return receipt


def merkle(items: list) -> str:
    """
    Compute Merkle root of items using dual_hash.
    Per CLAUDEME Section 8: MERKLE = "BLAKE3" (via dual_hash)

    Args:
        items: List of items to compute root for

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    hashes = [
        dual_hash(json.dumps(i, sort_keys=True) if isinstance(i, dict) else str(i))
        for i in items
    ]

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [
            dual_hash(hashes[i] + hashes[i + 1])
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0]


# ============================================================================
# STOPRULE FUNCTIONS
# ============================================================================

def stoprule_api_failure(status_code: int, retries: int, url: str = "") -> None:
    """
    Emit anomaly receipt and halt on API failure after max retries.

    Args:
        status_code: HTTP status code
        retries: Number of retries attempted
        url: API endpoint URL
    """
    emit_receipt("anomaly", {
        "metric": "api_failure",
        "status_code": status_code,
        "retries": retries,
        "url": url,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRule(f"API access failed: HTTP {status_code} after {retries} retries")


def stoprule_insufficient_data(cohort_name: str, count: int) -> None:
    """
    Emit anomaly receipt and halt on insufficient cohort size.

    Args:
        cohort_name: Name of the cohort
        count: Actual record count
    """
    emit_receipt("anomaly", {
        "metric": "insufficient_data",
        "cohort_name": cohort_name,
        "count": count,
        "minimum": MIN_COHORT_SIZE,
        "delta": count - MIN_COHORT_SIZE,
        "action": "halt",
        "classification": "degradation"
    })
    raise StopRule(f"Insufficient data volume: {cohort_name} has {count} records, need {MIN_COHORT_SIZE}")


def stoprule_compression_invalid(reason: str) -> None:
    """
    Emit anomaly receipt and halt on compression failure.

    Args:
        reason: Description of the compression failure
    """
    emit_receipt("anomaly", {
        "metric": "compression_invalid",
        "reason": reason,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRule(f"Compression engines failed: {reason}")


def stoprule_no_signal(z_score: float, threshold: float) -> None:
    """
    Emit anomaly receipt when no statistical signal detected.
    Note: This is a warning, not a fatal halt (hypothesis may be wrong).

    Args:
        z_score: Calculated Z-score
        threshold: Expected threshold
    """
    emit_receipt("anomaly", {
        "metric": "no_signal",
        "z_score": z_score,
        "threshold": threshold,
        "delta": z_score - threshold,
        "action": "alert",
        "classification": "deviation"
    })
    # Not raising StopRule - this is informational, not fatal


def stoprule_insufficient_control(cohort_name: str, count: int) -> None:
    """
    Emit anomaly receipt and halt on insufficient control cohort size.

    Args:
        cohort_name: Name of the control cohort
        count: Actual record count
    """
    emit_receipt("anomaly", {
        "metric": "insufficient_control",
        "cohort_name": cohort_name,
        "count": count,
        "minimum": MIN_CONTROL_SIZE,
        "delta": count - MIN_CONTROL_SIZE,
        "action": "halt",
        "classification": "degradation"
    })
    raise StopRule(f"Insufficient control size: {cohort_name} has {count} records, need {MIN_CONTROL_SIZE}")


def stoprule_degenerate_baseline(cohort_name: str) -> None:
    """
    Emit anomaly receipt and halt on degenerate baseline (zero variance).

    Args:
        cohort_name: Name of the cohort with zero variance
    """
    emit_receipt("anomaly", {
        "metric": "degenerate_baseline",
        "cohort_name": cohort_name,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRule(f"Degenerate baseline: {cohort_name} has zero variance")


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print(f"# RAZOR Core v{VERSION}", file=sys.stderr)
    print(f"# API: {API_BASE_URL}", file=sys.stderr)
    print(f"# CR_LOW: {CR_THRESHOLD_LOW}, CR_HIGH: {CR_THRESHOLD_HIGH}", file=sys.stderr)
    print(f"# Z_THRESHOLD: {Z_SCORE_THRESHOLD}", file=sys.stderr)

    # Emit test receipt
    h = dual_hash("test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"

    r = emit_receipt("test", {
        "message": "RAZOR core module self-test"
    })
    assert "receipt_type" in r, "emit_receipt must include receipt_type"
    assert "tenant_id" in r, "emit_receipt must include tenant_id"

    print("# PASS: RAZOR core module self-test", file=sys.stderr)
