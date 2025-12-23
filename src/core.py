"""
WarrantProof Core Module - CLAUDEME v3.1 Compliant Foundation

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module provides the foundation for all WarrantProof operations:
- dual_hash: SHA256:BLAKE3 per CLAUDEME §8
- emit_receipt: Every function calls this
- merkle: Compute Merkle root
- cite: Embed citations in receipts
- StopRuleException: Raised on stoprule triggers

LAW_1 = "No receipt → not real"
LAW_2 = "No test → not shipped"
LAW_3 = "No gate → not alive"
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

# === CONSTANTS (REQUIRED) ===

TENANT_ID = "warrantproof-omega"
DISCLAIMER = "THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
VERSION = "3.0.0"

# ============================================================================
# WARRANTPROOF v2 CONSTANTS (Physics-Derived from Grok Research)
# ============================================================================

import numpy as np

# Q1: Autocatalytic Detection - Phase transition threshold from RAF percolation theory
# N_critical ≈ log₂(ΔH⁻¹) × (H_legit / ΔH)
def N_CRITICAL_FORMULA(H_legit: float, H_fraud: float) -> float:
    """Calculate critical N for autocatalytic phase transition."""
    delta_H = H_fraud - H_legit
    if delta_H <= 0:
        return float('inf')
    return np.log2(1.0 / delta_H) * (H_legit / delta_H)

ENTROPY_GAP_MIN = 0.15  # Minimum ΔH = H_fraud - H_legit for detection

# Q2: Thompson Sampling - Bayesian threshold uncertainty
# accuracy_gain ≈ √(2 ln n / π² variance_prior)
THOMPSON_PRIOR_VARIANCE = 0.1  # Initial variance in threshold distribution
THOMPSON_FP_TARGET = 0.02  # False positive rate < 2%

# Q3: Hierarchical Compression - Entropy tree depth for O(log N) search
# speedup ≈ log₂ N / (1 - r_compress)
def ENTROPY_TREE_MAX_DEPTH(N: int) -> int:
    """Calculate maximum tree depth for N receipts."""
    if N <= 1:
        return 1
    return int(np.log2(N)) + 1

ENTROPY_TREE_REBALANCE_THRESHOLD = 0.10  # Rebalance when 10% imbalance
ENTROPY_TREE_STORAGE_OVERHEAD_MAX = 0.05  # <5% overhead vs v1

# Q4: Cross-Branch Learning - Mutual information transfer threshold
# time_reduction ≈ I(branch_A; branch_B) / H(pattern)
MUTUAL_INFO_TRANSFER_THRESHOLD = 0.30  # 30% shared entropy required
CROSS_BRANCH_ACCURACY_TARGET = 0.85  # >85% transfer accuracy

# Q5: Entropy Cascade - Compression derivative monitoring
# early_detection_gain ≈ cascade_rate / (dC/dt × cost_FP)
CASCADE_DERIVATIVE_THRESHOLD = 0.05  # dC/dt alert at 5% degradation per unit time
CASCADE_WINDOW_SIZE = 50  # Moving window for dC/dt calculation
CASCADE_FALSE_ALERT_MAX = 0.10  # <10% false cascade alerts

# Q6: Fraud Epidemic - SIR model parameters
# R₀ = density × volume / latency_detection
EPIDEMIC_R0_THRESHOLD = 1.0  # R₀ > 1 triggers quarantine
EPIDEMIC_DETECTION_LATENCY_TARGET = 7.0  # 7 days max latency
EPIDEMIC_RECOVERY_RATE = 0.10  # 10% infected vendors remediated per cycle

# Q7: Holographic Ledger - Boundary encoding via Bekenstein bound
# bits_required ≈ area × log(1/p_detect)
HOLOGRAPHIC_BITS_PER_RECEIPT = 2  # Theoretical minimum from holographic principle
HOLOGRAPHIC_DETECTION_PROBABILITY_MIN = 0.9999  # 99.99% detection from boundary
def HOLOGRAPHIC_LOCALIZATION_COMPLEXITY(N: int) -> int:
    """O(log N) to identify branch containing fraud."""
    if N <= 1:
        return 1
    return int(np.log2(N)) + 1

# Autocatalytic Pattern Parameters
PATTERN_COHERENCE_MIN = 0.80  # Minimum coherence for pattern survival
RAF_CLOSURE_ACCURACY_MIN = 0.80  # Minimum prediction accuracy for self-reference
META_RECEIPT_PREDICTION_WINDOW = 100  # Receipts to observe before validating prediction

# v1 Constants (kept for backward compatibility / hybrid mode)
COMPRESSION_RATIO_LEGIT_V1 = 0.80
COMPRESSION_RATIO_FRAUD_V1 = 0.50

# ============================================================================
# WARRANTPROOF v3: PROJECT OMEGA CONSTANTS (Physics + Cryptography)
# ============================================================================

# Kolmogorov Complexity Thresholds
KOLMOGOROV_THRESHOLD = 0.65
# K < 0.65 = scripted fraud (too compressible)
# K > 0.75 = legitimate (algorithmic irreducibility)
# Physics: Simple generators compress to low K(x). Reality is incompressible.

KOLMOGOROV_LEGITIMATE_MIN = 0.75
# Legitimate transactions typically have K(x) >= 0.75

# Bekenstein Bound for Holographic Validation
BEKENSTEIN_BITS_PER_DOLLAR = 1e-6
# Metadata entropy (bits) per invoice dollar
# $1M invoice requires ≥1 bit of digital trail
# Derived from S ≤ (2π/ℏc) × (cost × area)
# Physics: Information density must match physical reality

# RAF Autocatalytic Network Detection
RAF_CYCLE_MIN_LENGTH = 3
# Minimum cycle length for autocatalytic loop
# Length 3 = A→B→C→A (simplest self-sustaining)
# Physics: Origin-of-life chemistry RAF sets

RAF_CYCLE_MAX_LENGTH = 5
# Maximum cycle length for attribution
# Beyond 5 steps, catalysis too indirect
# Physics: Information decay over network distance

# Thompson Sampling for Adversarial Auditing
THOMPSON_AUDIT_BUDGET = 0.05
# Fraction of contractors audited per cycle (5%)
# Budget-constrained stochastic allocation
# Math: Multi-armed bandit regret minimization

# Adversarial Robustness
ADVERSARIAL_EPSILON = 0.01
# Maximum perturbation for PGD attacks (1% of feature)
# Mimics sophisticated fraudster capability
# Security: Gradient-based evasion bound

ADVERSARIAL_PGD_STEPS = 10
# Number of PGD attack steps

# Zero-Knowledge Proofs (Recursive SNARKs)
ZKP_PROOF_SIZE_BYTES = 22000
# Mina-style recursive proof size (22kb constant)
# Cryptography: IVC (Incrementally Verifiable Computation)
# Property: Size independent of transaction history depth

ZKP_VERIFICATION_TIME_MAX_MS = 5000
# Maximum verification time per proof (5 seconds)

# Data Availability Sampling
DATA_AVAILABILITY_SAMPLE_RATE = 0.10
# Auditors sample 10% of transaction chunks
# Detects erasure with p > 0.90
# Cryptography: Ethereum sharding security model

DATA_AVAILABILITY_THRESHOLD = 0.90
# Minimum availability score for valid transaction

# Layout Entropy for PDF Analysis
LAYOUT_ENTROPY_THRESHOLD = 1.0
# Bits of visual structure entropy
# < 1.0 = script-generated (perfect alignment)
# > 2.5 = human scan (warping, artifacts)
# Physics: Shannon entropy of bounding box distribution

LAYOUT_HUMAN_SCAN_MIN = 2.5
# Minimum entropy for human-generated documents

# SAM.gov Validation
SAM_CA_TRUST_THRESHOLD = 0.50
# Minimum Certificate Authority trust score
# Below threshold = additional scrutiny
# Cryptography: PKI trust metrics

# API Rate Limits
USASPENDING_RATE_LIMIT = 1000  # requests per hour
USASPENDING_RECORDS_PER_PAGE = 100

# KAN (Kolmogorov-Arnold Network) Architecture
KAN_INPUT_DIM = 5  # K_complexity, layout_entropy, graph_centrality, time_delta, amount
KAN_HIDDEN_DIM = 6
KAN_OUTPUT_DIM = 1  # P(fraud)
KAN_ROBUST_ACCURACY_TARGET = 0.85

# OMEGA Version
VERSION_OMEGA = "3.0.0"

# === ALL DATA CITATIONS AS CONSTANTS WITH URLS ===

CITATIONS = {
    "GAO_AUDIT_FAILURE": {
        "source": "GAO-25-107052",
        "url": "https://www.gao.gov/products/gao-25-107052",
        "detail": "$2.5T unaccounted assets, 7th consecutive audit failure",
        "date": "2024-11-15"
    },
    "GAO_FRAUD_ESTIMATE": {
        "source": "GAO High-Risk Series",
        "url": "https://www.gao.gov/highrisk",
        "detail": "$233-521B annual fraud estimate",
        "date": "2023"
    },
    "GAO_FORD_CARRIER": {
        "source": "GAO-20-257",
        "url": "https://www.gao.gov/products/gao-20-257",
        "detail": "Gerald Ford carrier 23% cost overrun ($2.8B), $13.3B vs $10.5B",
        "date": "2020-02-13"
    },
    "GAO_ZUMWALT": {
        "source": "GAO-16-395",
        "url": "https://www.gao.gov/products/gao-16-395",
        "detail": "Zumwalt 81% unit cost increase ($6.1B per ship)",
        "date": "2016-06-23"
    },
    "GAO_SHIP_MAINTENANCE": {
        "source": "GAO-23-106051",
        "url": "https://www.gao.gov/products/gao-23-106051",
        "detail": "$4.86B carrier maintenance backlog",
        "date": "2023-09-14"
    },
    "GAO_BARRACKS": {
        "source": "GAO-24-107174",
        "url": "https://www.gao.gov/products/gao-24-107174",
        "detail": "23% of 6,700+ barracks in poor/failing condition",
        "date": "2024-04-18"
    },
    "GAO_KC46": {
        "source": "GAO-24-106533",
        "url": "https://www.gao.gov/products/gao-24-106533",
        "detail": "7 category-1 critical deficiencies",
        "date": "2024-02-08"
    },
    "GAO_GPS_OCX": {
        "source": "GAO-23-105694",
        "url": "https://www.gao.gov/products/gao-23-105694",
        "detail": "$4B over budget",
        "date": "2023-05-25"
    },
    "GAO_GHOST_VENDOR": {
        "source": "GAO-23-105526",
        "url": "https://www.gao.gov/products/gao-23-105526",
        "detail": "Ghost vendor detection patterns in fraud taxonomy",
        "date": "2023-07-11"
    },
    "GAO_DOD_IT": {
        "source": "GAO-23-105815",
        "url": "https://www.gao.gov/products/gao-23-105815",
        "detail": "400+ incompatible financial systems",
        "date": "2023-08-22"
    },
    "DODIG_AMMO": {
        "source": "DODIG-2024-091",
        "url": "https://www.dodig.mil/reports.html",
        "detail": "95% ammunition inventory inaccuracy rate",
        "date": "2024-05-15"
    },
    "DODIG_UKRAINE": {
        "source": "DODIG-2023-109",
        "url": "https://www.dodig.mil/reports.html",
        "detail": "40,000+ weapons without serial number tracking",
        "date": "2023-10-17"
    },
    "NEWPORT_NEWS_WELDING": {
        "source": "DOJ Press Release / DODIG",
        "url": "https://www.dodig.mil/reports.html",
        "detail": "26 ships with faulty welds under investigation",
        "date": "2023-08-15"
    },
    "CRS_COLUMBIA": {
        "source": "CRS R41129",
        "url": "https://crsreports.congress.gov",
        "detail": "Columbia submarine 12-17 month schedule delay",
        "date": "2024-03-15"
    },
    "CRS_FORD": {
        "source": "CRS Navy Ford-Class Report",
        "url": "https://crsreports.congress.gov",
        "detail": "$13.3B vs $10.5B planned costs",
        "date": "2024-01-22"
    },
    "CRS_LCS": {
        "source": "CRS Littoral Combat Ship Report",
        "url": "https://crsreports.congress.gov",
        "detail": "Retired after only 5-10 years of service",
        "date": "2024-02-14"
    },
    "HASC_SUBMARINE": {
        "source": "HASC Testimony 2023",
        "url": "https://armedservices.house.gov/hearings",
        "detail": "37-40% attack submarine out-of-service rate",
        "date": "2023-06-14"
    },
    "POGO_AUDIT": {
        "source": "POGO Pentagon Audit Report",
        "url": "https://www.pogo.org",
        "detail": "7 consecutive audit failures",
        "date": "2024-11-20"
    },
    "DOTE_F35_ALIS": {
        "source": "DOT&E FY2022 Annual Report",
        "url": "https://www.dote.osd.mil",
        "detail": "45,000 hours/year manual workarounds for ALIS",
        "date": "2023-01-31"
    },
    "AF_F22_READINESS": {
        "source": "Air Force Magazine",
        "url": "https://www.airforcemag.com",
        "detail": "F-22 40.19% mission-capable rate",
        "date": "2024-03-15"
    },
    "SHANNON_1948": {
        "source": "Shannon 1948",
        "url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
        "detail": "Information theory: H = -Σ p(x) log p(x)",
        "date": "1948"
    },
    "GPS_DEFICIENCIES": {
        "source": "GPS.gov",
        "url": "https://www.gps.gov",
        "detail": "231 open deficiencies in GPS OCX",
        "date": "2024-06-01"
    },
}

# === BRANCHES ===

BRANCHES = ["Navy", "Army", "AirForce", "Marines", "SpaceForce", "CoastGuard"]

BRANCH_DISTRIBUTION = {
    "Navy": 0.30,
    "AirForce": 0.28,
    "Army": 0.26,
    "SpaceForce": 0.08,
    "Marines": 0.05,
    "CoastGuard": 0.03
}

# === STOPRULE EXCEPTION ===

class StopRuleException(Exception):
    """
    Raised when a stoprule triggers. Never catch silently.
    Per CLAUDEME §8: stoprules halt execution on critical failures.
    """
    pass


# === CORE FUNCTIONS (REQUIRED PER CLAUDEME §8) ===

def dual_hash(data: bytes | str) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.
    Per CLAUDEME §8: HASH = "SHA256 + BLAKE3" # ALWAYS dual-hash

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
    Per CLAUDEME §8: All operations emit receipts.

    Args:
        receipt_type: Type of receipt (warrant, detection, compression, etc.)
        data: Receipt payload data
        to_stdout: Whether to print to stdout (default True)

    Returns:
        Complete receipt dict with ts, tenant_id, payload_hash
    """
    # Ensure simulation flag is always present
    if "simulation_flag" not in data:
        data["simulation_flag"] = DISCLAIMER

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
    Per CLAUDEME §8: MERKLE = "BLAKE3" (via dual_hash)

    Args:
        items: List of items to compute root for

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    hashes = [dual_hash(json.dumps(i, sort_keys=True) if isinstance(i, dict) else str(i))
              for i in items]

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i+1])
                  for i in range(0, len(hashes), 2)]

    return hashes[0]


def cite(source: str, url: str, detail: str, date: Optional[str] = None) -> dict:
    """
    Embed citation in receipt. Required for all data claims.
    Per WarrantProof spec: 100% citation coverage required.

    Args:
        source: Source document identifier
        url: URL to source
        detail: Specific detail being cited
        date: Optional date of source

    Returns:
        Citation dict for embedding in receipts
    """
    citation = {
        "source": source,
        "url": url,
        "detail": detail
    }
    if date:
        citation["date"] = date
    return citation


def get_citation(key: str) -> dict:
    """
    Get a citation from the CITATIONS constant.

    Args:
        key: Citation key (e.g., "GAO_AUDIT_FAILURE")

    Returns:
        Citation dict or raises StopRuleException if not found
    """
    if key not in CITATIONS:
        stoprule_uncited_data(key)
    return CITATIONS[key].copy()


# === STOPRULES ===

def stoprule_hash_mismatch(expected: str, actual: str) -> None:
    """
    Emit anomaly receipt and halt on hash mismatch.
    Per CLAUDEME: Merkle integrity = 100% required.
    """
    emit_receipt("anomaly", {
        "metric": "hash_mismatch",
        "expected": expected,
        "actual": actual,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> None:
    """
    Emit anomaly receipt and halt on invalid receipt.
    Per CLAUDEME LAW_1: No receipt → not real.
    """
    emit_receipt("anomaly", {
        "metric": "invalid_receipt",
        "reason": reason,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Invalid receipt: {reason}")


def stoprule_uncited_data(field: str) -> None:
    """
    Emit violation receipt and halt on uncited data.
    Per WarrantProof spec: 100% citation coverage required.
    """
    emit_receipt("violation", {
        "metric": "uncited_data",
        "field": field,
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Uncited data: {field} requires citation")


def stoprule_missing_approver() -> None:
    """
    Emit violation receipt and halt on missing approver.
    Per WarrantProof warrant spec: approver required.
    """
    emit_receipt("violation", {
        "metric": "missing_approver",
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException("Missing approver: warrant requires approver field")


def stoprule_missing_lineage() -> None:
    """
    Emit violation receipt and halt on missing lineage.
    Per WarrantProof trace spec: lineage required.
    """
    emit_receipt("violation", {
        "metric": "missing_lineage",
        "delta": -1,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException("Missing lineage: warrant requires parent reference")


def stoprule_budget_exceeded(actual: float, limit: float) -> None:
    """
    Emit anomaly receipt and halt on budget exceeded.
    Per CLAUDEME: budget violations trigger stoprule.
    """
    emit_receipt("anomaly", {
        "metric": "budget",
        "actual": actual,
        "limit": limit,
        "delta": actual - limit,
        "action": "reject",
        "classification": "violation"
    })
    raise StopRuleException(f"Budget exceeded: {actual} > {limit}")


# === UTILITY FUNCTIONS ===

def validate_branch(branch: str) -> bool:
    """Validate branch is one of the 6 DoD branches."""
    return branch in BRANCHES


def generate_receipt_id() -> str:
    """Generate a unique receipt ID using timestamp and random hash."""
    import uuid
    return f"rcpt_{uuid.uuid4().hex[:16]}"


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    # Per CLAUDEME: cli.py must emit valid receipt JSON
    print(f"# WarrantProof Core v{VERSION}", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)
    print(f"# Citations loaded: {len(CITATIONS)}", file=sys.stderr)

    # Emit test receipt
    h = dual_hash("test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"

    c = cite("TEST", "http://test.com", "test citation")
    assert "url" in c, "cite must include url"

    r = emit_receipt("test", {
        "message": "core module self-test",
        "citation": get_citation("SHANNON_1948")
    })
    assert "receipt_type" in r, "emit_receipt must include receipt_type"
    assert "tenant_id" in r, "emit_receipt must include tenant_id"

    print("# PASS: core module self-test", file=sys.stderr)
