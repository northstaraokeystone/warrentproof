"""
Gov-OS Core Constants - CLAUDEME v3.1 Compliant

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

v5.1 Temporal Physics Constants:
- LAMBDA_NATURAL: Exponential decay rate from relationship attrition studies
- RESISTANCE_THRESHOLD: Anomaly detection threshold for decay resistance
- ZOMBIE_DAYS: Dormancy threshold for zombie entity detection
- CONTAGION_OVERLAP_MIN: Minimum shared entity ratio for cross-domain scan
- SHELL_PATTERN_THRESHOLD: Domains linked to classify entity as shell
- ENTROPY_RESISTANCE_MULTIPLIER: Sensitivity boost from resistance metric
"""

import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# === CORE CONSTANTS ===

TENANT_ID = "gov-os"
DISCLAIMER = "THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY. NOT REAL DATA."
VERSION = "5.1.0"

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

# === WARRANTPROOF v2 CONSTANTS ===

ENTROPY_GAP_MIN = 0.15
THOMPSON_PRIOR_VARIANCE = 0.1
THOMPSON_FP_TARGET = 0.02
ENTROPY_TREE_REBALANCE_THRESHOLD = 0.10
ENTROPY_TREE_STORAGE_OVERHEAD_MAX = 0.05
MUTUAL_INFO_TRANSFER_THRESHOLD = 0.30
CROSS_BRANCH_ACCURACY_TARGET = 0.85
CASCADE_DERIVATIVE_THRESHOLD = 0.05
CASCADE_WINDOW_SIZE = 50
CASCADE_FALSE_ALERT_MAX = 0.10
EPIDEMIC_R0_THRESHOLD = 1.0
EPIDEMIC_DETECTION_LATENCY_TARGET = 7.0
EPIDEMIC_RECOVERY_RATE = 0.10
HOLOGRAPHIC_BITS_PER_RECEIPT = 2
HOLOGRAPHIC_DETECTION_PROBABILITY_MIN = 0.9999
PATTERN_COHERENCE_MIN = 0.80
RAF_CLOSURE_ACCURACY_MIN = 0.80
META_RECEIPT_PREDICTION_WINDOW = 100
COMPRESSION_RATIO_LEGIT_V1 = 0.80
COMPRESSION_RATIO_FRAUD_V1 = 0.50

# Additional compression constants for test compatibility
COMPRESSION_LEGITIMATE_FLOOR = 0.85
COMPRESSION_FRAUD_CEILING = 0.60
VOLATILITY_ALPHA = 0.1
COMPLETENESS_THRESHOLD = 0.999
RAF_MIN_CYCLE_LENGTH = 3
HOLOGRAPHIC_DETECTION_PROB = 0.9999

# === OMEGA v3 CONSTANTS ===

KOLMOGOROV_THRESHOLD = 0.65
KOLMOGOROV_LEGITIMATE_MIN = 0.75
BEKENSTEIN_BITS_PER_DOLLAR = 1e-6
RAF_CYCLE_MIN_LENGTH = 3
RAF_CYCLE_MAX_LENGTH = 5
THOMPSON_AUDIT_BUDGET = 0.05
ADVERSARIAL_EPSILON = 0.01
ADVERSARIAL_PGD_STEPS = 10
ZKP_PROOF_SIZE_BYTES = 22000
ZKP_VERIFICATION_TIME_MAX_MS = 5000
DATA_AVAILABILITY_SAMPLE_RATE = 0.10
DATA_AVAILABILITY_THRESHOLD = 0.90
LAYOUT_ENTROPY_THRESHOLD = 1.0
LAYOUT_HUMAN_SCAN_MIN = 2.5
SAM_CA_TRUST_THRESHOLD = 0.50
USASPENDING_RATE_LIMIT = 1000
USASPENDING_RECORDS_PER_PAGE = 100
KAN_INPUT_DIM = 5
KAN_HIDDEN_DIM = 6
KAN_OUTPUT_DIM = 1
KAN_ROBUST_ACCURACY_TARGET = 0.85

# ============================================================================
# v5.1 TEMPORAL DECAY PHYSICS CONSTANTS
# ============================================================================

# Natural monthly decay rate
# Physics: Wt = W0 * e^(-lambda * t)
# Half-life = ln(2) / lambda ≈ 138 months (matches relationship attrition data)
LAMBDA_NATURAL = 0.005

# Threshold above which resistance flags anomaly
# Resistance = (observed_weight / expected_weight) - 1.0
# Resistance > 0.1 means observed weight is 10% higher than natural decay predicts
RESISTANCE_THRESHOLD = 0.1

# Days without transaction before entity classified as zombie if weight preserved
# Zombie = entity with zero activity but full edge weight = physically impossible
ZOMBIE_DAYS = 365

# Minimum shared entity overlap (5%) to trigger contagion scan
# Cross-domain contagion only meaningful when domains share significant entities
CONTAGION_OVERLAP_MIN = 0.05

# Entity linked to >= this many domains with resistance > threshold = shell candidate
SHELL_PATTERN_THRESHOLD = 2

# 22% sensitivity boost when resistance metric applied
# Entropy_final = Entropy_base * (1 + resistance * ENTROPY_RESISTANCE_MULTIPLIER)
ENTROPY_RESISTANCE_MULTIPLIER = 1.22

# ============================================================================
# v2 PHYSICS FORMULAS
# ============================================================================

def N_CRITICAL_FORMULA(H_legit: float, H_fraud: float) -> float:
    """Calculate critical N for autocatalytic phase transition."""
    delta_H = H_fraud - H_legit
    if delta_H <= 0:
        return float('inf')
    log2 = np.log2 if HAS_NUMPY else math.log2
    return log2(1.0 / delta_H) * (H_legit / delta_H)


def ENTROPY_TREE_MAX_DEPTH(N: int) -> int:
    """Calculate maximum tree depth for N receipts."""
    if N <= 1:
        return 1
    log2 = np.log2 if HAS_NUMPY else math.log2
    return int(log2(N)) + 1


def HOLOGRAPHIC_LOCALIZATION_COMPLEXITY(N: int) -> int:
    """O(log N) to identify branch containing fraud."""
    if N <= 1:
        return 1
    log2 = np.log2 if HAS_NUMPY else math.log2
    return int(log2(N)) + 1


def validate_branch(branch: str) -> bool:
    """Validate branch is one of the 6 DoD branches."""
    return branch in BRANCHES


# === CITATIONS (REQUIRED) ===

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
    "GAO_GHOST_VENDOR": {
        "source": "GAO-23-105526",
        "url": "https://www.gao.gov/products/gao-23-105526",
        "detail": "Ghost vendor detection patterns in fraud taxonomy",
        "date": "2023-07-11"
    },
    "SHANNON_1948": {
        "source": "Shannon 1948",
        "url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
        "detail": "Information theory: H = -Σ p(x) log p(x)",
        "date": "1948"
    },
    "GAO_ZUMWALT": {
        "source": "GAO-16-395",
        "url": "https://www.gao.gov/products/gao-16-395",
        "detail": "Zumwalt 81% unit cost increase ($6.1B per ship)",
        "date": "2016-06-23"
    },
    "NEWPORT_NEWS_WELDING": {
        "source": "DOJ Press Release / DODIG",
        "url": "https://www.dodig.mil/reports.html",
        "detail": "26 ships with faulty welds under investigation",
        "date": "2023-08-15"
    },
    "DODIG_AMMO": {
        "source": "DODIG-2024-091",
        "url": "https://www.dodig.mil/reports.html",
        "detail": "95% ammunition inventory inaccuracy rate",
        "date": "2024-05-15"
    },
}


def get_citation(key: str) -> dict:
    """
    Get a citation from the CITATIONS constant.

    Args:
        key: Citation key (e.g., "GAO_AUDIT_FAILURE")

    Returns:
        Citation dict or raises exception if not found
    """
    if key not in CITATIONS:
        raise KeyError(f"Citation not found: {key}")
    return CITATIONS[key].copy()
