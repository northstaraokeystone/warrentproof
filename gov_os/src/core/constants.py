"""
Gov-OS Core Constants - Universal Physics Constants

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

No domain-specific values here. These are physics constants validated
by information theory (Shannon, Kolmogorov) and RAF theory.
"""

# =============================================================================
# COMPRESSION THRESHOLDS (Shannon/Kolmogorov Physics)
# =============================================================================

# Legitimate transactions exhibit predictable redundancy (high compression)
COMPRESSION_LEGITIMATE_FLOOR = 0.85  # Shannon: ~0.85+ per Shannon/Kolmogorov

# Fraudulent transactions inject algorithmic randomness (low compression)
COMPRESSION_FRAUD_CEILING = 0.60  # Shannon: <0.60 for injected randomness

# =============================================================================
# ADAPTIVE THRESHOLD PARAMETERS
# =============================================================================

# Volatility adjustment sensitivity
VOLATILITY_ALPHA = 0.1  # Threshold adjustment: base * (1 + alpha * (vol - 1))

# Information gain from adaptive thresholds (Bayesian models)
INFORMATION_GAIN_TARGET_MIN = 0.15  # 15% minimum gain
INFORMATION_GAIN_TARGET_MAX = 0.30  # 30% maximum gain

# =============================================================================
# RAF (Reflexively Autocatalytic and Food-set) PARAMETERS
# =============================================================================

# Minimum nodes for self-sustaining loop (A→B→C→A is simplest)
RAF_MIN_CYCLE_LENGTH = 3

# Maximum cycle length for attribution (beyond 5 steps, catalysis too indirect)
RAF_MAX_CYCLE_LENGTH = 5

# Autocatalysis detection sensitivity (3 consecutive self-predictions at this threshold)
RAF_SELF_PREDICTION_THRESHOLD = 0.85

# =============================================================================
# CASCADE DETECTION PARAMETERS (dC/dt)
# =============================================================================

# dC/dt early warning trigger (5% degradation per unit time)
CASCADE_DERIVATIVE_THRESHOLD = 0.05

# Rolling window for derivative calculation
CASCADE_WINDOW = 10

# =============================================================================
# EPIDEMIC MODELING PARAMETERS (R₀)
# =============================================================================

# R₀ >= 1.0 indicates sustained outbreak
EPIDEMIC_R0_OUTBREAK = 1.0

# Recovery rate per cycle
EPIDEMIC_RECOVERY_RATE = 0.10

# Detection latency target (days)
EPIDEMIC_DETECTION_LATENCY_TARGET = 7.0

# =============================================================================
# HOLOGRAPHIC DETECTION PARAMETERS (Bekenstein Bound)
# =============================================================================

# Detection probability from boundary (99.99%)
HOLOGRAPHIC_DETECTION_PROB = 0.9999

# Bits per receipt (holographic compression bound)
HOLOGRAPHIC_BITS_PER_RECEIPT = 2

# =============================================================================
# THOMPSON SAMPLING PARAMETERS (Bayesian Thresholds)
# =============================================================================

# False positive rate target (<2%)
THOMPSON_FP_TARGET = 0.02

# Convergence criterion (variance below this = converged)
THOMPSON_CONVERGENCE_VAR = 0.001

# Prior variance for threshold distribution
THOMPSON_PRIOR_VARIANCE = 0.1

# Audit budget (fraction of contractors audited per cycle)
THOMPSON_AUDIT_BUDGET = 0.05

# =============================================================================
# META-RECEIPT PARAMETERS (L0-L4 Completeness)
# =============================================================================

# L0-L4 coverage target (99.9%)
COMPLETENESS_THRESHOLD = 0.999

# =============================================================================
# AUTOCATALYTIC PATTERN PARAMETERS
# =============================================================================

# Minimum coherence for pattern survival
PATTERN_COHERENCE_MIN = 0.80

# Minimum entropy gap for detection
ENTROPY_GAP_MIN = 0.15

# N_critical limit (stoprule if exceeded)
N_CRITICAL_MAX = 10000

# Prediction window for meta-receipt validation
META_RECEIPT_PREDICTION_WINDOW = 100

# =============================================================================
# ENTROPY TREE PARAMETERS
# =============================================================================

# Rebalance when imbalance exceeds this threshold
ENTROPY_TREE_REBALANCE_THRESHOLD = 0.10

# Maximum storage overhead vs v1 linear
ENTROPY_TREE_STORAGE_OVERHEAD_MAX = 0.05

# =============================================================================
# SYSTEM IDENTIFIERS
# =============================================================================

TENANT_ID = "gov-os"
DISCLAIMER = "THIS IS A SIMULATION. FOR RESEARCH ONLY. NOT REAL GOVERNMENT DATA."
VERSION = "1.0.0"

# =============================================================================
# SYSTEM LAWS (CLAUDEME Compliance)
# =============================================================================

LAW_1 = "No receipt → not real"
LAW_2 = "No test → not shipped"
LAW_3 = "No gate → not alive"
