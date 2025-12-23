"""
WarrantProof Military Accountability Simulation & Analysis Suite

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This package provides simulation and research tools for modeling
receipts-native accountability infrastructure in defense procurement.

All data is synthetic or derived from publicly available sources.
See CITATIONS.md for complete source list.

v2.0 PARADIGM: Thermodynamic autocatalytic fraud detection with
quantum-inspired threshold collapse and holographic ledger compression.
"""

__version__ = "2.0.0"
__disclaimer__ = "THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."

from .core import (
    TENANT_ID,
    DISCLAIMER,
    CITATIONS,
    VERSION,
    # v2 constants
    N_CRITICAL_FORMULA,
    THOMPSON_PRIOR_VARIANCE,
    ENTROPY_TREE_MAX_DEPTH,
    CASCADE_DERIVATIVE_THRESHOLD,
    EPIDEMIC_R0_THRESHOLD,
    HOLOGRAPHIC_BITS_PER_RECEIPT,
    PATTERN_COHERENCE_MIN,
    ENTROPY_GAP_MIN,
    MUTUAL_INFO_TRANSFER_THRESHOLD,
    CROSS_BRANCH_ACCURACY_TARGET,
    # Functions
    dual_hash,
    emit_receipt,
    merkle,
    cite,
    StopRuleException,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    stoprule_uncited_data,
)

# v2 modules
from .thompson import (
    ThresholdDistribution,
    sample_threshold,
    update_posterior,
    contextual_collapse,
    calibrate_prior,
)

from .autocatalytic import (
    compute_entropy_gap,
    calculate_N_critical,
    crystallize_pattern,
    pattern_coherence_score,
    autocatalytic_detect,
    EmergentPattern,
)

from .entropy_tree import (
    EntropyNode,
    build_entropy_tree,
    search_tree,
    entropy_bisect,
)

from .cascade import (
    calculate_compression_derivative,
    detect_cascade_onset,
    estimate_cascade_time,
    alert_early_warning,
)

from .epidemic import (
    calculate_R0,
    SIR_model_step,
    predict_spread,
    recommend_quarantine,
    VendorNetwork,
)

from .holographic import (
    compute_merkle_syndrome,
    detect_from_boundary,
    holographic_encode,
    MerkleRootHistory,
)

from .meta_receipt import (
    emit_meta_receipt,
    validate_self_reference,
    cluster_receipts_by_causality,
    test_autocatalytic_closure,
)
