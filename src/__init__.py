"""
WarrantProof Military Accountability Simulation & Analysis Suite

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This package provides simulation and research tools for modeling
receipts-native accountability infrastructure in defense procurement.

All data is synthetic or derived from publicly available sources.
See CITATIONS.md for complete source list.

v3.0 OMEGA PARADIGM: Deterministic zero-knowledge proof architecture with
Kolmogorov complexity, RAF network detection, data availability sampling,
and adversarial robustness. Transforms probabilistic detection into
cryptographic certainty.

Key OMEGA Features:
- Kolmogorov complexity replaces Shannon entropy
- Recursive ZK-SNARKs (Mina-style IVC)
- RAF autocatalytic network cycle detection
- Data Availability Sampling via erasure coding
- Adversarial robustness via PGD attacks
- USASpending.gov ETL integration
- PDF layout entropy analysis
- SAM.gov Certificate Authority validation
"""

__version__ = "3.0.0"
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
    # OMEGA v3 constants
    KOLMOGOROV_THRESHOLD,
    KOLMOGOROV_LEGITIMATE_MIN,
    BEKENSTEIN_BITS_PER_DOLLAR,
    RAF_CYCLE_MIN_LENGTH,
    RAF_CYCLE_MAX_LENGTH,
    THOMPSON_AUDIT_BUDGET,
    ADVERSARIAL_EPSILON,
    ZKP_PROOF_SIZE_BYTES,
    DATA_AVAILABILITY_SAMPLE_RATE,
    LAYOUT_ENTROPY_THRESHOLD,
    SAM_CA_TRUST_THRESHOLD,
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
    # OMEGA v3: Multi-armed bandit audit selection
    ContractorArm,
    thompson_audit_selection,
    update_audit_arms,
    emit_audit_selection_receipt,
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
    # OMEGA v3: Data Availability
    HolographicState,
    create_holographic_state,
    verify_data_availability,
    holographic_detect_with_da,
    holographic_audit,
    emit_holographic_da_receipt,
)

from .meta_receipt import (
    emit_meta_receipt,
    validate_self_reference,
    cluster_receipts_by_causality,
    test_autocatalytic_closure,
)

# OMEGA v3: New modules
from .kolmogorov import (
    calculate_kolmogorov,
    is_kolmogorov_fraud,
    compress_transaction_history,
    detect_generator_pattern,
    emit_kolmogorov_receipt,
)

from .zkp import (
    ZKProof,
    CircuitConstraints,
    generate_proof,
    verify_proof,
    recursive_compose,
    verify_recursive_chain,
    emit_zkp_receipt,
)

from .raf import (
    build_transaction_graph,
    add_catalytic_links,
    detect_cycles,
    identify_keystone_species,
    simulate_disruption,
    emit_raf_receipt,
)

from .das import (
    ErasureEncodedData,
    encode_with_erasure,
    sample_chunks,
    verify_availability,
    detect_erasure,
    light_client_audit,
    emit_das_receipt,
)

from .adversarial import (
    pgd_attack,
    fgsm_attack,
    generate_adversarial_dataset,
    evaluate_robustness,
    emit_adversarial_receipt,
)

from .usaspending_etl import (
    fetch_awards,
    fetch_transactions,
    fetch_federal_accounts,
    handle_pagination,
    detect_missing_fields,
    emit_usaspending_receipt,
)

from .layout_entropy import (
    extract_layout_features,
    calculate_layout_entropy,
    detect_scan_artifacts,
    detect_perfect_alignment,
    analyze_document,
    emit_layout_entropy_receipt,
)

from .sam_validator import (
    EntityData,
    ValidationResult,
    fetch_entity,
    validate_signature,
    reject_na_fields,
    calculate_ca_trust_score,
    validate_entity,
    emit_sam_validation_receipt,
)

from .bridge import (
    translate_receipt,
    verify_translation,
    create_bridge_proof,
    cross_branch_chain,
    mutual_information,
    cross_branch_learning,
    # OMEGA v3: Catalytic detection
    CatalyticLink,
    detect_shared_addresses,
    detect_board_connections,
    detect_ip_proximity,
    detect_temporal_patterns,
    detect_all_catalytic_links,
    integrate_catalytic_with_raf,
    cross_branch_learning_with_catalysis,
    emit_catalytic_receipt,
)

from .compress import (
    compress_receipt_with_entropy,
    calculate_compression_ratio,
    classify_by_compression,
    # OMEGA v3: Kolmogorov integration
    compress_receipt_kolmogorov,
)

from .detect import (
    detect_anomaly,
    scan_for_patterns,
    validate_cross_references,
    # OMEGA v3: ZKP gate
    zkp_verification_gate,
    detect_with_zkp_gate,
)

from .ledger import (
    Ledger,
    append_receipt,
    compute_merkle_root,
    verify_chain,
    # OMEGA v3: Bekenstein bound
    validate_bekenstein_bound,
    emit_bekenstein_receipt,
)

# Export all
__all__ = [
    # Version
    "__version__",
    "__disclaimer__",
    # Core
    "TENANT_ID",
    "DISCLAIMER",
    "VERSION",
    "dual_hash",
    "emit_receipt",
    "merkle",
    "cite",
    "StopRuleException",
    # OMEGA constants
    "KOLMOGOROV_THRESHOLD",
    "BEKENSTEIN_BITS_PER_DOLLAR",
    "RAF_CYCLE_MIN_LENGTH",
    "RAF_CYCLE_MAX_LENGTH",
    "ZKP_PROOF_SIZE_BYTES",
    "DATA_AVAILABILITY_SAMPLE_RATE",
    # New OMEGA modules
    "calculate_kolmogorov",
    "ZKProof",
    "generate_proof",
    "verify_proof",
    "build_transaction_graph",
    "detect_cycles",
    "encode_with_erasure",
    "verify_availability",
    "pgd_attack",
    "fetch_awards",
    "calculate_layout_entropy",
    "validate_entity",
    "detect_all_catalytic_links",
    "holographic_audit",
]
