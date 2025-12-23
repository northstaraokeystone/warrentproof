"""
Gov-OS Core Module - CLAUDEME v3.1 Compliant

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

v5.1 Temporal Physics:
- Exponential decay: Wt = W0 * e^(-λt)
- Resistance detection: static edges in dynamic environments
- Zombie entities: preserved weight with zero activity
- Cross-domain contagion: linked entropy pools
"""

from .constants import (
    TENANT_ID,
    DISCLAIMER,
    VERSION,
    BRANCHES,
    BRANCH_DISTRIBUTION,
    # v2 constants
    ENTROPY_GAP_MIN,
    THOMPSON_PRIOR_VARIANCE,
    THOMPSON_FP_TARGET,
    CASCADE_DERIVATIVE_THRESHOLD,
    CASCADE_WINDOW_SIZE,
    CASCADE_FALSE_ALERT_MAX,
    EPIDEMIC_R0_THRESHOLD,
    EPIDEMIC_DETECTION_LATENCY_TARGET,
    EPIDEMIC_RECOVERY_RATE,
    HOLOGRAPHIC_BITS_PER_RECEIPT,
    HOLOGRAPHIC_DETECTION_PROBABILITY_MIN,
    PATTERN_COHERENCE_MIN,
    RAF_CLOSURE_ACCURACY_MIN,
    META_RECEIPT_PREDICTION_WINDOW,
    MUTUAL_INFO_TRANSFER_THRESHOLD,
    CROSS_BRANCH_ACCURACY_TARGET,
    ENTROPY_TREE_REBALANCE_THRESHOLD,
    ENTROPY_TREE_STORAGE_OVERHEAD_MAX,
    COMPRESSION_RATIO_LEGIT_V1,
    COMPRESSION_RATIO_FRAUD_V1,
    # Test compatibility constants
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    VOLATILITY_ALPHA,
    COMPLETENESS_THRESHOLD,
    RAF_MIN_CYCLE_LENGTH,
    HOLOGRAPHIC_DETECTION_PROB,
    # OMEGA v3 constants
    KOLMOGOROV_THRESHOLD,
    KOLMOGOROV_LEGITIMATE_MIN,
    BEKENSTEIN_BITS_PER_DOLLAR,
    RAF_CYCLE_MIN_LENGTH,
    RAF_CYCLE_MAX_LENGTH,
    THOMPSON_AUDIT_BUDGET,
    ADVERSARIAL_EPSILON,
    ADVERSARIAL_PGD_STEPS,
    ZKP_PROOF_SIZE_BYTES,
    ZKP_VERIFICATION_TIME_MAX_MS,
    DATA_AVAILABILITY_SAMPLE_RATE,
    DATA_AVAILABILITY_THRESHOLD,
    LAYOUT_ENTROPY_THRESHOLD,
    LAYOUT_HUMAN_SCAN_MIN,
    SAM_CA_TRUST_THRESHOLD,
    USASPENDING_RATE_LIMIT,
    USASPENDING_RECORDS_PER_PAGE,
    KAN_INPUT_DIM,
    KAN_HIDDEN_DIM,
    KAN_OUTPUT_DIM,
    KAN_ROBUST_ACCURACY_TARGET,
    # v5.1 temporal constants
    LAMBDA_NATURAL,
    RESISTANCE_THRESHOLD,
    ZOMBIE_DAYS,
    CONTAGION_OVERLAP_MIN,
    SHELL_PATTERN_THRESHOLD,
    ENTROPY_RESISTANCE_MULTIPLIER,
    CITATIONS,
    # v2 formulas
    N_CRITICAL_FORMULA,
    ENTROPY_TREE_MAX_DEPTH,
    HOLOGRAPHIC_LOCALIZATION_COMPLEXITY,
    validate_branch,
    get_citation,
)

from .utils import (
    dual_hash,
    merkle,
    cite,
    generate_receipt_id,
)

from .receipt import (
    StopRuleException,
    StopRule,
    emit_receipt,
    emit_L0,
    emit_L1,
    emit_L2,
    emit_L3,
    emit_L4,
    completeness_check,
    emit_temporal_anomaly_receipt,
    emit_zombie_receipt,
    emit_contagion_receipt,
    emit_super_graph_receipt,
    emit_insight_receipt,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    stoprule_uncited_data,
    stoprule_missing_approver,
    stoprule_missing_lineage,
    stoprule_budget_exceeded,
)

from .volatility import (
    VolatilityIndex,
    MockVolatilityIndex,
)

from .temporal import (
    edge_weight_decay,
    resistance_to_decay,
    update_edge_with_decay,
    detect_zombies,
    identify_shell_entities,
    propagate_contagion,
    calculate_shared_entity_ratio,
)

from .insight import (
    explain_temporal_anomaly,
    explain_contagion,
    explain_zombie,
    format_insight,
    generate_executive_summary,
)

from .harness import (
    SimState,
    ScenarioResult,
    run_simulation,
)

from .raf import (
    build_transaction_graph,
    detect_cycles,
    emit_raf_receipt,
    detect_without_hardcode,
)

__all__ = [
    # Constants
    "TENANT_ID",
    "DISCLAIMER",
    "VERSION",
    "BRANCHES",
    "BRANCH_DISTRIBUTION",
    "ENTROPY_GAP_MIN",
    "PATTERN_COHERENCE_MIN",
    "KOLMOGOROV_THRESHOLD",
    "KOLMOGOROV_LEGITIMATE_MIN",
    "BEKENSTEIN_BITS_PER_DOLLAR",
    "RAF_CYCLE_MIN_LENGTH",
    "RAF_CYCLE_MAX_LENGTH",
    "THOMPSON_PRIOR_VARIANCE",
    "THOMPSON_AUDIT_BUDGET",
    "ADVERSARIAL_EPSILON",
    "ZKP_PROOF_SIZE_BYTES",
    "DATA_AVAILABILITY_SAMPLE_RATE",
    "LAYOUT_ENTROPY_THRESHOLD",
    "SAM_CA_TRUST_THRESHOLD",
    "CASCADE_DERIVATIVE_THRESHOLD",
    "EPIDEMIC_R0_THRESHOLD",
    "HOLOGRAPHIC_BITS_PER_RECEIPT",
    "MUTUAL_INFO_TRANSFER_THRESHOLD",
    "CROSS_BRANCH_ACCURACY_TARGET",
    # Test compatibility
    "COMPRESSION_LEGITIMATE_FLOOR",
    "COMPRESSION_FRAUD_CEILING",
    "VOLATILITY_ALPHA",
    "COMPLETENESS_THRESHOLD",
    "RAF_MIN_CYCLE_LENGTH",
    "HOLOGRAPHIC_DETECTION_PROB",
    # v2 formulas
    "N_CRITICAL_FORMULA",
    "ENTROPY_TREE_MAX_DEPTH",
    "HOLOGRAPHIC_LOCALIZATION_COMPLEXITY",
    "validate_branch",
    "get_citation",
    # v5.1 temporal
    "LAMBDA_NATURAL",
    "RESISTANCE_THRESHOLD",
    "ZOMBIE_DAYS",
    "CONTAGION_OVERLAP_MIN",
    "SHELL_PATTERN_THRESHOLD",
    "ENTROPY_RESISTANCE_MULTIPLIER",
    "CITATIONS",
    # Utilities
    "dual_hash",
    "merkle",
    "cite",
    "generate_receipt_id",
    # Receipts
    "StopRuleException",
    "StopRule",
    "emit_receipt",
    "emit_L0",
    "emit_L1",
    "emit_L2",
    "emit_L3",
    "emit_L4",
    "completeness_check",
    "emit_temporal_anomaly_receipt",
    "emit_zombie_receipt",
    "emit_contagion_receipt",
    "emit_super_graph_receipt",
    "emit_insight_receipt",
    "stoprule_hash_mismatch",
    "stoprule_invalid_receipt",
    "stoprule_uncited_data",
    "stoprule_missing_approver",
    "stoprule_missing_lineage",
    "stoprule_budget_exceeded",
    # Volatility
    "VolatilityIndex",
    "MockVolatilityIndex",
    # Temporal (v5.1)
    "edge_weight_decay",
    "resistance_to_decay",
    "update_edge_with_decay",
    "detect_zombies",
    "identify_shell_entities",
    "propagate_contagion",
    "calculate_shared_entity_ratio",
    # Insight (v5.1)
    "explain_temporal_anomaly",
    "explain_contagion",
    "explain_zombie",
    "format_insight",
    "generate_executive_summary",
    # Harness
    "SimState",
    "ScenarioResult",
    "run_simulation",
    # RAF
    "build_transaction_graph",
    "detect_cycles",
    "emit_raf_receipt",
    "detect_without_hardcode",
]
