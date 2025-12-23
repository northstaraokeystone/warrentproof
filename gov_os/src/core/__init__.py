"""
Gov-OS Core Module - Universal Physics Engine

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Core handles heavy physics (entropy, graphs, adaptive); modules only domain data.
"""

from .constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    VOLATILITY_ALPHA,
    RAF_MIN_CYCLE_LENGTH,
    RAF_SELF_PREDICTION_THRESHOLD,
    CASCADE_DERIVATIVE_THRESHOLD,
    CASCADE_WINDOW,
    EPIDEMIC_R0_OUTBREAK,
    HOLOGRAPHIC_DETECTION_PROB,
    THOMPSON_FP_TARGET,
    THOMPSON_CONVERGENCE_VAR,
    COMPLETENESS_THRESHOLD,
    TENANT_ID,
    DISCLAIMER,
    VERSION,
    ENTROPY_GAP_MIN,
    PATTERN_COHERENCE_MIN,
    INFORMATION_GAIN_TARGET_MIN,
    INFORMATION_GAIN_TARGET_MAX,
)

from .receipt import (
    emit_receipt,
    emit_L0,
    emit_L1,
    emit_L2,
    emit_L3,
    emit_L4,
    completeness_check,
    self_reference_achieved,
    godel_layer,
    StopRule,
    get_ledger_path,
    reset_ledger,
)

from .utils import dual_hash, merkle

from .ledger import (
    ingest,
    anchor,
    holographic_detect,
    boundary_encode,
    boundary_perturbation,
    verify_chain,
    compact,
)

from .compress import (
    compute_entropy_ratio,
    entropy_score,
    field_wise_compression,
    pattern_coherence,
    compression_derivative,
    build_entropy_tree,
    entropy_bisection,
    detect_hierarchical,
    speedup_factor,
    EntropyNode,
)

from .detect import (
    adaptive_threshold,
    thompson_sample,
    detect_anomaly,
    update_distribution,
    thompson_detect,
    false_positive_rate,
    convergence_check,
    ThresholdDistribution,
)

from .raf import (
    find_raf_cycles,
    build_graph,
    is_autocatalytic,
    coherence_score,
    pattern_birth,
    pattern_death,
    crystallize,
    detect_without_hardcode,
)

from .cascade import (
    compression_derivative as cascade_derivative,
    early_warning,
    cascade_prediction,
    trigger_alert,
    cascade_trace,
)

from .epidemic import (
    calculate_R0,
    infection_graph,
    patient_zero,
    outbreak_probability,
    quarantine_candidates,
    herd_immunity_threshold,
    spread_simulation,
)

from .volatility import VolatilityIndex

from .domain import (
    load_domain,
    register_domain,
    list_domains,
    get_volatility,
    get_schema,
    get_receipts,
    validate_receipt,
    DomainConfig,
)

from .harness import (
    monte_carlo_sim,
    run_simulation,
    simulate_cycle,
    inject_fraud,
    validate_constraints,
    entropy_conservation,
    run_scenario,
    run_all_scenarios,
    SimConfig,
    SimState,
    ScenarioResult,
)

__all__ = [
    # Constants
    "COMPRESSION_LEGITIMATE_FLOOR",
    "COMPRESSION_FRAUD_CEILING",
    "VOLATILITY_ALPHA",
    "RAF_MIN_CYCLE_LENGTH",
    "RAF_SELF_PREDICTION_THRESHOLD",
    "CASCADE_DERIVATIVE_THRESHOLD",
    "CASCADE_WINDOW",
    "EPIDEMIC_R0_OUTBREAK",
    "HOLOGRAPHIC_DETECTION_PROB",
    "THOMPSON_FP_TARGET",
    "THOMPSON_CONVERGENCE_VAR",
    "COMPLETENESS_THRESHOLD",
    "TENANT_ID",
    "DISCLAIMER",
    "VERSION",
    "ENTROPY_GAP_MIN",
    "PATTERN_COHERENCE_MIN",
    "INFORMATION_GAIN_TARGET_MIN",
    "INFORMATION_GAIN_TARGET_MAX",
    # Receipt
    "emit_receipt",
    "emit_L0",
    "emit_L1",
    "emit_L2",
    "emit_L3",
    "emit_L4",
    "completeness_check",
    "self_reference_achieved",
    "godel_layer",
    "StopRule",
    "get_ledger_path",
    "reset_ledger",
    # Utils
    "dual_hash",
    "merkle",
    # Ledger
    "ingest",
    "anchor",
    "holographic_detect",
    "boundary_encode",
    "boundary_perturbation",
    "verify_chain",
    "compact",
    # Compress
    "compute_entropy_ratio",
    "entropy_score",
    "field_wise_compression",
    "pattern_coherence",
    "compression_derivative",
    "build_entropy_tree",
    "entropy_bisection",
    "detect_hierarchical",
    "speedup_factor",
    "EntropyNode",
    # Detect
    "adaptive_threshold",
    "thompson_sample",
    "detect_anomaly",
    "update_distribution",
    "thompson_detect",
    "false_positive_rate",
    "convergence_check",
    "ThresholdDistribution",
    # RAF
    "find_raf_cycles",
    "build_graph",
    "is_autocatalytic",
    "coherence_score",
    "pattern_birth",
    "pattern_death",
    "crystallize",
    "detect_without_hardcode",
    # Cascade
    "cascade_derivative",
    "early_warning",
    "cascade_prediction",
    "trigger_alert",
    "cascade_trace",
    # Epidemic
    "calculate_R0",
    "infection_graph",
    "patient_zero",
    "outbreak_probability",
    "quarantine_candidates",
    "herd_immunity_threshold",
    "spread_simulation",
    # Volatility
    "VolatilityIndex",
    # Domain
    "load_domain",
    "register_domain",
    "list_domains",
    "get_volatility",
    "get_schema",
    "get_receipts",
    "validate_receipt",
    "DomainConfig",
    # Harness
    "monte_carlo_sim",
    "run_simulation",
    "simulate_cycle",
    "inject_fraud",
    "validate_constraints",
    "entropy_conservation",
    "run_scenario",
    "run_all_scenarios",
    "SimConfig",
    "SimState",
    "ScenarioResult",
]
