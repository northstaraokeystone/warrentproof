"""
Gov-OS v1.0: Universal Federal Fraud Detection Operating System

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Gov-OS:
- SIMULATES hypothetical fraud detection scenarios for academic research
- DOES NOT make claims about actual government programs, agencies, or contractors
- MODELS entropy-based detection using publicly available data only
- ALL detection results are simulation outputs, not allegations

Physics engine is universal. Domains supply volatility + schema.
One engine. Many domains. Zero duplication.
"""

from .core import (
    # Constants
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    VOLATILITY_ALPHA,
    RAF_MIN_CYCLE_LENGTH,
    CASCADE_DERIVATIVE_THRESHOLD,
    HOLOGRAPHIC_DETECTION_PROB,
    THOMPSON_FP_TARGET,
    COMPLETENESS_THRESHOLD,
    TENANT_ID,
    DISCLAIMER,
    VERSION,
    # Core functions
    dual_hash,
    merkle,
    emit_receipt,
    emit_L0,
    emit_L1,
    emit_L2,
    emit_L3,
    emit_L4,
    StopRule,
)

__version__ = "1.0.0"
__all__ = [
    "COMPRESSION_LEGITIMATE_FLOOR",
    "COMPRESSION_FRAUD_CEILING",
    "VOLATILITY_ALPHA",
    "RAF_MIN_CYCLE_LENGTH",
    "CASCADE_DERIVATIVE_THRESHOLD",
    "HOLOGRAPHIC_DETECTION_PROB",
    "THOMPSON_FP_TARGET",
    "COMPLETENESS_THRESHOLD",
    "TENANT_ID",
    "DISCLAIMER",
    "VERSION",
    "dual_hash",
    "merkle",
    "emit_receipt",
    "emit_L0",
    "emit_L1",
    "emit_L2",
    "emit_L3",
    "emit_L4",
    "StopRule",
]
