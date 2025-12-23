"""
WarrantProof Military Accountability Simulation & Analysis Suite

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This package provides simulation and research tools for modeling
receipts-native accountability infrastructure in defense procurement.

All data is synthetic or derived from publicly available sources.
See CITATIONS.md for complete source list.
"""

__version__ = "1.0.0"
__disclaimer__ = "THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."

from .core import (
    TENANT_ID,
    DISCLAIMER,
    CITATIONS,
    dual_hash,
    emit_receipt,
    merkle,
    cite,
    StopRuleException,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    stoprule_uncited_data,
)
