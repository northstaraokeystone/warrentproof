"""
ShieldProof v2.0 - Minimal Viable Truth for Defense Accountability

Components (3, not 20):
1. Immutable Receipts (contract, milestone, payment)
2. Automated Reconciliation
3. Public Audit Trail

"One receipt. One milestone. One truth."
"""

__version__ = "2.0.0"

from .core import (
    dual_hash,
    emit_receipt,
    merkle,
    StopRule,
    TENANT_ID,
    RECEIPT_TYPES,
    MILESTONE_STATES,
)
