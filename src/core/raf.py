"""
Gov-OS RAF (Reflexively Autocatalytic Food-Generated) Network Analysis - Core Re-export

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Re-exports RAF functionality from main module for backward compatibility.
"""

# Re-export from main raf module
from ..raf import (
    build_transaction_graph,
    detect_cycles,
    emit_raf_receipt,
)

# Additional function expected by domain modules
def detect_without_hardcode(receipts, **kwargs):
    """
    Detect RAF cycles without hardcoded patterns.

    This is a wrapper around detect_cycles that works with receipt lists.

    Args:
        receipts: List of receipt dicts
        **kwargs: Additional arguments

    Returns:
        List of detected cycles
    """
    from ..raf import build_transaction_graph, detect_cycles

    G = build_transaction_graph(receipts)
    cycles = detect_cycles(G)

    return [{"cycle": c, "receipt_type": "raf_detection"} for c in cycles]


__all__ = [
    "build_transaction_graph",
    "detect_cycles",
    "emit_raf_receipt",
    "detect_without_hardcode",
]
