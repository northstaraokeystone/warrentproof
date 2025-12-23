"""
Gov-OS Core Receipt - Universal Receipt Emission with L0-L4 Meta-Receipt

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Every function calls emit_receipt. No exceptions.
Per CLAUDEME §8: All operations emit receipts.

Receipt Levels:
- L0: Telemetry (ingest, compression) - Raw data receipts
- L1: Detection (detection, raf_cycle, cascade, epidemic) - Agent-level
- L2: Decision (anchor, violation) - Paradigm shift
- L3: Quality (effectiveness) - Quality measurement
- L4: Meta (meta_receipt, completeness) - Receipts about receipts
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from .constants import (
    TENANT_ID,
    DISCLAIMER,
    COMPLETENESS_THRESHOLD,
)
from .utils import dual_hash

# Global ledger path
_LEDGER_PATH: Optional[str] = None

# Receipt level counts for completeness tracking
_RECEIPT_COUNTS = {
    "L0": 0,
    "L1": 0,
    "L2": 0,
    "L3": 0,
    "L4": 0,
}


class StopRule(Exception):
    """
    Raised when a stoprule triggers. Never catch silently.
    Per CLAUDEME §8: stoprules halt execution on critical failures.
    """
    pass


def get_ledger_path() -> str:
    """Get the current ledger path."""
    global _LEDGER_PATH
    if _LEDGER_PATH is None:
        # Default to gov_os/receipts.jsonl
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _LEDGER_PATH = os.path.join(base_dir, "receipts.jsonl")
    return _LEDGER_PATH


def set_ledger_path(path: str) -> None:
    """Set the ledger path."""
    global _LEDGER_PATH
    _LEDGER_PATH = path


def reset_ledger() -> None:
    """Reset the ledger and receipt counts. Use for testing."""
    global _RECEIPT_COUNTS
    _RECEIPT_COUNTS = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    ledger_path = get_ledger_path()
    if os.path.exists(ledger_path):
        os.remove(ledger_path)


def emit_receipt(
    receipt_type: str,
    data: Dict[str, Any],
    level: int = 0,
    to_stdout: bool = False,
    to_ledger: bool = True,
) -> Dict[str, Any]:
    """
    Every function calls this. No exceptions.
    Per CLAUDEME §8: All operations emit receipts.

    Args:
        receipt_type: Type of receipt (ingest, detection, compression, etc.)
        data: Receipt payload data
        level: Receipt level 0-4 (default 0)
        to_stdout: Whether to print to stdout (default False)
        to_ledger: Whether to append to ledger file (default True)

    Returns:
        Complete receipt dict with ts, tenant_id, payload_hash, level
    """
    global _RECEIPT_COUNTS

    # Ensure simulation flag is always present
    if "simulation_flag" not in data:
        data["simulation_flag"] = DISCLAIMER

    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", TENANT_ID),
        "level": f"L{level}",
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data,
    }

    # Track receipt counts
    level_key = f"L{level}"
    if level_key in _RECEIPT_COUNTS:
        _RECEIPT_COUNTS[level_key] += 1

    # Write to stdout if requested
    if to_stdout:
        print(json.dumps(receipt), flush=True, file=sys.stdout)

    # Append to ledger file
    if to_ledger:
        try:
            ledger_path = get_ledger_path()
            os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
            with open(ledger_path, "a") as f:
                f.write(json.dumps(receipt) + "\n")
        except Exception:
            pass  # Don't fail on ledger write issues

    return receipt


def emit_L0(receipt_type: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Emit telemetry-level receipt (L0)."""
    return emit_receipt(receipt_type, data, level=0, **kwargs)


def emit_L1(receipt_type: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Emit agent/detection-level receipt (L1)."""
    return emit_receipt(receipt_type, data, level=1, **kwargs)


def emit_L2(receipt_type: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Emit paradigm-shift-level receipt (L2)."""
    return emit_receipt(receipt_type, data, level=2, **kwargs)


def emit_L3(receipt_type: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Emit quality-level receipt (L3)."""
    return emit_receipt(receipt_type, data, level=3, **kwargs)


def emit_L4(receipt_type: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Emit meta-level receipt (L4) - receipts about receipts."""
    return emit_receipt(receipt_type, data, level=4, **kwargs)


def completeness_check() -> Dict[str, Any]:
    """
    Return L0-L4 coverage scores.

    Returns:
        Dict with coverage scores per level and overall completeness
    """
    global _RECEIPT_COUNTS

    total = sum(_RECEIPT_COUNTS.values())

    if total == 0:
        coverage = {f"L{i}": 0.0 for i in range(5)}
        overall = 0.0
    else:
        coverage = {f"L{i}": _RECEIPT_COUNTS[f"L{i}"] / total for i in range(5)}
        # Overall completeness: all levels must have receipts
        levels_with_receipts = sum(1 for v in _RECEIPT_COUNTS.values() if v > 0)
        overall = levels_with_receipts / 5.0

    result = {
        "coverage": coverage,
        "overall_completeness": overall,
        "meets_threshold": overall >= COMPLETENESS_THRESHOLD,
        "total_receipts": total,
        "counts": _RECEIPT_COUNTS.copy(),
    }

    # Emit L4 meta-receipt about completeness
    emit_L4("completeness", result)

    return result


def self_reference_achieved() -> bool:
    """
    True when L4 feeds back to L0.
    Self-reference is achieved when:
    1. L4 receipts exist (meta level)
    2. Those L4 receipts influence L0 (telemetry level)
    """
    global _RECEIPT_COUNTS

    # Basic check: both L0 and L4 must have receipts
    return _RECEIPT_COUNTS["L0"] > 0 and _RECEIPT_COUNTS["L4"] > 0


def godel_layer() -> str:
    """
    Returns 'L0' (base layer hits undecidability first).
    Per Gödel: The lowest layer cannot prove its own consistency.
    """
    return "L0"


# =============================================================================
# STOPRULE HELPERS
# =============================================================================

def stoprule_hash_mismatch(expected: str, actual: str) -> None:
    """Emit anomaly receipt and halt on hash mismatch."""
    emit_L2("anomaly", {
        "metric": "hash_mismatch",
        "expected": expected,
        "actual": actual,
        "delta": -1,
        "action": "halt",
        "classification": "violation",
    })
    raise StopRule(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> None:
    """Emit anomaly receipt and halt on invalid receipt."""
    emit_L2("anomaly", {
        "metric": "invalid_receipt",
        "reason": reason,
        "delta": -1,
        "action": "halt",
        "classification": "violation",
    })
    raise StopRule(f"Invalid receipt: {reason}")


def stoprule_constraint_violated(constraint: str, actual: Any, expected: Any) -> None:
    """Emit violation receipt and halt on constraint violation."""
    emit_L2("violation", {
        "metric": "constraint_violated",
        "constraint": constraint,
        "actual": actual,
        "expected": expected,
        "action": "halt",
        "classification": "violation",
    })
    raise StopRule(f"Constraint violated: {constraint} - actual={actual}, expected={expected}")
