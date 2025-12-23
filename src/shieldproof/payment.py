"""
ShieldProof v2.0 Payment Module

Release payment only on verified milestone.
Per Grok: "On-chain payment release" - payment follows verification, never before.

"One receipt. One milestone. One truth."
"""

import json
from datetime import datetime
from typing import Any

from .core import (
    dual_hash,
    emit_receipt,
    query_receipts,
    StopRule,
)
from .contract import get_contract, get_contract_milestones
from .milestone import get_milestone


def release_payment(contract_id: str, milestone_id: str) -> dict:
    """
    Release payment for a verified milestone.
    Creates payment_receipt and updates milestone to PAID.

    Args:
        contract_id: Contract identifier
        milestone_id: Milestone identifier

    Returns:
        payment_receipt

    Stoprules:
        - stoprule_unverified_milestone: HALT if attempting payment on non-VERIFIED
        - stoprule_already_paid: If milestone already PAID
    """
    # Get milestone status
    milestone = get_milestone(contract_id, milestone_id)
    if not milestone:
        return stoprule_unverified_milestone(contract_id, milestone_id, "Milestone not found")

    # Check for existing payment FIRST (before status check)
    existing_payments = query_receipts("payment", contract_id=contract_id, milestone_id=milestone_id)
    if existing_payments:
        return stoprule_already_paid(contract_id, milestone_id)

    # Stoprule: Check milestone is verified (status must be VERIFIED, not PAID or other)
    if milestone.get("status") != "VERIFIED":
        return stoprule_unverified_milestone(
            contract_id,
            milestone_id,
            f"Status is {milestone.get('status')}, not VERIFIED"
        )

    amount = milestone.get("amount", 0)
    released_at = datetime.utcnow().isoformat() + "Z"

    # Create payment data for hashing
    payment_data = {
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "amount": amount,
        "released_at": released_at,
    }
    payment_hash = dual_hash(json.dumps(payment_data, sort_keys=True))

    # Create payment receipt
    receipt = emit_receipt("payment", {
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "amount": amount,
        "payment_hash": payment_hash,
        "released_at": released_at,
    })

    # Emit milestone receipt to update status to PAID
    emit_receipt("milestone", {
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "status": "PAID",
        "deliverable_hash": milestone.get("deliverable_hash"),
        "verifier_id": milestone.get("verifier_id"),
        "verification_ts": milestone.get("verification_ts"),
    })

    return receipt


def get_payment(contract_id: str, milestone_id: str) -> dict | None:
    """
    Retrieve payment status for a milestone.

    Args:
        contract_id: Contract identifier
        milestone_id: Milestone identifier

    Returns:
        Payment receipt or None if not paid
    """
    payments = query_receipts("payment", contract_id=contract_id, milestone_id=milestone_id)
    if not payments:
        return None
    return payments[-1]


def total_paid(contract_id: str) -> float:
    """
    Sum of released payments for a contract.

    Args:
        contract_id: Contract identifier

    Returns:
        Total amount paid
    """
    payments = query_receipts("payment", contract_id=contract_id)
    return sum(p.get("amount", 0) for p in payments)


def total_outstanding(contract_id: str) -> float:
    """
    Remaining unpaid amount for a contract.

    Args:
        contract_id: Contract identifier

    Returns:
        Outstanding amount
    """
    contract = get_contract(contract_id)
    if not contract:
        return 0.0

    total_amount = contract.get("amount_fixed", 0)
    paid = total_paid(contract_id)
    return total_amount - paid


def list_payments(contract_id: str | None = None) -> list[dict]:
    """
    List all payment receipts, optionally filtered by contract.

    Args:
        contract_id: Optional contract filter

    Returns:
        List of payment receipts
    """
    if contract_id:
        return query_receipts("payment", contract_id=contract_id)
    return query_receipts("payment")


# === STOPRULES ===

def stoprule_unverified_milestone(contract_id: str, milestone_id: str, reason: str) -> dict:
    """
    Emit anomaly receipt and HALT for unverified milestone payment attempt.
    This is a critical stoprule - payment without verification is the core problem.
    """
    receipt = emit_receipt("anomaly", {
        "metric": "unverified_milestone",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "reason": reason,
        "delta": -1,
        "action": "halt",
        "classification": "violation",
    })
    raise StopRule(f"HALT: Cannot pay unverified milestone {milestone_id} in {contract_id}: {reason}")


def stoprule_already_paid(contract_id: str, milestone_id: str) -> dict:
    """Emit anomaly receipt for already paid milestone."""
    receipt = emit_receipt("anomaly", {
        "metric": "already_paid",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Milestone already paid: {milestone_id} in contract {contract_id}")


def stoprule_amount_mismatch(contract_id: str, milestone_id: str, expected: float, actual: float) -> dict:
    """Emit anomaly receipt for payment amount mismatch."""
    receipt = emit_receipt("anomaly", {
        "metric": "amount_mismatch",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "expected": expected,
        "actual": actual,
        "delta": actual - expected,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Amount mismatch for {milestone_id}: expected {expected}, got {actual}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    from .core import clear_ledger
    from .contract import register_contract
    from .milestone import submit_deliverable, verify_milestone

    print("# Payment module self-test", file=sys.stderr)

    # Clear ledger for testing
    clear_ledger()

    # Register a test contract
    contract = register_contract(
        contractor="ACME Defense",
        amount=1000000.00,
        milestones=[
            {"id": "M1", "description": "Design review", "amount": 250000.00},
            {"id": "M2", "description": "Prototype delivery", "amount": 750000.00},
        ],
        terms={},
    )
    contract_id = contract["contract_id"]
    print(f"# Contract: {contract_id}", file=sys.stderr)

    # Submit and verify M1
    submit_deliverable(contract_id, "M1", b"Design document v1.0")
    verify_milestone(contract_id, "M1", "INSPECTOR-001", passed=True)

    # Test release_payment
    r = release_payment(contract_id, "M1")
    assert r["receipt_type"] == "payment"
    assert r["amount"] == 250000.00
    print(f"# Payment released: ${r['amount']}", file=sys.stderr)

    # Test get_payment
    p = get_payment(contract_id, "M1")
    assert p is not None
    print(f"# Payment retrieved: ${p['amount']}", file=sys.stderr)

    # Test total_paid
    paid = total_paid(contract_id)
    assert paid == 250000.00
    print(f"# Total paid: ${paid}", file=sys.stderr)

    # Test total_outstanding
    outstanding = total_outstanding(contract_id)
    assert outstanding == 750000.00
    print(f"# Outstanding: ${outstanding}", file=sys.stderr)

    # Test stoprule_unverified_milestone
    try:
        release_payment(contract_id, "M2")  # M2 is not verified
        print("# FAIL: Should have raised StopRule", file=sys.stderr)
    except StopRule as e:
        print(f"# Stoprule triggered correctly: {e}", file=sys.stderr)

    # Test stoprule_already_paid
    try:
        release_payment(contract_id, "M1")  # M1 already paid
        print("# FAIL: Should have raised StopRule", file=sys.stderr)
    except StopRule as e:
        print(f"# Stoprule triggered correctly: {e}", file=sys.stderr)

    print("# PASS: payment module self-test", file=sys.stderr)
