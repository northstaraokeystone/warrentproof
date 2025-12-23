"""
ShieldProof v2.0 Milestone Module

Track milestone delivery and verification.
Per Grok: "Automated reconciliation" - verify deliverables before payment.

"One receipt. One milestone. One truth."
"""

from datetime import datetime
from typing import Any

from .core import (
    dual_hash,
    emit_receipt,
    query_receipts,
    StopRule,
    MILESTONE_STATES,
)
from .contract import get_contract, get_contract_milestones


def submit_deliverable(
    contract_id: str,
    milestone_id: str,
    deliverable: bytes | str,
) -> dict:
    """
    Submit a deliverable for a milestone, changing status to DELIVERED.

    Args:
        contract_id: Contract identifier
        milestone_id: Milestone identifier
        deliverable: Deliverable content (bytes or string, will be hashed)

    Returns:
        milestone_receipt with status=DELIVERED

    Stoprules:
        - stoprule_unknown_contract: If contract_id not found
        - stoprule_unknown_milestone: If milestone_id not in contract
    """
    # Stoprule: Check contract exists
    contract = get_contract(contract_id)
    if not contract:
        return stoprule_unknown_contract(contract_id)

    # Stoprule: Check milestone exists in contract
    milestones = get_contract_milestones(contract_id)
    milestone = next((m for m in milestones if m["id"] == milestone_id), None)
    if not milestone:
        return stoprule_unknown_milestone(contract_id, milestone_id)

    # Hash the deliverable
    if isinstance(deliverable, str):
        deliverable = deliverable.encode('utf-8')
    deliverable_hash = dual_hash(deliverable)

    # Create milestone receipt
    receipt = emit_receipt("milestone", {
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "deliverable_hash": deliverable_hash,
        "status": "DELIVERED",
        "verifier_id": None,
        "verification_ts": None,
    })

    return receipt


def verify_milestone(
    contract_id: str,
    milestone_id: str,
    verifier_id: str,
    passed: bool,
) -> dict:
    """
    Verify a milestone deliverable, changing status to VERIFIED or DISPUTED.

    Args:
        contract_id: Contract identifier
        milestone_id: Milestone identifier
        verifier_id: ID of the verifying entity
        passed: Whether verification passed

    Returns:
        milestone_receipt with status=VERIFIED or DISPUTED

    Stoprules:
        - stoprule_unknown_contract: If contract_id not found
        - stoprule_unknown_milestone: If milestone_id not in contract
        - stoprule_already_verified: If milestone already VERIFIED or PAID
    """
    # Stoprule: Check contract exists
    contract = get_contract(contract_id)
    if not contract:
        return stoprule_unknown_contract(contract_id)

    # Stoprule: Check milestone exists
    milestones = get_contract_milestones(contract_id)
    milestone = next((m for m in milestones if m["id"] == milestone_id), None)
    if not milestone:
        return stoprule_unknown_milestone(contract_id, milestone_id)

    # Stoprule: Check not already verified or paid
    if milestone.get("status") in ["VERIFIED", "PAID"]:
        return stoprule_already_verified(contract_id, milestone_id)

    new_status = "VERIFIED" if passed else "DISPUTED"
    verification_ts = datetime.utcnow().isoformat() + "Z"

    # Create milestone receipt
    receipt = emit_receipt("milestone", {
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "deliverable_hash": milestone.get("deliverable_hash"),
        "status": new_status,
        "verifier_id": verifier_id,
        "verification_ts": verification_ts,
    })

    return receipt


def get_milestone(contract_id: str, milestone_id: str) -> dict | None:
    """
    Retrieve current milestone status.

    Args:
        contract_id: Contract identifier
        milestone_id: Milestone identifier

    Returns:
        Milestone dict with current status or None if not found
    """
    milestones = get_contract_milestones(contract_id)
    return next((m for m in milestones if m["id"] == milestone_id), None)


def list_pending() -> list[dict]:
    """
    List all milestones awaiting verification (status=DELIVERED).

    Returns:
        List of milestone dicts
    """
    pending = []

    # Get all contracts
    contracts = query_receipts("contract")
    for contract in contracts:
        contract_id = contract.get("contract_id")
        milestones = get_contract_milestones(contract_id)
        for m in milestones:
            if m.get("status") == "DELIVERED":
                pending.append({
                    "contract_id": contract_id,
                    "contractor": contract.get("contractor"),
                    **m
                })

    return pending


def list_verified() -> list[dict]:
    """
    List all milestones that have been verified (status=VERIFIED).

    Returns:
        List of milestone dicts
    """
    verified = []

    # Get all contracts
    contracts = query_receipts("contract")
    for contract in contracts:
        contract_id = contract.get("contract_id")
        milestones = get_contract_milestones(contract_id)
        for m in milestones:
            if m.get("status") == "VERIFIED":
                verified.append({
                    "contract_id": contract_id,
                    "contractor": contract.get("contractor"),
                    **m
                })

    return verified


def list_disputed() -> list[dict]:
    """
    List all disputed milestones (status=DISPUTED).

    Returns:
        List of milestone dicts
    """
    disputed = []

    # Get all contracts
    contracts = query_receipts("contract")
    for contract in contracts:
        contract_id = contract.get("contract_id")
        milestones = get_contract_milestones(contract_id)
        for m in milestones:
            if m.get("status") == "DISPUTED":
                disputed.append({
                    "contract_id": contract_id,
                    "contractor": contract.get("contractor"),
                    **m
                })

    return disputed


# === STOPRULES ===

def stoprule_unknown_contract(contract_id: str) -> dict:
    """Emit anomaly receipt for unknown contract."""
    receipt = emit_receipt("anomaly", {
        "metric": "unknown_contract",
        "contract_id": contract_id,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Unknown contract: {contract_id}")


def stoprule_unknown_milestone(contract_id: str, milestone_id: str) -> dict:
    """Emit anomaly receipt for unknown milestone."""
    receipt = emit_receipt("anomaly", {
        "metric": "unknown_milestone",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Unknown milestone: {milestone_id} in contract {contract_id}")


def stoprule_already_verified(contract_id: str, milestone_id: str) -> dict:
    """Emit anomaly receipt for already verified milestone."""
    receipt = emit_receipt("anomaly", {
        "metric": "already_verified",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Milestone already verified: {milestone_id} in contract {contract_id}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    from .core import clear_ledger
    from .contract import register_contract

    print("# Milestone module self-test", file=sys.stderr)

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

    # Test submit_deliverable
    r1 = submit_deliverable(contract_id, "M1", b"Design document v1.0")
    assert r1["receipt_type"] == "milestone"
    assert r1["status"] == "DELIVERED"
    print(f"# Deliverable submitted: {r1['milestone_id']}", file=sys.stderr)

    # Test list_pending
    pending = list_pending()
    assert len(pending) >= 1
    print(f"# Pending milestones: {len(pending)}", file=sys.stderr)

    # Test verify_milestone
    r2 = verify_milestone(contract_id, "M1", "INSPECTOR-001", passed=True)
    assert r2["receipt_type"] == "milestone"
    assert r2["status"] == "VERIFIED"
    print(f"# Milestone verified: {r2['milestone_id']}", file=sys.stderr)

    # Test list_verified
    verified = list_verified()
    assert len(verified) >= 1
    print(f"# Verified milestones: {len(verified)}", file=sys.stderr)

    # Test stoprule_already_verified
    try:
        verify_milestone(contract_id, "M1", "INSPECTOR-002", passed=True)
        print("# FAIL: Should have raised StopRule", file=sys.stderr)
    except StopRule as e:
        print(f"# Stoprule triggered correctly: {e}", file=sys.stderr)

    print("# PASS: milestone module self-test", file=sys.stderr)
