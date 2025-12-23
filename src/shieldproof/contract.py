"""
ShieldProof v2.0 Contract Module

Register contracts with fixed-price terms and milestone definitions.
Per Grok: "Fixed-price everything" - the SpaceX model.

"One receipt. One milestone. One truth."
"""

import json
import uuid
from typing import Any

from .core import (
    dual_hash,
    emit_receipt,
    query_receipts,
    StopRule,
    TENANT_ID,
)


def register_contract(
    contractor: str,
    amount: float,
    milestones: list[dict],
    terms: dict,
    contract_id: str | None = None,
) -> dict:
    """
    Register a fixed-price contract with milestone definitions.

    Args:
        contractor: Name of contractor
        amount: Total fixed-price amount
        milestones: List of milestone dicts with id, description, amount, due_date
        terms: Contract terms dict (will be hashed)
        contract_id: Optional contract ID (generated if not provided)

    Returns:
        contract_receipt

    Stoprules:
        - stoprule_duplicate_contract: If contract_id already exists
        - stoprule_invalid_amount: If amount <= 0 or milestones don't sum to amount
    """
    if contract_id is None:
        contract_id = f"C-{uuid.uuid4().hex[:12].upper()}"

    # Stoprule: Check for duplicate
    existing = query_receipts("contract", contract_id=contract_id)
    if existing:
        return stoprule_duplicate_contract(contract_id)

    # Stoprule: Validate amount
    if amount <= 0:
        return stoprule_invalid_amount(contract_id, "Amount must be positive")

    # Stoprule: Validate milestones sum to amount
    milestone_sum = sum(m.get("amount", 0) for m in milestones)
    if abs(milestone_sum - amount) > 0.01:  # Allow for floating point tolerance
        return stoprule_invalid_amount(
            contract_id,
            f"Milestone sum ({milestone_sum}) does not equal contract amount ({amount})"
        )

    # Normalize milestones with PENDING status
    normalized_milestones = []
    for m in milestones:
        normalized_milestones.append({
            "id": m.get("id", f"M{len(normalized_milestones)+1}"),
            "description": m.get("description", ""),
            "amount": m.get("amount", 0),
            "due_date": m.get("due_date"),
            "status": "PENDING",
        })

    # Create contract receipt
    receipt = emit_receipt("contract", {
        "contract_id": contract_id,
        "contractor": contractor,
        "amount_fixed": amount,
        "milestones": normalized_milestones,
        "terms_hash": dual_hash(json.dumps(terms, sort_keys=True)),
    })

    return receipt


def get_contract(contract_id: str) -> dict | None:
    """
    Retrieve contract by ID.

    Args:
        contract_id: Contract identifier

    Returns:
        Contract receipt or None if not found
    """
    contracts = query_receipts("contract", contract_id=contract_id)
    if not contracts:
        return None
    # Return the most recent contract receipt (in case of updates)
    return contracts[-1]


def list_contracts(status: str | None = None) -> list[dict]:
    """
    List all contracts, optionally filtered by milestone status.

    Args:
        status: Optional milestone status filter

    Returns:
        List of contract receipts
    """
    contracts = query_receipts("contract")

    if status is None:
        return contracts

    # Filter by contracts that have at least one milestone with the given status
    filtered = []
    for contract in contracts:
        milestones = contract.get("milestones", [])
        if any(m.get("status") == status for m in milestones):
            filtered.append(contract)

    return filtered


def get_contract_milestones(contract_id: str) -> list[dict]:
    """
    Get current milestone states for a contract.
    Merges original contract milestones with any milestone receipts.

    Args:
        contract_id: Contract identifier

    Returns:
        List of milestone dicts with current status
    """
    contract = get_contract(contract_id)
    if not contract:
        return []

    # Start with contract milestones
    milestones = {m["id"]: m.copy() for m in contract.get("milestones", [])}

    # Update with milestone receipts
    milestone_receipts = query_receipts("milestone", contract_id=contract_id)
    for mr in milestone_receipts:
        mid = mr.get("milestone_id")
        if mid in milestones:
            milestones[mid]["status"] = mr.get("status", milestones[mid]["status"])
            if mr.get("deliverable_hash"):
                milestones[mid]["deliverable_hash"] = mr["deliverable_hash"]
            if mr.get("verifier_id"):
                milestones[mid]["verifier_id"] = mr["verifier_id"]
            if mr.get("verification_ts"):
                milestones[mid]["verification_ts"] = mr["verification_ts"]

    return list(milestones.values())


# === STOPRULES ===

def stoprule_duplicate_contract(contract_id: str) -> dict:
    """Emit anomaly receipt for duplicate contract."""
    receipt = emit_receipt("anomaly", {
        "metric": "duplicate_contract",
        "contract_id": contract_id,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Duplicate contract: {contract_id} already exists")


def stoprule_invalid_amount(contract_id: str, reason: str) -> dict:
    """Emit anomaly receipt for invalid amount."""
    receipt = emit_receipt("anomaly", {
        "metric": "invalid_amount",
        "contract_id": contract_id,
        "reason": reason,
        "delta": -1,
        "action": "reject",
        "classification": "violation",
    })
    raise StopRule(f"Invalid amount for {contract_id}: {reason}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    from .core import clear_ledger

    print("# Contract module self-test", file=sys.stderr)

    # Clear ledger for testing
    clear_ledger()

    # Test contract registration
    r = register_contract(
        contractor="ACME Defense",
        amount=1000000.00,
        milestones=[
            {"id": "M1", "description": "Design review", "amount": 250000.00},
            {"id": "M2", "description": "Prototype delivery", "amount": 750000.00},
        ],
        terms={"payment_terms": "net30"},
    )
    assert r["receipt_type"] == "contract"
    assert r["contractor"] == "ACME Defense"
    assert r["amount_fixed"] == 1000000.00
    print(f"# Contract registered: {r['contract_id']}", file=sys.stderr)

    # Test get_contract
    c = get_contract(r["contract_id"])
    assert c is not None
    assert c["contract_id"] == r["contract_id"]
    print(f"# Contract retrieved: {c['contract_id']}", file=sys.stderr)

    # Test list_contracts
    contracts = list_contracts()
    assert len(contracts) >= 1
    print(f"# Contracts listed: {len(contracts)}", file=sys.stderr)

    # Test duplicate stoprule
    try:
        register_contract(
            contractor="ACME Defense",
            amount=1000000.00,
            milestones=[{"id": "M1", "amount": 1000000.00}],
            terms={},
            contract_id=r["contract_id"],
        )
        print("# FAIL: Should have raised StopRule", file=sys.stderr)
    except StopRule as e:
        print(f"# Stoprule triggered correctly: {e}", file=sys.stderr)

    print("# PASS: contract module self-test", file=sys.stderr)
