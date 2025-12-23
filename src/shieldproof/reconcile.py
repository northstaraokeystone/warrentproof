"""
ShieldProof v2.0 Reconciliation Module

Automated reconciliation of spending vs deliverables.
Per Grok: "Automated reconciliation" - detect waste before it happens.

"One receipt. One milestone. One truth."
"""

from typing import Any

from .core import (
    emit_receipt,
    query_receipts,
    StopRule,
)
from .contract import get_contract, get_contract_milestones
from .payment import total_paid, list_payments


def reconcile_contract(contract_id: str) -> dict:
    """
    Reconcile a single contract - compare paid vs verified milestones.

    Args:
        contract_id: Contract identifier

    Returns:
        Reconciliation report dict
    """
    contract = get_contract(contract_id)
    if not contract:
        return {
            "contract_id": contract_id,
            "error": "Contract not found",
            "status": "ERROR",
        }

    contractor = contract.get("contractor", "Unknown")
    amount_fixed = contract.get("amount_fixed", 0)
    milestones = get_contract_milestones(contract_id)
    payments = list_payments(contract_id)

    # Count milestone states
    milestones_pending = sum(1 for m in milestones if m.get("status") == "PENDING")
    milestones_delivered = sum(1 for m in milestones if m.get("status") == "DELIVERED")
    milestones_verified = sum(1 for m in milestones if m.get("status") == "VERIFIED")
    milestones_paid = sum(1 for m in milestones if m.get("status") == "PAID")
    milestones_disputed = sum(1 for m in milestones if m.get("status") == "DISPUTED")

    # Calculate amounts
    amount_paid = total_paid(contract_id)
    amount_verified = sum(
        m.get("amount", 0) for m in milestones
        if m.get("status") in ["VERIFIED", "PAID"]
    )

    # Determine status and discrepancy
    discrepancy = 0.0
    status = "ON_TRACK"
    anomalies = []

    # Check for overpayment
    if amount_paid > amount_verified:
        discrepancy = amount_paid - amount_verified
        status = "OVERPAID"
        anomalies.append(stoprule_overpayment(contract_id, amount_paid, amount_verified))

    # Check for unverified payments (payment without verification)
    for payment in payments:
        milestone_id = payment.get("milestone_id")
        milestone = next((m for m in milestones if m["id"] == milestone_id), None)
        if milestone and milestone.get("status") not in ["VERIFIED", "PAID"]:
            status = "UNVERIFIED_PAYMENT"
            anomalies.append(stoprule_unverified_payment(
                contract_id,
                milestone_id,
                payment.get("amount", 0)
            ))

    # Check for disputes
    if milestones_disputed > 0:
        status = "DISPUTED"

    report = {
        "contract_id": contract_id,
        "contractor": contractor,
        "amount_fixed": amount_fixed,
        "amount_paid": amount_paid,
        "amount_verified": amount_verified,
        "milestones_total": len(milestones),
        "milestones_pending": milestones_pending,
        "milestones_delivered": milestones_delivered,
        "milestones_verified": milestones_verified,
        "milestones_paid": milestones_paid,
        "milestones_disputed": milestones_disputed,
        "status": status,
        "discrepancy": discrepancy,
        "anomalies": len(anomalies),
    }

    return report


def reconcile_all() -> list[dict]:
    """
    Run reconciliation across all contracts.

    Returns:
        List of reconciliation reports
    """
    contracts = query_receipts("contract")
    reports = []

    seen_contracts = set()
    for contract in contracts:
        contract_id = contract.get("contract_id")
        if contract_id in seen_contracts:
            continue
        seen_contracts.add(contract_id)

        report = reconcile_contract(contract_id)
        reports.append(report)

    return reports


def flag_anomaly(contract_id: str, reason: str) -> dict:
    """
    Manually flag an anomaly for a contract.

    Args:
        contract_id: Contract identifier
        reason: Reason for flagging

    Returns:
        anomaly_receipt
    """
    receipt = emit_receipt("anomaly", {
        "metric": "manual_flag",
        "contract_id": contract_id,
        "reason": reason,
        "delta": -1,
        "action": "investigate",
        "classification": "suspicious",
    })
    return receipt


def get_waste_summary() -> dict:
    """
    Get aggregate waste summary across all contracts.

    Returns:
        Summary dict with waste metrics
    """
    reports = reconcile_all()

    total_contracts = len(reports)
    total_committed = sum(r.get("amount_fixed", 0) for r in reports)
    total_paid = sum(r.get("amount_paid", 0) for r in reports)
    total_verified = sum(r.get("amount_verified", 0) for r in reports)

    # Waste = paid without verification
    waste_identified = total_paid - total_verified if total_paid > total_verified else 0

    # Savings potential = pending milestones that could be converted to fixed-price
    milestones_pending = sum(r.get("milestones_pending", 0) for r in reports)
    milestones_disputed = sum(r.get("milestones_disputed", 0) for r in reports)

    contracts_on_track = sum(1 for r in reports if r.get("status") == "ON_TRACK")
    contracts_overpaid = sum(1 for r in reports if r.get("status") == "OVERPAID")
    contracts_unverified = sum(1 for r in reports if r.get("status") == "UNVERIFIED_PAYMENT")
    contracts_disputed = sum(1 for r in reports if r.get("status") == "DISPUTED")

    return {
        "total_contracts": total_contracts,
        "total_committed": total_committed,
        "total_paid": total_paid,
        "total_verified": total_verified,
        "waste_identified": waste_identified,
        "milestones_pending": milestones_pending,
        "milestones_disputed": milestones_disputed,
        "contracts_on_track": contracts_on_track,
        "contracts_overpaid": contracts_overpaid,
        "contracts_unverified": contracts_unverified,
        "contracts_disputed": contracts_disputed,
    }


# === STOPRULES (emit anomaly but don't halt - reconciliation is informational) ===

def stoprule_overpayment(contract_id: str, paid: float, verified: float) -> dict:
    """Emit anomaly receipt for overpayment."""
    receipt = emit_receipt("anomaly", {
        "metric": "overpayment",
        "contract_id": contract_id,
        "amount_paid": paid,
        "amount_verified": verified,
        "delta": paid - verified,
        "action": "investigate",
        "classification": "violation",
    })
    return receipt


def stoprule_unverified_payment(contract_id: str, milestone_id: str, amount: float) -> dict:
    """Emit anomaly receipt for payment without verification."""
    receipt = emit_receipt("anomaly", {
        "metric": "unverified_payment",
        "contract_id": contract_id,
        "milestone_id": milestone_id,
        "amount": amount,
        "delta": -1,
        "action": "investigate",
        "classification": "violation",
    })
    return receipt


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    from .core import clear_ledger
    from .contract import register_contract
    from .milestone import submit_deliverable, verify_milestone
    from .payment import release_payment

    print("# Reconcile module self-test", file=sys.stderr)

    # Clear ledger for testing
    clear_ledger()

    # Register test contracts
    c1 = register_contract(
        contractor="ACME Defense",
        amount=1000000.00,
        milestones=[
            {"id": "M1", "description": "Design", "amount": 250000.00},
            {"id": "M2", "description": "Build", "amount": 750000.00},
        ],
        terms={},
    )
    c2 = register_contract(
        contractor="Beta Corp",
        amount=500000.00,
        milestones=[
            {"id": "M1", "description": "Phase 1", "amount": 500000.00},
        ],
        terms={},
    )

    # Submit, verify, and pay M1 of contract 1
    submit_deliverable(c1["contract_id"], "M1", b"Design doc")
    verify_milestone(c1["contract_id"], "M1", "INSPECTOR-001", passed=True)
    release_payment(c1["contract_id"], "M1")

    # Test reconcile_contract
    report = reconcile_contract(c1["contract_id"])
    assert report["status"] == "ON_TRACK"
    assert report["amount_paid"] == 250000.00
    print(f"# Contract 1 status: {report['status']}", file=sys.stderr)

    # Test reconcile_all
    all_reports = reconcile_all()
    assert len(all_reports) >= 2
    print(f"# Reconciled {len(all_reports)} contracts", file=sys.stderr)

    # Test get_waste_summary
    summary = get_waste_summary()
    assert summary["total_contracts"] >= 2
    assert summary["waste_identified"] == 0  # No waste yet
    print(f"# Waste identified: ${summary['waste_identified']}", file=sys.stderr)

    # Test flag_anomaly
    anomaly = flag_anomaly(c2["contract_id"], "Manual review requested")
    assert anomaly["receipt_type"] == "anomaly"
    print(f"# Anomaly flagged: {anomaly['reason']}", file=sys.stderr)

    print("# PASS: reconcile module self-test", file=sys.stderr)
