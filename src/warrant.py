"""
WarrantProof Warrant Module - Receipt Generation for Military Transactions

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module generates cryptographic attestation receipts for military transactions.
Every decision point in procurement becomes a cryptographically signed warrant.

The warrant is NOT an authorization - it's an immutable record that authorization occurred.

Receipt Types:
- warrant_receipt: Per transaction
- quality_attestation_receipt: Per inspection
- milestone_receipt: Per program checkpoint
- cost_variance_receipt: When variance >= 5%
"""

import uuid
from datetime import datetime
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    BRANCHES,
    dual_hash,
    emit_receipt,
    get_citation,
    stoprule_missing_approver,
    stoprule_missing_lineage,
    stoprule_uncited_data,
    validate_branch,
)


def generate_warrant(
    transaction: dict,
    approver: str,
    branch: str,
    parent_receipt_id: Optional[str] = None,
    citation_key: Optional[str] = None
) -> dict:
    """
    Create warrant_receipt with dual-hash signature.
    Per spec: Every decision point becomes a cryptographic warrant.

    Args:
        transaction: Transaction details (amount, type, description)
        approver: Simulated official ID who approved
        branch: Military branch (Navy, Army, AirForce, Marines, SpaceForce, CoastGuard)
        parent_receipt_id: Optional parent receipt for lineage
        citation_key: Optional citation key from CITATIONS

    Returns:
        warrant_receipt dict

    SLO: Warrant generation <= 50ms
    """
    if not approver:
        stoprule_missing_approver()

    if not validate_branch(branch):
        raise ValueError(f"Invalid branch: {branch}. Must be one of {BRANCHES}")

    transaction_id = f"txn_{uuid.uuid4().hex[:16]}"

    # Build lineage
    decision_lineage = []
    if parent_receipt_id:
        decision_lineage.append(parent_receipt_id)

    # Build citation if provided
    citations = []
    if citation_key:
        citations.append(get_citation(citation_key))

    data = {
        "tenant_id": TENANT_ID,
        "transaction_id": transaction_id,
        "branch": branch,
        "transaction_type": transaction.get("type", "contract"),
        "amount_usd": transaction.get("amount", 0.0),
        "description": transaction.get("description", ""),
        "approver": approver,
        "decision_lineage": decision_lineage,
        "citations": citations,
        "simulation_flag": DISCLAIMER,
        "merkle_anchor": None,  # Populated by ledger.anchor()
    }

    return emit_receipt("warrant", data)


def quality_attestation(
    item: str,
    inspector: str,
    certification: dict,
    branch: str = "Navy",
    parent_receipt_id: Optional[str] = None,
    citation_key: str = "NEWPORT_NEWS_WELDING"
) -> dict:
    """
    Create quality_attestation_receipt for items like welding, materials.
    Per spec: Quality certifications become cryptographic attestations.

    Args:
        item: Item being inspected (e.g., "hull_weld_section_14")
        inspector: Simulated inspector ID
        certification: Certification details (passed, grade, notes)
        branch: Military branch
        parent_receipt_id: Optional parent receipt
        citation_key: Citation key for pattern source

    Returns:
        quality_attestation_receipt dict
    """
    if not inspector:
        stoprule_missing_approver()

    attestation_id = f"qa_{uuid.uuid4().hex[:16]}"

    data = {
        "tenant_id": TENANT_ID,
        "attestation_id": attestation_id,
        "item": item,
        "inspector": inspector,
        "branch": branch,
        "certification": {
            "passed": certification.get("passed", False),
            "grade": certification.get("grade", ""),
            "notes": certification.get("notes", ""),
            "inspection_date": certification.get("date", datetime.utcnow().isoformat()),
        },
        "decision_lineage": [parent_receipt_id] if parent_receipt_id else [],
        "citation": get_citation(citation_key),
        "simulation_flag": DISCLAIMER,
    }

    return emit_receipt("quality_attestation", data)


def milestone_warrant(
    program: str,
    milestone: str,
    status: dict,
    branch: str = "Navy",
    parent_receipt_id: Optional[str] = None,
    citation_key: str = "GAO_FORD_CARRIER"
) -> dict:
    """
    Create milestone_receipt for program checkpoints.
    Per spec: Every milestone generates a receipt for lineage tracking.

    Args:
        program: Program name (e.g., "Trump-class Battleship")
        milestone: Milestone identifier (e.g., "MS-B", "CDR", "Delivery")
        status: Status details (complete, on_schedule, notes)
        branch: Military branch
        parent_receipt_id: Optional parent receipt
        citation_key: Citation key for pattern source

    Returns:
        milestone_receipt dict
    """
    milestone_id = f"ms_{uuid.uuid4().hex[:16]}"

    data = {
        "tenant_id": TENANT_ID,
        "milestone_id": milestone_id,
        "program": program,
        "milestone": milestone,
        "branch": branch,
        "status": {
            "complete": status.get("complete", False),
            "on_schedule": status.get("on_schedule", True),
            "schedule_variance_days": status.get("schedule_variance_days", 0),
            "notes": status.get("notes", ""),
        },
        "decision_lineage": [parent_receipt_id] if parent_receipt_id else [],
        "citation": get_citation(citation_key),
        "simulation_flag": DISCLAIMER,
    }

    return emit_receipt("milestone", data)


def cost_variance_warrant(
    program: str,
    baseline: float,
    actual: float,
    variance_pct: float,
    branch: str = "Navy",
    parent_receipt_id: Optional[str] = None,
    citation_key: str = "GAO_ZUMWALT"
) -> dict:
    """
    Create cost_variance_receipt when costs deviate >= 5%.
    Per spec: Cost overruns generate receipts for cascade detection.

    Args:
        program: Program name
        baseline: Baseline cost in USD
        actual: Actual cost in USD
        variance_pct: Variance percentage (e.g., 23.0 for 23%)
        branch: Military branch
        parent_receipt_id: Optional parent receipt
        citation_key: Citation key for pattern source

    Returns:
        cost_variance_receipt dict
    """
    variance_id = f"cv_{uuid.uuid4().hex[:16]}"

    # Calculate severity based on variance
    if variance_pct >= 50:
        severity = "critical"
    elif variance_pct >= 25:
        severity = "high"
    elif variance_pct >= 10:
        severity = "medium"
    else:
        severity = "low"

    data = {
        "tenant_id": TENANT_ID,
        "variance_id": variance_id,
        "program": program,
        "branch": branch,
        "baseline_usd": baseline,
        "actual_usd": actual,
        "variance_usd": actual - baseline,
        "variance_pct": variance_pct,
        "severity": severity,
        "decision_lineage": [parent_receipt_id] if parent_receipt_id else [],
        "citation": get_citation(citation_key),
        "simulation_flag": DISCLAIMER,
    }

    return emit_receipt("cost_variance", data)


def contract_award_warrant(
    contract_number: str,
    vendor: str,
    amount: float,
    branch: str,
    approver: str,
    parent_receipt_id: Optional[str] = None
) -> dict:
    """
    Create warrant for contract award.

    Args:
        contract_number: Contract identifier
        vendor: Vendor name (simulated)
        amount: Contract value in USD
        branch: Military branch
        approver: Approving official
        parent_receipt_id: Optional parent receipt

    Returns:
        warrant_receipt for contract award
    """
    return generate_warrant(
        transaction={
            "type": "contract_award",
            "amount": amount,
            "description": f"Contract {contract_number} to {vendor}",
            "contract_number": contract_number,
            "vendor": vendor,
        },
        approver=approver,
        branch=branch,
        parent_receipt_id=parent_receipt_id,
        citation_key="GAO_AUDIT_FAILURE"
    )


def delivery_warrant(
    item: str,
    quantity: int,
    unit_cost: float,
    branch: str,
    approver: str,
    parent_receipt_id: Optional[str] = None
) -> dict:
    """
    Create warrant for delivery acceptance.

    Args:
        item: Item delivered
        quantity: Quantity delivered
        unit_cost: Cost per unit in USD
        branch: Military branch
        approver: Accepting official
        parent_receipt_id: Optional parent receipt

    Returns:
        warrant_receipt for delivery
    """
    return generate_warrant(
        transaction={
            "type": "delivery",
            "amount": quantity * unit_cost,
            "description": f"Delivery: {quantity}x {item}",
            "item": item,
            "quantity": quantity,
            "unit_cost": unit_cost,
        },
        approver=approver,
        branch=branch,
        parent_receipt_id=parent_receipt_id
    )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import time

    print(f"# WarrantProof Warrant Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test warrant generation latency
    t0 = time.time()
    w = generate_warrant(
        transaction={"type": "contract", "amount": 1000000.0, "description": "Test contract"},
        approver="SIM_OFFICIAL_001",
        branch="Navy"
    )
    latency_ms = (time.time() - t0) * 1000
    assert latency_ms <= 50, f"Warrant generation latency {latency_ms}ms > 50ms SLO"
    assert w["receipt_type"] == "warrant"
    assert "simulation_flag" in w

    # Test quality attestation
    qa = quality_attestation(
        item="hull_weld_section_14",
        inspector="SIM_INSPECTOR_001",
        certification={"passed": True, "grade": "A"}
    )
    assert qa["receipt_type"] == "quality_attestation"
    assert "citation" in qa

    # Test milestone
    ms = milestone_warrant(
        program="Trump-class Battleship",
        milestone="MS-B",
        status={"complete": True, "on_schedule": False, "schedule_variance_days": 45}
    )
    assert ms["receipt_type"] == "milestone"

    # Test cost variance
    cv = cost_variance_warrant(
        program="Zumwalt-class",
        baseline=3_400_000_000,
        actual=6_100_000_000,
        variance_pct=79.4
    )
    assert cv["receipt_type"] == "cost_variance"
    assert cv["severity"] == "critical"

    print(f"# PASS: warrant module self-test (latency: {latency_ms:.1f}ms)", file=sys.stderr)
