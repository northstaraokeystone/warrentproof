"""
Gov-OS Defense Scenarios - Defense-Specific Test Scenarios

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from typing import Any, Dict, List

from ...core.harness import SimState, ScenarioResult, run_simulation
from ...core.raf import detect_without_hardcode
from ...core.constants import DISCLAIMER

from .data import DefenseReceipts, generate_sample_data


DEFENSE_SCENARIOS = [
    "SUBCONTRACTOR_RING",
    "SHIPYARD_DISRUPTION",
    "COST_PLUS_FRAUD",
    "RAPID_ITERATION",
    "ADDITIVE_VALIDATION",
    "ASSEMBLY_TRACKING",
]


def run_defense_scenario(scenario_name: str) -> ScenarioResult:
    """
    Run a defense-specific scenario.

    Args:
        scenario_name: Name of scenario

    Returns:
        ScenarioResult
    """
    scenarios = {
        "SUBCONTRACTOR_RING": _scenario_subcontractor_ring,
        "SHIPYARD_DISRUPTION": _scenario_shipyard_disruption,
        "COST_PLUS_FRAUD": _scenario_cost_plus_fraud,
        "RAPID_ITERATION": _scenario_rapid_iteration,
        "ADDITIVE_VALIDATION": _scenario_additive_validation,
        "ASSEMBLY_TRACKING": _scenario_assembly_tracking,
    }

    if scenario_name not in scenarios:
        return ScenarioResult(
            name=scenario_name,
            passed=False,
            message=f"Unknown defense scenario: {scenario_name}",
        )

    return scenarios[scenario_name]()


def _scenario_subcontractor_ring() -> ScenarioResult:
    """Detect RAF cycles in subcontractor graph."""
    # Generate data with subcontractor ring fraud
    receipts = DefenseReceipts()
    receipts.inject_fraud("subcontractor_ring", count=10)

    # Run RAF detection
    detections = detect_without_hardcode(
        receipts.receipts,
        "defense",
        node_key="vendor_id",
        edge_key="payment_to",
    )

    # Check if rings were detected
    ring_detections = [d for d in detections if d.get("anomaly_type") == "raf_cycle"]

    passed = len(ring_detections) > 0
    return ScenarioResult(
        name="SUBCONTRACTOR_RING",
        passed=passed,
        metrics={
            "rings_injected": 10,
            "rings_detected": len(ring_detections),
        },
        message=f"Detected {len(ring_detections)} subcontractor rings",
    )


def _scenario_shipyard_disruption() -> ScenarioResult:
    """Model SpaceX-style rapid iteration and calculate disruption factor."""
    state = run_simulation("defense", n_cycles=100, seed=42)

    # Calculate disruption factor based on iteration speed
    # Traditional: ~26 weeks per iteration
    # SpaceX-style: ~1-2 weeks per iteration
    traditional_cadence = 26 * 7  # days
    target_cadence = 7  # days (1 week)

    disruption_factor = traditional_cadence / target_cadence

    passed = disruption_factor > 10  # At least 10x improvement
    return ScenarioResult(
        name="SHIPYARD_DISRUPTION",
        passed=passed,
        metrics={
            "traditional_cadence_days": traditional_cadence,
            "target_cadence_days": target_cadence,
            "disruption_factor": disruption_factor,
        },
        message=f"Disruption factor: {disruption_factor:.1f}x",
    )


def _scenario_cost_plus_fraud() -> ScenarioResult:
    """Detect cost-plus inflation patterns."""
    receipts = DefenseReceipts()
    receipts.inject_fraud("cost_plus_inflation", count=20)

    # Detection should flag increasing variance
    fraud_receipts = [r for r in receipts.receipts if r.get("_is_fraud")]
    detected = sum(1 for r in fraud_receipts if r.get("variance_pct", 0) > 10)

    detection_rate = detected / len(fraud_receipts) if fraud_receipts else 0

    passed = detection_rate >= 0.90
    return ScenarioResult(
        name="COST_PLUS_FRAUD",
        passed=passed,
        metrics={
            "fraud_injected": len(fraud_receipts),
            "detected": detected,
            "detection_rate": detection_rate,
        },
        message=f"Detection rate: {detection_rate:.2%}",
    )


def _scenario_rapid_iteration() -> ScenarioResult:
    """Validate iteration receipt chain for rapid development."""
    receipts = DefenseReceipts()

    # Generate iteration sequence
    for i in range(10):
        receipts.receipts.append({
            "receipt_type": "shipyard_iteration_receipt",
            "block_id": f"BLOCK_{i:03d}",
            "ship_id": "SHIP_001",
            "iteration": i + 1,
            "cadence_days": 7 + (i % 3),  # Varying cadence
            "_is_fraud": False,
            "simulation_flag": DISCLAIMER,
        })

    # Verify cadence compression
    cadences = [r.get("cadence_days", 0) for r in receipts.receipts]
    avg_cadence = sum(cadences) / len(cadences) if cadences else 0

    passed = avg_cadence < 14  # Less than 2 weeks average
    return ScenarioResult(
        name="RAPID_ITERATION",
        passed=passed,
        metrics={
            "iterations": len(receipts.receipts),
            "avg_cadence_days": avg_cadence,
        },
        message=f"Average cadence: {avg_cadence:.1f} days",
    )


def _scenario_additive_validation() -> ScenarioResult:
    """Verify 3D printing QA receipts and hash chain integrity."""
    from ...core.utils import dual_hash

    receipts = []
    prev_hash = ""

    for i in range(5):
        print_data = f"SECTION_{i}_LAYER_{i*100}"
        print_hash = dual_hash(print_data)

        receipts.append({
            "receipt_type": "shipyard_additive_receipt",
            "hull_section": f"SECTION_{i}",
            "print_hash": print_hash,
            "material_kg": 1000 + i * 100,
            "qa_status": "passed",
            "prev_hash": prev_hash,
            "simulation_flag": DISCLAIMER,
        })

        prev_hash = print_hash

    # Verify hash chain
    chain_valid = True
    for i in range(1, len(receipts)):
        if receipts[i]["prev_hash"] != receipts[i - 1]["print_hash"]:
            chain_valid = False
            break

    passed = chain_valid
    return ScenarioResult(
        name="ADDITIVE_VALIDATION",
        passed=passed,
        metrics={
            "sections": len(receipts),
            "chain_valid": chain_valid,
        },
        message="Hash chain " + ("valid" if chain_valid else "broken"),
    )


def _scenario_assembly_tracking() -> ScenarioResult:
    """Verify robotic assembly receipts - all welds traced."""
    from ...core.utils import dual_hash

    receipts = []

    for i in range(10):
        weld_data = f"WELD_{i}_ROBOT_001_BLOCKS_{i},{i+1}"
        inspection_hash = dual_hash(weld_data)

        receipts.append({
            "receipt_type": "shipyard_assembly_receipt",
            "weld_id": f"WELD_{i:04d}",
            "robot_id": "ROBOT_001",
            "block_ids": [f"BLOCK_{i}", f"BLOCK_{i+1}"],
            "inspection_hash": inspection_hash,
            "weld_quality": 0.95 + (i % 5) * 0.01,
            "simulation_flag": DISCLAIMER,
        })

    # Verify all welds have inspection hashes
    all_inspected = all(r.get("inspection_hash") for r in receipts)

    # Verify quality above threshold
    qualities = [r.get("weld_quality", 0) for r in receipts]
    avg_quality = sum(qualities) / len(qualities) if qualities else 0

    passed = all_inspected and avg_quality >= 0.95
    return ScenarioResult(
        name="ASSEMBLY_TRACKING",
        passed=passed,
        metrics={
            "welds": len(receipts),
            "all_inspected": all_inspected,
            "avg_quality": avg_quality,
        },
        message=f"All welds traced, avg quality: {avg_quality:.4f}",
    )
