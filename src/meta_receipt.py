"""
WarrantProof Meta Receipt Module - Receipts-About-Receipts

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements meta-receipts: receipts that reference other receipts
and emit judgments about them. Required for autocatalytic closure (RAF) -
receipt clusters must predict their own anomalies.

Physics Foundation:
- RAF (Reflexively Autocatalytic Food sets) require self-reference
- Meta-receipts enable prediction validation
- Closure achieved when ≥80% predictions validate

SLOs:
- Meta-receipts cannot form cycles (DAG structure)
- Prediction accuracy ≥80% for RAF closure
- Validation requires outcome observation
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    RAF_CLOSURE_ACCURACY_MIN,
    META_RECEIPT_PREDICTION_WINDOW,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class MetaReceiptPrediction:
    """A prediction made by a meta-receipt about parent receipts."""
    prediction_id: str
    parent_cluster_ids: list
    prediction_type: str  # "anomaly", "pattern", "outcome"
    predicted_value: dict
    created_at: str = ""
    validated: Optional[bool] = None
    actual_value: Optional[dict] = None


@dataclass
class CausalCluster:
    """A cluster of receipts linked by causal relationships."""
    cluster_id: str
    receipt_ids: list = field(default_factory=list)
    creation_order: list = field(default_factory=list)  # Temporal order
    causal_links: dict = field(default_factory=dict)  # id -> [caused_ids]


def emit_meta_receipt(
    parent_receipts: list,
    prediction: dict
) -> dict:
    """
    Create receipt referencing parent receipt cluster.
    Prediction = "this cluster will exhibit pattern X."

    Args:
        parent_receipts: List of parent receipt dicts or IDs
        prediction: Prediction about the parent cluster

    Returns:
        meta_receipt dict
    """
    # Extract parent IDs
    parent_ids = []
    for p in parent_receipts:
        if isinstance(p, dict):
            parent_ids.append(p.get("payload_hash", dual_hash(str(p))))
        else:
            parent_ids.append(str(p))

    # Generate prediction ID
    prediction_id = dual_hash(str(prediction) + str(parent_ids))[:16]

    return emit_receipt("meta", {
        "tenant_id": TENANT_ID,
        "parent_receipt_cluster": parent_ids,
        "prediction": prediction,
        "prediction_id": prediction_id,
        "prediction_validated": None,  # To be filled later
        "self_reference_closure": False,  # To be validated
        "RAF_contribution": 0.0,  # To be calculated
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def validate_self_reference(
    meta_receipt: dict,
    parent_receipts: list
) -> bool:
    """
    Check closure: does meta-receipt's prediction match parent cluster's
    actual behavior? If yes, RAF closure achieved.

    Args:
        meta_receipt: Meta-receipt with prediction
        parent_receipts: Parent receipts to validate against

    Returns:
        True if prediction validated (RAF closure)
    """
    prediction = meta_receipt.get("prediction", {})

    if not prediction:
        return False

    # Extract prediction type and value
    pred_type = prediction.get("type", "")
    pred_value = prediction.get("value")

    # Validate based on prediction type
    if pred_type == "anomaly_count":
        # Prediction: cluster will have X anomalies
        actual_anomalies = sum(1 for r in parent_receipts if r.get("_is_fraud"))
        return actual_anomalies == pred_value

    elif pred_type == "compression_range":
        # Prediction: cluster compression will be in range [low, high]
        from .compress import compress_receipt_stream
        result = compress_receipt_stream(parent_receipts)
        ratio = result.get("compression_ratio", 0)
        low, high = pred_value.get("low", 0), pred_value.get("high", 1)
        return low <= ratio <= high

    elif pred_type == "entropy_threshold":
        # Prediction: cluster entropy will exceed threshold
        from .compress import entropy_score
        actual_entropy = entropy_score(parent_receipts)
        return actual_entropy >= pred_value

    elif pred_type == "pattern_present":
        # Prediction: pattern X will be present in cluster
        pattern_id = pred_value
        # Check if any receipt matches pattern
        for r in parent_receipts:
            if r.get("_fraud_pattern") == pattern_id:
                return True
        return False

    elif pred_type == "branch_distribution":
        # Prediction: branch distribution will match
        from collections import Counter
        branches = Counter(r.get("branch", "") for r in parent_receipts)
        for branch, expected_ratio in pred_value.items():
            actual_ratio = branches.get(branch, 0) / len(parent_receipts) if parent_receipts else 0
            if abs(actual_ratio - expected_ratio) > 0.1:  # 10% tolerance
                return False
        return True

    # Default: cannot validate
    return False


def cluster_receipts_by_causality(receipts: list) -> list:
    """
    Group receipts by causal relationships (time sequence, vendor links,
    transaction chains).

    Args:
        receipts: All receipts to cluster

    Returns:
        List of CausalCluster objects
    """
    if not receipts:
        return []

    clusters = []
    processed = set()

    # Build adjacency map from lineage
    lineage_map = defaultdict(set)  # parent -> children
    reverse_map = defaultdict(set)  # child -> parents

    for r in receipts:
        receipt_id = r.get("payload_hash", dual_hash(str(r)))
        for parent_id in r.get("decision_lineage", []):
            lineage_map[parent_id].add(receipt_id)
            reverse_map[receipt_id].add(parent_id)

    # Find connected components
    def dfs(start_id: str, visited: set, component: list):
        if start_id in visited:
            return
        visited.add(start_id)
        component.append(start_id)

        # Follow forward (children)
        for child in lineage_map.get(start_id, []):
            dfs(child, visited, component)

        # Follow backward (parents)
        for parent in reverse_map.get(start_id, []):
            dfs(parent, visited, component)

    # Build clusters
    receipt_ids = {r.get("payload_hash", dual_hash(str(r))) for r in receipts}

    for r in receipts:
        receipt_id = r.get("payload_hash", dual_hash(str(r)))

        if receipt_id in processed:
            continue

        component = []
        dfs(receipt_id, processed, component)

        if component:
            cluster = CausalCluster(
                cluster_id=dual_hash(str(component))[:16],
                receipt_ids=component,
                creation_order=sorted(component),  # Simplified ordering
            )
            clusters.append(cluster)

    return clusters


def test_autocatalytic_closure(
    cluster: CausalCluster,
    meta_receipts: list,
    parent_receipts: list
) -> bool:
    """
    For cluster: find meta-receipts about cluster, validate predictions.
    Return true if ≥80% predictions correct (self-referencing).

    Args:
        cluster: CausalCluster to test
        meta_receipts: Meta-receipts that might reference this cluster
        parent_receipts: Actual receipts in cluster

    Returns:
        True if autocatalytic closure achieved
    """
    # Find meta-receipts about this cluster
    relevant_metas = []
    cluster_id_set = set(cluster.receipt_ids)

    for meta in meta_receipts:
        parent_cluster = set(meta.get("parent_receipt_cluster", []))
        # Meta is relevant if it references any receipts in this cluster
        if parent_cluster & cluster_id_set:
            relevant_metas.append(meta)

    if not relevant_metas:
        return False

    # Validate each prediction
    validated_count = 0

    for meta in relevant_metas:
        # Get receipts referenced by this meta
        meta_parents = meta.get("parent_receipt_cluster", [])
        referenced_receipts = [
            r for r in parent_receipts
            if r.get("payload_hash") in meta_parents
        ]

        is_valid = validate_self_reference(meta, referenced_receipts)
        if is_valid:
            validated_count += 1

    # Check if ≥80% validated
    validation_ratio = validated_count / len(relevant_metas) if relevant_metas else 0

    return validation_ratio >= RAF_CLOSURE_ACCURACY_MIN


def calculate_RAF_contribution(
    meta_receipt: dict,
    validation_result: bool,
    cluster_size: int
) -> float:
    """
    Calculate RAF contribution score for a meta-receipt.

    Args:
        meta_receipt: Meta-receipt to score
        validation_result: Whether prediction validated
        cluster_size: Size of referenced cluster

    Returns:
        RAF contribution score 0-1
    """
    if not validation_result:
        return 0.0

    # Base score for validation
    base_score = 0.5

    # Bonus for larger clusters (more meaningful predictions)
    size_bonus = min(0.3, cluster_size / 100)

    # Bonus for specific prediction types
    prediction = meta_receipt.get("prediction", {})
    pred_type = prediction.get("type", "")

    type_bonus = {
        "anomaly_count": 0.2,
        "compression_range": 0.15,
        "pattern_present": 0.2,
        "entropy_threshold": 0.1,
    }.get(pred_type, 0.05)

    return min(1.0, base_score + size_bonus + type_bonus)


def check_acyclic(meta_receipts: list) -> bool:
    """
    Verify meta-receipts form a DAG (no cycles).

    Args:
        meta_receipts: List of meta-receipts to check

    Returns:
        True if acyclic
    """
    # Build graph: meta_id -> parent_ids
    graph = {}
    for meta in meta_receipts:
        meta_id = meta.get("payload_hash", meta.get("prediction_id", ""))
        parent_ids = meta.get("parent_receipt_cluster", [])
        graph[meta_id] = set(parent_ids)

    # DFS cycle detection
    visited = set()
    rec_stack = set()

    def has_cycle(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if has_cycle(node):
                return False

    return True


def emit_meta_validation_receipt(
    meta_receipt: dict,
    validation_result: bool,
    RAF_score: float
) -> dict:
    """
    Emit receipt documenting meta-receipt validation.

    Args:
        meta_receipt: Meta-receipt that was validated
        validation_result: Whether prediction validated
        RAF_score: RAF contribution score

    Returns:
        meta_validation_receipt dict
    """
    return emit_receipt("meta_validation", {
        "tenant_id": TENANT_ID,
        "meta_receipt_id": meta_receipt.get("payload_hash", ""),
        "prediction_id": meta_receipt.get("prediction_id", ""),
        "prediction_validated": validation_result,
        "RAF_contribution": round(RAF_score, 4),
        "self_reference_achieved": validation_result and RAF_score >= 0.5,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_cyclic_meta(cycle_nodes: list) -> None:
    """Meta-receipts cannot form cycles (DAG violation)."""
    emit_receipt("anomaly", {
        "metric": "cyclic_meta",
        "cycle_nodes": cycle_nodes[:10],  # First 10
        "action": "halt",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Cyclic meta-receipt detected: {cycle_nodes[:3]}")


def stoprule_prediction_accuracy_low(accuracy: float, threshold: float) -> None:
    """Prediction accuracy must be ≥ threshold."""
    if accuracy < threshold:
        emit_receipt("anomaly", {
            "metric": "prediction_accuracy_low",
            "accuracy": accuracy,
            "threshold": threshold,
            "delta": accuracy - threshold,
            "action": "recalibrate",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Prediction accuracy {accuracy:.2%} below threshold {threshold:.0%}"
        )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Meta Receipt Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test meta-receipt emission
    parent_receipts = [
        {"payload_hash": "parent_1", "branch": "Navy"},
        {"payload_hash": "parent_2", "branch": "Navy"},
        {"payload_hash": "parent_3", "branch": "Army"},
    ]

    prediction = {
        "type": "branch_distribution",
        "value": {"Navy": 0.67, "Army": 0.33}
    }

    meta = emit_meta_receipt(parent_receipts, prediction)
    print(f"# Meta receipt: {meta['receipt_type']}", file=sys.stderr)
    assert meta["receipt_type"] == "meta"
    assert len(meta["parent_receipt_cluster"]) == 3

    # Test self-reference validation
    valid = validate_self_reference(meta, parent_receipts)
    print(f"# Branch distribution validation: {valid}", file=sys.stderr)
    assert valid == True  # 2/3 Navy, 1/3 Army matches prediction

    # Test causal clustering
    receipts_with_lineage = [
        {"payload_hash": "r1", "decision_lineage": []},
        {"payload_hash": "r2", "decision_lineage": ["r1"]},
        {"payload_hash": "r3", "decision_lineage": ["r1"]},
        {"payload_hash": "r4", "decision_lineage": ["r2"]},
        {"payload_hash": "r5", "decision_lineage": []},  # Separate cluster
    ]

    clusters = cluster_receipts_by_causality(receipts_with_lineage)
    print(f"# Clusters found: {len(clusters)}", file=sys.stderr)

    # Test acyclic check
    meta_receipts = [
        {"payload_hash": "m1", "parent_receipt_cluster": ["p1", "p2"]},
        {"payload_hash": "m2", "parent_receipt_cluster": ["p3"]},
    ]
    is_acyclic = check_acyclic(meta_receipts)
    print(f"# DAG structure valid: {is_acyclic}", file=sys.stderr)
    assert is_acyclic == True

    # Test RAF contribution
    raf_score = calculate_RAF_contribution(meta, True, 50)
    print(f"# RAF contribution score: {raf_score:.4f}", file=sys.stderr)
    assert 0 <= raf_score <= 1

    # Test validation receipt
    validation_receipt = emit_meta_validation_receipt(meta, True, raf_score)
    assert validation_receipt["receipt_type"] == "meta_validation"

    print(f"# PASS: meta_receipt module self-test", file=sys.stderr)
