"""
Gov-OS Core Receipt System - CLAUDEME v3.1 Compliant

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

LAW_1 = "No receipt → not real"

Receipt Types for v5.1 Temporal Physics:
- temporal_anomaly_receipt: Resistance to natural decay detected
- zombie_receipt: Dormant entity with preserved weight
- contagion_receipt: Cross-domain fraud propagation
- super_graph_receipt: Multi-domain graph merge result
- insight_receipt: Plain-English explanation for audit
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from .constants import DISCLAIMER, TENANT_ID
from .utils import dual_hash


class StopRuleException(Exception):
    """
    Raised when a stoprule triggers. Never catch silently.
    Per CLAUDEME §8: stoprules halt execution on critical failures.
    """
    pass


def emit_receipt(
    receipt_type: str,
    data: Dict[str, Any],
    to_stdout: bool = True,
    level: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Every function calls this. No exceptions.
    Per CLAUDEME §8: All operations emit receipts.

    Args:
        receipt_type: Type of receipt
        data: Receipt payload data
        to_stdout: Whether to print to stdout
        level: Optional level (0-4) for hierarchical receipts

    Returns:
        Complete receipt dict with ts, tenant_id, payload_hash, receipt_id
    """
    from .utils import generate_receipt_id

    if "simulation_flag" not in data:
        data["simulation_flag"] = DISCLAIMER

    ts = datetime.utcnow().isoformat() + "Z"
    receipt = {
        "receipt_type": receipt_type,
        "receipt_id": generate_receipt_id(),
        "ts": ts,
        "timestamp": ts,  # Alias for backward compatibility
        "tenant_id": data.get("tenant_id", TENANT_ID),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }

    if level is not None:
        receipt["level"] = level

    if to_stdout:
        print(json.dumps(receipt), flush=True, file=sys.stdout)

    return receipt


def emit_L0(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit Level-0 receipt (base layer).
    L0 receipts are the foundation of the receipt chain.

    Args:
        receipt_type: Type of receipt
        data: Receipt payload

    Returns:
        L0 receipt dict with level=0
    """
    return emit_receipt(receipt_type, data, to_stdout=False, level=0)


# ============================================================================
# v5.1 TEMPORAL PHYSICS RECEIPT TYPES
# ============================================================================

def emit_temporal_anomaly_receipt(
    from_node: str,
    to_node: str,
    resistance: float,
    days_since_last: int,
    expected_weight: float,
    observed_weight: float,
    domain: str = "unknown",
) -> Dict[str, Any]:
    """
    Emit receipt when resistance to natural decay is detected.

    Physics: Edge weight should decay as Wt = W0 * e^(-λt)
    Resistance > threshold indicates artificial weight preservation.

    Args:
        from_node: Source entity
        to_node: Target entity
        resistance: Observed/Expected ratio - 1.0
        days_since_last: Days since last transaction
        expected_weight: Predicted weight after decay
        observed_weight: Actual weight in graph

    Returns:
        temporal_anomaly_receipt dict
    """
    return emit_receipt("temporal_anomaly_receipt", {
        "from_node": from_node,
        "to_node": to_node,
        "resistance": resistance,
        "days_since_last": days_since_last,
        "expected_weight": expected_weight,
        "observed_weight": observed_weight,
        "domain": domain,
        "anomaly_class": "decay_resistance",
        "physics_violation": "static_edge_in_dynamic_environment",
    }, to_stdout=False)


def emit_zombie_receipt(
    entity_id: str,
    days_dormant: int,
    preserved_weight: float,
    domain: str,
    linked_domains: list,
) -> Dict[str, Any]:
    """
    Emit receipt when zombie entity is detected.

    Zombie = entity with zero activity but preserved edge weight.
    This is physically impossible in natural systems.

    Args:
        entity_id: Zombie entity identifier
        days_dormant: Days without activity
        preserved_weight: Total weight of preserved edges
        domain: Primary domain of entity
        linked_domains: Other domains entity links to

    Returns:
        zombie_receipt dict
    """
    return emit_receipt("zombie_receipt", {
        "entity_id": entity_id,
        "days_dormant": days_dormant,
        "preserved_weight": preserved_weight,
        "domain": domain,
        "linked_domains": linked_domains,
        "anomaly_class": "temporal_rigidity",
        "physics_violation": "zero_decay_with_zero_activity",
    }, to_stdout=False)


def emit_contagion_receipt(
    source_domain: str,
    target_domain: str,
    shell_entity: str,
    propagation_path: list,
    pre_invoice_flag: bool,
) -> Dict[str, Any]:
    """
    Emit receipt when cross-domain contagion is detected.

    Contagion: When a shell entity links domains, collapse in one
    propagates temporal rigidity to the other.

    Args:
        source_domain: Domain where fraud was detected
        target_domain: Domain receiving contagion signal
        shell_entity: Entity linking the domains
        propagation_path: Path through super-graph
        pre_invoice_flag: True if target flagged before local evidence

    Returns:
        contagion_receipt dict
    """
    return emit_receipt("contagion_receipt", {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "shell_entity": shell_entity,
        "propagation_path": propagation_path,
        "pre_invoice_flag": pre_invoice_flag,
        "detection_type": "cross_domain_contagion",
        "physics_principle": "linked_entropy_pools",
    }, to_stdout=False)


def emit_super_graph_receipt(
    domains: list,
    total_nodes: int,
    total_edges: int,
    shared_entities: int,
    cycles_detected: int,
) -> Dict[str, Any]:
    """
    Emit receipt when super-graph is built from multiple domains.

    Args:
        domains: List of domains merged
        total_nodes: Total nodes in super-graph
        total_edges: Total edges in super-graph
        shared_entities: Entities appearing in multiple domains
        cycles_detected: RAF cycles found in super-graph

    Returns:
        super_graph_receipt dict
    """
    return emit_receipt("super_graph_receipt", {
        "domains": domains,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "shared_entities": shared_entities,
        "cycles_detected": cycles_detected,
        "cross_domain_enabled": True,
    }, to_stdout=False)


def emit_insight_receipt(
    anomaly_type: str,
    plain_english: str,
    technical_summary: Dict[str, Any],
    confidence: float,
) -> Dict[str, Any]:
    """
    Emit receipt with plain-English explanation for audit transparency.

    Args:
        anomaly_type: Type of anomaly being explained
        plain_english: Human-readable explanation
        technical_summary: Technical details for analysts
        confidence: Confidence score 0.0-1.0

    Returns:
        insight_receipt dict
    """
    return emit_receipt("insight_receipt", {
        "anomaly_type": anomaly_type,
        "plain_english": plain_english,
        "technical_summary": technical_summary,
        "confidence": confidence,
        "audit_ready": True,
    }, to_stdout=False)


# ============================================================================
# STOPRULES
# ============================================================================

def stoprule_hash_mismatch(expected: str, actual: str) -> None:
    """Emit anomaly receipt and halt on hash mismatch."""
    emit_receipt("anomaly", {
        "metric": "hash_mismatch",
        "expected": expected,
        "actual": actual,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> None:
    """Emit anomaly receipt and halt on invalid receipt."""
    emit_receipt("anomaly", {
        "metric": "invalid_receipt",
        "reason": reason,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Invalid receipt: {reason}")


def stoprule_uncited_data(field: str) -> None:
    """Emit violation receipt and halt on uncited data."""
    emit_receipt("violation", {
        "metric": "uncited_data",
        "field": field,
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException(f"Uncited data: {field} requires citation")


def stoprule_missing_approver() -> None:
    """Emit violation receipt and halt on missing approver."""
    emit_receipt("violation", {
        "metric": "missing_approver",
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException("Missing approver: warrant requires approver field")


def stoprule_missing_lineage() -> None:
    """Emit violation receipt and halt on missing lineage."""
    emit_receipt("violation", {
        "metric": "missing_lineage",
        "action": "halt",
        "classification": "violation"
    })
    raise StopRuleException("Missing lineage: warrant requires parent reference")


def stoprule_budget_exceeded(actual: float, limit: float) -> None:
    """Emit anomaly receipt and halt on budget exceeded."""
    emit_receipt("anomaly", {
        "metric": "budget",
        "actual": actual,
        "limit": limit,
        "delta": actual - limit,
        "action": "reject",
        "classification": "violation"
    })
    raise StopRuleException(f"Budget exceeded: {actual} > {limit}")


# ============================================================================
# LEVEL RECEIPTS (L0-L4) FOR HIERARCHICAL PROCESSING
# ============================================================================

def emit_L1(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit Level-1 receipt (first aggregation layer).
    L1 receipts aggregate multiple L0 receipts.
    """
    data["level"] = 1
    return emit_receipt(receipt_type, data, to_stdout=False)


def emit_L2(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit Level-2 receipt (pattern detection layer).
    L2 receipts represent detected patterns across L1 receipts.
    """
    data["level"] = 2
    return emit_receipt(receipt_type, data, to_stdout=False)


def emit_L3(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit Level-3 receipt (cross-domain correlation layer).
    L3 receipts capture relationships between domains.
    """
    data["level"] = 3
    return emit_receipt(receipt_type, data, to_stdout=False)


def emit_L4(receipt_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit Level-4 receipt (executive summary layer).
    L4 receipts provide audit-ready summaries.
    """
    data["level"] = 4
    return emit_receipt(receipt_type, data, to_stdout=False)


def completeness_check(
    receipts: Optional[list] = None,
    required_fields: Optional[list] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Check completeness of receipt chain.

    Args:
        receipts: List of receipts to check (optional, returns default if None)
        required_fields: Fields that must be present (optional)
        threshold: Minimum completeness ratio (optional, uses COMPLETENESS_THRESHOLD)

    Returns:
        Dict with completeness metrics
    """
    from .constants import COMPLETENESS_THRESHOLD

    if threshold is None:
        threshold = COMPLETENESS_THRESHOLD

    # If no receipts provided, return a default result
    if receipts is None:
        return {
            "complete": True,
            "ratio": 1.0,
            "threshold": threshold,
            "missing": [],
        }

    if required_fields is None:
        required_fields = ["receipt_type", "ts", "tenant_id"]

    if not receipts:
        return {"complete": False, "ratio": 0.0, "missing": required_fields, "threshold": threshold}

    total_fields = len(receipts) * len(required_fields)
    present_fields = 0
    missing = []

    for receipt in receipts:
        for field in required_fields:
            if field in receipt and receipt[field] is not None:
                present_fields += 1
            else:
                missing.append(f"{receipt.get('receipt_type', 'unknown')}:{field}")

    ratio = present_fields / total_fields if total_fields > 0 else 0.0
    complete = ratio >= threshold

    return {
        "complete": complete,
        "ratio": ratio,
        "threshold": threshold,
        "missing": missing[:10] if not complete else [],  # Limit output
    }


# Alias for backward compatibility
StopRule = StopRuleException
