"""
Gov-OS Temporal Physics Module - Universal Decay for RAF Graphs

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

v5.1 Temporal Decay Physics:

The Physics Frame:
- Legitimate relationships in complex systems decay exponentially without
  reinforcement: Wt = W0 * e^(-λt)
- This is universal: radioactive decay, synaptic pruning, market churn
- Fraudulent constructs RESIST this decay—maintaining perfect weight over
  time is a low-entropy, high-information anomaly

Key Insight:
- "Static edges in dynamic environments are algorithmically improbable.
   The absence of expected decay is stronger evidence than presence of activity."

Detection Signals:
- Resistance > 0: Edge weight higher than natural decay predicts
- Zombie entities: Zero activity but preserved weight = impossible naturally
- Contagion: When shell entity links domains, collapse in one propagates
  temporal rigidity to the other—flagging pre-invoice fraud

Constants (from core.constants):
- LAMBDA_NATURAL = 0.005 (half-life ≈ 138 months)
- RESISTANCE_THRESHOLD = 0.1 (flag when resistance exceeds 10%)
- ZOMBIE_DAYS = 365 (dormancy threshold)
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .constants import (
    LAMBDA_NATURAL,
    RESISTANCE_THRESHOLD,
    ZOMBIE_DAYS,
    CONTAGION_OVERLAP_MIN,
    SHELL_PATTERN_THRESHOLD,
    DISCLAIMER,
)
from .receipt import (
    StopRuleException,
    emit_temporal_anomaly_receipt,
    emit_zombie_receipt,
    emit_contagion_receipt,
    emit_receipt,
)


# ============================================================================
# CORE DECAY PHYSICS
# ============================================================================

def edge_weight_decay(
    initial_weight: float,
    days_since_last: int,
    lambda_decay: float = LAMBDA_NATURAL,
) -> float:
    """
    Calculate expected edge weight after exponential decay.

    Physics: Wt = W0 × e^(-λt)
    - Same formula as radioactive decay, synaptic pruning, relationship attrition
    - λ = 0.005 per month → half-life ≈ 138 months (ln(2)/0.005 ≈ 138.6)

    Args:
        initial_weight: Original edge weight (W0)
        days_since_last: Time since last transaction (t in days)
        lambda_decay: Decay rate per day (default: LAMBDA_NATURAL/30 for daily)

    Returns:
        Expected weight after decay (Wt)

    Raises:
        StopRuleException: If decay would increase weight (physics violation)
    """
    if days_since_last < 0:
        stoprule_invalid_dates(days_since_last)

    if days_since_last == 0:
        return initial_weight

    # Convert monthly lambda to daily rate
    daily_lambda = lambda_decay / 30.0

    if HAS_NUMPY:
        decayed_weight = initial_weight * np.exp(-daily_lambda * days_since_last)
    else:
        decayed_weight = initial_weight * math.exp(-daily_lambda * days_since_last)

    # Stoprule: decay cannot increase weight
    if decayed_weight > initial_weight:
        stoprule_negative_decay(initial_weight, decayed_weight)

    return decayed_weight


def resistance_to_decay(
    expected_weight: float,
    observed_weight: float,
) -> float:
    """
    Calculate resistance to natural decay.

    Resistance = max(0, (observed / expected) - 1.0)
    - Resistance = 0: Edge decayed naturally
    - Resistance > 0: Edge maintained artificially (fraud signal)
    - Resistance > 1: Edge weight doubled vs expectation (strong signal)

    Physics interpretation:
    - Resistance is the "entropy debt" accumulated by maintaining dead connections
    - Natural systems shed unused structure; fraud accumulates disorder

    Args:
        expected_weight: Weight predicted by decay formula
        observed_weight: Actual weight in graph

    Returns:
        Resistance value (0.0 or positive)
    """
    if expected_weight <= 0:
        # Edge should have decayed to zero but exists
        return float('inf') if observed_weight > 0 else 0.0

    ratio = observed_weight / expected_weight

    # Resistance is always non-negative (floor at 0)
    resistance = max(0.0, ratio - 1.0)

    return resistance


def update_edge_with_decay(
    graph: 'nx.DiGraph',
    from_node: str,
    to_node: str,
    current_date: datetime,
    last_seen_date: datetime,
    domain: str = "unknown",
) -> float:
    """
    Apply decay to edge and detect anomalies.

    Called by raf.py on each transaction. Calculates expected decay,
    compares to observed weight, and emits temporal_anomaly_receipt
    if resistance exceeds threshold.

    Args:
        graph: NetworkX DiGraph to update
        from_node: Source node
        to_node: Target node
        current_date: Current transaction date
        last_seen_date: Date of last transaction on this edge
        domain: Domain identifier

    Returns:
        Resistance value for this edge

    Emits:
        temporal_anomaly_receipt if resistance > RESISTANCE_THRESHOLD
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for temporal analysis")

    if not graph.has_edge(from_node, to_node):
        return 0.0

    # Calculate days since last transaction
    days_since_last = (current_date - last_seen_date).days

    if days_since_last <= 0:
        return 0.0

    # Get observed weight
    observed_weight = graph[from_node][to_node].get("weight", 1.0)
    initial_weight = graph[from_node][to_node].get("initial_weight", observed_weight)

    # Calculate expected weight after decay
    expected_weight = edge_weight_decay(initial_weight, days_since_last)

    # Calculate resistance
    resistance = resistance_to_decay(expected_weight, observed_weight)

    # Store temporal metadata
    graph[from_node][to_node]["expected_weight"] = expected_weight
    graph[from_node][to_node]["resistance"] = resistance
    graph[from_node][to_node]["days_since_last"] = days_since_last

    # Emit anomaly receipt if resistance exceeds threshold
    if resistance > RESISTANCE_THRESHOLD:
        emit_temporal_anomaly_receipt(
            from_node=from_node,
            to_node=to_node,
            resistance=resistance,
            days_since_last=days_since_last,
            expected_weight=expected_weight,
            observed_weight=observed_weight,
            domain=domain,
        )

    return resistance


# ============================================================================
# ZOMBIE DETECTION
# ============================================================================

def detect_zombies(
    graph: 'nx.DiGraph',
    current_date: datetime,
    zombie_days: int = ZOMBIE_DAYS,
    weight_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Find zombie entities: dormant but weight preserved.

    Zombie = entity with:
    - Days since last activity > ZOMBIE_DAYS
    - Edge weight > weight_threshold (not decayed)
    - This is physically impossible for legitimate relationships

    Args:
        graph: NetworkX DiGraph to analyze
        current_date: Current date for age calculation
        zombie_days: Threshold for dormancy (default: 365)
        weight_threshold: Minimum weight to consider preserved

    Returns:
        List of zombie_receipt dicts

    Emits:
        zombie_receipt for each detected zombie
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for zombie detection")

    zombies = []

    for node in graph.nodes():
        # Check all edges from this node
        zombie_edges = []
        max_dormant_days = 0

        for successor in graph.successors(node):
            edge_data = graph[node][successor]
            last_seen = edge_data.get("last_seen_date")

            if last_seen is None:
                continue

            # Calculate dormancy
            if isinstance(last_seen, str):
                last_seen = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            elif isinstance(last_seen, datetime):
                pass
            else:
                continue

            days_dormant = (current_date - last_seen).days

            if days_dormant > zombie_days:
                weight = edge_data.get("weight", 0)
                if weight > weight_threshold:
                    zombie_edges.append({
                        "to": successor,
                        "weight": weight,
                        "days_dormant": days_dormant,
                    })
                    max_dormant_days = max(max_dormant_days, days_dormant)

        if zombie_edges:
            # Calculate total preserved weight
            preserved_weight = sum(e["weight"] for e in zombie_edges)

            # Identify linked domains
            linked_domains = set()
            for successor in graph.successors(node):
                domain = graph[node][successor].get("domain", "unknown")
                linked_domains.add(domain)

            zombie = emit_zombie_receipt(
                entity_id=node,
                days_dormant=max_dormant_days,
                preserved_weight=preserved_weight,
                domain=graph.nodes[node].get("domain", "unknown"),
                linked_domains=list(linked_domains),
            )
            zombies.append(zombie)

    return zombies


# ============================================================================
# CONTAGION PROPAGATION
# ============================================================================

def identify_shell_entities(
    super_graph: 'nx.DiGraph',
    min_domains: int = SHELL_PATTERN_THRESHOLD,
    require_resistance: bool = False,
) -> List[str]:
    """
    Identify potential shell entities in super-graph.

    Shell entity = node appearing in multiple domains (and optionally
    with high resistance on edges). These entities link domain entropy pools.

    Args:
        super_graph: Merged multi-domain graph
        min_domains: Minimum domains to qualify as shell
        require_resistance: If True, also requires resistance > threshold

    Returns:
        List of shell entity IDs
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for shell detection")

    shell_entities = []

    for node in super_graph.nodes():
        # Check node attributes first - may already be marked
        node_domains = super_graph.nodes[node].get("domains", [])
        if isinstance(node_domains, str):
            node_domains = [node_domains]

        # If node is explicitly marked as shared or potential shell
        if super_graph.nodes[node].get("is_shared") or \
           super_graph.nodes[node].get("is_potential_shell"):
            shell_entities.append(node)
            continue

        # Count domains this entity participates in from edges
        domains = set(node_domains)

        for successor in super_graph.successors(node):
            domain = super_graph[node][successor].get("domain", "unknown")
            if domain != "unknown":
                domains.add(domain)

        for predecessor in super_graph.predecessors(node):
            domain = super_graph[predecessor][node].get("domain", "unknown")
            if domain != "unknown":
                domains.add(domain)

        # Check if connected to enough domains
        if len(domains) >= min_domains:
            if require_resistance:
                # Also check for high resistance on any edge
                max_resistance = 0
                for successor in super_graph.successors(node):
                    resistance = super_graph[node][successor].get("resistance", 0)
                    max_resistance = max(max_resistance, resistance)

                if max_resistance > RESISTANCE_THRESHOLD:
                    shell_entities.append(node)
            else:
                # Just require multi-domain connection
                shell_entities.append(node)

    return shell_entities


def propagate_contagion(
    super_graph: 'nx.DiGraph',
    collapsed_domain: str,
    shell_entity: str,
) -> List[str]:
    """
    Propagate temporal signal through shell entity to linked domains.

    When a fraud ring collapses in one domain, the temporal rigidity
    propagates through shell entities to other domains.

    Physics: Shell entity = thermodynamic link between entropy pools.
    Collapse in one → pressure spike in linked domains.

    Args:
        super_graph: Merged multi-domain graph
        collapsed_domain: Domain where fraud was detected
        shell_entity: Shell entity linking domains

    Returns:
        List of flagged node IDs in other domains

    Emits:
        contagion_receipt for each propagation path
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for contagion analysis")

    flagged_nodes = []

    # Find all nodes connected to shell in other domains
    for successor in super_graph.successors(shell_entity):
        target_domain = super_graph[shell_entity][successor].get("domain", "unknown")

        if target_domain != collapsed_domain and target_domain != "unknown":
            # Propagation detected
            propagation_path = [collapsed_domain, shell_entity, successor, target_domain]

            emit_contagion_receipt(
                source_domain=collapsed_domain,
                target_domain=target_domain,
                shell_entity=shell_entity,
                propagation_path=propagation_path,
                pre_invoice_flag=True,  # Flagged before local evidence
            )

            flagged_nodes.append(successor)

    # Also check predecessors
    for predecessor in super_graph.predecessors(shell_entity):
        source_domain = super_graph[predecessor][shell_entity].get("domain", "unknown")

        if source_domain != collapsed_domain and source_domain != "unknown":
            propagation_path = [collapsed_domain, shell_entity, predecessor, source_domain]

            emit_contagion_receipt(
                source_domain=collapsed_domain,
                target_domain=source_domain,
                shell_entity=shell_entity,
                propagation_path=propagation_path,
                pre_invoice_flag=True,
            )

            flagged_nodes.append(predecessor)

    return flagged_nodes


def calculate_shared_entity_ratio(
    graph1: 'nx.DiGraph',
    graph2: 'nx.DiGraph',
) -> float:
    """
    Calculate ratio of shared entities between two domain graphs.

    Used to determine if cross-domain scan is warranted.
    Grok: "8% shell overlap between Defense/Medicaid empirically validated"

    Args:
        graph1: First domain graph
        graph2: Second domain graph

    Returns:
        Ratio of shared nodes (0.0 to 1.0)
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required")

    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())

    if not nodes1 or not nodes2:
        return 0.0

    shared = nodes1.intersection(nodes2)
    total = nodes1.union(nodes2)

    return len(shared) / len(total) if total else 0.0


# ============================================================================
# STOPRULES
# ============================================================================

def stoprule_negative_decay(initial: float, result: float) -> None:
    """
    Halt if decay formula increases weight (physics violation).
    """
    emit_receipt("anomaly", {
        "metric": "negative_decay",
        "initial_weight": initial,
        "result_weight": result,
        "action": "halt",
        "classification": "physics_violation",
        "physics": "decay_cannot_increase_weight",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Physics violation: decay increased weight from {initial} to {result}")


def stoprule_invalid_dates(days: int) -> None:
    """
    Halt if dates are invalid (current_date < last_seen_date).
    """
    emit_receipt("anomaly", {
        "metric": "invalid_dates",
        "days_since_last": days,
        "action": "halt",
        "classification": "violation",
        "reason": "current_date_before_last_seen",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Invalid dates: days_since_last = {days} (negative)")


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("# Gov-OS Temporal Physics Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Decay formula
    w30 = edge_weight_decay(1.0, 30, LAMBDA_NATURAL)
    print(f"# 30-day decay: 1.0 → {w30:.6f}", file=sys.stderr)
    assert 0.99 < w30 < 1.0, f"30-day decay should be ~0.995, got {w30}"

    # Test 2: Zero days = no decay
    w0 = edge_weight_decay(1.0, 0, LAMBDA_NATURAL)
    assert w0 == 1.0, f"Zero days should = 1.0, got {w0}"

    # Test 3: Large decay
    w3000 = edge_weight_decay(1.0, 3000, LAMBDA_NATURAL)
    print(f"# 3000-day decay: 1.0 → {w3000:.6f}", file=sys.stderr)
    assert w3000 < 0.1, f"3000-day decay should be < 0.1, got {w3000}"

    # Test 4: Resistance - normal
    r_normal = resistance_to_decay(0.5, 0.5)
    assert r_normal == 0.0, f"Equal weights should = 0 resistance, got {r_normal}"

    # Test 5: Resistance - anomaly
    r_anomaly = resistance_to_decay(0.5, 1.0)
    assert r_anomaly == 1.0, f"Double weight should = 1.0 resistance, got {r_anomaly}"

    # Test 6: Resistance - floor at zero
    r_floor = resistance_to_decay(0.5, 0.3)
    assert r_floor == 0.0, f"Lower weight should = 0 resistance, got {r_floor}"

    print("# PASS: temporal module self-test", file=sys.stderr)
