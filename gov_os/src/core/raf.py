"""
Gov-OS Core RAF - Universal Reflexively Autocatalytic and Food-set Detection

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

RAF Theory: Patterns are not programmed—they crystallize from receipt clusters
that achieve self-referencing closure. Polynomial-time detection on directed graphs.

Domain-agnostic RAF: nodes=vendors/providers, edges=payments/referrals.
The ONLY domain-specific inputs are node_key and edge_key from config.yaml.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .constants import (
    RAF_MIN_CYCLE_LENGTH,
    RAF_MAX_CYCLE_LENGTH,
    RAF_SELF_PREDICTION_THRESHOLD,
    PATTERN_COHERENCE_MIN,
    ENTROPY_GAP_MIN,
    N_CRITICAL_MAX,
    DISCLAIMER,
    TENANT_ID,
)
from .utils import dual_hash
from .receipt import emit_L1, emit_L2


@dataclass
class EmergentPattern:
    """A fraud pattern that emerged from data without hardcoding."""
    pattern_id: str
    description: str
    fingerprint: Dict[str, Any]
    entropy_gap: float
    N_observed: int
    N_critical: int
    coherence: float
    RAF_closure: bool
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "fingerprint": self.fingerprint,
            "entropy_gap": self.entropy_gap,
            "N_observed": self.N_observed,
            "N_critical": self.N_critical,
            "coherence": self.coherence,
            "RAF_closure": self.RAF_closure,
        }


def build_graph(
    receipts: List[Dict[str, Any]],
    node_key: str = "vendor_id",
    edge_key: str = "payment_to",
) -> Any:
    """
    Build directed graph from receipts. Domain supplies key names.

    Args:
        receipts: List of receipts
        node_key: Field name for node identifier (e.g., "vendor_id", "provider_npi")
        edge_key: Field name for edge target (e.g., "payment_to", "referral_to")

    Returns:
        NetworkX DiGraph (or dict if networkx not available)
    """
    if HAS_NETWORKX:
        graph = nx.DiGraph()

        for r in receipts:
            source = r.get(node_key)
            target = r.get(edge_key)

            if source:
                graph.add_node(source, **{k: v for k, v in r.items() if k not in [node_key, edge_key]})

            if source and target:
                graph.add_edge(source, target, receipt=r)

        return graph
    else:
        # Fallback dict-based graph
        graph = {"nodes": set(), "edges": defaultdict(list)}
        for r in receipts:
            source = r.get(node_key)
            target = r.get(edge_key)
            if source:
                graph["nodes"].add(source)
            if source and target:
                graph["edges"][source].append(target)
                graph["nodes"].add(target)
        return graph


def find_raf_cycles(graph: Any) -> List[List[str]]:
    """
    Find all simple cycles. Emit raf_cycle_receipt for each.

    Args:
        graph: NetworkX DiGraph or dict-based graph

    Returns:
        List of cycles (each cycle is a list of node IDs)
    """
    cycles = []

    if HAS_NETWORKX and isinstance(graph, nx.DiGraph):
        try:
            for cycle in nx.simple_cycles(graph):
                if RAF_MIN_CYCLE_LENGTH <= len(cycle) <= RAF_MAX_CYCLE_LENGTH:
                    cycles.append(cycle)

                    # Emit RAF cycle receipt
                    emit_L1("raf_cycle_receipt", {
                        "nodes": cycle,
                        "cycle_length": len(cycle),
                        "graph_hash": dual_hash(str(sorted(graph.nodes())))[:16],
                        "tenant_id": TENANT_ID,
                        "simulation_flag": DISCLAIMER,
                    })
        except Exception:
            pass  # Graph may have issues
    else:
        # Simple DFS-based cycle detection for dict graph
        if isinstance(graph, dict):
            visited: Set[str] = set()
            rec_stack: Set[str] = set()
            path: List[str] = []

            def dfs(node: str) -> None:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in graph.get("edges", {}).get(node, []):
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Found cycle
                        try:
                            idx = path.index(neighbor)
                            cycle = path[idx:]
                            if RAF_MIN_CYCLE_LENGTH <= len(cycle) <= RAF_MAX_CYCLE_LENGTH:
                                cycles.append(cycle)
                        except ValueError:
                            pass

                path.pop()
                rec_stack.remove(node)

            for node in graph.get("nodes", set()):
                if node not in visited:
                    dfs(node)

    return cycles


def is_autocatalytic(
    pattern: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> bool:
    """
    True if pattern achieves self-reference.
    Requires 3 consecutive self-predictions at RAF_SELF_PREDICTION_THRESHOLD.

    Args:
        pattern: Pattern receipts
        history: Historical receipts

    Returns:
        True if autocatalytic
    """
    if len(pattern) < 3:
        return False

    # Extract pattern fingerprint
    fingerprint = _extract_fingerprint(pattern)

    # Test self-prediction on sliding window
    consecutive_predictions = 0
    required_consecutive = 3

    for i in range(len(history) - len(pattern)):
        window = history[i:i + len(pattern)]
        score = coherence_score(fingerprint, window)

        if score >= RAF_SELF_PREDICTION_THRESHOLD:
            consecutive_predictions += 1
            if consecutive_predictions >= required_consecutive:
                return True
        else:
            consecutive_predictions = 0

    return False


def coherence_score(
    fingerprint: Dict[str, Any],
    receipts: List[Dict[str, Any]],
) -> float:
    """
    Coherence score 0-1 based on causal self-references.

    Args:
        fingerprint: Pattern fingerprint
        receipts: Receipts to compare

    Returns:
        Coherence score 0-1
    """
    if not receipts or not fingerprint:
        return 0.5

    scores = []

    # Check vendor uniqueness match
    vendors = [r.get("vendor", "") for r in receipts if r.get("vendor")]
    if vendors and "vendor_uniqueness" in fingerprint:
        new_uniqueness = len(set(vendors)) / len(vendors)
        match = 1.0 - abs(new_uniqueness - fingerprint["vendor_uniqueness"])
        scores.append(match)

    # Check amount distribution match
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    if amounts and "amount_cv" in fingerprint:
        new_cv = np.std(amounts) / (np.mean(amounts) + 1e-10)
        cv_match = 1.0 - min(1.0, abs(new_cv - fingerprint["amount_cv"]) / (fingerprint["amount_cv"] + 0.1))
        scores.append(cv_match)

    # Check lineage ratio match
    with_lineage = sum(1 for r in receipts if r.get("decision_lineage"))
    if receipts and "lineage_ratio" in fingerprint:
        new_ratio = with_lineage / len(receipts)
        lineage_match = 1.0 - abs(new_ratio - fingerprint["lineage_ratio"])
        scores.append(lineage_match)

    return sum(scores) / len(scores) if scores else 0.5


def pattern_birth(pattern: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Emit pattern_birth_receipt when autocatalysis threshold crossed.

    Args:
        pattern: Pattern receipts

    Returns:
        Birth receipt
    """
    fingerprint = _extract_fingerprint(pattern)

    return emit_L2("pattern_birth", {
        "pattern_id": dual_hash(str(fingerprint))[:16],
        "fingerprint": fingerprint,
        "receipt_count": len(pattern),
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })


def pattern_death(pattern: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Emit pattern_death_receipt when coherence lost → SUPERPOSITION.

    Args:
        pattern: Pattern receipts

    Returns:
        Death receipt
    """
    fingerprint = _extract_fingerprint(pattern)

    return emit_L2("pattern_death", {
        "pattern_id": dual_hash(str(fingerprint))[:16],
        "reason": "coherence_lost",
        "state": "SUPERPOSITION",
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })


def crystallize(
    receipts: List[Dict[str, Any]],
    domain: str,
) -> List[EmergentPattern]:
    """
    Identify emerging patterns without hardcoding.

    Args:
        receipts: Receipts to analyze
        domain: Domain identifier

    Returns:
        List of EmergentPatterns
    """
    patterns = []

    if not receipts:
        return patterns

    # Compute entropy gap
    entropy_gap = _compute_entropy_gap(receipts)

    if entropy_gap < ENTROPY_GAP_MIN:
        return patterns

    # Calculate N_critical
    N_critical = _calculate_N_critical(entropy_gap)

    if len(receipts) < N_critical:
        return patterns

    # Extract fingerprint
    fingerprint = _extract_fingerprint(receipts)

    # Calculate coherence
    coherence = coherence_score(fingerprint, receipts)

    if coherence < PATTERN_COHERENCE_MIN:
        return patterns

    # Check RAF closure
    RAF_closure = _detect_RAF_closure(receipts)

    # Create emergent pattern
    pattern = EmergentPattern(
        pattern_id=dual_hash(str(fingerprint))[:16],
        description=_generate_description(fingerprint),
        fingerprint=fingerprint,
        entropy_gap=entropy_gap,
        N_observed=len(receipts),
        N_critical=N_critical,
        coherence=coherence,
        RAF_closure=RAF_closure,
    )

    patterns.append(pattern)

    # Emit receipt
    emit_L1("autocatalytic", {
        "pattern_emerged": pattern.description,
        "pattern_id": pattern.pattern_id,
        "N_receipts": pattern.N_observed,
        "N_critical": pattern.N_critical,
        "entropy_gap": round(pattern.entropy_gap, 4),
        "RAF_closure": pattern.RAF_closure,
        "coherence": round(pattern.coherence, 4),
        "domain": domain,
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })

    return patterns


def detect_without_hardcode(
    receipts: List[Dict[str, Any]],
    domain: str,
    node_key: str = "vendor_id",
    edge_key: str = "payment_to",
) -> List[Dict[str, Any]]:
    """
    Full RAF detection: build graph, find cycles, check autocatalysis.

    Args:
        receipts: Receipts to analyze
        domain: Domain identifier
        node_key: Field name for node identifier
        edge_key: Field name for edge target

    Returns:
        List of detected anomalies
    """
    detections = []

    # Build graph
    graph = build_graph(receipts, node_key, edge_key)

    # Find RAF cycles
    cycles = find_raf_cycles(graph)

    for cycle in cycles:
        detections.append({
            "anomaly_type": "raf_cycle",
            "cycle_nodes": cycle,
            "cycle_length": len(cycle),
            "domain": domain,
            "confidence": 0.85,
        })

    # Crystallize patterns
    patterns = crystallize(receipts, domain)

    for pattern in patterns:
        detections.append({
            "anomaly_type": "emergent_pattern",
            "pattern_id": pattern.pattern_id,
            "description": pattern.description,
            "coherence": pattern.coherence,
            "RAF_closure": pattern.RAF_closure,
            "confidence": pattern.coherence,
        })

    return detections


def _extract_fingerprint(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract characteristic fingerprint from receipt cluster."""
    fingerprint: Dict[str, Any] = {}

    # Amount distribution
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    if amounts:
        fingerprint["amount_mean"] = float(np.mean(amounts))
        fingerprint["amount_std"] = float(np.std(amounts))
        fingerprint["amount_cv"] = fingerprint["amount_std"] / (fingerprint["amount_mean"] + 1e-10)

    # Vendor distribution
    vendors = [r.get("vendor", "") for r in receipts if r.get("vendor")]
    if vendors:
        fingerprint["vendor_uniqueness"] = len(set(vendors)) / len(vendors)
        fingerprint["vendor_count"] = len(set(vendors))

    # Lineage characteristics
    with_lineage = sum(1 for r in receipts if r.get("decision_lineage"))
    fingerprint["lineage_ratio"] = with_lineage / len(receipts) if receipts else 0

    # Receipt type distribution
    types = Counter(r.get("receipt_type", "") for r in receipts)
    fingerprint["type_entropy"] = _entropy_from_counter(types)

    return fingerprint


def _entropy_from_counter(counter: Counter) -> float:
    """Calculate entropy from Counter object."""
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _compute_entropy_gap(receipts: List[Dict[str, Any]]) -> float:
    """Calculate entropy gap from receipts."""
    if len(receipts) < 10:
        return ENTROPY_GAP_MIN

    # Estimate from field variance
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    vendors = [r.get("vendor", "") for r in receipts if r.get("vendor")]

    vendor_uniqueness = len(set(vendors)) / len(vendors) if vendors else 0
    amount_cv = np.std(amounts) / (np.mean(amounts) + 1e-10) if amounts else 0

    return max(ENTROPY_GAP_MIN, vendor_uniqueness * 0.3 + min(1.0, amount_cv) * 0.2)


def _calculate_N_critical(entropy_gap: float) -> int:
    """Calculate N_critical from entropy gap."""
    if entropy_gap <= 0:
        return N_CRITICAL_MAX

    # N_critical ≈ log₂(ΔH⁻¹) × (H_legit / ΔH)
    H_legit = 3.0  # Typical legitimate entropy
    N = math.log2(1.0 / entropy_gap) * (H_legit / entropy_gap)

    return int(min(N_CRITICAL_MAX, max(1, N)))


def _detect_RAF_closure(receipts: List[Dict[str, Any]]) -> bool:
    """Test if receipts form self-referencing cluster."""
    if len(receipts) < 3:
        return False

    # Check lineage cycles
    ids = {r.get("payload_hash") for r in receipts if r.get("payload_hash")}
    lineage_refs = set()
    for r in receipts:
        for parent in r.get("decision_lineage", []):
            if parent in ids:
                lineage_refs.add(parent)

    # Check vendor repetition
    vendors = [r.get("vendor") for r in receipts if r.get("vendor")]
    vendor_repetition = len(vendors) - len(set(vendors)) if vendors else 0

    closure_score = (
        (len(lineage_refs) / len(ids) if ids else 0) * 0.5 +
        (vendor_repetition / len(receipts)) * 0.5
    )

    return closure_score > 0.3


def _generate_description(fingerprint: Dict[str, Any]) -> str:
    """Generate human-readable description from fingerprint."""
    parts = []

    if fingerprint.get("vendor_uniqueness", 0) > 0.8:
        parts.append("high-vendor-churn")
    if fingerprint.get("amount_cv", 0) > 1.0:
        parts.append("high-amount-variance")
    if fingerprint.get("lineage_ratio", 0) < 0.3:
        parts.append("orphan-transactions")
    if fingerprint.get("type_entropy", 0) > 2.0:
        parts.append("mixed-transaction-types")

    return "-".join(parts) if parts else "emergent-cluster"
