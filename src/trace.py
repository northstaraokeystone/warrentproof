"""
WarrantProof Trace Module - Decision Lineage Reconstruction

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module reconstructs decision lineage from receipt chains.
Who approved what, when, why - traced through warrant lineage.

Key Insight: Every warrant references its parent decision. By following
parent pointers, we reconstruct the full decision chain from initial
requirement to final delivery.

SLOs:
- Lineage construction <= 500ms per receipt
- Gap detection <= 1s
- Support chains up to 100 levels deep
"""

from collections import defaultdict
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    dual_hash,
    emit_receipt,
)


# === CORE FUNCTIONS ===

def build_lineage_graph(receipts: list, target_receipt_id: Optional[str] = None) -> dict:
    """
    Construct DAG of all parent/child relationships.
    Per spec: Return graph as adjacency list.

    Args:
        receipts: List of receipts to build graph from
        target_receipt_id: Optional target to focus on

    Returns:
        Graph as adjacency list with metadata
    """
    # Build index
    receipt_index = {}
    for receipt in receipts:
        receipt_id = receipt.get("payload_hash", dual_hash(str(receipt)))
        receipt_index[receipt_id] = receipt

    # Build adjacency lists
    parents = defaultdict(list)  # child -> [parents]
    children = defaultdict(list)  # parent -> [children]

    for receipt in receipts:
        receipt_id = receipt.get("payload_hash", dual_hash(str(receipt)))
        lineage = receipt.get("decision_lineage", [])

        for parent_id in lineage:
            parents[receipt_id].append(parent_id)
            children[parent_id].append(receipt_id)

    return {
        "parents": dict(parents),
        "children": dict(children),
        "receipt_index": receipt_index,
        "node_count": len(receipt_index),
        "edge_count": sum(len(v) for v in parents.values()),
    }


def find_root_decision(receipt_id: str, graph: dict) -> str:
    """
    Trace back to initial requirement receipt.
    Per spec: Find the root of the decision tree.

    Args:
        receipt_id: Receipt to trace from
        graph: Lineage graph from build_lineage_graph

    Returns:
        Root receipt ID
    """
    current = receipt_id
    visited = set()
    depth = 0
    max_depth = 100  # Per spec: support chains up to 100 levels

    while depth < max_depth:
        if current in visited:
            break  # Cycle detected
        visited.add(current)

        parent_ids = graph.get("parents", {}).get(current, [])
        if not parent_ids:
            break  # Found root

        current = parent_ids[0]  # Follow first parent
        depth += 1

    return current


def find_approval_chain(receipt_id: str, graph: dict) -> list[dict]:
    """
    Return ordered list of approvers from root to leaf.
    Per spec: Trace approval chain for accountability.

    Args:
        receipt_id: Receipt to trace
        graph: Lineage graph

    Returns:
        List of approver records from root to target
    """
    # Find root first
    root_id = find_root_decision(receipt_id, graph)

    # Traverse from root to target
    chain = []
    current = root_id
    visited = set()
    receipt_index = graph.get("receipt_index", {})

    while current and current not in visited:
        visited.add(current)
        receipt = receipt_index.get(current, {})

        if receipt:
            chain.append({
                "receipt_id": current,
                "approver": receipt.get("approver", receipt.get("inspector", "unknown")),
                "receipt_type": receipt.get("receipt_type"),
                "ts": receipt.get("ts"),
                "branch": receipt.get("branch"),
            })

        if current == receipt_id:
            break

        # Find path to target
        children = graph.get("children", {}).get(current, [])
        if not children:
            break

        # Simple BFS to find path to target
        next_current = None
        for child in children:
            if child == receipt_id or _has_path_to(child, receipt_id, graph, visited.copy()):
                next_current = child
                break

        current = next_current

    return chain


def _has_path_to(start: str, target: str, graph: dict, visited: set) -> bool:
    """Check if path exists from start to target."""
    if start == target:
        return True
    if start in visited:
        return False

    visited.add(start)
    children = graph.get("children", {}).get(start, [])

    for child in children:
        if _has_path_to(child, target, graph, visited):
            return True

    return False


def gap_detection(graph: dict) -> list[dict]:
    """
    Find missing links in approval chain.
    Per spec: Detect gaps for audit completeness.

    Args:
        graph: Lineage graph

    Returns:
        List of gap descriptions
    """
    gaps = []
    receipt_index = graph.get("receipt_index", {})
    parents = graph.get("parents", {})

    for receipt_id, parent_ids in parents.items():
        for parent_id in parent_ids:
            if parent_id not in receipt_index:
                receipt = receipt_index.get(receipt_id, {})
                gaps.append({
                    "gap_type": "missing_parent",
                    "child_receipt_id": receipt_id,
                    "missing_parent_id": parent_id,
                    "child_type": receipt.get("receipt_type"),
                    "child_branch": receipt.get("branch"),
                    "description": f"Receipt {receipt_id[:16]}... references missing parent {parent_id[:16]}...",
                })

    # Check for orphan receipts (no parents, not roots)
    children_set = set()
    for child_list in graph.get("children", {}).values():
        children_set.update(child_list)

    for receipt_id in receipt_index:
        if receipt_id not in parents and receipt_id in children_set:
            # This is a root, which is fine
            pass
        elif receipt_id not in parents and receipt_id not in children_set:
            # Orphan receipt
            receipt = receipt_index.get(receipt_id, {})
            if receipt.get("receipt_type") not in ["anchor", "ingest", "test"]:
                gaps.append({
                    "gap_type": "orphan_receipt",
                    "receipt_id": receipt_id,
                    "receipt_type": receipt.get("receipt_type"),
                    "description": f"Receipt {receipt_id[:16]}... has no lineage connections",
                })

    return gaps


def visualize_lineage(graph: dict, target_receipt_id: Optional[str] = None, max_depth: int = 10) -> str:
    """
    Generate ASCII tree diagram of decision chain.
    Per spec: Human-readable lineage visualization.

    Args:
        graph: Lineage graph
        target_receipt_id: Optional focus on specific receipt
        max_depth: Maximum depth to render

    Returns:
        ASCII tree string
    """
    lines = []
    lines.append("# Decision Lineage Tree")
    lines.append(f"# {DISCLAIMER}")
    lines.append("")

    receipt_index = graph.get("receipt_index", {})

    if target_receipt_id:
        root_id = find_root_decision(target_receipt_id, graph)
        _render_subtree(root_id, graph, receipt_index, lines, "", 0, max_depth, set())
    else:
        # Find all roots
        parents = graph.get("parents", {})
        all_nodes = set(receipt_index.keys())
        child_nodes = set(parents.keys())
        roots = all_nodes - child_nodes

        for root in list(roots)[:5]:  # Limit to 5 roots
            _render_subtree(root, graph, receipt_index, lines, "", 0, max_depth, set())
            lines.append("")

    return "\n".join(lines)


def _render_subtree(node_id: str, graph: dict, receipt_index: dict,
                    lines: list, prefix: str, depth: int, max_depth: int, visited: set):
    """Render a subtree recursively."""
    if depth >= max_depth or node_id in visited:
        return

    visited.add(node_id)
    receipt = receipt_index.get(node_id, {})

    # Format node
    receipt_type = receipt.get("receipt_type", "unknown")
    approver = receipt.get("approver", receipt.get("inspector", ""))[:12]
    branch = receipt.get("branch", "")[:6]
    node_str = f"[{receipt_type}] {branch} {approver}"

    lines.append(f"{prefix}├── {node_str}")

    # Render children
    children = graph.get("children", {}).get(node_id, [])
    for i, child_id in enumerate(children[:5]):  # Limit children
        child_prefix = prefix + "│   " if i < len(children) - 1 else prefix + "    "
        _render_subtree(child_id, graph, receipt_index, lines, child_prefix, depth + 1, max_depth, visited)


# === LINEAGE RECEIPT ===

def emit_lineage_receipt(target_receipt_id: str, graph: dict) -> dict:
    """
    Emit lineage_receipt summarizing trace results.

    Args:
        target_receipt_id: Receipt that was traced
        graph: Lineage graph

    Returns:
        lineage_receipt dict
    """
    root_id = find_root_decision(target_receipt_id, graph)
    approval_chain = find_approval_chain(target_receipt_id, graph)
    gaps = gap_detection(graph)

    return emit_receipt("lineage", {
        "tenant_id": TENANT_ID,
        "target_receipt": target_receipt_id,
        "root_decision": root_id,
        "approval_chain": [a["approver"] for a in approval_chain],
        "depth": len(approval_chain),
        "gaps_detected": gaps[:10],  # First 10 gaps
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === COMPLETENESS CHECK ===

def lineage_completeness(receipts: list) -> dict:
    """
    Calculate lineage completeness score.
    Per spec: Lineage completeness >= 95%.

    Args:
        receipts: All receipts to check

    Returns:
        Completeness metrics
    """
    if not receipts:
        return {"completeness": 1.0, "total": 0, "complete": 0, "incomplete": 0}

    graph = build_lineage_graph(receipts)
    gaps = gap_detection(graph)

    # Receipts that should have lineage
    should_have_lineage = [
        r for r in receipts
        if r.get("receipt_type") in ["warrant", "quality_attestation", "milestone", "cost_variance"]
    ]

    # Receipts with valid lineage
    complete_count = 0
    for receipt in should_have_lineage:
        lineage = receipt.get("decision_lineage", [])
        if lineage or receipt.get("receipt_type") == "warrant":  # Root warrants OK
            complete_count += 1

    total = len(should_have_lineage) if should_have_lineage else 1
    completeness = complete_count / total

    return {
        "completeness": completeness,
        "total": len(should_have_lineage),
        "complete": complete_count,
        "incomplete": len(should_have_lineage) - complete_count,
        "gaps": len(gaps),
        "simulation_flag": DISCLAIMER,
    }


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import time

    print(f"# WarrantProof Trace Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Create test receipts with lineage
    test_receipts = [
        {"payload_hash": "root_001", "receipt_type": "warrant", "approver": "SECNAV",
         "branch": "Navy", "decision_lineage": []},
        {"payload_hash": "child_001", "receipt_type": "milestone", "approver": "PEO_Ships",
         "branch": "Navy", "decision_lineage": ["root_001"]},
        {"payload_hash": "child_002", "receipt_type": "cost_variance", "approver": "PM_CVN",
         "branch": "Navy", "decision_lineage": ["child_001"]},
        {"payload_hash": "leaf_001", "receipt_type": "quality_attestation", "inspector": "QA_001",
         "branch": "Navy", "decision_lineage": ["child_002"]},
    ]

    # Test graph building
    t0 = time.time()
    graph = build_lineage_graph(test_receipts)
    build_time = (time.time() - t0) * 1000
    assert graph["node_count"] == 4
    assert graph["edge_count"] == 3

    # Test root finding
    t0 = time.time()
    root = find_root_decision("leaf_001", graph)
    trace_time = (time.time() - t0) * 1000
    assert root == "root_001"
    assert trace_time <= 500, f"Trace time {trace_time}ms > 500ms SLO"

    # Test approval chain
    chain = find_approval_chain("leaf_001", graph)
    assert len(chain) == 4
    assert chain[0]["approver"] == "SECNAV"

    # Test gap detection
    t0 = time.time()
    gaps = gap_detection(graph)
    gap_time = (time.time() - t0) * 1000
    assert gap_time <= 1000, f"Gap detection {gap_time}ms > 1000ms SLO"

    # Test visualization
    viz = visualize_lineage(graph, "leaf_001")
    assert "Decision Lineage Tree" in viz

    # Test completeness
    completeness = lineage_completeness(test_receipts)
    assert completeness["completeness"] >= 0.95

    print(f"# PASS: trace module self-test (build: {build_time:.1f}ms, trace: {trace_time:.1f}ms)",
          file=sys.stderr)
