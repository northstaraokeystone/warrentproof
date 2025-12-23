"""
WarrantProof RAF Module - Reflexively Autocatalytic Food-generated Network Detection

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements RAF (Reflexively Autocatalytic Food-generated) network
detection for corruption metabolism identification. Corruption is not viral
(external agent) - it's autocatalytic (self-sustaining metabolic network).

Key Insight:
- Molecules: Contractors, sub-contractors, procurement officers, shell companies
- Reactions: Contract awards, modifications, invoicing, hiring
- Catalysts: Bribes, "revolving door" jobs, insider info
- Food: Congressional appropriations (the budget)

RAF Property: Reaction outputs become catalysts for other reactions.
Self-sustaining closed loop (A->B->C->A).

Detection: Cycles of length 3-5 in transaction graph where edges are
transactions AND catalytic links (shared addresses, board members, IP proximity).

OMEGA Citation:
"Systemic corruption is not viral; it is Autocatalytic. It is a
self-sustaining metabolic network."
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .core import (
    TENANT_ID,
    DISCLAIMER,
    RAF_CYCLE_MIN_LENGTH,
    RAF_CYCLE_MAX_LENGTH,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class CatalyticLink:
    """Non-financial link that enables catalysis."""
    source: str
    target: str
    link_type: str  # shared_address, board_overlap, ip_proximity, revolving_door
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)


def build_transaction_graph(transactions: list) -> 'nx.DiGraph':
    """
    NetworkX directed graph. Nodes = DUNS (entities). Edges = transactions.

    Args:
        transactions: List of transaction dicts with source/target entities

    Returns:
        NetworkX DiGraph
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for RAF analysis. Install: pip install networkx")

    G = nx.DiGraph()

    for tx in transactions:
        # Extract source and target entities
        source = tx.get("source_duns") or tx.get("vendor") or tx.get("from")
        target = tx.get("target_duns") or tx.get("recipient") or tx.get("to")

        if not source or not target:
            continue

        # Add nodes with attributes
        if not G.has_node(source):
            G.add_node(source, entity_type=tx.get("source_type", "unknown"))
        if not G.has_node(target):
            G.add_node(target, entity_type=tx.get("target_type", "unknown"))

        # Add edge with transaction metadata
        if G.has_edge(source, target):
            # Update existing edge
            G[source][target]["weight"] += tx.get("amount_usd", 0)
            G[source][target]["transaction_count"] += 1
        else:
            G.add_edge(
                source, target,
                weight=tx.get("amount_usd", 0),
                transaction_count=1,
                edge_type="financial"
            )

    return G


def add_catalytic_links(
    graph: 'nx.DiGraph',
    entities: list,
    link_types: list = None
) -> 'nx.DiGraph':
    """
    Add non-financial edges: shared addresses, board members, IP proximity.
    These are catalytic links that enable reactions.

    Args:
        graph: Existing transaction graph
        entities: List of entity dicts with metadata
        link_types: Types of links to detect (default: all)

    Returns:
        Graph with catalytic links added
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for RAF analysis")

    if link_types is None:
        link_types = ["shared_address", "board_overlap", "ip_proximity", "revolving_door"]

    # Index entities by various attributes
    by_address = defaultdict(list)
    by_board_member = defaultdict(list)
    by_ip_prefix = defaultdict(list)
    by_former_employee = defaultdict(list)

    for entity in entities:
        entity_id = entity.get("duns") or entity.get("id")
        if not entity_id:
            continue

        # Index by address (ZIP code prefix)
        address = entity.get("address", "")
        if address:
            zip_prefix = address[-5:][:3] if len(address) >= 5 else ""
            if zip_prefix:
                by_address[zip_prefix].append(entity_id)

        # Index by board members
        for member in entity.get("board_members", []):
            by_board_member[member].append(entity_id)

        # Index by IP prefix
        ip = entity.get("ip_address", "")
        if ip:
            ip_prefix = ".".join(ip.split(".")[:3])  # /24 prefix
            by_ip_prefix[ip_prefix].append(entity_id)

        # Index by former employees (revolving door)
        for employee in entity.get("former_gov_employees", []):
            by_former_employee[employee].append(entity_id)

    # Add catalytic links
    catalytic_links = []

    if "shared_address" in link_types:
        for zip_prefix, entity_ids in by_address.items():
            for i, e1 in enumerate(entity_ids):
                for e2 in entity_ids[i+1:]:
                    if graph.has_node(e1) and graph.has_node(e2):
                        if not graph.has_edge(e1, e2):
                            graph.add_edge(e1, e2, edge_type="catalytic", link_type="shared_address")
                        if not graph.has_edge(e2, e1):
                            graph.add_edge(e2, e1, edge_type="catalytic", link_type="shared_address")
                        catalytic_links.append(CatalyticLink(e1, e2, "shared_address", 0.6))

    if "board_overlap" in link_types:
        for member, entity_ids in by_board_member.items():
            for i, e1 in enumerate(entity_ids):
                for e2 in entity_ids[i+1:]:
                    if graph.has_node(e1) and graph.has_node(e2):
                        if not graph.has_edge(e1, e2):
                            graph.add_edge(e1, e2, edge_type="catalytic", link_type="board_overlap")
                        if not graph.has_edge(e2, e1):
                            graph.add_edge(e2, e1, edge_type="catalytic", link_type="board_overlap")
                        catalytic_links.append(CatalyticLink(e1, e2, "board_overlap", 0.8))

    if "ip_proximity" in link_types:
        for ip_prefix, entity_ids in by_ip_prefix.items():
            for i, e1 in enumerate(entity_ids):
                for e2 in entity_ids[i+1:]:
                    if graph.has_node(e1) and graph.has_node(e2):
                        if not graph.has_edge(e1, e2):
                            graph.add_edge(e1, e2, edge_type="catalytic", link_type="ip_proximity")
                        if not graph.has_edge(e2, e1):
                            graph.add_edge(e2, e1, edge_type="catalytic", link_type="ip_proximity")
                        catalytic_links.append(CatalyticLink(e1, e2, "ip_proximity", 0.4))

    if "revolving_door" in link_types:
        for employee, entity_ids in by_former_employee.items():
            for i, e1 in enumerate(entity_ids):
                for e2 in entity_ids[i+1:]:
                    if graph.has_node(e1) and graph.has_node(e2):
                        if not graph.has_edge(e1, e2):
                            graph.add_edge(e1, e2, edge_type="catalytic", link_type="revolving_door")
                        if not graph.has_edge(e2, e1):
                            graph.add_edge(e2, e1, edge_type="catalytic", link_type="revolving_door")
                        catalytic_links.append(CatalyticLink(e1, e2, "revolving_door", 0.9))

    # Store catalytic links as graph attribute
    graph.graph["catalytic_links"] = catalytic_links

    return graph


def detect_cycles(
    graph: 'nx.DiGraph',
    min_length: int = RAF_CYCLE_MIN_LENGTH,
    max_length: int = RAF_CYCLE_MAX_LENGTH
) -> List[List[str]]:
    """
    Find all simple cycles of length [min_length, max_length].

    Args:
        graph: NetworkX DiGraph
        min_length: Minimum cycle length (default 3)
        max_length: Maximum cycle length (default 5)

    Returns:
        List of cycle node sequences
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for RAF analysis")

    cycles = []

    # Use NetworkX simple_cycles with length filtering
    try:
        for cycle in nx.simple_cycles(graph):
            cycle_length = len(cycle)
            if min_length <= cycle_length <= max_length:
                # Add closing node to make cycle explicit
                cycles.append(cycle + [cycle[0]])
    except Exception:
        # Fallback: DFS-based cycle detection for smaller graphs
        cycles = _dfs_find_cycles(graph, min_length, max_length)

    return cycles


def _dfs_find_cycles(
    graph: 'nx.DiGraph',
    min_length: int,
    max_length: int
) -> List[List[str]]:
    """Fallback DFS-based cycle detection."""
    cycles = []
    visited = set()

    def dfs(node: str, path: List[str], depth: int):
        if depth > max_length:
            return

        for neighbor in graph.successors(node):
            if neighbor == path[0] and depth >= min_length:
                # Found cycle back to start
                cycles.append(path + [path[0]])
            elif neighbor not in path:
                dfs(neighbor, path + [neighbor], depth + 1)

    for node in graph.nodes():
        if node not in visited:
            dfs(node, [node], 1)
            visited.add(node)

    return cycles


def identify_keystone_species(
    graph: 'nx.DiGraph',
    cycles: List[List[str]]
) -> List[str]:
    """
    Find nodes appearing in most cycles. These are "keystone species"
    whose removal collapses the RAF.

    Args:
        graph: NetworkX DiGraph
        cycles: List of detected cycles

    Returns:
        List of keystone species (DUNS IDs) sorted by cycle participation
    """
    if not cycles:
        return []

    # Count cycle participation
    participation = defaultdict(int)
    for cycle in cycles:
        for node in cycle[:-1]:  # Exclude closing node
            participation[node] += 1

    # Identify keystones (top 10% of participation)
    if not participation:
        return []

    sorted_nodes = sorted(participation.items(), key=lambda x: x[1], reverse=True)
    threshold_count = len(sorted_nodes) // 10 + 1  # Top 10%

    keystones = [node for node, count in sorted_nodes[:threshold_count]]

    return keystones


def simulate_disruption(
    graph: 'nx.DiGraph',
    remove_nodes: List[str],
    cycles: List[List[str]]
) -> dict:
    """
    Remove keystone nodes. Measure cascade failure (how many cycles collapse).

    Args:
        graph: NetworkX DiGraph
        remove_nodes: Nodes to remove
        cycles: Original cycles

    Returns:
        impact_metrics dict
    """
    if not cycles:
        return {
            "cycles_before": 0,
            "cycles_after": 0,
            "collapse_ratio": 0.0,
            "nodes_removed": len(remove_nodes),
        }

    # Count cycles affected by removal
    affected_cycles = 0
    for cycle in cycles:
        cycle_nodes = set(cycle[:-1])
        if cycle_nodes.intersection(remove_nodes):
            affected_cycles += 1

    collapse_ratio = affected_cycles / len(cycles) if cycles else 0.0

    # Create graph copy without keystones
    if HAS_NETWORKX:
        G_disrupted = graph.copy()
        for node in remove_nodes:
            if G_disrupted.has_node(node):
                G_disrupted.remove_node(node)

        # Re-detect cycles
        remaining_cycles = detect_cycles(G_disrupted)
    else:
        remaining_cycles = []

    return {
        "cycles_before": len(cycles),
        "cycles_after": len(remaining_cycles),
        "collapse_ratio": collapse_ratio,
        "nodes_removed": len(remove_nodes),
        "cascade_failure_potential": collapse_ratio,
    }


def raf_closure_test(graph: 'nx.DiGraph', cycles: List[List[str]]) -> bool:
    """
    Test if any cycles exist (RAF closure property).

    Args:
        graph: NetworkX DiGraph
        cycles: Detected cycles

    Returns:
        True if RAF closure exists (corruption is self-sustaining)
    """
    return len(cycles) > 0


def emit_raf_receipt(
    graph: 'nx.DiGraph',
    cycles: List[List[str]],
    keystone_species: List[str]
) -> dict:
    """
    Emit raf_receipt documenting network analysis.

    Args:
        graph: Analyzed graph
        cycles: Detected cycles
        keystone_species: Identified keystones

    Returns:
        raf_receipt dict
    """
    catalytic_links = graph.graph.get("catalytic_links", []) if HAS_NETWORKX else []

    return emit_receipt("raf", {
        "tenant_id": TENANT_ID,
        "nodes_count": graph.number_of_nodes() if HAS_NETWORKX else 0,
        "edges_count": graph.number_of_edges() if HAS_NETWORKX else 0,
        "cycles_detected": len(cycles),
        "cycle_lengths": [len(c) - 1 for c in cycles],  # Exclude closing node
        "catalytic_links": len(catalytic_links),
        "keystone_species": keystone_species[:5],  # Top 5
        "cascade_failure_potential": len(cycles) / max(1, graph.number_of_nodes()) if HAS_NETWORKX else 0,
        "raf_closure": raf_closure_test(graph, cycles),
        "min_cycle_length": RAF_CYCLE_MIN_LENGTH,
        "max_cycle_length": RAF_CYCLE_MAX_LENGTH,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_graph_construction_failed(reason: str) -> None:
    """If transaction graph incomplete."""
    emit_receipt("anomaly", {
        "metric": "graph_construction_failed",
        "reason": reason,
        "action": "halt",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"RAF graph construction failed: {reason}")


def stoprule_no_catalytic_links() -> None:
    """If only financial edges, RAF detection impossible."""
    emit_receipt("anomaly", {
        "metric": "no_catalytic_links",
        "action": "enrich_data",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    # This is a warning, not a hard stop
    pass


def stoprule_cascade_impact_low(collapse_ratio: float) -> None:
    """If keystone removal doesn't collapse cycles, not true RAF."""
    if collapse_ratio < 0.3:
        emit_receipt("anomaly", {
            "metric": "cascade_impact_low",
            "collapse_ratio": collapse_ratio,
            "threshold": 0.3,
            "action": "reconsider_keystones",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof RAF Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    if not HAS_NETWORKX:
        print("# SKIP: NetworkX not installed", file=sys.stderr)
        sys.exit(0)

    # Test 1: Build simple graph with cycle
    transactions = [
        {"source_duns": "A", "target_duns": "B", "amount_usd": 1000000},
        {"source_duns": "B", "target_duns": "C", "amount_usd": 500000},
        {"source_duns": "C", "target_duns": "A", "amount_usd": 250000},  # Closes cycle
    ]

    G = build_transaction_graph(transactions)
    print(f"# Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3

    # Test 2: Detect cycles
    cycles = detect_cycles(G, min_length=3, max_length=5)
    print(f"# Cycles detected: {len(cycles)}", file=sys.stderr)
    assert len(cycles) == 1
    assert cycles[0] == ['A', 'B', 'C', 'A'] or cycles[0] == ['B', 'C', 'A', 'B'] or cycles[0] == ['C', 'A', 'B', 'C']

    # Test 3: Keystone species
    keystones = identify_keystone_species(G, cycles)
    print(f"# Keystone species: {keystones}", file=sys.stderr)
    assert len(keystones) > 0

    # Test 4: Simulate disruption
    disruption = simulate_disruption(G, keystones[:1], cycles)
    print(f"# Disruption impact: {disruption['collapse_ratio']:.2f}", file=sys.stderr)
    assert disruption["collapse_ratio"] > 0

    # Test 5: RAF closure test
    has_raf = raf_closure_test(G, cycles)
    print(f"# RAF closure: {has_raf}", file=sys.stderr)
    assert has_raf == True

    # Test 6: Add catalytic links
    entities = [
        {"duns": "A", "address": "12345 Street, 20001"},
        {"duns": "B", "address": "67890 Avenue, 20001"},
        {"duns": "C", "board_members": ["John Smith"]},
    ]
    G = add_catalytic_links(G, entities)
    catalytic_count = len(G.graph.get("catalytic_links", []))
    print(f"# Catalytic links added: {catalytic_count}", file=sys.stderr)

    # Test 7: Receipt emission
    receipt = emit_raf_receipt(G, cycles, keystones)
    assert receipt["receipt_type"] == "raf"
    assert receipt["cycles_detected"] == 1
    assert receipt["raf_closure"] == True

    print(f"# PASS: raf module self-test", file=sys.stderr)
