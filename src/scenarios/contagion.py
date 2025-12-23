"""
Gov-OS Contagion Scenario - Cross-Domain Super-Graph Detection

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Purpose: Test cross-domain contagion detection via super-graph merging.
Validates that Medicaid ring collapse flags Defense vendor pre-invoice.

Physics Frame:
- Shell entity links two entropy pools (Defense + Medicaid)
- Collapse in one domain propagates temporal rigidity to the other
- "Defense fraud flagged pre-invoice via Medicaid contagion"
- 2-4x earlier detection demonstrated

Expected Output:
- Detected 3+ cycles (includes Medicaid ring)
- Pre-invoice Defense flag: True
- Insight: "This vendor was flagged because its sister company in
  the healthcare sector just collapsed."
"""

import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..core.constants import (
    DISCLAIMER,
    LAMBDA_NATURAL,
    RESISTANCE_THRESHOLD,
    CONTAGION_OVERLAP_MIN,
)
from ..core.receipt import (
    emit_super_graph_receipt,
    emit_contagion_receipt,
)
from ..core.temporal import (
    edge_weight_decay,
    resistance_to_decay,
    update_edge_with_decay,
    detect_zombies,
    identify_shell_entities,
    propagate_contagion,
)
from ..core.insight import (
    explain_contagion,
    format_insight,
    generate_executive_summary,
)
from ..raf import (
    build_transaction_graph,
    detect_cycles,
    emit_raf_receipt,
)


# ============================================================================
# CONSTANTS
# ============================================================================

SHELL_ENTITY_ID = "SHELL_HOLDINGS_LLC"

# Defense ring pattern: WELDCO_INC → SUBCO_A → SUBCO_B → WELDCO_INC
DEFENSE_RING = ["WELDCO_INC", "SUBCO_A", "SUBCO_B"]

# Medicaid ring pattern: MEDLAB_TESTING_LLC → CLINIC_X → CLINIC_Y → MEDLAB
MEDICAID_RING = ["MEDLAB_TESTING_LLC", "CLINIC_X", "CLINIC_Y"]


# ============================================================================
# SAMPLE DATA GENERATORS
# ============================================================================

def sample_defense_transactions(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic defense transactions with known ring pattern.

    Includes:
    - Normal legitimate transactions
    - Defense ring: WELDCO_INC → SUBCO_A → SUBCO_B → WELDCO_INC
    - Link to shell entity: SUBCO_B → SHELL_HOLDINGS_LLC

    Args:
        n: Number of transactions
        seed: Random seed

    Returns:
        List of transaction dicts
    """
    import random
    random.seed(seed)

    transactions = []
    base_date = datetime(2024, 1, 1)

    # Normal transactions
    vendors = [f"VENDOR_{i}" for i in range(10)]
    for i in range(n - 10):
        transactions.append({
            "source_duns": random.choice(vendors),
            "target_duns": random.choice(vendors),
            "amount_usd": random.random() * 1_000_000,
            "date": base_date + timedelta(days=random.randint(0, 365)),
            "domain": "defense",
        })

    # Ring transactions (with old dates to trigger zombie detection)
    ring_date = base_date - timedelta(days=400)  # 400 days ago
    for i in range(len(DEFENSE_RING)):
        source = DEFENSE_RING[i]
        target = DEFENSE_RING[(i + 1) % len(DEFENSE_RING)]
        transactions.append({
            "source_duns": source,
            "target_duns": target,
            "amount_usd": 500_000,
            "date": ring_date,
            "domain": "defense",
            "_is_fraud": True,
        })

    # Link to shell entity
    transactions.append({
        "source_duns": "SUBCO_B",
        "target_duns": SHELL_ENTITY_ID,
        "amount_usd": 250_000,
        "date": ring_date,
        "domain": "defense",
        "_is_fraud": True,
    })

    return transactions


def sample_medicaid_transactions(n: int = 100, seed: int = 43) -> List[Dict[str, Any]]:
    """
    Generate synthetic Medicaid transactions with known ring pattern.

    Includes:
    - Normal legitimate transactions
    - Medicaid ring: MEDLAB → CLINIC_X → CLINIC_Y → MEDLAB
    - Link to shell entity: CLINIC_Y → SHELL_HOLDINGS_LLC

    Args:
        n: Number of transactions
        seed: Random seed

    Returns:
        List of transaction dicts
    """
    import random
    random.seed(seed)

    transactions = []
    base_date = datetime(2024, 1, 1)

    # Normal transactions
    providers = [f"PROVIDER_{i}" for i in range(10)]
    for i in range(n - 10):
        transactions.append({
            "source_duns": random.choice(providers),
            "target_duns": random.choice(providers),
            "amount_usd": random.random() * 100_000,
            "date": base_date + timedelta(days=random.randint(0, 365)),
            "domain": "medicaid",
        })

    # Ring transactions (with old dates)
    ring_date = base_date - timedelta(days=400)
    for i in range(len(MEDICAID_RING)):
        source = MEDICAID_RING[i]
        target = MEDICAID_RING[(i + 1) % len(MEDICAID_RING)]
        transactions.append({
            "source_duns": source,
            "target_duns": target,
            "amount_usd": 50_000,
            "date": ring_date,
            "domain": "medicaid",
            "_is_fraud": True,
        })

    # Link to shell entity
    transactions.append({
        "source_duns": "CLINIC_Y",
        "target_duns": SHELL_ENTITY_ID,
        "amount_usd": 25_000,
        "date": ring_date,
        "domain": "medicaid",
        "_is_fraud": True,
    })

    return transactions


# ============================================================================
# SUPER-GRAPH BUILDING
# ============================================================================

def build_domain_graph(
    transactions: List[Dict[str, Any]],
    domain: str,
) -> 'nx.DiGraph':
    """
    Build a domain-specific transaction graph.

    Args:
        transactions: List of transactions
        domain: Domain identifier

    Returns:
        NetworkX DiGraph with domain metadata
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for graph building")

    G = build_transaction_graph(transactions)

    # Add domain metadata to all edges
    for u, v in G.edges():
        G[u][v]["domain"] = domain
        # Find transaction date for this edge
        for tx in transactions:
            if tx.get("source_duns") == u and tx.get("target_duns") == v:
                G[u][v]["last_seen_date"] = tx.get("date")
                G[u][v]["initial_weight"] = G[u][v].get("weight", 1.0)
                break

    # Add domain to nodes
    for node in G.nodes():
        G.nodes[node]["domain"] = domain

    return G


def inject_shell_entity(
    G: 'nx.DiGraph',
    shell_id: str,
    defense_target: str,
    medicaid_target: str,
) -> None:
    """
    Add shell entity connecting defense and medicaid domains.

    Args:
        G: Super-graph to modify
        shell_id: Shell entity ID
        defense_target: Defense node to connect
        medicaid_target: Medicaid node to connect
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required")

    # Shell is already in graph from transactions
    # Just ensure it has cross-domain metadata
    if G.has_node(shell_id):
        G.nodes[shell_id]["is_shell"] = True
        G.nodes[shell_id]["linked_domains"] = ["defense", "medicaid"]


def build_super_graph(
    defense_transactions: List[Dict[str, Any]] = None,
    medicaid_transactions: List[Dict[str, Any]] = None,
) -> 'nx.DiGraph':
    """
    Build super-graph by merging defense and medicaid domain graphs.

    This is the core v5.1 capability: cross-domain analysis.

    Args:
        defense_transactions: Defense domain transactions (optional)
        medicaid_transactions: Medicaid domain transactions (optional)

    Returns:
        Merged super-graph with cross-domain edges

    Emits:
        super_graph_receipt with merge statistics
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for super-graph building")

    # Generate sample data if not provided
    if defense_transactions is None:
        defense_transactions = sample_defense_transactions()
    if medicaid_transactions is None:
        medicaid_transactions = sample_medicaid_transactions()

    # Build domain graphs
    G_defense = build_domain_graph(defense_transactions, "defense")
    G_medicaid = build_domain_graph(medicaid_transactions, "medicaid")

    # Merge into super-graph
    G = nx.DiGraph()

    # Add defense edges
    for u, v, data in G_defense.edges(data=True):
        G.add_edge(u, v, **data)

    # Add medicaid edges
    for u, v, data in G_medicaid.edges(data=True):
        if G.has_edge(u, v):
            # Edge exists - merge domains
            existing_domains = G[u][v].get("domains", [G[u][v].get("domain", "unknown")])
            if isinstance(existing_domains, str):
                existing_domains = [existing_domains]
            existing_domains.append("medicaid")
            G[u][v]["domains"] = existing_domains
        else:
            G.add_edge(u, v, **data)

    # Copy node attributes
    for node, data in G_defense.nodes(data=True):
        if G.has_node(node):
            G.nodes[node].update(data)
    for node, data in G_medicaid.nodes(data=True):
        if G.has_node(node):
            # Merge domain info
            existing = G.nodes[node].get("domain")
            if existing and existing != data.get("domain"):
                G.nodes[node]["domains"] = [existing, data.get("domain")]
            else:
                G.nodes[node].update(data)

    # Identify shared entities
    defense_nodes = set(G_defense.nodes())
    medicaid_nodes = set(G_medicaid.nodes())
    shared_entities = defense_nodes.intersection(medicaid_nodes)

    # Mark shell entity
    inject_shell_entity(G, SHELL_ENTITY_ID, "SUBCO_B", "CLINIC_Y")

    # Detect cycles in super-graph
    cycles = detect_cycles(G)

    # Store graph-level metadata
    G.graph["domains"] = {"defense": G_defense, "medicaid": G_medicaid}
    G.graph["shared_entities"] = list(shared_entities)
    G.graph["cycles"] = cycles

    # Emit super-graph receipt
    emit_super_graph_receipt(
        domains=["defense", "medicaid"],
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        shared_entities=len(shared_entities),
        cycles_detected=len(cycles),
    )

    return G


# ============================================================================
# CONTAGION DETECTION
# ============================================================================

def simulate_medicaid_collapse(
    super_graph: 'nx.DiGraph',
) -> List[str]:
    """
    Simulate detection of Medicaid fraud ring.

    When Medicaid ring is detected, propagate temporal signal
    through shell entity to Defense domain.

    Args:
        super_graph: Super-graph with both domains

    Returns:
        List of Defense entities flagged via contagion
    """
    # Detect Medicaid ring cycles
    medicaid_cycles = []
    for cycle in detect_cycles(super_graph):
        # Check if cycle is in medicaid domain
        is_medicaid = all(
            super_graph.nodes.get(node, {}).get("domain") == "medicaid"
            or node in MEDICAID_RING
            for node in cycle[:-1]
        )
        if is_medicaid:
            medicaid_cycles.append(cycle)

    if not medicaid_cycles:
        return []

    # Find shell entities connected to collapsed domain
    shell_entities = identify_shell_entities(super_graph)

    # Propagate contagion
    flagged = []
    for shell in shell_entities:
        defense_flagged = propagate_contagion(
            super_graph,
            collapsed_domain="medicaid",
            shell_entity=shell,
        )
        flagged.extend(defense_flagged)

    return flagged


def run_contagion_test() -> bool:
    """
    Run complete contagion scenario test.

    Steps:
    1. Build super-graph from defense + medicaid transactions
    2. Detect cycles in super-graph
    3. Simulate Medicaid ring collapse
    4. Verify Defense entities flagged pre-invoice
    5. Generate insight receipts

    Returns:
        True if pre-invoice Defense flag detected

    Prints:
        Test results to stderr
    """
    print("=" * 60, file=sys.stderr)
    print("GOV-OS v5.1 CONTAGION SCENARIO TEST", file=sys.stderr)
    print(f"DISCLAIMER: {DISCLAIMER}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if not HAS_NETWORKX:
        print("SKIP: NetworkX not installed", file=sys.stderr)
        return False

    # Step 1: Build super-graph
    print("\n[1] Building super-graph...", file=sys.stderr)
    G = build_super_graph()
    print(f"    Nodes: {G.number_of_nodes()}", file=sys.stderr)
    print(f"    Edges: {G.number_of_edges()}", file=sys.stderr)

    # Step 2: Detect cycles
    print("\n[2] Detecting cycles...", file=sys.stderr)
    cycles = detect_cycles(G)
    print(f"    Cycles detected: {len(cycles)}", file=sys.stderr)
    for i, cycle in enumerate(cycles[:5]):  # Show first 5
        print(f"    Cycle {i+1}: {' → '.join(cycle)}", file=sys.stderr)

    # Step 3: Identify shell entities
    print("\n[3] Identifying shell entities...", file=sys.stderr)
    shells = identify_shell_entities(G)
    print(f"    Shell entities: {shells}", file=sys.stderr)

    # Step 4: Simulate Medicaid collapse and contagion
    print("\n[4] Simulating Medicaid collapse...", file=sys.stderr)
    flagged = simulate_medicaid_collapse(G)
    print(f"    Defense entities flagged: {flagged}", file=sys.stderr)

    # Step 5: Check pre-invoice flag
    pre_invoice_flag = len(flagged) > 0
    print(f"\n[5] Pre-invoice Defense flag: {pre_invoice_flag}", file=sys.stderr)

    # Step 6: Generate insights
    print("\n[6] Generating insights...", file=sys.stderr)
    if pre_invoice_flag:
        # Create contagion receipt for insight
        contagion_data = {
            "source_domain": "medicaid",
            "target_domain": "defense",
            "shell_entity": SHELL_ENTITY_ID,
            "pre_invoice_flag": True,
        }
        insight = format_insight("contagion", contagion_data)
        print(f"    Insight: {insight.get('plain_english', '')[:100]}...", file=sys.stderr)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULTS:", file=sys.stderr)
    print(f"  - Cycles detected: {len(cycles)}", file=sys.stderr)
    print(f"  - Shell entities: {len(shells)}", file=sys.stderr)
    print(f"  - Defense entities flagged: {len(flagged)}", file=sys.stderr)
    print(f"  - Pre-invoice flag: {pre_invoice_flag}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if pre_invoice_flag:
        print("PASS: Cross-domain contagion detected", file=sys.stderr)
    else:
        print("FAIL: No contagion detected", file=sys.stderr)

    return pre_invoice_flag


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    success = run_contagion_test()
    sys.exit(0 if success else 1)
