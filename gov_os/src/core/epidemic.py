"""
Gov-OS Core Epidemic - R₀ Spread Modeling for Entity Contagion

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

R₀ (basic reproduction number) modeling for vendor-to-vendor or
provider-to-provider fraud spread using SIR model (Susceptible → Infected → Removed).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .constants import (
    EPIDEMIC_R0_OUTBREAK,
    EPIDEMIC_RECOVERY_RATE,
    EPIDEMIC_DETECTION_LATENCY_TARGET,
    DISCLAIMER,
    TENANT_ID,
)
from .receipt import emit_L1


@dataclass
class EntityNetwork:
    """Graph representation of entity (vendor/provider) relationships."""
    entities: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    infection_status: Dict[str, str] = field(default_factory=dict)  # "S", "I", or "R"

    def add_entity(self, entity_id: str):
        self.entities.add(entity_id)
        if entity_id not in self.infection_status:
            self.infection_status[entity_id] = "S"  # Susceptible

    def add_edge(self, entity1: str, entity2: str):
        self.add_entity(entity1)
        self.add_entity(entity2)
        self.edges[entity1].add(entity2)
        self.edges[entity2].add(entity1)

    @property
    def density(self) -> float:
        """Network density: actual edges / possible edges."""
        n = len(self.entities)
        if n <= 1:
            return 0.0
        possible = n * (n - 1) / 2
        actual = sum(len(neighbors) for neighbors in self.edges.values()) / 2
        return actual / possible if possible > 0 else 0.0

    @property
    def volume(self) -> int:
        return len(self.entities)

    def infected_entities(self) -> List[str]:
        return [e for e, status in self.infection_status.items() if status == "I"]


def calculate_R0(
    graph: Any,
    infected: Set[str],
    window_days: int = 7,
) -> float:
    """
    Basic reproduction number: R₀ = density × volume / detection_latency.

    Args:
        graph: EntityNetwork or NetworkX graph
        infected: Set of infected entity IDs
        window_days: Detection latency in days

    Returns:
        R₀ value (>=1.0 means outbreak)
    """
    if isinstance(graph, EntityNetwork):
        density = graph.density
        volume = graph.volume
    elif HAS_NETWORKX and isinstance(graph, nx.Graph):
        density = nx.density(graph)
        volume = graph.number_of_nodes()
    else:
        return 0.0

    if window_days <= 0:
        window_days = EPIDEMIC_DETECTION_LATENCY_TARGET

    # R₀ formula
    R0 = (density * volume) / window_days

    # Scale to realistic range [0, 5]
    return min(5.0, R0 * 0.1)


def infection_graph(
    receipts: List[Dict[str, Any]],
    source: str,
    entity_key: str = "vendor_id",
) -> Any:
    """
    Build spread graph from source entity.

    Args:
        receipts: List of receipts
        source: Source entity ID
        entity_key: Field name for entity identifier

    Returns:
        EntityNetwork or NetworkX graph
    """
    network = EntityNetwork()

    # Build entity relationships
    entities_by_context: Dict[str, List[str]] = defaultdict(list)

    for r in receipts:
        entity = r.get(entity_key)
        context = r.get("branch", "") or r.get("domain", "default")

        if entity:
            network.add_entity(entity)
            entities_by_context[context].append(entity)

    # Connect entities in same context
    for context, entities in entities_by_context.items():
        unique = list(set(entities))
        for i in range(len(unique)):
            for j in range(i + 1, min(i + 5, len(unique))):
                network.add_edge(unique[i], unique[j])

    # Mark source as infected
    if source in network.entities:
        network.infection_status[source] = "I"

    return network


def patient_zero(graph: Any) -> Optional[str]:
    """
    Identify origin node (first infected).

    Args:
        graph: EntityNetwork or NetworkX graph

    Returns:
        Entity ID of patient zero, or None
    """
    if isinstance(graph, EntityNetwork):
        infected = graph.infected_entities()
        return infected[0] if infected else None
    elif HAS_NETWORKX and isinstance(graph, nx.DiGraph):
        # Find node with no incoming infection edges
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.nodes[node].get("infected"):
                return node
    return None


def outbreak_probability(R0: float) -> float:
    """
    Probability of sustained outbreak.
    P(outbreak) ≈ 1 - 1/R₀ for R₀ > 1.

    Args:
        R0: Basic reproduction number

    Returns:
        Outbreak probability 0-1
    """
    if R0 <= 1.0:
        return 0.0
    return 1.0 - (1.0 / R0)


def quarantine_candidates(
    graph: Any,
    infected: Set[str],
    max_candidates: int = 10,
) -> List[str]:
    """
    Nodes to isolate to contain spread.

    Args:
        graph: EntityNetwork or NetworkX graph
        infected: Set of infected entity IDs
        max_candidates: Maximum candidates to return

    Returns:
        List of entity IDs to quarantine
    """
    candidates = []

    if isinstance(graph, EntityNetwork):
        # Quarantine infected + their direct neighbors
        for entity in infected:
            if entity not in candidates:
                candidates.append(entity)
            for neighbor in graph.edges.get(entity, set()):
                if neighbor not in candidates:
                    candidates.append(neighbor)
    elif HAS_NETWORKX and isinstance(graph, nx.Graph):
        for entity in infected:
            if entity not in candidates and entity in graph:
                candidates.append(entity)
            for neighbor in graph.neighbors(entity) if entity in graph else []:
                if neighbor not in candidates:
                    candidates.append(neighbor)

    return candidates[:max_candidates]


def herd_immunity_threshold(R0: float) -> float:
    """
    Fraction of network needed for containment.
    H = 1 - 1/R₀.

    Args:
        R0: Basic reproduction number

    Returns:
        Herd immunity threshold 0-1
    """
    if R0 <= 1.0:
        return 0.0
    return 1.0 - (1.0 / R0)


def spread_simulation(
    graph: Any,
    R0: float,
    days: int = 30,
) -> List[Dict[str, int]]:
    """
    Monte Carlo spread projection using SIR model.

    Args:
        graph: EntityNetwork or NetworkX graph
        R0: Basic reproduction number
        days: Number of days to simulate

    Returns:
        List of SIR state dicts per day
    """
    if isinstance(graph, EntityNetwork):
        N = graph.volume
        I = len(graph.infected_entities())
    elif HAS_NETWORKX and isinstance(graph, nx.Graph):
        N = graph.number_of_nodes()
        I = sum(1 for _, data in graph.nodes(data=True) if data.get("infected"))
    else:
        return []

    S = N - I
    R = 0

    trajectory = [{"S": S, "I": I, "R": R}]

    for _ in range(days):
        S, I, R = _SIR_step(S, I, R, R0)
        trajectory.append({"S": S, "I": I, "R": R})

        if I == 0:
            break

    # Emit epidemic receipt
    emit_L1("epidemic_receipt", {
        "R0": round(R0, 4),
        "days_simulated": len(trajectory) - 1,
        "final_infected": trajectory[-1]["I"],
        "total_infected": trajectory[-1]["R"],
        "outbreak_probability": round(outbreak_probability(R0), 4),
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })

    return trajectory


def _SIR_step(
    S: int,
    I: int,
    R: int,
    R0: float,
    recovery_rate: float = EPIDEMIC_RECOVERY_RATE,
) -> tuple:
    """Single SIR iteration."""
    N = S + I + R
    if N == 0:
        return (0, 0, 0)

    beta = R0 * recovery_rate

    # New infections
    new_infections = min(S, int(beta * S * I / N))

    # New recoveries
    new_recoveries = int(recovery_rate * I)

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return (max(0, S_new), max(0, I_new), max(0, R_new))


def build_entity_network(
    receipts: List[Dict[str, Any]],
    entity_key: str = "vendor_id",
) -> EntityNetwork:
    """
    Build entity network from receipts.

    Args:
        receipts: List of receipts
        entity_key: Field name for entity identifier

    Returns:
        EntityNetwork
    """
    network = EntityNetwork()

    # Group by context
    entities_by_context: Dict[str, List[str]] = defaultdict(list)

    for r in receipts:
        entity = r.get(entity_key)
        context = r.get("branch", "") or r.get("domain", "default")

        if entity:
            network.add_entity(entity)
            entities_by_context[context].append(entity)

    # Connect entities in same context
    for context, entities in entities_by_context.items():
        unique = list(set(entities))
        for i in range(len(unique)):
            for j in range(i + 1, min(i + 5, len(unique))):
                network.add_edge(unique[i], unique[j])

    return network


def mark_infected(
    network: EntityNetwork,
    infected_ids: List[str],
) -> EntityNetwork:
    """
    Mark entities as infected.

    Args:
        network: EntityNetwork to update
        infected_ids: List of infected entity IDs

    Returns:
        Updated network
    """
    for entity_id in infected_ids:
        if entity_id in network.entities:
            network.infection_status[entity_id] = "I"
    return network
