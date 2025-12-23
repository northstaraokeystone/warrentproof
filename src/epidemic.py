"""
WarrantProof Epidemic Module - R₀ Modeling for Vendor Spread

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements R₀ (basic reproduction number) modeling for
vendor-to-vendor fraud spread. Fraud spreads like epidemic through
vendor network using SIR model (Susceptible → Infected → Removed).

Physics Foundation:
- R₀ = density × volume / latency_detection
- If R₀ > 1, epidemic spreads. If R₀ < 1, containment achieved.
- Quarantine (vendor isolation) when R₀ exceeds threshold

SLOs:
- R₀ > 1.0 triggers quarantine action
- Quarantine should reduce R₀ below 1.0 within 30 days
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    EPIDEMIC_R0_THRESHOLD,
    EPIDEMIC_DETECTION_LATENCY_TARGET,
    EPIDEMIC_RECOVERY_RATE,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class VendorNetwork:
    """Graph representation of vendor transaction relationships."""
    vendors: set = field(default_factory=set)
    edges: dict = field(default_factory=lambda: defaultdict(set))  # vendor -> connected vendors
    infection_status: dict = field(default_factory=dict)  # vendor -> "S", "I", or "R"

    def add_vendor(self, vendor_id: str):
        self.vendors.add(vendor_id)
        if vendor_id not in self.infection_status:
            self.infection_status[vendor_id] = "S"  # Susceptible

    def add_edge(self, vendor1: str, vendor2: str):
        self.add_vendor(vendor1)
        self.add_vendor(vendor2)
        self.edges[vendor1].add(vendor2)
        self.edges[vendor2].add(vendor1)

    @property
    def density(self) -> float:
        """Calculate network density: actual edges / possible edges."""
        n = len(self.vendors)
        if n <= 1:
            return 0.0
        possible = n * (n - 1) / 2
        actual = sum(len(neighbors) for neighbors in self.edges.values()) / 2
        return actual / possible if possible > 0 else 0.0

    @property
    def volume(self) -> int:
        """Network volume = number of vendors."""
        return len(self.vendors)

    def infected_vendors(self) -> list:
        """Return list of infected vendor IDs."""
        return [v for v, status in self.infection_status.items() if status == "I"]


def calculate_R0(
    vendor_network: VendorNetwork,
    fraud_cases: list,
    detection_latency: float
) -> float:
    """
    Compute R₀ = (vendor_density × network_volume) / detection_latency.

    Args:
        vendor_network: VendorNetwork object
        fraud_cases: List of fraud detection cases
        detection_latency: Average days from fraud to detection

    Returns:
        R₀ value (epidemic threshold = 1.0)
    """
    if detection_latency <= 0:
        detection_latency = EPIDEMIC_DETECTION_LATENCY_TARGET

    density = vendor_network.density
    volume = vendor_network.volume

    # R₀ formula from Grok Q6
    R0 = (density * volume) / detection_latency

    # Scale to realistic epidemic numbers
    # Typically R₀ should be in range 0-5 for most epidemics
    R0 = min(5.0, R0 * 0.1)  # Scaling factor

    return R0


def SIR_model_step(
    susceptible: int,
    infected: int,
    removed: int,
    R0: float,
    recovery_rate: Optional[float] = None
) -> tuple:
    """
    Single SIR iteration.

    Args:
        susceptible: Number of susceptible entities
        infected: Number of infected entities
        removed: Number of removed (recovered/dead) entities
        R0: Basic reproduction number
        recovery_rate: Rate of recovery per timestep (default from constants)

    Returns:
        (S', I', R') for next timestep
    """
    if recovery_rate is None:
        recovery_rate = EPIDEMIC_RECOVERY_RATE

    N = susceptible + infected + removed
    if N == 0:
        return (0, 0, 0)

    # SIR differential equations (discrete)
    # Beta = R0 * recovery_rate (transmission rate)
    beta = R0 * recovery_rate

    # New infections
    new_infections = min(
        susceptible,
        int(beta * susceptible * infected / N)
    )

    # New recoveries
    new_recoveries = int(recovery_rate * infected)

    S_new = susceptible - new_infections
    I_new = infected + new_infections - new_recoveries
    R_new = removed + new_recoveries

    # Ensure non-negative
    return (max(0, S_new), max(0, I_new), max(0, R_new))


def predict_spread(
    initial_infected: int,
    R0: float,
    network_size: int,
    timesteps: int = 30
) -> list:
    """
    Run SIR forward to predict epidemic curve.

    Args:
        initial_infected: Number of initially infected vendors
        R0: Basic reproduction number
        network_size: Total network size
        timesteps: Number of timesteps to simulate

    Returns:
        List of (S, I, R) tuples over time
    """
    S = network_size - initial_infected
    I = initial_infected
    R = 0

    trajectory = [(S, I, R)]

    for _ in range(timesteps):
        S, I, R = SIR_model_step(S, I, R, R0)
        trajectory.append((S, I, R))

        # Stop if epidemic ended
        if I == 0:
            break

    return trajectory


def recommend_quarantine(
    R0: float,
    infected_vendors: list
) -> dict:
    """
    If R₀ > EPIDEMIC_R0_THRESHOLD (1.0), recommend quarantine.

    Args:
        R0: Current R₀ value
        infected_vendors: List of infected vendor IDs

    Returns:
        Quarantine recommendation dict
    """
    should_quarantine = R0 > EPIDEMIC_R0_THRESHOLD

    recommendation = {
        "quarantine_recommended": should_quarantine,
        "R0": R0,
        "threshold": EPIDEMIC_R0_THRESHOLD,
        "infected_vendors": infected_vendors,
        "quarantine_count": len(infected_vendors) if should_quarantine else 0,
        "action": "isolate_vendors" if should_quarantine else "continue_monitoring",
    }

    return recommendation


def estimate_detection_latency_required(
    vendor_density: float,
    network_volume: int,
    target_R0: float = 0.9
) -> float:
    """
    Calculate required detection speed to contain spread (R₀ < 1).

    Args:
        vendor_density: Network density
        network_volume: Number of vendors
        target_R0: Target R₀ (default 0.9 for safety margin)

    Returns:
        Required detection latency in days
    """
    if target_R0 <= 0:
        target_R0 = 0.9

    # From R₀ = density × volume / latency
    # latency = density × volume / R₀
    required_latency = (vendor_density * network_volume * 0.1) / target_R0

    return max(0.5, required_latency)  # Minimum 0.5 days


def build_vendor_network(receipts: list) -> VendorNetwork:
    """
    Build vendor network from receipts.

    Args:
        receipts: List of receipts with vendor information

    Returns:
        VendorNetwork object
    """
    network = VendorNetwork()

    # Extract vendors
    for r in receipts:
        vendor = r.get("vendor")
        if vendor:
            network.add_vendor(vendor)

    # Build edges from co-occurrence (same branch, similar time)
    vendors_by_branch = defaultdict(list)
    for r in receipts:
        vendor = r.get("vendor")
        branch = r.get("branch", "unknown")
        if vendor:
            vendors_by_branch[branch].append(vendor)

    # Connect vendors in same branch
    for branch, vendors in vendors_by_branch.items():
        unique_vendors = list(set(vendors))
        for i in range(len(unique_vendors)):
            for j in range(i + 1, min(i + 5, len(unique_vendors))):  # Connect nearby vendors
                network.add_edge(unique_vendors[i], unique_vendors[j])

    return network


def mark_infected_vendors(
    network: VendorNetwork,
    fraud_receipts: list
) -> VendorNetwork:
    """
    Mark vendors associated with fraud as infected.

    Args:
        network: VendorNetwork to update
        fraud_receipts: Receipts flagged as fraud

    Returns:
        Updated VendorNetwork
    """
    for r in fraud_receipts:
        vendor = r.get("vendor")
        if vendor and vendor in network.vendors:
            network.infection_status[vendor] = "I"

    return network


def emit_epidemic_receipt(
    R0: float,
    network: VendorNetwork,
    detection_latency: float,
    spread_prediction: Optional[list] = None
) -> dict:
    """
    Emit epidemic_receipt documenting R₀ analysis.

    Args:
        R0: Calculated R₀
        network: VendorNetwork analyzed
        detection_latency: Detection latency used
        spread_prediction: Optional SIR trajectory

    Returns:
        epidemic_receipt dict
    """
    infected = network.infected_vendors()
    quarantine = recommend_quarantine(R0, infected)

    return emit_receipt("epidemic", {
        "tenant_id": TENANT_ID,
        "R0": round(R0, 4),
        "vendor_density": round(network.density, 4),
        "network_volume": network.volume,
        "detection_latency_days": round(detection_latency, 2),
        "infected_vendors": infected[:10],  # Limit to 10 for receipt size
        "infected_count": len(infected),
        "quarantine_recommended": quarantine["quarantine_recommended"],
        "spread_prediction": [(s, i, r) for s, i, r in (spread_prediction or [])[:10]],
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_epidemic_runaway(R0: float) -> None:
    """Epidemic out of control if R₀ > 3.0."""
    if R0 > 3.0:
        emit_receipt("anomaly", {
            "metric": "epidemic_runaway",
            "R0": R0,
            "threshold": 3.0,
            "delta": R0 - 3.0,
            "action": "emergency_quarantine",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"Epidemic runaway: R₀ = {R0} > 3.0")


def stoprule_quarantine_ineffective(
    R0_before: float,
    R0_after: float,
    days_elapsed: int
) -> None:
    """Quarantine must reduce R₀ below 1.0 within 30 days."""
    if days_elapsed >= 30 and R0_after >= EPIDEMIC_R0_THRESHOLD:
        emit_receipt("anomaly", {
            "metric": "quarantine_ineffective",
            "R0_before": R0_before,
            "R0_after": R0_after,
            "days_elapsed": days_elapsed,
            "action": "escalate",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Quarantine ineffective: R₀ still {R0_after} after {days_elapsed} days"
        )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Epidemic Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test vendor network construction
    test_receipts = [
        {"vendor": "Vendor_A", "branch": "Navy"},
        {"vendor": "Vendor_B", "branch": "Navy"},
        {"vendor": "Vendor_C", "branch": "Navy"},
        {"vendor": "Vendor_D", "branch": "Army"},
        {"vendor": "Vendor_E", "branch": "Army"},
    ]

    network = build_vendor_network(test_receipts)
    print(f"# Network: {network.volume} vendors, density={network.density:.4f}", file=sys.stderr)
    assert network.volume == 5

    # Test R₀ calculation
    R0 = calculate_R0(network, [], detection_latency=7.0)
    print(f"# R₀: {R0:.4f}", file=sys.stderr)

    # Test SIR model
    S, I, R = SIR_model_step(100, 10, 0, R0=1.5)
    print(f"# SIR step: S={S}, I={I}, R={R}", file=sys.stderr)

    # Test spread prediction
    trajectory = predict_spread(5, R0=1.5, network_size=100, timesteps=30)
    print(f"# Trajectory length: {len(trajectory)}", file=sys.stderr)
    final_S, final_I, final_R = trajectory[-1]
    assert final_S + final_I + final_R == 100

    # Test quarantine recommendation
    network.infection_status["Vendor_A"] = "I"
    network.infection_status["Vendor_B"] = "I"

    infected = network.infected_vendors()
    recommendation = recommend_quarantine(1.5, infected)
    print(f"# Quarantine recommended: {recommendation['quarantine_recommended']}", file=sys.stderr)

    # Test detection latency estimation
    required_latency = estimate_detection_latency_required(
        vendor_density=0.3,
        network_volume=50,
        target_R0=0.9
    )
    print(f"# Required detection latency: {required_latency:.2f} days", file=sys.stderr)

    # Test receipt emission
    receipt = emit_epidemic_receipt(R0, network, 7.0, trajectory[:5])
    assert receipt["receipt_type"] == "epidemic"

    print(f"# PASS: epidemic module self-test", file=sys.stderr)
