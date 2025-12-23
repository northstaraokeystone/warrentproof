"""
Gov-OS Defense Data - Ingest Adapters for Defense Domain

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.constants import DISCLAIMER, TENANT_ID
from ...core.receipt import emit_L0
from ...core.utils import dual_hash

from .schema import validate_contract, validate_vendor, validate_block


@dataclass
class DefenseReceipts:
    """Container for defense receipts."""
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    contracts: List[Dict[str, Any]] = field(default_factory=list)
    vendors: List[Dict[str, Any]] = field(default_factory=list)
    shipyard_blocks: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, receipt: Dict[str, Any]) -> None:
        self.receipts.append(receipt)

    def inject_fraud(self, fraud_type: str, count: int = 5) -> List[Dict[str, Any]]:
        """Inject synthetic fraud for testing."""
        fraudulent = []

        for i in range(count):
            if fraud_type == "kickback":
                fraudulent.append(self._generate_kickback_fraud(i))
            elif fraud_type == "shell":
                fraudulent.append(self._generate_shell_vendor_fraud(i))
            elif fraud_type == "cost_plus_inflation":
                fraudulent.append(self._generate_cost_plus_fraud(i))
            elif fraud_type == "subcontractor_ring":
                fraudulent.append(self._generate_subcontractor_ring_fraud(i))
            elif fraud_type == "iteration_fraud":
                fraudulent.append(self._generate_iteration_fraud(i))

        self.receipts.extend(fraudulent)
        return fraudulent

    def _generate_kickback_fraud(self, idx: int) -> Dict[str, Any]:
        return {
            "receipt_type": "defense_ingest_receipt",
            "contract_id": f"KICKBACK_{idx}",
            "vendor_id": f"VENDOR_A_{idx}",
            "payment_to": f"VENDOR_B_{idx}",  # Circular
            "amount_usd": random.random() * 1_000_000,
            "_is_fraud": True,
            "fraud_type": "kickback",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_shell_vendor_fraud(self, idx: int) -> Dict[str, Any]:
        return {
            "receipt_type": "defense_ingest_receipt",
            "contract_id": f"SHELL_{idx}",
            "vendor_id": f"SHELL_VENDOR_{random.randint(10000, 99999)}",
            "amount_usd": random.random() * 5_000_000,
            "_is_fraud": True,
            "fraud_type": "shell",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_cost_plus_fraud(self, idx: int) -> Dict[str, Any]:
        return {
            "receipt_type": "defense_ingest_receipt",
            "contract_id": f"COSTPLUS_{idx}",
            "contract_type": "cost_plus",
            "vendor_id": f"VENDOR_{idx % 5}",
            "amount_usd": random.random() * 2_000_000 * (1.2 + idx * 0.05),  # Increasing
            "variance_pct": 15 + idx * 3,  # Increasing variance
            "_is_fraud": True,
            "fraud_type": "cost_plus_inflation",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_subcontractor_ring_fraud(self, idx: int) -> Dict[str, Any]:
        # A→B→C→A cycle
        ring_vendors = [f"RING_A_{idx}", f"RING_B_{idx}", f"RING_C_{idx}"]
        return {
            "receipt_type": "defense_raf_receipt",
            "contract_id": f"RING_{idx}",
            "vendor_id": ring_vendors[idx % 3],
            "payment_to": ring_vendors[(idx + 1) % 3],
            "amount_usd": random.random() * 500_000,
            "_is_fraud": True,
            "fraud_type": "subcontractor_ring",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_iteration_fraud(self, idx: int) -> Dict[str, Any]:
        return {
            "receipt_type": "shipyard_iteration_receipt",
            "block_id": f"FAKE_BLOCK_{idx}",
            "ship_id": "FAKE_SHIP_001",
            "iteration": idx + 1,
            "cadence_days": 0,  # Impossibly fast
            "_is_fraud": True,
            "fraud_type": "iteration_fraud",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


def ingest_contract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and emit defense_ingest_receipt for contract."""
    if not validate_contract(data):
        raise ValueError("Invalid contract data")

    receipt = emit_L0("defense_ingest_receipt", {
        "contract_id": data["contract_id"],
        "vendor_id": data["vendor_id"],
        "program": data["program"],
        "contract_type": data.get("contract_type", "fixed"),
        "amount_usd": data["value"],
        "domain": "defense",
        "tenant_id": "gov-os-defense",
        "simulation_flag": DISCLAIMER,
    })

    return receipt


def ingest_vendor(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and emit vendor receipt."""
    if not validate_vendor(data):
        raise ValueError("Invalid vendor data")

    receipt = emit_L0("defense_ingest_receipt", {
        "vendor_id": data["vendor_id"],
        "vendor_name": data["name"],
        "tier": data.get("tier", "prime"),
        "cage_code": data.get("cage_code", ""),
        "domain": "defense",
        "tenant_id": "gov-os-defense",
        "simulation_flag": DISCLAIMER,
    })

    return receipt


def ingest_shipyard_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and emit shipyard receipt."""
    if not validate_block(data):
        raise ValueError("Invalid shipyard block data")

    receipt = emit_L0("shipyard_block_receipt", {
        "block_id": data["block_id"],
        "ship_id": data["ship_id"],
        "phase": data["phase"],
        "iteration": data.get("iteration", 1),
        "method": data.get("method", "traditional"),
        "domain": "defense",
        "tenant_id": "gov-os-defense",
        "simulation_flag": DISCLAIMER,
    })

    return receipt


def generate_sample_data(n_contracts: int = 10, n_vendors: int = 5) -> DefenseReceipts:
    """Generate sample defense data for testing."""
    receipts = DefenseReceipts()

    # Generate vendors
    vendors = []
    for i in range(n_vendors):
        vendor = {
            "vendor_id": f"VENDOR_{i:03d}",
            "name": f"Defense Contractor {i}",
            "tier": "prime" if i < n_vendors // 2 else "sub",
            "cage_code": f"CAGE{i:05d}",
        }
        vendors.append(vendor)
        ingest_vendor(vendor)
        receipts.vendors.append(vendor)

    # Generate contracts
    for i in range(n_contracts):
        vendor = random.choice(vendors)
        contract = {
            "contract_id": f"CONTRACT_{i:05d}",
            "vendor_id": vendor["vendor_id"],
            "program": random.choice(["PROGRAM_A", "PROGRAM_B", "PROGRAM_C"]),
            "value": random.random() * 10_000_000,
            "start_date": "2024-01-01",
            "contract_type": random.choice(["fixed", "cost_plus"]),
        }
        ingest_contract(contract)
        receipts.contracts.append(contract)

    return receipts
