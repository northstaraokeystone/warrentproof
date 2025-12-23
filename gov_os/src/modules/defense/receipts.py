"""
Gov-OS Defense Receipts - Domain-Specific Receipt Extensions

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...core.receipt import emit_L0, emit_L1
from ...core.constants import DISCLAIMER


@dataclass
class DefenseIngestReceipt:
    """Extends ingest_receipt for defense domain."""
    contract_id: str
    vendor_id: str
    program: str
    contract_type: str = "fixed"
    amount_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "defense_ingest_receipt",
            "contract_id": self.contract_id,
            "vendor_id": self.vendor_id,
            "program": self.program,
            "contract_type": self.contract_type,
            "amount_usd": self.amount_usd,
            "domain": "defense",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class DefenseRafReceipt:
    """Extends raf_cycle_receipt for defense domain."""
    subcontractor_ring: List[str]
    prime_contractor: str
    cycle_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "defense_raf_receipt",
            "subcontractor_ring": self.subcontractor_ring,
            "prime_contractor": self.prime_contractor,
            "cycle_value": self.cycle_value,
            "cycle_length": len(self.subcontractor_ring),
            "domain": "defense",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class ShipyardIterationReceipt:
    """Shipyard iteration tracking receipt."""
    build_phase: str
    iteration_count: int
    cadence_days: float
    disruption_factor: float = 1.0
    ship_id: str = ""
    block_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "shipyard_iteration_receipt",
            "build_phase": self.build_phase,
            "iteration_count": self.iteration_count,
            "cadence_days": self.cadence_days,
            "disruption_factor": self.disruption_factor,
            "ship_id": self.ship_id,
            "block_id": self.block_id,
            "domain": "defense",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class ShipyardAdditiveReceipt:
    """3D printing (LFAM) hull validation receipt."""
    hull_section: str
    print_hash: str
    material_kg: float
    qa_status: str = "pending"
    layer_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "shipyard_additive_receipt",
            "hull_section": self.hull_section,
            "print_hash": self.print_hash,
            "material_kg": self.material_kg,
            "qa_status": self.qa_status,
            "layer_count": self.layer_count,
            "domain": "defense",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class ShipyardAssemblyReceipt:
    """Robotic assembly tracking receipt."""
    weld_id: str
    robot_id: str
    block_ids: List[str]
    inspection_hash: str
    weld_quality: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "shipyard_assembly_receipt",
            "weld_id": self.weld_id,
            "robot_id": self.robot_id,
            "block_ids": self.block_ids,
            "inspection_hash": self.inspection_hash,
            "weld_quality": self.weld_quality,
            "domain": "defense",
            "tenant_id": "gov-os-defense",
            "simulation_flag": DISCLAIMER,
        }


def emit_defense_receipt(receipt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit a defense-specific receipt."""
    receipt_type = receipt_data.get("receipt_type", "defense_ingest_receipt")

    # Add domain fields
    receipt_data["domain"] = "defense"
    receipt_data["tenant_id"] = receipt_data.get("tenant_id", "gov-os-defense")
    receipt_data["simulation_flag"] = DISCLAIMER

    # Determine level based on type
    if receipt_type in ["defense_raf_receipt"]:
        return emit_L1(receipt_type, receipt_data)
    else:
        return emit_L0(receipt_type, receipt_data)
