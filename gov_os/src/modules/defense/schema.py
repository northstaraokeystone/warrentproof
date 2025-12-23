"""
Gov-OS Defense Schema - Contract, Vendor, and Shipyard Data Schemas

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContractSchema:
    """Defense contract schema."""
    contract_id: str
    vendor_id: str
    program: str
    value: float
    start_date: str
    contract_type: str = "fixed"  # "fixed" or "cost_plus"
    end_date: Optional[str] = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractSchema":
        return cls(
            contract_id=data.get("contract_id", ""),
            vendor_id=data.get("vendor_id", ""),
            program=data.get("program", ""),
            value=data.get("value", 0.0),
            start_date=data.get("start_date", ""),
            contract_type=data.get("contract_type", "fixed"),
            end_date=data.get("end_date"),
            description=data.get("description", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "vendor_id": self.vendor_id,
            "program": self.program,
            "value": self.value,
            "start_date": self.start_date,
            "contract_type": self.contract_type,
            "end_date": self.end_date,
            "description": self.description,
        }


@dataclass
class VendorSchema:
    """Defense vendor schema."""
    vendor_id: str
    name: str
    tier: str = "prime"  # "prime" or "sub"
    cage_code: str = ""
    address: str = ""
    enrolled_date: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VendorSchema":
        return cls(
            vendor_id=data.get("vendor_id", ""),
            name=data.get("name", ""),
            tier=data.get("tier", "prime"),
            cage_code=data.get("cage_code", ""),
            address=data.get("address", ""),
            enrolled_date=data.get("enrolled_date", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor_id": self.vendor_id,
            "name": self.name,
            "tier": self.tier,
            "cage_code": self.cage_code,
            "address": self.address,
            "enrolled_date": self.enrolled_date,
        }


@dataclass
class SubcontractorSchema:
    """Subcontractor relationship schema."""
    sub_id: str
    prime_id: str
    value: float
    percentage: float  # 0-100
    contract_id: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubcontractorSchema":
        return cls(
            sub_id=data.get("sub_id", ""),
            prime_id=data.get("prime_id", ""),
            value=data.get("value", 0.0),
            percentage=data.get("percentage", 0.0),
            contract_id=data.get("contract_id", ""),
        )


@dataclass
class ShipyardSchema:
    """Shipyard block schema."""
    block_id: str
    ship_id: str
    phase: str  # DESIGN, KEEL_LAYING, BLOCK_ASSEMBLY, etc.
    iteration: int = 1
    method: str = "traditional"  # "traditional" or "additive"
    weight_kg: float = 0.0
    weld_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShipyardSchema":
        return cls(
            block_id=data.get("block_id", ""),
            ship_id=data.get("ship_id", ""),
            phase=data.get("phase", ""),
            iteration=data.get("iteration", 1),
            method=data.get("method", "traditional"),
            weight_kg=data.get("weight_kg", 0.0),
            weld_count=data.get("weld_count", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "ship_id": self.ship_id,
            "phase": self.phase,
            "iteration": self.iteration,
            "method": self.method,
            "weight_kg": self.weight_kg,
            "weld_count": self.weld_count,
        }


def validate_contract(data: Dict[str, Any]) -> bool:
    """Validate data against ContractSchema."""
    required = ["contract_id", "vendor_id", "program", "value", "start_date"]
    for field in required:
        if field not in data or not data[field]:
            return False

    # Validate contract_type
    if data.get("contract_type") not in ["fixed", "cost_plus", None, ""]:
        return False

    # Validate value is positive
    if data.get("value", 0) <= 0:
        return False

    return True


def validate_vendor(data: Dict[str, Any]) -> bool:
    """Validate data against VendorSchema."""
    required = ["vendor_id", "name"]
    for field in required:
        if field not in data or not data[field]:
            return False

    # Validate tier
    if data.get("tier") and data["tier"] not in ["prime", "sub"]:
        return False

    return True


def validate_block(data: Dict[str, Any]) -> bool:
    """Validate data against ShipyardSchema."""
    required = ["block_id", "ship_id", "phase"]
    for field in required:
        if field not in data or not data[field]:
            return False

    # Validate method
    if data.get("method") and data["method"] not in ["traditional", "additive"]:
        return False

    return True
