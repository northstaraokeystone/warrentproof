"""
Gov-OS Medicaid Receipts - Domain-Specific Receipt Extensions

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ...core.receipt import emit_L0, emit_L1
from ...core.constants import DISCLAIMER


@dataclass
class MedicaidIngestReceipt:
    """Extends ingest_receipt for medicaid domain."""
    claim_id: str
    provider_npi: str
    cpt_codes: List[str]
    amount: float
    beneficiary_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "medicaid_ingest_receipt",
            "claim_id": self.claim_id,
            "provider_npi": self.provider_npi,
            "cpt_codes": self.cpt_codes,
            "amount": self.amount,
            "beneficiary_id": self.beneficiary_id,
            "domain": "medicaid",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class MedicaidRafReceipt:
    """Extends raf_cycle_receipt for medicaid domain."""
    referral_ring: List[str]  # List of provider NPIs
    patient_count: int
    ring_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "medicaid_raf_receipt",
            "referral_ring": self.referral_ring,
            "patient_count": self.patient_count,
            "ring_value": self.ring_value,
            "cycle_length": len(self.referral_ring),
            "domain": "medicaid",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }


@dataclass
class ReferralRingReceipt:
    """Referral ring detection receipt."""
    providers: List[str]
    referral_pattern: str
    ring_value: float
    patient_ids: List[str] = None

    def __post_init__(self):
        if self.patient_ids is None:
            self.patient_ids = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_type": "referral_ring_receipt",
            "providers": self.providers,
            "referral_pattern": self.referral_pattern,
            "ring_value": self.ring_value,
            "patient_count": len(self.patient_ids),
            "domain": "medicaid",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }


def emit_medicaid_receipt(receipt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit a medicaid-specific receipt."""
    receipt_type = receipt_data.get("receipt_type", "medicaid_ingest_receipt")

    # Add domain fields
    receipt_data["domain"] = "medicaid"
    receipt_data["tenant_id"] = receipt_data.get("tenant_id", "gov-os-medicaid")
    receipt_data["simulation_flag"] = DISCLAIMER

    # Determine level based on type
    if receipt_type in ["medicaid_raf_receipt", "referral_ring_receipt"]:
        return emit_L1(receipt_type, receipt_data)
    else:
        return emit_L0(receipt_type, receipt_data)
