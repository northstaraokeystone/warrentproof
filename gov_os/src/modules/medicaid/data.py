"""
Gov-OS Medicaid Data - Ingest Adapters for Medicaid Domain

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ...core.constants import DISCLAIMER
from ...core.receipt import emit_L0

from .schema import validate_claim, validate_provider


@dataclass
class MedicaidReceipts:
    """Container for medicaid receipts."""
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    providers: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, receipt: Dict[str, Any]) -> None:
        self.receipts.append(receipt)

    def inject_fraud(self, fraud_type: str, count: int = 5) -> List[Dict[str, Any]]:
        """Inject synthetic fraud for testing."""
        fraudulent = []

        for i in range(count):
            if fraud_type == "upcoding":
                fraudulent.append(self._generate_upcoding_fraud(i))
            elif fraud_type == "phantom_billing":
                fraudulent.append(self._generate_phantom_billing_fraud(i))
            elif fraud_type == "provider_ring":
                fraudulent.append(self._generate_provider_ring_fraud(i))
            elif fraud_type == "kickback_chain":
                fraudulent.append(self._generate_kickback_fraud(i))
            elif fraud_type == "identity_theft":
                fraudulent.append(self._generate_identity_fraud(i))

        self.receipts.extend(fraudulent)
        return fraudulent

    def _generate_upcoding_fraud(self, idx: int) -> Dict[str, Any]:
        # Upcoding: billing for more expensive procedure than performed
        return {
            "receipt_type": "medicaid_ingest_receipt",
            "claim_id": f"UPCODE_{idx}",
            "provider_npi": f"12345{idx:05d}",
            "cpt_codes": ["99215", "99214", "99213"],  # Multiple high-level codes
            "amount": 500 + idx * 50,  # Inflated amount
            "_is_fraud": True,
            "fraud_type": "upcoding",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_phantom_billing_fraud(self, idx: int) -> Dict[str, Any]:
        # Phantom billing: services not rendered
        return {
            "receipt_type": "medicaid_ingest_receipt",
            "claim_id": f"PHANTOM_{idx}",
            "provider_npi": f"99999{idx:05d}",
            "beneficiary_id": f"DECEASED_{idx:05d}",  # Deceased beneficiary
            "cpt_codes": ["99213"],
            "amount": random.random() * 1000,
            "_is_fraud": True,
            "fraud_type": "phantom_billing",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_provider_ring_fraud(self, idx: int) -> Dict[str, Any]:
        # Provider ring: circular referrals
        ring_providers = [f"1111100000{idx % 3}", f"2222200000{idx % 3}", f"3333300000{idx % 3}"]
        return {
            "receipt_type": "medicaid_raf_receipt",
            "claim_id": f"RING_{idx}",
            "provider_npi": ring_providers[idx % 3],
            "referral_to": ring_providers[(idx + 1) % 3],
            "amount": random.random() * 500,
            "_is_fraud": True,
            "fraud_type": "provider_ring",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_kickback_fraud(self, idx: int) -> Dict[str, Any]:
        # Kickback: payment loops between providers
        return {
            "receipt_type": "medicaid_raf_receipt",
            "claim_id": f"KICKBACK_{idx}",
            "provider_npi": f"44444{idx:05d}",
            "referral_to": f"55555{(idx + 1):05d}",
            "kickback_amount": random.random() * 100,
            "amount": random.random() * 500,
            "_is_fraud": True,
            "fraud_type": "kickback_chain",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }

    def _generate_identity_fraud(self, idx: int) -> Dict[str, Any]:
        # Identity theft: claims for non-existent beneficiaries
        return {
            "receipt_type": "medicaid_ingest_receipt",
            "claim_id": f"IDENTITY_{idx}",
            "provider_npi": f"66666{idx:05d}",
            "beneficiary_id": f"FAKE_{random.randint(100000, 999999)}",
            "cpt_codes": ["99213"],
            "amount": random.random() * 300,
            "_is_fraud": True,
            "fraud_type": "identity_theft",
            "tenant_id": "gov-os-medicaid",
            "simulation_flag": DISCLAIMER,
        }


def ingest_claim(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and emit medicaid_ingest_receipt for claim."""
    if not validate_claim(data):
        raise ValueError("Invalid claim data")

    receipt = emit_L0("medicaid_ingest_receipt", {
        "claim_id": data["claim_id"],
        "provider_npi": data["provider_npi"],
        "beneficiary_id": data.get("beneficiary_id", ""),
        "cpt_codes": data.get("cpt_codes", []),
        "amount": data["amount"],
        "domain": "medicaid",
        "tenant_id": "gov-os-medicaid",
        "simulation_flag": DISCLAIMER,
    })

    return receipt


def ingest_provider(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and emit provider receipt."""
    if not validate_provider(data):
        raise ValueError("Invalid provider data")

    receipt = emit_L0("medicaid_ingest_receipt", {
        "provider_npi": data["npi"],
        "provider_name": data["name"],
        "specialty": data.get("specialty", ""),
        "domain": "medicaid",
        "tenant_id": "gov-os-medicaid",
        "simulation_flag": DISCLAIMER,
    })

    return receipt


def generate_sample_data(n_claims: int = 10, n_providers: int = 5) -> MedicaidReceipts:
    """Generate sample medicaid data for testing."""
    receipts = MedicaidReceipts()

    # Generate providers
    providers = []
    specialties = ["General Practice", "Cardiology", "Orthopedics", "Neurology", "Internal Medicine"]
    for i in range(n_providers):
        provider = {
            "npi": f"{(i + 1) * 1000000000:010d}",
            "name": f"Dr. Provider {i}",
            "specialty": specialties[i % len(specialties)],
        }
        providers.append(provider)
        ingest_provider(provider)
        receipts.providers.append(provider)

    # Generate claims
    cpt_codes = ["99213", "99214", "99215", "99201", "99202"]
    for i in range(n_claims):
        provider = random.choice(providers)
        claim = {
            "claim_id": f"CLAIM_{i:06d}",
            "provider_npi": provider["npi"],
            "beneficiary_id": f"BEN_{random.randint(100000, 999999)}",
            "cpt_codes": random.sample(cpt_codes, random.randint(1, 3)),
            "amount": random.random() * 500,
            "date": "2024-01-15",
        }
        ingest_claim(claim)
        receipts.claims.append(claim)

    return receipts
