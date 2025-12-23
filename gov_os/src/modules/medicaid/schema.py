"""
Gov-OS Medicaid Schema - Claim, Provider, and CPT Data Schemas

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


@dataclass
class ClaimSchema:
    """Medicaid claim schema."""
    claim_id: str
    provider_npi: str
    beneficiary_id: str
    cpt_codes: List[str]
    amount: float
    date: str
    diagnosis_codes: List[str] = field(default_factory=list)
    referral_to: Optional[str] = None  # NPI of referred provider

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimSchema":
        return cls(
            claim_id=data.get("claim_id", ""),
            provider_npi=data.get("provider_npi", ""),
            beneficiary_id=data.get("beneficiary_id", ""),
            cpt_codes=data.get("cpt_codes", []),
            amount=data.get("amount", 0.0),
            date=data.get("date", ""),
            diagnosis_codes=data.get("diagnosis_codes", []),
            referral_to=data.get("referral_to"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "provider_npi": self.provider_npi,
            "beneficiary_id": self.beneficiary_id,
            "cpt_codes": self.cpt_codes,
            "amount": self.amount,
            "date": self.date,
            "diagnosis_codes": self.diagnosis_codes,
            "referral_to": self.referral_to,
        }


@dataclass
class ProviderSchema:
    """Healthcare provider schema."""
    npi: str
    name: str
    specialty: str = ""
    address: str = ""
    enrolled_date: str = ""
    taxonomy_code: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderSchema":
        return cls(
            npi=data.get("npi", ""),
            name=data.get("name", ""),
            specialty=data.get("specialty", ""),
            address=data.get("address", ""),
            enrolled_date=data.get("enrolled_date", ""),
            taxonomy_code=data.get("taxonomy_code", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "npi": self.npi,
            "name": self.name,
            "specialty": self.specialty,
            "address": self.address,
            "enrolled_date": self.enrolled_date,
            "taxonomy_code": self.taxonomy_code,
        }


@dataclass
class CPTSchema:
    """CPT code schema."""
    code: str
    description: str
    base_rate: float = 0.0
    modifier: str = ""
    category: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPTSchema":
        return cls(
            code=data.get("code", ""),
            description=data.get("description", ""),
            base_rate=data.get("base_rate", 0.0),
            modifier=data.get("modifier", ""),
            category=data.get("category", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "base_rate": self.base_rate,
            "modifier": self.modifier,
            "category": self.category,
        }


def validate_claim(data: Dict[str, Any]) -> bool:
    """Validate data against ClaimSchema."""
    required = ["claim_id", "provider_npi", "beneficiary_id", "amount", "date"]
    for fld in required:
        if fld not in data or not data[fld]:
            return False

    # Validate amount is positive
    if data.get("amount", 0) <= 0:
        return False

    # Validate NPI format (10 digits)
    npi = data.get("provider_npi", "")
    if not _validate_npi(npi):
        return False

    return True


def validate_provider(data: Dict[str, Any]) -> bool:
    """Validate data against ProviderSchema."""
    required = ["npi", "name"]
    for fld in required:
        if fld not in data or not data[fld]:
            return False

    # Validate NPI format
    if not _validate_npi(data.get("npi", "")):
        return False

    return True


def validate_cpt(data: Dict[str, Any]) -> bool:
    """Validate data against CPTSchema."""
    required = ["code"]
    for fld in required:
        if fld not in data or not data[fld]:
            return False

    # CPT codes are 5 digits
    code = data.get("code", "")
    if not re.match(r"^\d{5}$", code):
        return False

    return True


def _validate_npi(npi: str) -> bool:
    """Validate NPI format (10 digits starting with 1 or 2)."""
    if not npi:
        return False
    # For simulation, accept any 10-digit string
    return len(npi) == 10 and npi.isdigit()
