"""
WarrantProof SAM Validator Module - SAM.gov Entity Validation as Certificate Authority

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements SAM.gov entity validation as a Certificate Authority
in the ZKP infrastructure. All vendor attributes must be cryptographically
signed. N/A fields are rejected with zero tolerance.

Key Insight:
SAM.gov acts as a Certificate Authority. Vendor identity claims must be
cryptographically verified, not just checked against a database.

OMEGA Citation:
"SAM.gov entity validation as Certificate Authority. Reject N/A fields.
Cryptographic signatures required."
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .core import (
    TENANT_ID,
    DISCLAIMER,
    SAM_CA_TRUST_THRESHOLD,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class EntityData:
    """SAM.gov entity data structure."""
    duns: str = ""
    cage: str = ""
    uei: str = ""  # Unique Entity Identifier
    legal_business_name: str = ""
    dba_name: str = ""
    physical_address: Dict[str, str] = field(default_factory=dict)
    registration_date: str = ""
    expiration_date: str = ""
    active_status: str = ""
    exclusion_status: str = ""
    ownership_details: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[bytes] = None


@dataclass
class ValidationResult:
    """Result of SAM.gov validation."""
    entity_valid: bool = False
    signature_verified: bool = False
    na_fields_found: int = 0
    na_fields_rejected: bool = False
    expired: bool = False
    excluded: bool = False
    trust_score: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)


# Fields that require cryptographic signatures
SIGNATURE_REQUIRED_FIELDS = [
    "legal_business_name",
    "physical_address",
    "ownership_details",
]

# Fields that cannot be N/A
ZERO_TOLERANCE_FIELDS = [
    "legal_business_name",
    "duns",
    "cage",
    "uei",
    "physical_address",
    "active_status",
]


def fetch_entity(
    duns: Optional[str] = None,
    cage: Optional[str] = None,
    uei: Optional[str] = None,
    _simulate: bool = True
) -> EntityData:
    """
    Query SAM.gov entity API.

    Args:
        duns: DUNS number
        cage: CAGE code
        uei: Unique Entity Identifier
        _simulate: If True, return simulated data

    Returns:
        EntityData from SAM.gov
    """
    if _simulate:
        return _simulate_entity(duns, cage, uei)

    # In production: Query SAM.gov API
    # https://api.sam.gov/entity-information/v3/entities
    raise NotImplementedError("Production SAM.gov API integration not implemented")


def validate_signature(
    entity_data: EntityData,
    signature: Optional[bytes] = None
) -> bool:
    """
    Verify cryptographic signature on entity attributes.
    In production, this would verify against SAM.gov's public key.

    Args:
        entity_data: Entity data to verify
        signature: Optional signature (uses entity_data.signature if not provided)

    Returns:
        True if signature is valid
    """
    sig = signature or entity_data.signature

    if not sig:
        return False

    # Simulate signature verification
    # In production: Use cryptographic library to verify against SAM.gov CA
    signed_data = json.dumps({
        "legal_business_name": entity_data.legal_business_name,
        "physical_address": entity_data.physical_address,
        "ownership_details": entity_data.ownership_details,
    }, sort_keys=True)

    expected_sig_prefix = hashlib.sha256(signed_data.encode()).hexdigest()[:16]

    # Check if signature matches (simulated)
    sig_hex = sig.hex() if isinstance(sig, bytes) else str(sig)
    return sig_hex.startswith(expected_sig_prefix)


def reject_na_fields(entity_data: EntityData) -> List[str]:
    """
    Check for "N/A", "Not Available", null fields.
    Reject entity if any found in zero-tolerance fields.

    Args:
        entity_data: Entity data to check

    Returns:
        List of fields with N/A values
    """
    na_fields = []
    na_indicators = ["n/a", "not available", "null", "none", "unknown", "-", ""]

    for field_name in ZERO_TOLERANCE_FIELDS:
        value = getattr(entity_data, field_name, None)

        if value is None:
            na_fields.append(field_name)
        elif isinstance(value, str):
            if value.lower().strip() in na_indicators:
                na_fields.append(field_name)
        elif isinstance(value, dict):
            if not value or all(v in na_indicators or v is None for v in value.values()):
                na_fields.append(field_name)

    return na_fields


def calculate_ca_trust_score(entity_data: EntityData) -> float:
    """
    Compute trust score based on registration age, update frequency, verification status.

    Args:
        entity_data: Entity data

    Returns:
        Trust score 0-1
    """
    scores = []

    # 1. Registration age score
    if entity_data.registration_date:
        try:
            from datetime import datetime
            reg_date = datetime.strptime(entity_data.registration_date[:10], "%Y-%m-%d")
            age_days = (datetime.now() - reg_date).days
            age_score = min(1.0, age_days / 365)  # Max score after 1 year
            scores.append(age_score)
        except ValueError:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 2. Expiration status score
    if entity_data.expiration_date:
        try:
            from datetime import datetime
            exp_date = datetime.strptime(entity_data.expiration_date[:10], "%Y-%m-%d")
            if exp_date > datetime.now():
                days_until_exp = (exp_date - datetime.now()).days
                exp_score = min(1.0, days_until_exp / 180)  # Full score if >6 months
                scores.append(exp_score)
            else:
                scores.append(0.0)  # Expired
        except ValueError:
            scores.append(0.5)
    else:
        scores.append(0.5)

    # 3. Active status score
    if entity_data.active_status:
        if entity_data.active_status.lower() == "active":
            scores.append(1.0)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 4. Exclusion status score
    if entity_data.exclusion_status:
        if entity_data.exclusion_status.lower() in ["none", "not excluded", ""]:
            scores.append(1.0)
        else:
            scores.append(0.0)  # Excluded entity
    else:
        scores.append(0.5)

    # 5. Signature presence score
    scores.append(1.0 if entity_data.signature else 0.0)

    # 6. Data completeness score
    total_fields = len(ZERO_TOLERANCE_FIELDS)
    na_fields = len(reject_na_fields(entity_data))
    completeness = (total_fields - na_fields) / total_fields
    scores.append(completeness)

    return sum(scores) / len(scores) if scores else 0.0


def validate_entity(entity_data: EntityData) -> ValidationResult:
    """
    Full entity validation including signature, N/A check, trust score.

    Args:
        entity_data: Entity to validate

    Returns:
        ValidationResult with all checks
    """
    result = ValidationResult()

    # Check for N/A fields
    na_fields = reject_na_fields(entity_data)
    result.na_fields_found = len(na_fields)
    result.na_fields_rejected = len(na_fields) > 0

    # Verify signature
    result.signature_verified = validate_signature(entity_data)

    # Check expiration
    if entity_data.expiration_date:
        try:
            from datetime import datetime
            exp_date = datetime.strptime(entity_data.expiration_date[:10], "%Y-%m-%d")
            result.expired = exp_date < datetime.now()
        except ValueError:
            result.expired = False

    # Check exclusion
    if entity_data.exclusion_status:
        result.excluded = entity_data.exclusion_status.lower() not in ["none", "not excluded", ""]

    # Calculate trust score
    result.trust_score = calculate_ca_trust_score(entity_data)

    # Overall validity
    result.entity_valid = (
        not result.na_fields_rejected
        and result.signature_verified
        and not result.expired
        and not result.excluded
        and result.trust_score >= SAM_CA_TRUST_THRESHOLD
    )

    result.validation_details = {
        "na_fields": na_fields,
        "signature_present": entity_data.signature is not None,
        "active_status": entity_data.active_status,
    }

    return result


def emit_sam_validation_receipt(
    entity_data: EntityData,
    validation_result: Optional[ValidationResult] = None
) -> dict:
    """
    Emit sam_validation_receipt documenting entity validation.

    Args:
        entity_data: Entity that was validated
        validation_result: Pre-computed validation result

    Returns:
        sam_validation_receipt dict
    """
    if validation_result is None:
        validation_result = validate_entity(entity_data)

    return emit_receipt("sam_validation", {
        "tenant_id": TENANT_ID,
        "duns": entity_data.duns,
        "cage": entity_data.cage,
        "uei": entity_data.uei,
        "entity_valid": validation_result.entity_valid,
        "signature_verified": validation_result.signature_verified,
        "na_fields_found": validation_result.na_fields_found,
        "na_fields_rejected": validation_result.na_fields_rejected,
        "ca_trust_score": round(validation_result.trust_score, 4),
        "trust_threshold": SAM_CA_TRUST_THRESHOLD,
        "expired": validation_result.expired,
        "excluded": validation_result.excluded,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === SIMULATION DATA ===

def _simulate_entity(
    duns: Optional[str] = None,
    cage: Optional[str] = None,
    uei: Optional[str] = None
) -> EntityData:
    """Generate simulated entity data."""
    import hashlib
    seed = hashlib.sha256((duns or cage or uei or "default").encode()).hexdigest()

    # Create signed data for signature
    entity = EntityData(
        duns=duns or f"DUNS{seed[:9].upper()}",
        cage=cage or seed[:5].upper(),
        uei=uei or f"UEI{seed[:12].upper()}",
        legal_business_name=f"Simulated Contractor {seed[:4].upper()}",
        dba_name=f"SimCo {seed[:3]}",
        physical_address={
            "street": f"{seed[:3]} Main Street",
            "city": "Washington",
            "state": "DC",
            "zip": "20001",
        },
        registration_date="2020-01-15",
        expiration_date="2025-12-31",
        active_status="Active",
        exclusion_status="None",
        ownership_details={"type": "Corporation", "country": "USA"},
    )

    # Generate simulated signature
    signed_data = json.dumps({
        "legal_business_name": entity.legal_business_name,
        "physical_address": entity.physical_address,
        "ownership_details": entity.ownership_details,
    }, sort_keys=True)
    entity.signature = hashlib.sha256(signed_data.encode()).digest()

    return entity


# === STOPRULES ===

def stoprule_na_fields(na_fields: List[str]) -> None:
    """Reject transaction if N/A fields present."""
    if na_fields:
        emit_receipt("anomaly", {
            "metric": "na_fields_detected",
            "fields": na_fields,
            "action": "reject_transaction",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"N/A fields detected (zero tolerance): {na_fields}")


def stoprule_signature_invalid(entity_id: str) -> None:
    """Reject if signature fails verification."""
    emit_receipt("anomaly", {
        "metric": "signature_invalid",
        "entity_id": entity_id,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Signature verification failed for entity: {entity_id}")


def stoprule_entity_expired(entity_id: str, expiration_date: str) -> None:
    """Reject if SAM.gov registration expired."""
    emit_receipt("anomaly", {
        "metric": "entity_expired",
        "entity_id": entity_id,
        "expiration_date": expiration_date,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Entity registration expired: {entity_id} on {expiration_date}")


def stoprule_entity_excluded(entity_id: str) -> None:
    """Reject if entity is excluded from contracting."""
    emit_receipt("anomaly", {
        "metric": "entity_excluded",
        "entity_id": entity_id,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Entity is excluded from contracting: {entity_id}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof SAM Validator Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Fetch simulated entity
    entity = fetch_entity(duns="123456789", _simulate=True)
    print(f"# Entity: {entity.legal_business_name}", file=sys.stderr)
    assert entity.duns is not None

    # Test 2: Validate signature
    sig_valid = validate_signature(entity)
    print(f"# Signature valid: {sig_valid}", file=sys.stderr)
    assert sig_valid == True

    # Test 3: Check N/A fields on valid entity
    na_fields = reject_na_fields(entity)
    print(f"# N/A fields found: {len(na_fields)}", file=sys.stderr)
    assert len(na_fields) == 0

    # Test 4: Check N/A fields on invalid entity
    invalid_entity = EntityData(
        duns="N/A",
        cage="",
        legal_business_name="Not Available",
    )
    na_invalid = reject_na_fields(invalid_entity)
    print(f"# Invalid entity N/A fields: {na_invalid}", file=sys.stderr)
    assert len(na_invalid) > 0

    # Test 5: Calculate trust score
    trust_score = calculate_ca_trust_score(entity)
    print(f"# Trust score: {trust_score:.4f}", file=sys.stderr)
    assert trust_score > 0

    # Test 6: Full validation on valid entity
    result = validate_entity(entity)
    print(f"# Entity valid: {result.entity_valid}", file=sys.stderr)
    print(f"# Trust score: {result.trust_score:.4f}", file=sys.stderr)
    assert result.entity_valid == True

    # Test 7: Full validation on invalid entity
    result_invalid = validate_entity(invalid_entity)
    print(f"# Invalid entity valid: {result_invalid.entity_valid}", file=sys.stderr)
    assert result_invalid.entity_valid == False

    # Test 8: Emit receipt
    receipt = emit_sam_validation_receipt(entity, result)
    assert receipt["receipt_type"] == "sam_validation"
    assert receipt["entity_valid"] == True

    print(f"# PASS: sam_validator module self-test", file=sys.stderr)
