"""
Gov-OS Medicaid Module - CMS/HHS Healthcare Domain

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Medicaid domain module for:
- Healthcare claim fraud detection
- Provider referral ring detection (RAF cycles)
- Upcoding patterns
- Phantom billing detection

Data Sources (All Public):
- CMS/HHS published statistics
- OIG audit reports
- Academic research on healthcare fraud
"""

from .volatility import (
    MedicalCPI,
    DrugPriceIndex,
    ProcedureCostIndex,
    get_primary_volatility,
)

from .schema import (
    ClaimSchema,
    ProviderSchema,
    CPTSchema,
    validate_claim,
    validate_provider,
    validate_cpt,
)

from .data import (
    MedicaidReceipts,
    ingest_claim,
    ingest_provider,
)

from .receipts import (
    MedicaidIngestReceipt,
    MedicaidRafReceipt,
    ReferralRingReceipt,
    emit_medicaid_receipt,
)

from .scenarios import MEDICAID_SCENARIOS, run_medicaid_scenario

__all__ = [
    # Volatility
    "MedicalCPI",
    "DrugPriceIndex",
    "ProcedureCostIndex",
    "get_primary_volatility",
    # Schema
    "ClaimSchema",
    "ProviderSchema",
    "CPTSchema",
    "validate_claim",
    "validate_provider",
    "validate_cpt",
    # Data
    "MedicaidReceipts",
    "ingest_claim",
    "ingest_provider",
    # Receipts
    "MedicaidIngestReceipt",
    "MedicaidRafReceipt",
    "ReferralRingReceipt",
    "emit_medicaid_receipt",
    # Scenarios
    "MEDICAID_SCENARIOS",
    "run_medicaid_scenario",
]
