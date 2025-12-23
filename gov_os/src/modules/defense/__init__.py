"""
Gov-OS Defense Module - DoD + Shipyard Domain

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Defense domain module for:
- Defense contract fraud detection
- Shipyard/battleship program tracking
- Subcontractor ring detection (RAF cycles)
- Cost-plus inflation patterns

Data Sources (All Public):
- GAO reports (shipbuilding, procurement)
- OIG audit reports
- DoD public testimony
"""

from .volatility import (
    SteelVolatility,
    CommodityIndex,
    LaborIndex,
    get_primary_volatility,
)

from .schema import (
    ContractSchema,
    VendorSchema,
    ShipyardSchema,
    validate_contract,
    validate_vendor,
    validate_block,
)

from .data import (
    DefenseReceipts,
    ingest_contract,
    ingest_vendor,
    ingest_shipyard_block,
)

from .receipts import (
    DefenseIngestReceipt,
    DefenseRafReceipt,
    ShipyardIterationReceipt,
    ShipyardAdditiveReceipt,
    ShipyardAssemblyReceipt,
    emit_defense_receipt,
)

from .scenarios import DEFENSE_SCENARIOS, run_defense_scenario

__all__ = [
    # Volatility
    "SteelVolatility",
    "CommodityIndex",
    "LaborIndex",
    "get_primary_volatility",
    # Schema
    "ContractSchema",
    "VendorSchema",
    "ShipyardSchema",
    "validate_contract",
    "validate_vendor",
    "validate_block",
    # Data
    "DefenseReceipts",
    "ingest_contract",
    "ingest_vendor",
    "ingest_shipyard_block",
    # Receipts
    "DefenseIngestReceipt",
    "DefenseRafReceipt",
    "ShipyardIterationReceipt",
    "ShipyardAdditiveReceipt",
    "ShipyardAssemblyReceipt",
    "emit_defense_receipt",
    # Scenarios
    "DEFENSE_SCENARIOS",
    "run_defense_scenario",
]
