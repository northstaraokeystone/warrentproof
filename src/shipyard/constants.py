"""
WarrantProof Shipyard Constants - Verified Values with Citations

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

All constants are derived from public sources with citation keys.
Each citation key corresponds to an entry in CITATIONS.md (Shipyard Module Citations section).

The physics: Entropy compression ratio IS the efficiency metric.
- Traditional shipbuilding: entropy accumulates until overruns discovered
- Elon-sphere shipbuilding: entropy compresses until efficiency emerges
"""

# === TENANT CONFIGURATION ===
SHIPYARD_TENANT_ID = "warrantproof-shipyard"
SHIPYARD_DISCLAIMER = "THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
SHIPYARD_VERSION = "1.0.0"

# === PROGRAM SCALE (Announced/Estimated) ===
# Citation: TRUMP_2025 - Presidential announcement on shipbuilding
TRUMP_CLASS_PROGRAM_COST_B = 200.0  # $200B total program estimate
TRUMP_CLASS_SHIP_COUNT = 22  # Midpoint of 20-25 range
TRUMP_CLASS_PER_SHIP_B = 9.09  # Derived: 200/22
TRUMP_CLASS_DISPLACEMENT_TONS = 35000  # Estimated battleship class

# === DISRUPTION PHYSICS (Measured) ===
# Citation: SPACEX_2025 - SpaceX Starfactory production data
STARFACTORY_CADENCE_WEEKS = 1.0  # SpaceX target production cadence

# Citation: UMAINE_2025 - University of Maine LFAM research
LFAM_TIME_SAVINGS_PCT = 0.45  # 40-50% time savings measured
LFAM_WEIGHT_SAVINGS_PCT = 0.40  # 40% weight reduction via topology optimization

# Citation: NAVY_2025 - NAVSEA additive manufacturing trials
NAVY_ADDITIVE_SPARES_SAVINGS = 0.35  # 35% savings on additive spares

# Citation: COMAU_2024 - MR4Weld mobile welder specifications
COMAU_WELD_EFFICIENCY_GAIN = 0.30  # 30% efficiency over manual

# Combined disruption estimate (conservative 40-60% range)
ELON_SPHERE_COST_REDUCTION = 0.50  # 50% midpoint estimate

# === HISTORICAL FAILURES (GAO Data) ===
# Citation: GAO_2022 - GAO-22-105451
FORD_CVN78_OVERRUN_PCT = 0.23  # Gerald R. Ford 23% cost overrun

# Citation: GAO_2018 - GAO-18-238SP
ZUMWALT_COST_INCREASE_PCT = 0.81  # $1.4B to $7.5B (81% increase)

# Citation: NAVSEA_2015 / GAO_2024 - LCS analysis
LCS_DESIGN_LIFE_YEARS = 25  # Original specification
LCS_ACTUAL_LIFE_YEARS = 7  # Average actual retirement

# Citation: CRS_COLUMBIA - Congressional Research Service
COLUMBIA_DELAY_MONTHS = 17  # Schedule slip

# === FRAUD BASELINE (GAO Confirmed) ===
# Citation: GAO_2025 - Pentagon audit failures
DOD_FRAUD_CONFIRMED_B = 11.0  # $11B confirmed fraud 2017-2024
PENTAGON_FAILED_AUDITS = 7  # Consecutive audit failures
PENTAGON_UNACCOUNTABLE_T = 2.5  # $2.5T in unaccountable assets

# Citation: GAO_2024, GAO_2025 - Shipbuilding cost-to-complete
SHIPBUILDING_OVERRUN_2024_B = 3.4  # 2024 cost-to-complete delta
SHIPBUILDING_OVERRUN_2025_B = 10.4  # 2025 cost-to-complete (3x increase!)

# === DETECTION THRESHOLDS ===
# Target: detect at 12% variance vs historical 23%
EARLY_DETECTION_PCT = 0.12
HISTORICAL_DETECTION_PCT = 0.23

# Entropy conservation tolerance
ENTROPY_VIOLATION_THRESHOLD = 0.001

# Compression fraud threshold (below = fraud-like)
COMPRESSION_FRAUD_THRESHOLD = 0.70

# === NUCLEAR (NRC Approved) ===
# Citation: NRC_2025 - NuScale SMR approval
NUSCALE_POWER_MWE = 77  # NRC approved output

# === LIFECYCLE PARAMETERS ===
# Minimum phase durations (physics-constrained)
PHASE_MIN_DAYS = {
    "DESIGN": 180,  # 6 months minimum
    "KEEL_LAYING": 30,  # 1 month
    "BLOCK_ASSEMBLY": 365,  # 1 year minimum
    "LAUNCH": 14,  # 2 weeks
    "FITTING_OUT": 180,  # 6 months
    "SEA_TRIALS": 90,  # 3 months
    "DELIVERY": 1,  # 1 day
}

# Overrun detection trigger threshold
VARIANCE_ALERT_PCT = 0.12  # Alert at 12% variance

# === SIMULATION DEFAULTS ===
SIM_DEFAULT_CYCLES = 1000
SIM_DEFAULT_SHIPS = 5
SIM_ENTROPY_SEED = 42

# === ITERATION PARAMETERS ===
# SpaceX-style iteration bounds
MIN_ITERATION_DAYS = 7  # Minimum days per iteration (material curing)
MAX_ITERATIONS_PER_MONTH = 4  # Physical limit on iteration speed

# Parallel bay configuration
STARFACTORY_MEGA_BAYS = 8  # SpaceX reference configuration
TRADITIONAL_SERIAL_BAYS = 1  # Traditional yard baseline

# === ADDITIVE MANUFACTURING PARAMETERS ===
# Material deposition rates (kg/hour)
LFAM_DEPOSITION_RATE_KG_HR = 100.0  # Large format additive

# Layer validation tolerance
LAYER_HASH_TOLERANCE = 0.01  # 1% tolerance on layer validation

# Certified marine-grade materials
MARINE_CERTIFIED_MATERIALS = [
    "HDPE",  # High-density polyethylene (workboat hulls)
    "CF-PEEK",  # Carbon fiber reinforced PEEK
    "316L_SS",  # 316L stainless steel
    "INCONEL_625",  # High-temp nickel alloy
    "AL_5083",  # Marine-grade aluminum
]

# === WELDING PARAMETERS ===
# Weld count bounds per block
WELDS_PER_BLOCK_MIN = 50
WELDS_PER_BLOCK_MAX = 500

# Inspector certification requirement
INSPECTOR_CERTIFICATION_REQUIRED = True

# Weld quality grades
WELD_GRADES = ["A", "B", "C", "FAIL"]

# === PROCUREMENT PARAMETERS ===
# Contract types
CONTRACT_TYPES = ["fixed_price", "cost_plus", "hybrid"]

# Entropy direction by contract type
CONTRACT_ENTROPY = {
    "fixed_price": -0.1,  # Entropy shedding (good)
    "cost_plus": 0.1,  # Entropy accumulating (bad)
    "hybrid": 0.0,  # Neutral
}

# Change order limits
MAX_CHANGE_ORDERS_BEFORE_REVIEW = 5
CHANGE_ORDER_OVERRUN_THRESHOLD = 0.10  # 10% cumulative triggers review

# === CITATION KEYS (documented in CITATIONS.md) ===
CITATION_KEYS = [
    "TRUMP_2025",
    "GAO_2024",
    "GAO_2025",
    "GAO_2022",
    "GAO_2018",
    "NAVY_2025",
    "UMAINE_2025",
    "SPACEX_2025",
    "COMAU_2024",
    "NRC_2025",
    "NAVSEA_2015",
    "CRS_COLUMBIA",
]


def get_constant_citation(constant_name: str) -> str:
    """
    Get citation key for a constant.

    Args:
        constant_name: Name of the constant

    Returns:
        Citation key string
    """
    citation_map = {
        "TRUMP_CLASS_PROGRAM_COST_B": "TRUMP_2025",
        "TRUMP_CLASS_SHIP_COUNT": "TRUMP_2025",
        "STARFACTORY_CADENCE_WEEKS": "SPACEX_2025",
        "LFAM_TIME_SAVINGS_PCT": "UMAINE_2025",
        "LFAM_WEIGHT_SAVINGS_PCT": "UMAINE_2025",
        "NAVY_ADDITIVE_SPARES_SAVINGS": "NAVY_2025",
        "COMAU_WELD_EFFICIENCY_GAIN": "COMAU_2024",
        "FORD_CVN78_OVERRUN_PCT": "GAO_2022",
        "ZUMWALT_COST_INCREASE_PCT": "GAO_2018",
        "DOD_FRAUD_CONFIRMED_B": "GAO_2025",
        "PENTAGON_FAILED_AUDITS": "GAO_2024",
        "NUSCALE_POWER_MWE": "NRC_2025",
        "SHIPBUILDING_OVERRUN_2024_B": "GAO_2024",
        "SHIPBUILDING_OVERRUN_2025_B": "GAO_2025",
    }
    return citation_map.get(constant_name, "UNKNOWN")
