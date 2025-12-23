"""
RAZOR Cohorts Module - Historical Fraud Cohort Definitions

Centralized definitions for all historical fraud cohorts and their controls.
These are KNOWN fraud cases with legal adjudication - ground truth for
calibrating the Kolmogorov complexity hypothesis.

COHORTS:
  1. Fat Leonard (GDMA) - Repetitive husbanding invoices
  2. TransDigm - Monopoly pricing on simple parts
  3. Boeing KC-767 - Conflict-of-interest sole-source awards

THE PHYSICS:
  Each cohort represents a different fraud signature:
  - Fat Leonard: Copy-paste fraud (zlib catches it)
  - TransDigm: Value-complexity decoupling (price >> description complexity)
  - Boeing: Boilerplate justifications (template fraud)
"""

from typing import Dict, List, Optional

from .core import (
    FAT_LEONARD_START,
    FAT_LEONARD_END,
    TRANSDIGM_START,
    TRANSDIGM_END,
    BOEING_START,
    BOEING_END,
    emit_receipt,
    StopRule,
)

# ============================================================================
# FRAUD COHORT CONFIGURATIONS
# ============================================================================

FAT_LEONARD_CONFIG = {
    "keywords": ["husbanding", "cht", "sewage", "fender", "force protection"],
    "recipient_search_text": ["Glenn Defense Marine Asia", "GDMA"],
    "agencies": [
        {"type": "awarding", "tier": "toptier", "name": "Department of the Navy"}
    ],
    "time_period": [
        {"start_date": FAT_LEONARD_START, "end_date": FAT_LEONARD_END}
    ],
    "award_type_codes": ["A", "B", "C", "D"],  # Contracts only
}

TRANSDIGM_CONFIG = {
    # TransDigm subsidiaries identified by CAGE codes
    "recipient_search_text": ["TransDigm", "MarathonNorco", "TA Aerospace", "Whippany Actuation"],
    "agencies": [
        {"type": "awarding", "tier": "toptier", "name": "Defense Logistics Agency"},
        {"type": "awarding", "tier": "toptier", "name": "Department of the Air Force"},
    ],
    "time_period": [
        {"start_date": TRANSDIGM_START, "end_date": TRANSDIGM_END}
    ],
    "award_type_codes": ["A", "B", "C", "D"],
}

BOEING_CONFIG = {
    "keywords": ["KC-767", "tanker", "lease"],
    "recipient_search_text": ["The Boeing Company", "Boeing"],
    "agencies": [
        {"type": "awarding", "tier": "toptier", "name": "Department of the Air Force"}
    ],
    "time_period": [
        {"start_date": BOEING_START, "end_date": BOEING_END}
    ],
    "award_type_codes": ["A", "B"],  # Major contracts
}

# ============================================================================
# COHORT REGISTRY
# ============================================================================

COHORTS: Dict[str, dict] = {
    "fat_leonard": {
        "name": "Fat Leonard (GDMA)",
        "fraud_config": FAT_LEONARD_CONFIG,
        "control_config": {
            "naics_codes": ["488390", "488320"],  # Water transportation support
            "psc_codes": ["M1", "J0"],  # Operation of facilities, maintenance
            "exclude_keywords": ["Glenn Defense Marine Asia", "GDMA"],
            "time_period": {"start_date": FAT_LEONARD_START, "end_date": FAT_LEONARD_END},
            "agencies": [
                {"type": "awarding", "tier": "toptier", "name": "Department of the Navy"}
            ],
        },
        "hypothesis": "Repetitive husbanding invoices compress better than legitimate port services",
        "expected_z_score": "< -2.5",
        "keywords_expected": ["husbanding", "cht", "sewage"],
        "fraud_type": "copy_paste",  # zlib catches it
        "description": (
            "Leonard Glenn Francis bribed Navy officials to steer contracts to "
            "Glenn Defense Marine Asia (GDMA) for port services in the Pacific. "
            "Invoices were repetitive and inflated, billing for services not rendered."
        ),
    },
    "transdigm": {
        "name": "TransDigm Monopoly Pricing",
        "fraud_config": TRANSDIGM_CONFIG,
        "control_config": {
            "naics_codes": ["336413", "336412"],  # Aircraft parts manufacturing
            "psc_codes": ["1680", "1560"],  # Aircraft accessories and components
            "exclude_keywords": ["TransDigm", "MarathonNorco", "TA Aerospace", "Whippany"],
            "time_period": {"start_date": TRANSDIGM_START, "end_date": TRANSDIGM_END},
            "agencies": [
                {"type": "awarding", "tier": "toptier", "name": "Defense Logistics Agency"}
            ],
        },
        "hypothesis": (
            "Simple part descriptions (low K(x)) with high prices. "
            "Ratio log(amount)/K(description) is anomalous"
        ),
        "expected_z_score": "< -1.5",
        "price_complexity_ratio": "> 3.0",  # Novel metric: log(price) / CR
        "fraud_type": "value_decoupling",  # Price >> complexity
        "description": (
            "TransDigm acquired sole-source suppliers and raised prices by 100-1000%. "
            "Simple parts like 'Clutch Disc' priced at $7,000 vs fair market $173. "
            "Low complexity descriptions masking extreme markup."
        ),
    },
    "boeing": {
        "name": "Boeing KC-767 Tanker Conflict",
        "fraud_config": BOEING_CONFIG,
        "control_config": {
            "naics_codes": ["336411"],  # Aircraft Manufacturing
            "psc_codes": ["1510", "1520"],  # Aircraft, Fixed Wing
            "exclude_keywords": ["The Boeing Company", "Boeing"],
            "time_period": {"start_date": BOEING_START, "end_date": BOEING_END},
            "agencies": [
                {"type": "awarding", "tier": "toptier", "name": "Department of the Air Force"}
            ],
        },
        "hypothesis": (
            "Sole-source justifications use boilerplate language (low complexity) "
            "vs competitive awards"
        ),
        "expected_z_score": "< -1.8",
        "focus_field": "description",  # Award justification text
        "fraud_type": "template_fraud",  # bz2 catches structural reordering
        "description": (
            "Boeing CFO and Air Force official conspired on KC-767 tanker lease deal. "
            "Sole-source justifications used templated language to avoid competition. "
            "Both were later convicted."
        ),
    },
}

# ============================================================================
# COHORT FUNCTIONS
# ============================================================================

def get_cohort_config(cohort_name: str) -> dict:
    """
    Return fraud + control configs for named cohort.

    Args:
        cohort_name: One of 'fat_leonard', 'transdigm', 'boeing'

    Returns:
        Complete cohort configuration dict

    Raises:
        StopRule: If cohort name is invalid
    """
    if cohort_name not in COHORTS:
        emit_receipt("anomaly", {
            "metric": "invalid_cohort",
            "cohort_name": cohort_name,
            "valid_cohorts": list(COHORTS.keys()),
            "action": "halt",
            "classification": "violation"
        })
        raise StopRule(f"Invalid cohort: {cohort_name}. Valid: {list(COHORTS.keys())}")

    return COHORTS[cohort_name].copy()


def list_cohorts() -> List[str]:
    """
    Return all cohort names.

    Returns:
        List of cohort name strings
    """
    return list(COHORTS.keys())


def validate_cohort(cohort_name: str) -> bool:
    """
    Check that cohort exists and has valid config.

    Args:
        cohort_name: Cohort name to validate

    Returns:
        True if valid, False otherwise
    """
    if cohort_name not in COHORTS:
        return False

    cohort = COHORTS[cohort_name]
    required_keys = ["name", "fraud_config", "control_config", "hypothesis"]

    for key in required_keys:
        if key not in cohort:
            return False

    # Validate fraud_config has time_period
    if "time_period" not in cohort["fraud_config"]:
        return False

    return True


def get_cohort_description(cohort_name: str) -> str:
    """
    Get human-readable description of a cohort.

    Args:
        cohort_name: Cohort name

    Returns:
        Description string
    """
    if cohort_name not in COHORTS:
        return f"Unknown cohort: {cohort_name}"

    cohort = COHORTS[cohort_name]
    return cohort.get("description", cohort.get("hypothesis", "No description"))


def get_fraud_type(cohort_name: str) -> str:
    """
    Get the fraud type classification for a cohort.

    Args:
        cohort_name: Cohort name

    Returns:
        Fraud type string (copy_paste, value_decoupling, template_fraud)
    """
    if cohort_name not in COHORTS:
        return "unknown"

    return COHORTS[cohort_name].get("fraud_type", "unknown")


def get_expected_signal(cohort_name: str) -> dict:
    """
    Get expected statistical signal for a cohort.

    Args:
        cohort_name: Cohort name

    Returns:
        Dict with expected Z-score and other metrics
    """
    if cohort_name not in COHORTS:
        return {}

    cohort = COHORTS[cohort_name]
    return {
        "expected_z_score": cohort.get("expected_z_score"),
        "hypothesis": cohort.get("hypothesis"),
        "fraud_type": cohort.get("fraud_type"),
    }


# ============================================================================
# COHORT SUMMARY RECEIPT
# ============================================================================

def emit_cohort_summary_receipt() -> dict:
    """
    Emit a summary receipt of all available cohorts.

    Returns:
        Receipt dict with cohort summary
    """
    summary = {
        "cohort_count": len(COHORTS),
        "cohorts": {}
    }

    for name, config in COHORTS.items():
        summary["cohorts"][name] = {
            "display_name": config["name"],
            "fraud_type": config.get("fraud_type"),
            "expected_z_score": config.get("expected_z_score"),
            "time_period": config["fraud_config"]["time_period"],
        }

    return emit_receipt("cohort_summary", summary, to_stdout=False)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print(f"# RAZOR Cohorts Module", file=sys.stderr)
    print(f"# Available cohorts: {list_cohorts()}", file=sys.stderr)

    for name in list_cohorts():
        assert validate_cohort(name), f"Invalid cohort config: {name}"
        config = get_cohort_config(name)
        print(f"#   {name}: {config['name']}", file=sys.stderr)
        print(f"#     Fraud type: {get_fraud_type(name)}", file=sys.stderr)
        print(f"#     Hypothesis: {config['hypothesis'][:60]}...", file=sys.stderr)

    print("# PASS: RAZOR cohorts module self-test", file=sys.stderr)
