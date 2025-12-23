"""
Gov-OS Modules - Domain-Specific Configurations

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Each module provides:
- config.yaml: Domain configuration
- volatility.py: Domain-specific volatility index
- schema.py: Data schemas
- data.py: Ingest adapters
- receipts.py: Domain-specific receipt types
- scenarios.py: Domain-specific test scenarios

Modules supply only volatility + schema. Core handles heavy physics.
"""

from typing import List

AVAILABLE_MODULES = ["defense", "medicaid"]


def list_modules() -> List[str]:
    """Return list of available domain modules."""
    return AVAILABLE_MODULES.copy()
