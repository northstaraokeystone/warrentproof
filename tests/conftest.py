"""
WarrantProof Test Configuration

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import DISCLAIMER, TENANT_ID
from src.ledger import reset_ledger


@pytest.fixture(autouse=True)
def reset_state():
    """Reset ledger state before each test."""
    reset_ledger()
    yield
    reset_ledger()


@pytest.fixture
def disclaimer():
    """Return the standard disclaimer."""
    return DISCLAIMER


@pytest.fixture
def tenant_id():
    """Return the standard tenant ID."""
    return TENANT_ID
