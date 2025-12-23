"""
RAZOR Test Fixtures
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.fixture
def repetitive_text():
    """Highly repetitive text that should compress well."""
    return "husbanding services for naval vessel " * 100


@pytest.fixture
def random_text():
    """Pseudo-random text that should not compress well."""
    import random
    random.seed(12345)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    # Generate truly random string with no repeating pattern
    return "".join(random.choice(chars) for _ in range(3000))


@pytest.fixture
def sample_fraud_data():
    """Synthetic fraud cohort data for testing."""
    if not HAS_PANDAS:
        pytest.skip("pandas required")

    import random
    random.seed(42)

    data = {
        "award_id": [f"AWARD-{i:04d}" for i in range(100)],
        "recipient_name": ["FRAUD CORP"] * 100,
        "description": ["husbanding services for ship " * 10] * 100,
        "total_obligation": [random.uniform(10000, 100000) for _ in range(100)],
        "naics_code": ["488390"] * 100,
        "psc_code": ["M1"] * 100,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_control_data():
    """Synthetic control cohort data for testing."""
    if not HAS_PANDAS:
        pytest.skip("pandas required")

    import random
    random.seed(43)

    # More varied descriptions for control
    descriptions = [
        "Maritime support services including fuel provisioning and waste management",
        "Port logistics coordination and vessel scheduling assistance",
        "Dockside repair and maintenance of naval equipment systems",
        "Harbor pilot services and navigation assistance for vessel approach",
        "Emergency response and damage control supplies procurement",
    ]

    data = {
        "award_id": [f"CTRL-{i:04d}" for i in range(200)],
        "recipient_name": [f"VENDOR-{i % 20}" for i in range(200)],
        "description": [descriptions[i % len(descriptions)] for i in range(200)],
        "total_obligation": [random.uniform(5000, 50000) for _ in range(200)],
        "naics_code": ["488390"] * 200,
        "psc_code": ["M1"] * 200,
    }
    return pd.DataFrame(data)


@pytest.fixture
def fraud_cr_values():
    """Synthetic fraud cohort compression ratios."""
    import random
    random.seed(42)
    # Lower values = more compressible = fraud-like
    return [random.gauss(0.35, 0.08) for _ in range(75)]


@pytest.fixture
def control_cr_values():
    """Synthetic control cohort compression ratios."""
    import random
    random.seed(43)
    # Higher values = less compressible = legitimate
    return [random.gauss(0.65, 0.10) for _ in range(150)]
