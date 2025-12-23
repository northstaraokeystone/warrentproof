"""
Gov-OS Core Volatility Index - Base Classes for Domain-Specific Indices

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class VolatilityIndex(ABC):
    """
    Base class for volatility indices.
    Tracks price/cost changes over time for anomaly detection.
    """
    name: str
    base_date: datetime
    base_value: float = 100.0

    @abstractmethod
    def get_current_value(self) -> float:
        """Get current index value."""
        pass

    @abstractmethod
    def get_change_pct(self, days: int = 30) -> float:
        """Get percentage change over period."""
        pass


class MockVolatilityIndex(VolatilityIndex):
    """
    Mock volatility index for simulation.
    Returns configurable static values for testing.
    """
    name: str = "mock_index"
    base_date: datetime = datetime(2024, 1, 1)
    base_value: float = 100.0
    current_value: float = 105.0
    change_pct: float = 5.0

    def __init__(
        self,
        name: str = "mock_index",
        current_value: float = 105.0,
        change_pct: float = 5.0,
    ):
        self.name = name
        self.base_date = datetime(2024, 1, 1)
        self.base_value = 100.0
        self.current_value = current_value
        self.change_pct = change_pct

    def get_current_value(self) -> float:
        return self.current_value

    def get_change_pct(self, days: int = 30) -> float:
        return self.change_pct
