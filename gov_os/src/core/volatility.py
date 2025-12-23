"""
Gov-OS Core Volatility - Abstract Base Class for Domain-Specific Indices

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Volatility index is the ONLY domain-specific parameter.
- Steel price index for shipbuilding (defense)
- Medical CPI for healthcare (medicaid)
Different indices, same adaptive_threshold() function.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class VolatilityIndex(ABC):
    """
    Abstract base class for domain-specific volatility indices.

    Domain implementations inherit and override:
    - modules/defense/volatility.py: SteelVolatility, CommodityIndex
    - modules/medicaid/volatility.py: MedicalCPI, DrugPriceIndex
    """

    def __init__(self, name: str, baseline: float = 1.0):
        """
        Initialize volatility index.

        Args:
            name: Index name
            baseline: Baseline value (1.0 = no volatility adjustment)
        """
        self._name = name
        self._baseline = baseline
        self._current_value = baseline
        self._history: Dict[str, float] = {}

    @property
    def name(self) -> str:
        """Return index name."""
        return self._name

    @abstractmethod
    def current(self) -> float:
        """
        Return current volatility factor.
        1.0 = baseline, >1.0 = above average volatility, <1.0 = below.

        Returns:
            Current volatility factor
        """
        pass

    @abstractmethod
    def historical(self, date: str) -> float:
        """
        Return historical volatility for date.

        Args:
            date: ISO date string (YYYY-MM-DD)

        Returns:
            Historical volatility factor
        """
        pass

    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update index with new data.

        Args:
            data: New data dict (format depends on index type)
        """
        pass

    @abstractmethod
    def source(self) -> str:
        """
        Return data source URL/path.

        Returns:
            Source location string
        """
        pass


class MockVolatilityIndex(VolatilityIndex):
    """
    Mock volatility index for testing.
    Returns configurable volatility without real data.
    """

    def __init__(
        self,
        name: str = "mock",
        current_value: float = 1.0,
        source_path: str = "mock://volatility",
    ):
        super().__init__(name)
        self._current_value = current_value
        self._source_path = source_path

    def current(self) -> float:
        """Return current mock volatility."""
        return self._current_value

    def historical(self, date: str) -> float:
        """Return historical mock volatility."""
        return self._history.get(date, self._baseline)

    def update(self, data: Dict[str, Any]) -> None:
        """Update mock volatility."""
        if "value" in data:
            self._current_value = data["value"]
        if "date" in data and "value" in data:
            self._history[data["date"]] = data["value"]

    def source(self) -> str:
        """Return mock source."""
        return self._source_path

    def set_current(self, value: float) -> None:
        """Set current value (for testing)."""
        self._current_value = value


class CompositeVolatilityIndex(VolatilityIndex):
    """
    Composite index combining multiple volatility sources.
    Weighted average of component indices.
    """

    def __init__(
        self,
        name: str,
        components: Dict[str, tuple],  # {name: (index, weight)}
    ):
        """
        Initialize composite index.

        Args:
            name: Composite index name
            components: Dict mapping component names to (VolatilityIndex, weight) tuples
        """
        super().__init__(name)
        self._components = components

    def current(self) -> float:
        """Return weighted average of component current values."""
        if not self._components:
            return self._baseline

        total_weight = sum(weight for _, weight in self._components.values())
        if total_weight == 0:
            return self._baseline

        weighted_sum = sum(
            index.current() * weight
            for index, weight in self._components.values()
        )

        return weighted_sum / total_weight

    def historical(self, date: str) -> float:
        """Return weighted average of component historical values."""
        if not self._components:
            return self._baseline

        total_weight = sum(weight for _, weight in self._components.values())
        if total_weight == 0:
            return self._baseline

        weighted_sum = sum(
            index.historical(date) * weight
            for index, weight in self._components.values()
        )

        return weighted_sum / total_weight

    def update(self, data: Dict[str, Any]) -> None:
        """Update component specified in data."""
        component_name = data.get("component")
        if component_name and component_name in self._components:
            index, _ = self._components[component_name]
            index.update(data)

    def source(self) -> str:
        """Return all component sources."""
        sources = [
            f"{name}: {index.source()}"
            for name, (index, _) in self._components.items()
        ]
        return "; ".join(sources)

    def add_component(
        self,
        name: str,
        index: VolatilityIndex,
        weight: float,
    ) -> None:
        """Add a component index."""
        self._components[name] = (index, weight)

    def remove_component(self, name: str) -> None:
        """Remove a component index."""
        if name in self._components:
            del self._components[name]
