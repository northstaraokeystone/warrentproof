"""
Gov-OS Medicaid Volatility - Medical CPI and Drug Price Indices

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import csv
import os
from typing import Any, Dict, Optional

from ...core.volatility import VolatilityIndex, MockVolatilityIndex


class MedicalCPI(VolatilityIndex):
    """Medical Consumer Price Index."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("medical_cpi")
        self._data_path = data_path or self._default_path()
        self._load_data()

    def _default_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(base, "data", "volatility_indices", "medical_cpi.csv")

    def _load_data(self) -> None:
        """Load historical data from CSV."""
        self._history = {}
        if os.path.exists(self._data_path):
            try:
                with open(self._data_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date = row.get("date", "")
                        value = float(row.get("index", 1.0))
                        self._history[date] = value
                        self._current_value = value
            except Exception:
                self._current_value = 1.0

    def current(self) -> float:
        return self._current_value

    def historical(self, date: str) -> float:
        return self._history.get(date, self._baseline)

    def update(self, data: Dict[str, Any]) -> None:
        if "value" in data:
            self._current_value = data["value"]
        if "date" in data and "value" in data:
            self._history[data["date"]] = data["value"]

    def source(self) -> str:
        return self._data_path


class DrugPriceIndex(VolatilityIndex):
    """Prescription drug price index."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("drug_price_index")
        self._data_path = data_path or self._default_path()
        self._current_value = 1.0
        self._history = {}

    def _default_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(base, "data", "volatility_indices", "drug_price_index.csv")

    def current(self) -> float:
        return self._current_value

    def historical(self, date: str) -> float:
        return self._history.get(date, self._baseline)

    def update(self, data: Dict[str, Any]) -> None:
        if "value" in data:
            self._current_value = data["value"]
        if "date" in data and "value" in data:
            self._history[data["date"]] = data["value"]

    def source(self) -> str:
        return self._data_path


class ProcedureCostIndex(VolatilityIndex):
    """Medical procedure cost index."""

    def __init__(self):
        super().__init__("procedure_cost_index")
        self._current_value = 1.0
        self._history = {}

    def current(self) -> float:
        return self._current_value

    def historical(self, date: str) -> float:
        return self._history.get(date, self._baseline)

    def update(self, data: Dict[str, Any]) -> None:
        if "value" in data:
            self._current_value = data["value"]
        if "date" in data and "value" in data:
            self._history[data["date"]] = data["value"]

    def source(self) -> str:
        return "CMS Procedure Cost Data"


def get_primary_volatility() -> VolatilityIndex:
    """Get the primary volatility index for medicaid domain."""
    cpi = MedicalCPI()
    if cpi.current() == 1.0 and not cpi._history:
        return MockVolatilityIndex(name="medicaid_mock", current_value=1.0)
    return cpi
