"""
Gov-OS Simulation Harness - CLAUDEME v3.1 Compliant

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Provides simulation state management and scenario execution for
domain-specific test scenarios.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from .constants import DISCLAIMER, TENANT_ID
from .receipt import emit_receipt


@dataclass
class SimState:
    """
    Simulation state container.

    Tracks receipts, entities, and metrics across a simulation run.
    """
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    domain: str = "unknown"
    simulation_flag: str = DISCLAIMER

    def add_receipt(self, receipt: Dict[str, Any]) -> None:
        """Add a receipt to the simulation state."""
        self.receipts.append(receipt)

    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """Add or update an entity in the simulation."""
        self.entities[entity_id] = entity_data

    def set_metric(self, name: str, value: float) -> None:
        """Set a metric value."""
        self.metrics[name] = value

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()


@dataclass
class ScenarioResult:
    """
    Result of a scenario execution.

    Contains detection results, metrics, and audit trail.
    """
    scenario_name: str
    success: bool
    detection_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    simulation_flag: str = DISCLAIMER

    def accuracy(self) -> float:
        """Calculate detection accuracy."""
        total = self.detection_count + self.false_positive_count + self.false_negative_count
        if total == 0:
            return 1.0
        correct = self.detection_count
        return correct / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "detection_count": self.detection_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "accuracy": self.accuracy(),
            "metrics": self.metrics,
            "errors": self.errors,
            "simulation_flag": self.simulation_flag,
        }


def run_simulation(
    scenario_name: str,
    data: List[Dict[str, Any]],
    detector: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    ground_truth: Optional[List[str]] = None,
    domain: str = "unknown",
) -> ScenarioResult:
    """
    Run a simulation scenario.

    Args:
        scenario_name: Name of the scenario
        data: Input data for the simulation
        detector: Detection function to run
        ground_truth: Optional list of known fraud entity IDs
        domain: Domain identifier

    Returns:
        ScenarioResult with detection metrics
    """
    state = SimState(domain=domain)
    state.start_time = datetime.utcnow()

    result = ScenarioResult(scenario_name=scenario_name, success=False)

    try:
        # Run detector
        detections = detector(data)
        result.receipts = detections
        result.detection_count = len(detections)

        # Calculate false positives/negatives if ground truth provided
        if ground_truth is not None:
            detected_ids = set()
            for det in detections:
                if "entity_id" in det:
                    detected_ids.add(det["entity_id"])
                elif "vendor_id" in det:
                    detected_ids.add(det["vendor_id"])

            ground_truth_set = set(ground_truth)
            true_positives = detected_ids.intersection(ground_truth_set)
            false_positives = detected_ids - ground_truth_set
            false_negatives = ground_truth_set - detected_ids

            result.detection_count = len(true_positives)
            result.false_positive_count = len(false_positives)
            result.false_negative_count = len(false_negatives)

        result.success = True

    except Exception as e:
        result.errors.append(str(e))
        result.success = False

    state.end_time = datetime.utcnow()
    result.metrics["elapsed_seconds"] = state.elapsed_seconds()

    # Emit simulation result receipt
    emit_receipt("simulation_result", {
        "scenario_name": scenario_name,
        "success": result.success,
        "detection_count": result.detection_count,
        "accuracy": result.accuracy(),
        "domain": domain,
    }, to_stdout=False)

    return result
