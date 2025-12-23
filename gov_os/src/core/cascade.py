"""
Gov-OS Core Cascade - dC/dt Derivative Detection for Early Warning

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Compression ratio C is a trailing indicator. dC/dt is a LEADING indicator:
accelerating incompressibility signals intensifying fraud before C crosses threshold.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .constants import (
    CASCADE_DERIVATIVE_THRESHOLD,
    CASCADE_WINDOW,
    DISCLAIMER,
    TENANT_ID,
)
from .receipt import emit_L1


@dataclass
class CompressionHistory:
    """Tracks compression ratio history for derivative calculation."""
    ratios: deque = field(default_factory=lambda: deque(maxlen=CASCADE_WINDOW))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=CASCADE_WINDOW))

    def add(self, ratio: float, timestamp: Optional[str] = None):
        """Add compression ratio observation."""
        self.ratios.append(ratio)
        self.timestamps.append(timestamp or datetime.utcnow().isoformat())

    @property
    def size(self) -> int:
        return len(self.ratios)


def compression_derivative(
    history: List[float],
    window: int = CASCADE_WINDOW,
) -> float:
    """
    Compute dC/dt over rolling window.
    Negative dC/dt = fraud accumulating.

    Args:
        history: List of compression ratios (oldest to newest)
        window: Size of moving window for calculation

    Returns:
        Rate of change dC/dt (negative = degrading)
    """
    if len(history) < 2:
        return 0.0

    window_data = history[-window:]
    if len(window_data) < 2:
        return 0.0

    # Linear regression for slope
    n = len(window_data)
    x = np.arange(n)
    y = np.array(window_data)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def early_warning(
    current_ratio: float,
    history: List[float],
    threshold: float = CASCADE_DERIVATIVE_THRESHOLD,
) -> bool:
    """
    True if cascade imminent.
    dC/dt < -threshold indicates rapidly degrading compression.

    Args:
        current_ratio: Current compression ratio
        history: Historical compression ratios
        threshold: dC/dt threshold for warning

    Returns:
        True if early warning triggered
    """
    if not history:
        return False

    all_history = list(history) + [current_ratio]
    dC_dt = compression_derivative(all_history)

    return dC_dt < -threshold


def cascade_prediction(history: List[float]) -> Dict[str, Any]:
    """
    Predict days until cascade breach.

    Args:
        history: Historical compression ratios

    Returns:
        Prediction dict with urgency_days
    """
    if len(history) < 3:
        return {"urgency_days": -1, "status": "insufficient_data"}

    dC_dt = compression_derivative(history)
    current_C = history[-1] if history else 0.85

    if dC_dt >= 0:
        return {"urgency_days": -1, "status": "stable", "dC_dt": dC_dt}

    # Linear extrapolation to fraud threshold (0.60)
    fraud_threshold = 0.60
    if current_C <= fraud_threshold:
        return {"urgency_days": 0, "status": "breached", "dC_dt": dC_dt}

    days_to_breach = (current_C - fraud_threshold) / abs(dC_dt)

    return {
        "urgency_days": round(days_to_breach, 2),
        "status": "degrading",
        "dC_dt": dC_dt,
        "current_C": current_C,
    }


def trigger_alert(
    source: Dict[str, Any],
    urgency_days: float,
) -> Dict[str, Any]:
    """
    Emit cascade_alert_receipt.

    Args:
        source: Source receipt that triggered alert
        urgency_days: Estimated days until breach

    Returns:
        Alert receipt
    """
    return emit_L1("cascade_alert", {
        "source_hash": source.get("payload_hash", ""),
        "urgency_days": urgency_days,
        "severity": _urgency_to_severity(urgency_days),
        "action": "investigate",
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })


def cascade_trace(
    source: Dict[str, Any],
    graph: Any,
) -> List[str]:
    """
    Trace cascade path through graph.
    Identifies root cause by following edges backward.

    Args:
        source: Source receipt/node
        graph: NetworkX DiGraph or dict-based graph

    Returns:
        List of node IDs in cascade path
    """
    path = []

    if HAS_NETWORKX and isinstance(graph, nx.DiGraph):
        source_id = source.get("vendor_id") or source.get("provider_npi") or source.get("id")
        if source_id and source_id in graph:
            # BFS backward from source
            try:
                predecessors = list(nx.ancestors(graph, source_id))
                path = predecessors[:10]  # Limit to 10
            except Exception:
                pass
    elif isinstance(graph, dict):
        # Simple backward trace for dict graph
        source_id = source.get("vendor_id") or source.get("provider_npi") or source.get("id")
        edges = graph.get("edges", {})

        # Build reverse edges
        reverse_edges: Dict[str, List[str]] = {}
        for src, targets in edges.items():
            for tgt in targets:
                if tgt not in reverse_edges:
                    reverse_edges[tgt] = []
                reverse_edges[tgt].append(src)

        # BFS backward
        if source_id:
            visited = set()
            queue = [source_id]
            while queue and len(path) < 10:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                path.append(node)
                for predecessor in reverse_edges.get(node, []):
                    if predecessor not in visited:
                        queue.append(predecessor)

    return path


def analyze_cascade_pattern(
    history: List[float],
    detect_acceleration: bool = True,
) -> Dict[str, Any]:
    """
    Analyze cascade pattern including second derivative (acceleration).

    Args:
        history: List of compression ratios
        detect_acceleration: Whether to compute d²C/dt²

    Returns:
        Cascade analysis dict
    """
    result: Dict[str, Any] = {
        "dC_dt": 0.0,
        "d2C_dt2": 0.0,
        "cascade_phase": "stable",
        "severity": "none",
    }

    if len(history) < 3:
        return result

    # First derivative
    dC_dt = compression_derivative(history)
    result["dC_dt"] = dC_dt

    if detect_acceleration and len(history) >= 5:
        # Second derivative: rate of change of dC/dt
        mid = len(history) // 2
        dC_dt_early = compression_derivative(history[:mid])
        dC_dt_late = compression_derivative(history[mid:])

        d2C_dt2 = dC_dt_late - dC_dt_early
        result["d2C_dt2"] = d2C_dt2

        # Classify phase
        if dC_dt < -CASCADE_DERIVATIVE_THRESHOLD:
            if d2C_dt2 < 0:
                result["cascade_phase"] = "accelerating"
                result["severity"] = "critical"
            else:
                result["cascade_phase"] = "decelerating"
                result["severity"] = "high"
        elif dC_dt < 0:
            result["cascade_phase"] = "degrading"
            result["severity"] = "medium"
        else:
            result["cascade_phase"] = "stable"
            result["severity"] = "none"

    return result


def _urgency_to_severity(urgency_days: float) -> str:
    """Convert urgency days to severity level."""
    if urgency_days < 0:
        return "none"
    elif urgency_days == 0:
        return "critical"
    elif urgency_days <= 7:
        return "high"
    elif urgency_days <= 30:
        return "medium"
    else:
        return "low"


def monitor_stream(
    receipts: List[Dict[str, Any]],
    history: Optional[CompressionHistory] = None,
) -> Dict[str, Any]:
    """
    Monitor a stream of receipts for cascade onset.

    Args:
        receipts: Recent receipts to analyze
        history: Optional existing compression history

    Returns:
        Monitoring result with cascade status
    """
    from .compress import compute_entropy_ratio

    if history is None:
        history = CompressionHistory()

    # Calculate compression for batch
    if receipts:
        ratio = compute_entropy_ratio(receipts[-1], receipts[:-1])
        history.add(ratio)

    ratios_list = list(history.ratios)
    dC_dt = compression_derivative(ratios_list)
    current_C = ratios_list[-1] if ratios_list else 0.85

    cascade_detected = dC_dt < -CASCADE_DERIVATIVE_THRESHOLD
    prediction = cascade_prediction(ratios_list)

    return {
        "dC_dt": dC_dt,
        "current_C": current_C,
        "cascade_detected": cascade_detected,
        "urgency_days": prediction.get("urgency_days", -1),
        "history_size": history.size,
    }
