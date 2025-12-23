"""
WarrantProof Cascade Module - dC/dt Monitoring for Early Alerts

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements dC/dt (compression derivative) monitoring.
Compression ratio C is a trailing indicator. dC/dt is a LEADING indicator:
accelerating incompressibility signals intensifying fraud before C crosses threshold.

Physics Foundation:
- First derivative (dC/dt) leads C by cascade timescale
- early_detection_gain ≈ cascade_rate / (dC/dt × cost_FP)
- Negative dC/dt indicates degrading compression = fraud cascade

SLOs:
- Alert threshold: dC/dt < -0.05 (5% degradation per unit time)
- False cascade alerts < 10%
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    CASCADE_DERIVATIVE_THRESHOLD,
    CASCADE_WINDOW_SIZE,
    CASCADE_FALSE_ALERT_MAX,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class CompressionHistory:
    """Tracks compression ratio history for derivative calculation."""
    ratios: deque = field(default_factory=lambda: deque(maxlen=CASCADE_WINDOW_SIZE))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=CASCADE_WINDOW_SIZE))

    def add(self, ratio: float, timestamp: Optional[str] = None):
        """Add compression ratio observation."""
        self.ratios.append(ratio)
        self.timestamps.append(timestamp or datetime.utcnow().isoformat())

    @property
    def size(self) -> int:
        return len(self.ratios)


def calculate_compression_derivative(
    compression_history: list,
    time_delta: float = 1.0
) -> float:
    """
    Compute dC/dt from recent history using moving window.

    Args:
        compression_history: List of compression ratios (oldest to newest)
        time_delta: Time interval between observations

    Returns:
        Rate of change dC/dt (negative = degrading)
    """
    if len(compression_history) < 2:
        return 0.0

    # Use recent window
    window = compression_history[-CASCADE_WINDOW_SIZE:]

    if len(window) < 2:
        return 0.0

    # Simple linear regression for slope
    n = len(window)
    x = np.arange(n) * time_delta
    y = np.array(window)

    # Least squares: dC/dt = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    dC_dt = numerator / denominator

    return dC_dt


def detect_cascade_onset(dC_dt: float, threshold: Optional[float] = None) -> bool:
    """
    Check if dC/dt < -CASCADE_DERIVATIVE_THRESHOLD.
    Compression rapidly degrading → cascade starting.

    Args:
        dC_dt: Compression derivative
        threshold: Optional custom threshold (default from constants)

    Returns:
        True if cascade onset detected
    """
    if threshold is None:
        threshold = CASCADE_DERIVATIVE_THRESHOLD

    # Negative dC/dt means compression is degrading
    # Alert when degradation exceeds threshold
    return dC_dt < -threshold


def estimate_cascade_time(
    dC_dt: float,
    current_C: float,
    fraud_threshold: float = 0.50
) -> float:
    """
    Extrapolate: time until C crosses fraud threshold at current dC/dt.
    Early warning metric.

    Args:
        dC_dt: Current compression derivative
        current_C: Current compression ratio
        fraud_threshold: Threshold indicating fraud (default 0.50)

    Returns:
        Estimated time to breach (seconds). -1 if not converging to fraud.
    """
    if dC_dt >= 0:
        # Compression improving or stable - no fraud convergence
        return -1.0

    if current_C <= fraud_threshold:
        # Already at or below threshold
        return 0.0

    # Linear extrapolation: time = (current_C - fraud_threshold) / |dC/dt|
    time_to_breach = (current_C - fraud_threshold) / abs(dC_dt)

    return time_to_breach


def calculate_early_detection_gain(
    cascade_rate: float,
    dC_dt: float,
    cost_FP: float = 1.0
) -> float:
    """
    Calculate early detection gain.
    Formula: gain ≈ cascade_rate / (|dC/dt| × cost_FP)

    Args:
        cascade_rate: Rate of fraud cascade propagation
        dC_dt: Compression derivative
        cost_FP: Cost of false positive (default 1.0)

    Returns:
        Early detection gain factor
    """
    if dC_dt == 0 or cost_FP == 0:
        return 0.0

    gain = cascade_rate / (abs(dC_dt) * cost_FP)

    return min(10.0, gain)  # Cap at 10x


def alert_early_warning(
    cascade_detected: bool,
    time_to_breach: float,
    dC_dt: float,
    current_C: float
) -> dict:
    """
    Emit cascade_receipt if cascade onset detected.
    Include time estimate for breach.

    Args:
        cascade_detected: Whether cascade was detected
        time_to_breach: Estimated time to fraud threshold
        dC_dt: Current compression derivative
        current_C: Current compression ratio

    Returns:
        cascade_receipt dict
    """
    early_gain = calculate_early_detection_gain(
        cascade_rate=abs(dC_dt) * 2,  # Estimate cascade rate from derivative
        dC_dt=dC_dt,
        cost_FP=1.0
    )

    return emit_receipt("cascade", {
        "tenant_id": TENANT_ID,
        "dC_dt": round(dC_dt, 6),
        "cascade_detected": cascade_detected,
        "time_to_breach_estimate": round(time_to_breach, 2) if time_to_breach >= 0 else None,
        "compression_current": round(current_C, 4),
        "alert_triggered": cascade_detected,
        "early_detection_gain": round(early_gain, 4),
        "threshold_used": CASCADE_DERIVATIVE_THRESHOLD,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def monitor_compression_stream(
    receipts: list,
    history: Optional[CompressionHistory] = None
) -> dict:
    """
    Monitor a stream of receipts for cascade onset.

    Args:
        receipts: Recent receipts to analyze
        history: Optional existing compression history

    Returns:
        Monitoring result with cascade status
    """
    if history is None:
        history = CompressionHistory()

    # Calculate compression ratio for receipts
    from .compress import compress_receipt_stream

    compression_result = compress_receipt_stream(receipts)
    current_C = compression_result.get("compression_ratio", 0.80)

    # Add to history
    history.add(current_C)

    # Calculate derivative
    ratios_list = list(history.ratios)
    dC_dt = calculate_compression_derivative(ratios_list)

    # Check for cascade
    cascade_detected = detect_cascade_onset(dC_dt)

    # Estimate time to breach
    time_to_breach = estimate_cascade_time(dC_dt, current_C)

    # Emit alert if cascade detected
    if cascade_detected:
        alert_early_warning(cascade_detected, time_to_breach, dC_dt, current_C)

    return {
        "dC_dt": dC_dt,
        "current_C": current_C,
        "cascade_detected": cascade_detected,
        "time_to_breach": time_to_breach,
        "history_size": history.size,
    }


def analyze_cascade_pattern(
    compression_history: list,
    detect_acceleration: bool = True
) -> dict:
    """
    Analyze cascade pattern including second derivative (acceleration).

    Args:
        compression_history: List of compression ratios
        detect_acceleration: Whether to compute d²C/dt²

    Returns:
        Cascade analysis dict
    """
    result = {
        "dC_dt": 0.0,
        "d2C_dt2": 0.0,
        "cascade_phase": "stable",
        "severity": "none",
    }

    if len(compression_history) < 3:
        return result

    # First derivative
    dC_dt = calculate_compression_derivative(compression_history)
    result["dC_dt"] = dC_dt

    if detect_acceleration and len(compression_history) >= 5:
        # Second derivative: rate of change of dC/dt
        # Calculate dC/dt for first half and second half
        mid = len(compression_history) // 2
        dC_dt_early = calculate_compression_derivative(compression_history[:mid])
        dC_dt_late = calculate_compression_derivative(compression_history[mid:])

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


# === STOPRULES ===

def stoprule_compression_collapse(current_C: float) -> None:
    """Compression below 0.30 indicates catastrophic fraud."""
    if current_C < 0.30:
        emit_receipt("anomaly", {
            "metric": "compression_collapse",
            "current_C": current_C,
            "threshold": 0.30,
            "delta": current_C - 0.30,
            "action": "halt",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"Compression collapsed to {current_C}")


def stoprule_false_cascade_rate(false_alerts: int, total_alerts: int) -> None:
    """False cascade alerts must be < 10%."""
    if total_alerts == 0:
        return

    fp_rate = false_alerts / total_alerts

    if fp_rate > CASCADE_FALSE_ALERT_MAX:
        emit_receipt("anomaly", {
            "metric": "false_cascade_rate",
            "false_alerts": false_alerts,
            "total_alerts": total_alerts,
            "fp_rate": fp_rate,
            "threshold": CASCADE_FALSE_ALERT_MAX,
            "action": "recalibrate",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"False cascade rate {fp_rate:.2%} exceeds {CASCADE_FALSE_ALERT_MAX:.0%}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Cascade Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test compression derivative calculation
    # Stable compression
    stable_history = [0.85, 0.84, 0.85, 0.86, 0.84, 0.85]
    dC_dt_stable = calculate_compression_derivative(stable_history)
    print(f"# Stable dC/dt: {dC_dt_stable:.6f}", file=sys.stderr)
    assert abs(dC_dt_stable) < CASCADE_DERIVATIVE_THRESHOLD

    # Degrading compression (cascade)
    degrading_history = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    dC_dt_degrading = calculate_compression_derivative(degrading_history)
    print(f"# Degrading dC/dt: {dC_dt_degrading:.6f}", file=sys.stderr)
    assert dC_dt_degrading < 0

    # Test cascade detection
    cascade_stable = detect_cascade_onset(dC_dt_stable)
    cascade_degrading = detect_cascade_onset(dC_dt_degrading)
    print(f"# Cascade stable: {cascade_stable}, degrading: {cascade_degrading}", file=sys.stderr)
    assert not cascade_stable
    assert cascade_degrading

    # Test time estimation
    time_stable = estimate_cascade_time(dC_dt_stable, 0.80)
    time_degrading = estimate_cascade_time(dC_dt_degrading, 0.60)
    print(f"# Time to breach - stable: {time_stable:.2f}, degrading: {time_degrading:.2f}", file=sys.stderr)
    assert time_stable == -1.0  # No convergence
    assert time_degrading >= 0  # Will breach

    # Test early detection gain
    gain = calculate_early_detection_gain(0.10, dC_dt_degrading, 1.0)
    print(f"# Early detection gain: {gain:.4f}", file=sys.stderr)
    assert gain > 0

    # Test alert emission
    alert = alert_early_warning(True, 100.0, dC_dt_degrading, 0.60)
    assert alert["receipt_type"] == "cascade"
    assert alert["cascade_detected"] == True

    # Test cascade pattern analysis
    analysis = analyze_cascade_pattern(degrading_history)
    print(f"# Cascade phase: {analysis['cascade_phase']}, severity: {analysis['severity']}", file=sys.stderr)

    # Test compression history
    history = CompressionHistory()
    for ratio in stable_history:
        history.add(ratio)
    assert history.size == len(stable_history)

    print(f"# PASS: cascade module self-test", file=sys.stderr)
