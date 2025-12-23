"""
Gov-OS Core Detect - Adaptive Threshold Detection with Bayesian Sampling

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Thresholds are not fixed scalars—they are probability distributions that
update via Bayesian inference. Thompson sampling provides O(1/√n) convergence.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    VOLATILITY_ALPHA,
    THOMPSON_FP_TARGET,
    THOMPSON_CONVERGENCE_VAR,
    THOMPSON_PRIOR_VARIANCE,
    DISCLAIMER,
    TENANT_ID,
)
from .receipt import emit_L1


@dataclass
class ThresholdDistribution:
    """
    Maintains (mean, variance) for compression thresholds.
    Initialized with THOMPSON_PRIOR_VARIANCE.
    """
    mean: float = COMPRESSION_LEGITIMATE_FLOOR
    variance: float = THOMPSON_PRIOR_VARIANCE
    observations: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    false_positives: int = 0
    true_positives: int = 0

    def sample(self) -> float:
        """Draw a sample from the distribution."""
        sampled = np.random.normal(self.mean, math.sqrt(self.variance))
        return max(0.0, min(1.0, sampled))


def adaptive_threshold(
    base: float,
    volatility_factor: float,
    alpha: float = VOLATILITY_ALPHA,
) -> float:
    """
    Dynamic threshold: base * (1 + alpha * (vol - 1)).
    Adjusts threshold based on external volatility index.

    Args:
        base: Base threshold value
        volatility_factor: Current volatility factor (1.0 = baseline)
        alpha: Adjustment sensitivity (default from constants)

    Returns:
        Adjusted threshold
    """
    adjustment = alpha * (volatility_factor - 1.0)
    adjusted = base * (1.0 + adjustment)
    return max(0.0, min(1.0, adjusted))


def thompson_sample(distribution: ThresholdDistribution) -> float:
    """
    Sample threshold from posterior distribution.
    Thompson sampling: draw from posterior for exploration/exploitation.

    Args:
        distribution: ThresholdDistribution to sample from

    Returns:
        Sampled threshold value in [0, 1]
    """
    return distribution.sample()


def update_distribution(
    dist: ThresholdDistribution,
    observed_compression: float,
    was_fraud: bool,
) -> ThresholdDistribution:
    """
    Bayesian update after observation.
    Formula: accuracy_gain ≈ √(2 ln n / π² variance_prior)

    Args:
        dist: Current distribution
        observed_compression: Observed compression ratio
        was_fraud: Whether the receipt was actually fraudulent

    Returns:
        Updated ThresholdDistribution
    """
    n = dist.observations + 1

    # Bayesian update for threshold mean
    learning_rate = 1.0 / (n + 1)

    if was_fraud:
        # Receipt was fraud - threshold should be higher than observed
        target = observed_compression + 0.1
        dist.true_positives += 1
    else:
        # Receipt was legitimate - threshold should be lower than observed
        target = observed_compression - 0.1
        dist.false_positives += 1 if observed_compression < dist.mean else 0

    # Update mean toward target
    new_mean = dist.mean + learning_rate * (target - dist.mean)
    new_mean = max(0.0, min(1.0, new_mean))

    # Variance decreases with observations (Bayesian concentration)
    if n > 1:
        accuracy_gain = math.sqrt(2 * math.log(n) / (math.pi ** 2 * dist.variance))
        new_variance = dist.variance / (1 + accuracy_gain * 0.01)
    else:
        new_variance = dist.variance * 0.99

    return ThresholdDistribution(
        mean=new_mean,
        variance=max(THOMPSON_CONVERGENCE_VAR, new_variance),
        observations=n,
        context=dist.context,
        false_positives=dist.false_positives,
        true_positives=dist.true_positives,
    )


def detect_anomaly(
    receipt: Dict[str, Any],
    history: List[Dict[str, Any]],
    volatility_factor: float = 1.0,
    distribution: Optional[ThresholdDistribution] = None,
) -> bool:
    """
    Full detection: entropy + threshold + volatility.
    Emit detection_receipt.

    Args:
        receipt: Receipt to analyze
        history: Historical receipts
        volatility_factor: Current volatility (from domain VolatilityIndex)
        distribution: Optional threshold distribution for Thompson sampling

    Returns:
        True if anomaly detected
    """
    from .compress import compute_entropy_ratio, pattern_coherence

    # Compute compression ratio
    ratio = compute_entropy_ratio(receipt, history)

    # Get threshold
    if distribution is not None:
        base_threshold = thompson_sample(distribution)
    else:
        base_threshold = COMPRESSION_LEGITIMATE_FLOOR

    threshold = adaptive_threshold(base_threshold, volatility_factor)

    # Check coherence
    coherence = pattern_coherence([receipt] + history[-10:])

    # Determine if anomaly
    is_anomaly = ratio < threshold or coherence < 0.5

    # Check for cascade - need compression ratio history, not raw receipts
    from .cascade import early_warning
    # Extract compression ratios from recent history
    ratio_history = []
    if history:
        for i, h in enumerate(history[-20:]):  # Last 20 for efficiency
            h_ratio = compute_entropy_ratio(h, history[:max(0, i)])
            ratio_history.append(h_ratio)
    cascade_flag = early_warning(ratio, ratio_history)

    # Emit detection receipt
    emit_L1("detection_receipt", {
        "anomaly_detected": is_anomaly,
        "compression_ratio": round(ratio, 4),
        "threshold": round(threshold, 4),
        "volatility_factor": round(volatility_factor, 4),
        "coherence": round(coherence, 4),
        "cascade_flag": cascade_flag,
        "receipt_hash": receipt.get("payload_hash", ""),
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })

    return is_anomaly


def thompson_detect(
    receipt: Dict[str, Any],
    history: List[Dict[str, Any]],
    dist: ThresholdDistribution,
) -> Tuple[bool, float]:
    """
    Detection with Thompson sampling. Returns (is_anomaly, sampled_threshold).
    Updates distribution based on observation.

    Args:
        receipt: Receipt to analyze
        history: Historical receipts
        dist: ThresholdDistribution for sampling

    Returns:
        (is_anomaly, sampled_threshold)
    """
    from .compress import compute_entropy_ratio

    # Sample threshold
    threshold = thompson_sample(dist)

    # Compute compression
    ratio = compute_entropy_ratio(receipt, history)

    # Determine anomaly
    is_anomaly = ratio < threshold

    return is_anomaly, threshold


def false_positive_rate(dist: ThresholdDistribution) -> float:
    """
    Current FP rate estimate.

    Args:
        dist: ThresholdDistribution

    Returns:
        False positive rate 0-1
    """
    total = dist.false_positives + dist.true_positives
    if total == 0:
        return 0.0
    return dist.false_positives / total


def convergence_check(dist: ThresholdDistribution) -> bool:
    """
    True when variance < THOMPSON_CONVERGENCE_VAR.

    Args:
        dist: ThresholdDistribution

    Returns:
        True if converged
    """
    return dist.variance <= THOMPSON_CONVERGENCE_VAR


def create_distribution(
    context: Optional[Dict[str, Any]] = None,
    initial_mean: Optional[float] = None,
) -> ThresholdDistribution:
    """
    Create a new threshold distribution.

    Args:
        context: Optional context dict
        initial_mean: Optional initial mean (default COMPRESSION_LEGITIMATE_FLOOR)

    Returns:
        New ThresholdDistribution
    """
    return ThresholdDistribution(
        mean=initial_mean or COMPRESSION_LEGITIMATE_FLOOR,
        variance=THOMPSON_PRIOR_VARIANCE,
        observations=0,
        context=context or {},
    )


def calibrate_from_history(
    receipts: List[Dict[str, Any]],
    labels: Optional[List[bool]] = None,
) -> ThresholdDistribution:
    """
    Bootstrap distribution from historical data.

    Args:
        receipts: Historical receipts
        labels: Optional fraud labels (True = fraud)

    Returns:
        Calibrated ThresholdDistribution
    """
    from .compress import compute_entropy_ratio

    if not receipts:
        return create_distribution()

    ratios = []
    fraud_ratios = []
    legit_ratios = []

    for i, r in enumerate(receipts):
        ratio = compute_entropy_ratio(r, receipts[:i])
        ratios.append(ratio)

        if labels and i < len(labels):
            if labels[i]:
                fraud_ratios.append(ratio)
            else:
                legit_ratios.append(ratio)

    if fraud_ratios and legit_ratios:
        # Set threshold between fraud and legitimate means
        threshold_mean = (np.mean(fraud_ratios) + np.mean(legit_ratios)) / 2
    elif ratios:
        threshold_mean = np.mean(ratios)
    else:
        threshold_mean = COMPRESSION_LEGITIMATE_FLOOR

    variance = np.var(ratios) if len(ratios) > 1 else THOMPSON_PRIOR_VARIANCE

    return ThresholdDistribution(
        mean=max(0.1, min(0.9, threshold_mean)),
        variance=max(THOMPSON_CONVERGENCE_VAR, min(THOMPSON_PRIOR_VARIANCE, variance)),
        observations=len(ratios),
    )
