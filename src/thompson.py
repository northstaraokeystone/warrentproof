"""
WarrantProof Thompson Module - Bayesian Threshold Sampling

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements Thompson sampling for contextual threshold collapse.
Thresholds are not fixed scalars—they are probability distributions that
update via Bayesian inference. Each audit query "measures" the threshold,
collapsing it contextually like quantum wavefunction collapse.

Physics Foundation:
- Thresholds exist in superposition until measurement (audit query)
- accuracy_gain ≈ √(2 ln n / π² variance_prior) from Grok Q2
- Variance decreases monotonically with observations (convergence)

SLOs:
- False positive rate < 2% (Grok Q2 kill shot survival)
- Variance convergence: monotonically decreasing
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    THOMPSON_PRIOR_VARIANCE,
    THOMPSON_FP_TARGET,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class ThresholdDistribution:
    """
    Maintains (mean, variance) for compression thresholds.
    Initialized with THOMPSON_PRIOR_VARIANCE.
    """
    mean: float = 0.75  # Default threshold mean
    variance: float = THOMPSON_PRIOR_VARIANCE
    observations: int = 0
    context: dict = field(default_factory=dict)

    def sample(self) -> float:
        """Draw a sample from the distribution."""
        # Use normal distribution centered at mean with sqrt(variance) std
        sampled = np.random.normal(self.mean, math.sqrt(self.variance))
        # Clamp to valid threshold range [0, 1]
        return max(0.0, min(1.0, sampled))


def sample_threshold(distribution: ThresholdDistribution, context: dict) -> float:
    """
    Thompson sampling: draw from posterior distribution given context.
    Returns threshold value for this measurement.

    Args:
        distribution: ThresholdDistribution to sample from
        context: Context dict (branch, vendor_type, time_period)

    Returns:
        Sampled threshold value in [0, 1]
    """
    # Adjust variance based on context matching
    context_match = sum(
        1 for k, v in context.items()
        if distribution.context.get(k) == v
    )

    # Lower variance (more confidence) when context matches
    effective_variance = distribution.variance / (1 + context_match * 0.1)

    # Thompson sampling: draw from posterior
    sampled = np.random.normal(distribution.mean, math.sqrt(effective_variance))
    return max(0.0, min(1.0, sampled))


def update_posterior(
    distribution: ThresholdDistribution,
    observed_compression: float,
    was_fraud: bool
) -> ThresholdDistribution:
    """
    Bayesian update: observed outcome updates (mean, variance).
    Formula: accuracy_gain ≈ √(2 ln n / π² variance_prior)

    Args:
        distribution: Current distribution
        observed_compression: Observed compression ratio
        was_fraud: Whether the receipt was actually fraudulent

    Returns:
        Updated ThresholdDistribution
    """
    n = distribution.observations + 1

    # Bayesian update for threshold mean
    # If fraud detected correctly, shift mean toward observed
    # If false positive, shift away
    learning_rate = 1.0 / (n + 1)

    if was_fraud:
        # Receipt was fraud - threshold should be higher than observed
        target = observed_compression + 0.1
    else:
        # Receipt was legitimate - threshold should be lower than observed
        target = observed_compression - 0.1

    # Update mean toward target
    new_mean = distribution.mean + learning_rate * (target - distribution.mean)
    new_mean = max(0.0, min(1.0, new_mean))

    # Variance decreases with observations (Bayesian concentration)
    # accuracy_gain ≈ √(2 ln n / π² variance_prior)
    if n > 1:
        accuracy_gain = math.sqrt(2 * math.log(n) / (math.pi**2 * distribution.variance))
        new_variance = distribution.variance / (1 + accuracy_gain * 0.01)
    else:
        new_variance = distribution.variance * 0.99

    # Enforce monotonic variance decrease (stoprule)
    if new_variance >= distribution.variance and n > 1:
        stoprule_divergent_variance(distribution.variance, new_variance)

    return ThresholdDistribution(
        mean=new_mean,
        variance=max(0.001, new_variance),  # Minimum variance
        observations=n,
        context=distribution.context
    )


def contextual_collapse(
    distributions: dict,
    receipt: dict
) -> float:
    """
    Given receipt context, select relevant distribution, sample threshold.
    This is the "measurement" that collapses superposition.

    Args:
        distributions: Dict mapping context keys to ThresholdDistribution
        receipt: Receipt dict with context fields

    Returns:
        Collapsed threshold value
    """
    # Extract context from receipt
    context = {
        "branch": receipt.get("branch", "unknown"),
        "vendor_type": receipt.get("vendor_type", "general"),
        "time_period": receipt.get("ts", "")[:7]  # YYYY-MM
    }

    # Find best matching distribution
    best_distribution = None
    best_match_score = -1

    for key, dist in distributions.items():
        match_score = sum(
            1 for k, v in context.items()
            if dist.context.get(k) == v
        )
        if match_score > best_match_score:
            best_match_score = match_score
            best_distribution = dist

    # If no matching distribution, use default
    if best_distribution is None:
        best_distribution = ThresholdDistribution()

    # Sample threshold (collapse superposition)
    return sample_threshold(best_distribution, context)


def calibrate_prior(historical_receipts: list) -> ThresholdDistribution:
    """
    Bootstrap initial distribution from historical data.
    Compute empirical mean/variance of legitimate vs fraud compression ratios.

    Args:
        historical_receipts: List of receipts with compression data

    Returns:
        Calibrated ThresholdDistribution
    """
    if not historical_receipts:
        return ThresholdDistribution()

    # Extract compression ratios
    legit_ratios = []
    fraud_ratios = []

    for receipt in historical_receipts:
        ratio = receipt.get("compression_ratio")
        if ratio is None:
            continue

        if receipt.get("_is_fraud") or receipt.get("classification") == "fraudulent":
            fraud_ratios.append(ratio)
        else:
            legit_ratios.append(ratio)

    if not legit_ratios and not fraud_ratios:
        return ThresholdDistribution()

    # Calculate mean threshold at boundary between legit and fraud
    if legit_ratios and fraud_ratios:
        mean_legit = np.mean(legit_ratios)
        mean_fraud = np.mean(fraud_ratios)
        threshold_mean = (mean_legit + mean_fraud) / 2
    elif legit_ratios:
        threshold_mean = np.mean(legit_ratios) - 0.1
    else:
        threshold_mean = np.mean(fraud_ratios) + 0.1

    # Calculate variance from combined data
    all_ratios = legit_ratios + fraud_ratios
    if len(all_ratios) > 1:
        variance = np.var(all_ratios)
    else:
        variance = THOMPSON_PRIOR_VARIANCE

    return ThresholdDistribution(
        mean=max(0.1, min(0.9, threshold_mean)),
        variance=max(0.01, min(THOMPSON_PRIOR_VARIANCE, variance)),
        observations=len(all_ratios)
    )


def emit_thompson_receipt(
    context: dict,
    threshold_sampled: float,
    distribution: ThresholdDistribution,
    posterior_updated: bool = False
) -> dict:
    """
    Emit thompson_receipt documenting threshold sampling.

    Args:
        context: Context used for sampling
        threshold_sampled: Sampled threshold value
        distribution: Distribution sampled from
        posterior_updated: Whether posterior was updated

    Returns:
        thompson_receipt dict
    """
    return emit_receipt("thompson", {
        "tenant_id": TENANT_ID,
        "context": context,
        "threshold_sampled": round(threshold_sampled, 4),
        "distribution_mean": round(distribution.mean, 4),
        "distribution_variance": round(distribution.variance, 6),
        "observations": distribution.observations,
        "posterior_updated": posterior_updated,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_divergent_variance(old_variance: float, new_variance: float) -> None:
    """Variance must decrease monotonically with observations."""
    emit_receipt("anomaly", {
        "metric": "divergent_variance",
        "old_variance": old_variance,
        "new_variance": new_variance,
        "delta": new_variance - old_variance,
        "action": "halt",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"Variance diverged: {old_variance} -> {new_variance}")


def stoprule_false_positive_rate(observed_fp: float) -> None:
    """False positive rate must remain below target."""
    if observed_fp > THOMPSON_FP_TARGET:
        emit_receipt("anomaly", {
            "metric": "false_positive_rate",
            "observed": observed_fp,
            "target": THOMPSON_FP_TARGET,
            "delta": observed_fp - THOMPSON_FP_TARGET,
            "action": "recalibrate",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"FP rate {observed_fp} exceeds target {THOMPSON_FP_TARGET}")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Thompson Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test distribution creation
    dist = ThresholdDistribution(mean=0.75, variance=0.1)
    assert dist.mean == 0.75
    assert dist.variance == 0.1

    # Test sampling
    sample = sample_threshold(dist, {"branch": "Navy"})
    assert 0.0 <= sample <= 1.0

    # Test posterior update
    updated = update_posterior(dist, 0.65, was_fraud=True)
    assert updated.observations == 1
    # Variance should decrease
    assert updated.variance <= dist.variance

    # Test contextual collapse
    distributions = {
        "default": ThresholdDistribution(mean=0.75, variance=0.1)
    }
    receipt = {"branch": "Navy", "ts": "2024-01-15T10:00:00Z"}
    collapsed = contextual_collapse(distributions, receipt)
    assert 0.0 <= collapsed <= 1.0

    # Test calibration
    historical = [
        {"compression_ratio": 0.85, "_is_fraud": False},
        {"compression_ratio": 0.82, "_is_fraud": False},
        {"compression_ratio": 0.45, "_is_fraud": True},
        {"compression_ratio": 0.40, "_is_fraud": True},
    ]
    calibrated = calibrate_prior(historical)
    assert 0.4 < calibrated.mean < 0.85

    # Test receipt emission
    receipt = emit_thompson_receipt(
        context={"branch": "Navy"},
        threshold_sampled=0.72,
        distribution=dist
    )
    assert receipt["receipt_type"] == "thompson"

    print(f"# PASS: thompson module self-test", file=sys.stderr)
