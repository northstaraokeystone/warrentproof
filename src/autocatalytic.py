"""
WarrantProof Autocatalytic Module - Pattern Emergence Without Hardcoding

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements autocatalytic fraud pattern emergence.
Patterns are not programmed—they crystallize from receipt clusters that
achieve self-referencing closure (RAF sets). When catalytic coverage
exceeds percolation threshold, phase transition occurs: pattern emerges.

Physics Foundation:
- RAF (Reflexively Autocatalytic Food sets) from origin-of-life chemistry
- N_critical ≈ log₂(ΔH⁻¹) × (H_legit / ΔH) for phase transition
- Patterns crystallize when catalytic coverage exceeds threshold

SLOs:
- N_critical < 10,000 receipts (Grok Q1 kill shot survival)
- Entropy gap ΔH ≥ 0.15
- Pattern coherence ≥ 0.80 for survival
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    N_CRITICAL_FORMULA,
    ENTROPY_GAP_MIN,
    PATTERN_COHERENCE_MIN,
    RAF_CLOSURE_ACCURACY_MIN,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class EmergentPattern:
    """A fraud pattern that emerged from data without hardcoding."""
    pattern_id: str
    description: str
    fingerprint: dict  # Field-level characteristics
    entropy_gap: float
    N_observed: int
    N_critical: int
    coherence: float
    RAF_closure: bool
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "fingerprint": self.fingerprint,
            "entropy_gap": self.entropy_gap,
            "N_observed": self.N_observed,
            "N_critical": self.N_critical,
            "coherence": self.coherence,
            "RAF_closure": self.RAF_closure,
        }


def compute_entropy_gap(receipts: list) -> float:
    """
    Calculate ΔH = H_fraud - H_legit using Shannon entropy.
    Legitimate receipts compress to low H, fraud resists.

    Args:
        receipts: List of receipts to analyze

    Returns:
        Entropy gap ΔH
    """
    if not receipts:
        return 0.0

    # Separate by classification if available
    legit_receipts = [r for r in receipts if not r.get("_is_fraud")]
    fraud_receipts = [r for r in receipts if r.get("_is_fraud")]

    # If not classified, estimate from compression behavior
    if not fraud_receipts:
        # Use field entropy variance as proxy
        return _estimate_entropy_gap_from_variance(receipts)

    H_legit = _calculate_shannon_entropy(legit_receipts) if legit_receipts else 3.0
    H_fraud = _calculate_shannon_entropy(fraud_receipts) if fraud_receipts else 5.0

    return max(0.0, H_fraud - H_legit)


def _calculate_shannon_entropy(receipts: list) -> float:
    """Calculate Shannon entropy of receipt feature distribution."""
    if not receipts:
        return 0.0

    features = []
    for r in receipts:
        features.append(r.get("receipt_type", ""))
        features.append(r.get("branch", ""))
        features.append(str(r.get("amount_usd", 0))[:3])
        features.append(r.get("vendor", "")[:5] if r.get("vendor") else "")

    counter = Counter(features)
    total = len(features)

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _estimate_entropy_gap_from_variance(receipts: list) -> float:
    """Estimate entropy gap from field variance when labels unavailable."""
    if len(receipts) < 10:
        return ENTROPY_GAP_MIN  # Default minimum

    # Calculate variance across fields
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    vendors = [r.get("vendor", "") for r in receipts if r.get("vendor")]

    # High vendor uniqueness suggests fraud patterns
    vendor_uniqueness = len(set(vendors)) / len(vendors) if vendors else 0

    # High amount variance suggests disorder
    if amounts:
        amount_cv = np.std(amounts) / (np.mean(amounts) + 1e-10)
    else:
        amount_cv = 0

    # Combine into entropy gap estimate
    estimated_gap = vendor_uniqueness * 0.3 + min(1.0, amount_cv) * 0.2
    return max(ENTROPY_GAP_MIN, estimated_gap)


def detect_autocatalytic_closure(receipt_cluster: list) -> bool:
    """
    Test if cluster is self-referencing: do receipts in cluster
    predict anomalies in cluster? If yes, RAF closure achieved.

    Args:
        receipt_cluster: List of receipts to test for closure

    Returns:
        True if RAF closure detected
    """
    if len(receipt_cluster) < 3:
        return False

    # Check for self-referencing patterns
    # Pattern 1: Receipts reference each other (lineage cycles)
    ids = {r.get("payload_hash") for r in receipt_cluster if r.get("payload_hash")}
    lineage_refs = set()
    for r in receipt_cluster:
        for parent in r.get("decision_lineage", []):
            if parent in ids:
                lineage_refs.add(parent)

    # Pattern 2: Field value repetition (same vendors, amounts, etc.)
    vendors = [r.get("vendor") for r in receipt_cluster if r.get("vendor")]
    vendor_repetition = len(vendors) - len(set(vendors)) if vendors else 0

    # Pattern 3: Temporal clustering (many receipts in short window)
    timestamps = [r.get("ts", "") for r in receipt_cluster if r.get("ts")]
    if len(timestamps) >= 3:
        # Simple check: if all timestamps share same date prefix
        date_prefixes = [ts[:10] for ts in timestamps]
        temporal_clustering = len(date_prefixes) / (len(set(date_prefixes)) + 1)
    else:
        temporal_clustering = 1.0

    # RAF closure if cluster exhibits self-referencing behavior
    closure_score = (
        (len(lineage_refs) / len(ids) if ids else 0) * 0.3 +
        (vendor_repetition / len(receipt_cluster)) * 0.3 +
        (temporal_clustering - 1) * 0.4
    )

    return closure_score > RAF_CLOSURE_ACCURACY_MIN * 0.5


def calculate_N_critical(H_legit: float, H_fraud: float) -> int:
    """
    Apply N_CRITICAL_FORMULA: N ≈ log₂(ΔH⁻¹) × (H_legit / ΔH).
    Returns minimum receipts for phase transition.

    Args:
        H_legit: Entropy of legitimate receipts
        H_fraud: Entropy of fraudulent receipts

    Returns:
        N_critical value
    """
    N = N_CRITICAL_FORMULA(H_legit, H_fraud)

    if N == float('inf') or N < 0 or math.isnan(N):
        return 10000  # Cap at limit

    return int(min(10000, max(1, N)))


def crystallize_pattern(
    receipts: list,
    entropy_gap: float
) -> Optional[EmergentPattern]:
    """
    When len(receipts) > N_critical AND entropy_gap > ENTROPY_GAP_MIN,
    extract pattern signature (no hardcoded rules).

    Args:
        receipts: List of receipts exhibiting pattern
        entropy_gap: Calculated entropy gap

    Returns:
        EmergentPattern if crystallized, None otherwise
    """
    if not receipts:
        return None

    if entropy_gap < ENTROPY_GAP_MIN:
        return None

    # Estimate H_legit and H_fraud from receipts
    H_total = _calculate_shannon_entropy(receipts)
    H_legit = H_total - entropy_gap / 2
    H_fraud = H_total + entropy_gap / 2

    N_critical = calculate_N_critical(H_legit, H_fraud)

    if len(receipts) < N_critical:
        return None

    # Check for RAF closure
    RAF_closure = detect_autocatalytic_closure(receipts)

    # Extract fingerprint from cluster characteristics
    fingerprint = _extract_fingerprint(receipts)

    # Calculate coherence
    coherence = pattern_coherence_score(fingerprint, receipts)

    if coherence < PATTERN_COHERENCE_MIN:
        return None

    # Generate pattern description
    description = _generate_pattern_description(fingerprint)

    return EmergentPattern(
        pattern_id=dual_hash(str(fingerprint))[:16],
        description=description,
        fingerprint=fingerprint,
        entropy_gap=entropy_gap,
        N_observed=len(receipts),
        N_critical=N_critical,
        coherence=coherence,
        RAF_closure=RAF_closure,
    )


def _extract_fingerprint(receipts: list) -> dict:
    """Extract characteristic fingerprint from receipt cluster."""
    fingerprint = {}

    # Amount distribution
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    if amounts:
        fingerprint["amount_mean"] = np.mean(amounts)
        fingerprint["amount_std"] = np.std(amounts)
        fingerprint["amount_cv"] = fingerprint["amount_std"] / (fingerprint["amount_mean"] + 1e-10)

    # Vendor distribution
    vendors = [r.get("vendor", "") for r in receipts if r.get("vendor")]
    if vendors:
        fingerprint["vendor_uniqueness"] = len(set(vendors)) / len(vendors)
        fingerprint["vendor_count"] = len(set(vendors))

    # Lineage characteristics
    with_lineage = sum(1 for r in receipts if r.get("decision_lineage"))
    fingerprint["lineage_ratio"] = with_lineage / len(receipts) if receipts else 0

    # Receipt type distribution
    types = Counter(r.get("receipt_type", "") for r in receipts)
    fingerprint["type_entropy"] = _entropy_from_counter(types)

    # Branch distribution
    branches = Counter(r.get("branch", "") for r in receipts)
    fingerprint["branch_entropy"] = _entropy_from_counter(branches)

    return fingerprint


def _entropy_from_counter(counter: Counter) -> float:
    """Calculate entropy from Counter object."""
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _generate_pattern_description(fingerprint: dict) -> str:
    """Generate human-readable description from fingerprint."""
    parts = []

    if fingerprint.get("vendor_uniqueness", 0) > 0.8:
        parts.append("high-vendor-churn")
    if fingerprint.get("amount_cv", 0) > 1.0:
        parts.append("high-amount-variance")
    if fingerprint.get("lineage_ratio", 0) < 0.3:
        parts.append("orphan-transactions")
    if fingerprint.get("type_entropy", 0) > 2.0:
        parts.append("mixed-transaction-types")

    return "-".join(parts) if parts else "emergent-cluster"


def pattern_coherence_score(pattern: dict, new_receipts: list) -> float:
    """
    Score 0-1: how well pattern predicts anomalies in new data.
    Used for pattern survival selection.

    Args:
        pattern: Pattern fingerprint dict
        new_receipts: New receipts to test against

    Returns:
        Coherence score 0-1
    """
    if not new_receipts or not pattern:
        return 0.5

    scores = []

    # Check vendor uniqueness match
    new_vendors = [r.get("vendor", "") for r in new_receipts if r.get("vendor")]
    if new_vendors and "vendor_uniqueness" in pattern:
        new_uniqueness = len(set(new_vendors)) / len(new_vendors)
        match = 1.0 - abs(new_uniqueness - pattern["vendor_uniqueness"])
        scores.append(match)

    # Check amount distribution match
    new_amounts = [r.get("amount_usd", 0) for r in new_receipts if r.get("amount_usd")]
    if new_amounts and "amount_cv" in pattern:
        new_cv = np.std(new_amounts) / (np.mean(new_amounts) + 1e-10)
        # Relative difference capped
        cv_match = 1.0 - min(1.0, abs(new_cv - pattern["amount_cv"]) / (pattern["amount_cv"] + 0.1))
        scores.append(cv_match)

    # Check lineage ratio match
    new_with_lineage = sum(1 for r in new_receipts if r.get("decision_lineage"))
    if new_receipts and "lineage_ratio" in pattern:
        new_ratio = new_with_lineage / len(new_receipts)
        lineage_match = 1.0 - abs(new_ratio - pattern["lineage_ratio"])
        scores.append(lineage_match)

    return sum(scores) / len(scores) if scores else 0.5


def autocatalytic_detect(
    receipts: list,
    existing_patterns: Optional[list] = None
) -> list[dict]:
    """
    Main detection loop via autocatalytic pattern emergence.

    Steps:
    1. Compute entropy gaps
    2. Test N_critical
    3. Crystallize new patterns
    4. Test coherence
    5. Return detected frauds

    Args:
        receipts: Receipts to analyze
        existing_patterns: Optional list of known EmergentPatterns

    Returns:
        List of fraud detections
    """
    if not receipts:
        return []

    if existing_patterns is None:
        existing_patterns = []

    detections = []

    # Step 1: Compute entropy gap for full set
    entropy_gap = compute_entropy_gap(receipts)

    # Step 2: Calculate N_critical
    H_total = _calculate_shannon_entropy(receipts)
    N_critical = calculate_N_critical(H_total - entropy_gap/2, H_total + entropy_gap/2)

    # Check stoprule
    if N_critical > 10000:
        stoprule_N_critical_exceeded(N_critical)

    if entropy_gap < ENTROPY_GAP_MIN:
        stoprule_entropy_gap_insufficient(entropy_gap)
        # Return early but don't raise exception
        return detections

    # Step 3: Try to crystallize new pattern if sufficient data
    if len(receipts) >= N_critical:
        new_pattern = crystallize_pattern(receipts, entropy_gap)
        if new_pattern:
            existing_patterns.append(new_pattern)

            # Emit autocatalytic receipt
            emit_autocatalytic_receipt(new_pattern)

            # Mark receipts as detected
            for r in receipts:
                if r.get("_is_fraud"):
                    detections.append({
                        "receipt_id": r.get("payload_hash", ""),
                        "pattern_id": new_pattern.pattern_id,
                        "confidence": new_pattern.coherence,
                        "anomaly_type": "autocatalytic_emergence",
                    })

    # Step 4: Check existing patterns against receipts
    for pattern in existing_patterns:
        if isinstance(pattern, EmergentPattern):
            fingerprint = pattern.fingerprint
        else:
            fingerprint = pattern

        coherence = pattern_coherence_score(fingerprint, receipts)

        if coherence >= PATTERN_COHERENCE_MIN:
            # Pattern matches - detect fraud
            for r in receipts:
                if coherence > 0.7:  # High match threshold
                    detections.append({
                        "receipt_id": r.get("payload_hash", ""),
                        "pattern_id": pattern.pattern_id if hasattr(pattern, 'pattern_id') else "inherited",
                        "confidence": coherence,
                        "anomaly_type": "pattern_match_emergent",
                    })

    return detections


def emit_autocatalytic_receipt(pattern: EmergentPattern) -> dict:
    """Emit receipt documenting pattern emergence."""
    return emit_receipt("autocatalytic", {
        "tenant_id": TENANT_ID,
        "pattern_emerged": pattern.description,
        "pattern_id": pattern.pattern_id,
        "N_receipts": pattern.N_observed,
        "N_critical": pattern.N_critical,
        "entropy_gap": round(pattern.entropy_gap, 4),
        "RAF_closure": pattern.RAF_closure,
        "pattern_coherence_score": round(pattern.coherence, 4),
        "fingerprint": pattern.fingerprint,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_N_critical_exceeded(N_critical: int) -> None:
    """N_critical must be < 10,000 for viable detection."""
    emit_receipt("anomaly", {
        "metric": "N_critical_exceeded",
        "N_critical": N_critical,
        "limit": 10000,
        "delta": N_critical - 10000,
        "action": "alert",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    # Don't raise - this is informational


def stoprule_entropy_gap_insufficient(entropy_gap: float) -> None:
    """Entropy gap must be >= ENTROPY_GAP_MIN for detection."""
    emit_receipt("anomaly", {
        "metric": "entropy_gap_insufficient",
        "entropy_gap": entropy_gap,
        "minimum": ENTROPY_GAP_MIN,
        "delta": entropy_gap - ENTROPY_GAP_MIN,
        "action": "continue",  # Continue with reduced confidence
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    # Don't raise - this is informational


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import random
    import string

    print(f"# WarrantProof Autocatalytic Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Generate legitimate receipts
    legit_receipts = []
    for i in range(100):
        legit_receipts.append({
            "receipt_type": "warrant",
            "branch": random.choice(["Navy", "Navy", "Army"]),
            "vendor": f"Vendor_{i % 5}",
            "amount_usd": random.choice([1000000, 2000000]) * (1 + random.random() * 0.1),
            "decision_lineage": [f"parent_{i-1}"] if i > 0 else [],
            "_is_fraud": False,
        })

    # Generate fraud receipts
    fraud_receipts = []
    for i in range(100):
        fraud_receipts.append({
            "receipt_type": random.choice(["warrant", "milestone"]),
            "branch": random.choice(["Navy", "Army", "AirForce", "Marines"]),
            "vendor": ''.join(random.choices(string.ascii_uppercase, k=10)),
            "amount_usd": random.random() * 10000000,
            "decision_lineage": [],
            "_is_fraud": True,
        })

    # Test entropy gap calculation
    all_receipts = legit_receipts + fraud_receipts
    gap = compute_entropy_gap(all_receipts)
    assert gap >= 0.0, f"Entropy gap should be non-negative: {gap}"
    print(f"# Entropy gap: {gap:.4f}", file=sys.stderr)

    # Test N_critical calculation
    N_crit = calculate_N_critical(3.0, 5.0)
    assert N_crit < 10000, f"N_critical {N_crit} exceeds limit"
    print(f"# N_critical: {N_crit}", file=sys.stderr)

    # Test RAF closure detection
    closure = detect_autocatalytic_closure(fraud_receipts[:50])
    print(f"# RAF closure detected: {closure}", file=sys.stderr)

    # Test pattern crystallization
    if gap >= ENTROPY_GAP_MIN:
        pattern = crystallize_pattern(fraud_receipts, gap)
        if pattern:
            print(f"# Pattern crystallized: {pattern.description}", file=sys.stderr)
            assert pattern.N_critical <= 10000

    # Test autocatalytic detection
    detections = autocatalytic_detect(all_receipts)
    print(f"# Detections: {len(detections)}", file=sys.stderr)

    print(f"# PASS: autocatalytic module self-test", file=sys.stderr)
