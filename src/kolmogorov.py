"""
WarrantProof Kolmogorov Module - Algorithmic Complexity via Compression

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements Kolmogorov complexity estimation for fraud detection.
The key insight: Fraud = low algorithmic complexity masquerading as high
Shannon entropy. A fraudster using a script to generate 10,000 invoices
adds Gaussian noise (high H) but the underlying generator is simple.

K(x) = compressed_size / original_size

- K < 0.65 = scripted fraud (too compressible)
- K > 0.75 = legitimate (algorithmically irreducible)

Physics Foundation:
- Kolmogorov complexity measures minimum program length to generate string
- Shannon entropy measures uncertainty in random variable
- Fraud = low K(x) despite high H(x) (mimicry attack)
- Reality is incompressible (weather, human error, mechanical failures)

OMEGA Citation:
"Sophisticated fraud often exhibits lower Kolmogorov complexity than
legitimate chaos."
"""

import gzip
import json
import lzma
import zlib
from typing import Optional, Union

from .core import (
    TENANT_ID,
    DISCLAIMER,
    KOLMOGOROV_THRESHOLD,
    KOLMOGOROV_LEGITIMATE_MIN,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


def calculate_kolmogorov(data: Union[str, bytes]) -> float:
    """
    Apply zlib/lzma compression. Return K(x) = compressed_size / original_size.
    Range 0-1 where lower = more compressible = more suspicious.

    Args:
        data: String or bytes to analyze

    Returns:
        K(x) complexity ratio (0-1)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')

    if len(data) == 0:
        return 1.0

    original_size = len(data)

    # Use multiple compression algorithms for robustness
    try:
        zlib_compressed = zlib.compress(data, level=9)
        zlib_ratio = len(zlib_compressed) / original_size
    except Exception:
        zlib_ratio = 1.0

    try:
        lzma_compressed = lzma.compress(data, preset=9)
        lzma_ratio = len(lzma_compressed) / original_size
    except Exception:
        lzma_ratio = 1.0

    # Use the better (lower) compression ratio
    # Better compression = simpler underlying structure
    K = min(zlib_ratio, lzma_ratio)

    # Clamp to [0, 1] range (can exceed 1 for very small inputs)
    return max(0.0, min(1.0, K))


def compress_transaction_history(transactions: list) -> float:
    """
    Serialize transaction history, compress, return ratio.
    Low ratio = scripted transactions (suspicious).

    Args:
        transactions: List of transaction dicts

    Returns:
        K(x) complexity ratio for the entire history
    """
    if not transactions:
        return 1.0

    # Serialize to JSON with sorted keys for consistency
    data = json.dumps(transactions, sort_keys=True)
    return calculate_kolmogorov(data)


def detect_generator_pattern(transactions: list) -> dict:
    """
    If K < KOLMOGOROV_THRESHOLD, analyze for loop structures.
    Return generator detection result.

    Args:
        transactions: List of transaction dicts

    Returns:
        Dict with generator_detected, pattern_description, K_complexity
    """
    K = compress_transaction_history(transactions)

    result = {
        "K_complexity": round(K, 4),
        "generator_detected": False,
        "pattern_description": "",
        "fraud_flag": K < KOLMOGOROV_THRESHOLD,
    }

    if K >= KOLMOGOROV_THRESHOLD:
        result["pattern_description"] = "Transactions appear algorithmically irreducible"
        return result

    # Analyze patterns when K is suspiciously low
    result["generator_detected"] = True

    # Check for repeating structures
    patterns_found = []

    # Pattern 1: Check for repeating vendor sequences
    vendors = [t.get("vendor", "") for t in transactions if t.get("vendor")]
    if vendors:
        vendor_set = set(vendors)
        if len(vendor_set) <= len(vendors) * 0.1:  # <10% unique vendors
            patterns_found.append(f"Rotating vendors ({len(vendor_set)} unique in {len(vendors)} transactions)")

    # Pattern 2: Check for amount patterns
    amounts = [t.get("amount_usd", 0) for t in transactions if t.get("amount_usd")]
    if amounts:
        # Check if amounts follow a simple distribution (uniform, linear)
        amount_set = set(amounts)
        if len(amount_set) <= 5:
            patterns_found.append(f"Limited amount values ({len(amount_set)} unique)")

    # Pattern 3: Check for timestamp patterns
    timestamps = [t.get("ts", "") for t in transactions if t.get("ts")]
    if len(timestamps) >= 2:
        # Check for uniform intervals
        try:
            from datetime import datetime
            dts = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps[:20]]
            if len(dts) >= 2:
                intervals = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)]
                if intervals and max(intervals) - min(intervals) < 60:  # Same interval within 1 min
                    patterns_found.append(f"Uniform timestamp intervals (~{intervals[0]:.0f}s)")
        except Exception:
            pass

    if patterns_found:
        result["pattern_description"] = "; ".join(patterns_found)
    else:
        result["pattern_description"] = f"Low K(x)={K:.3f} suggests simple generator"

    return result


def compare_to_legitimate_distribution(
    K_vendor: float,
    K_baseline: float = 0.80
) -> float:
    """
    Z-score comparison. Legitimate approx 0.75-0.95.

    Args:
        K_vendor: Vendor's transaction K complexity
        K_baseline: Baseline legitimate K (default 0.80)

    Returns:
        Deviation score (higher = more deviation from legitimate)
    """
    # Assume legitimate K has std dev of ~0.10
    std_dev = 0.10
    z_score = (K_baseline - K_vendor) / std_dev

    # Positive z_score means K_vendor is lower than baseline (suspicious)
    return max(0.0, z_score)


def kolmogorov_compress(receipt: dict) -> tuple:
    """
    Serialize receipt, compress with zlib/lzma. Return K(x).
    Replacement for Shannon entropy in compress.py.

    Args:
        receipt: Single receipt to analyze

    Returns:
        (K_complexity, compression_ratio, field_ratios)
    """
    # Calculate overall K
    data = json.dumps(receipt, sort_keys=True)
    K = calculate_kolmogorov(data)

    # Calculate field-wise K
    field_ratios = {}
    for field, value in receipt.items():
        if field in ["payload_hash", "ts", "tenant_id", "simulation_flag"]:
            continue

        if isinstance(value, (dict, list)):
            field_data = json.dumps(value, sort_keys=True)
        else:
            field_data = str(value)

        if len(field_data) >= 10:
            field_ratios[field] = round(calculate_kolmogorov(field_data), 4)

    return (round(K, 4), round(K, 4), field_ratios)


def emit_kolmogorov_receipt(
    entity_id: str,
    transactions: Optional[list] = None,
    K_complexity: Optional[float] = None
) -> dict:
    """
    Emit kolmogorov_receipt documenting complexity analysis.

    Args:
        entity_id: DUNS or Award_ID
        transactions: Optional transaction history
        K_complexity: Pre-calculated K (if transactions not provided)

    Returns:
        kolmogorov_receipt dict
    """
    if K_complexity is None and transactions:
        K_complexity = compress_transaction_history(transactions)
    elif K_complexity is None:
        K_complexity = 1.0

    detection = detect_generator_pattern(transactions or [])

    return emit_receipt("kolmogorov", {
        "tenant_id": TENANT_ID,
        "entity_id": entity_id,
        "K_complexity": round(K_complexity, 4),
        "compression_ratio": round(K_complexity, 4),
        "algorithm_detected": detection.get("generator_detected", False),
        "generator_pattern": detection.get("pattern_description", ""),
        "fraud_flag": K_complexity < KOLMOGOROV_THRESHOLD,
        "threshold": KOLMOGOROV_THRESHOLD,
        "legitimate_min": KOLMOGOROV_LEGITIMATE_MIN,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_compression_failure(ratio: float) -> None:
    """If compression ratio > 1.0, data corrupted."""
    if ratio > 1.0:
        emit_receipt("anomaly", {
            "metric": "compression_failure",
            "ratio": ratio,
            "delta": ratio - 1.0,
            "action": "halt",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(f"Compression ratio {ratio} > 1.0 indicates data corruption")


def stoprule_threshold_invalid(K_threshold: float, baseline_K: float) -> None:
    """If threshold not calibrated on legitimate baseline."""
    if K_threshold >= baseline_K:
        emit_receipt("anomaly", {
            "metric": "threshold_invalid",
            "threshold": K_threshold,
            "baseline": baseline_K,
            "delta": K_threshold - baseline_K,
            "action": "recalibrate",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Threshold {K_threshold} >= baseline {baseline_K}, must recalibrate"
        )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import os

    print(f"# WarrantProof Kolmogorov Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Highly compressible data (scripted)
    scripted = b"x" * 10000
    K_scripted = calculate_kolmogorov(scripted)
    print(f"# Scripted K(x): {K_scripted:.4f} (should be < 0.10)", file=sys.stderr)
    assert K_scripted < 0.10, f"Scripted K={K_scripted} should be very low"

    # Test 2: Random data (incompressible)
    random_data = os.urandom(10000)
    K_random = calculate_kolmogorov(random_data)
    print(f"# Random K(x): {K_random:.4f} (should be > 0.90)", file=sys.stderr)
    assert K_random > 0.90, f"Random K={K_random} should be high"

    # Test 3: Transaction history
    scripted_transactions = [
        {"vendor": f"Vendor_{i % 3}", "amount_usd": 1000000, "ts": f"2024-01-{(i % 28)+1:02d}T10:00:00Z"}
        for i in range(100)
    ]
    K_scripted_tx = compress_transaction_history(scripted_transactions)
    print(f"# Scripted transactions K(x): {K_scripted_tx:.4f}", file=sys.stderr)
    assert K_scripted_tx < KOLMOGOROV_LEGITIMATE_MIN, "Scripted transactions should compress well"

    # Test 4: Generator detection
    detection = detect_generator_pattern(scripted_transactions)
    print(f"# Generator detected: {detection['generator_detected']}", file=sys.stderr)
    print(f"# Pattern: {detection['pattern_description']}", file=sys.stderr)
    assert detection["generator_detected"] == True

    # Test 5: Receipt emission
    receipt = emit_kolmogorov_receipt("DUNS_123456", scripted_transactions)
    assert receipt["receipt_type"] == "kolmogorov"
    assert "K_complexity" in receipt
    assert receipt["fraud_flag"] == True

    # Test 6: Z-score comparison
    deviation = compare_to_legitimate_distribution(K_scripted_tx)
    print(f"# Deviation score: {deviation:.2f}", file=sys.stderr)
    assert deviation > 0, "Low K should show positive deviation"

    print(f"# PASS: kolmogorov module self-test", file=sys.stderr)
