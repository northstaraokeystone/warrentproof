"""
WarrantProof Detect Module - Pattern-Based Anomaly Detection

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module provides anomaly detection based on GAO fraud taxonomy.
Flags unusual spending patterns for review.

Detection Taxonomy (Based on GAO Fraud Categories):
- Ghost Vendor: Payments to non-existent vendors
- Time Anomaly: Unusual timing patterns
- Cost Cascade: Overruns that propagate across milestones
- Certification Fraud: Quality certs without supporting data
- Inventory Ghost: Items recorded but non-existent

SLOs:
- Scan <= 100ms per 1000 receipts
- Alert generation <= 5s after detection
- False positive rate <= 5%
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    CITATIONS,
    PATTERN_COHERENCE_MIN,
    KOLMOGOROV_THRESHOLD,
    dual_hash,
    emit_receipt,
    get_citation,
)

# OMEGA v3 imports
from .zkp import generate_proof, verify_proof, ZKProof
from .kolmogorov import calculate_kolmogorov, detect_generator_pattern

# Import v2 autocatalytic functions
from .autocatalytic import (
    compute_entropy_gap,
    calculate_N_critical,
    crystallize_pattern,
    pattern_coherence_score,
    detect_autocatalytic_closure,
    autocatalytic_detect as _autocatalytic_detect_internal,
)


# === DETECTION TAXONOMY ===

ANOMALY_TYPES = {
    "ghost_vendor": {
        "description": "Payments to non-existent vendors",
        "detection_method": "Compression failure on vendor metadata",
        "citation_key": "GAO_GHOST_VENDOR",
    },
    "time_anomaly": {
        "description": "Unusual timing patterns (e.g., end-of-quarter spikes)",
        "detection_method": "Temporal clustering analysis",
        "citation_key": "GAO_FRAUD_ESTIMATE",
    },
    "cost_cascade": {
        "description": "Overruns that propagate across milestones",
        "detection_method": "Graph analysis of cost_variance_receipts",
        "citation_key": "GAO_ZUMWALT",
    },
    "cert_fraud": {
        "description": "Quality certs without supporting data",
        "detection_method": "Compression failure on attestation chains",
        "citation_key": "NEWPORT_NEWS_WELDING",
    },
    "inventory_ghost": {
        "description": "Items recorded but non-existent",
        "detection_method": "Compression failure on location metadata",
        "citation_key": "DODIG_AMMO",
    },
}

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]


# === CORE DETECTION FUNCTIONS ===

def scan(receipts: list, patterns: Optional[list[str]] = None) -> list[dict]:
    """
    Pattern match on receipt stream.
    Per spec: Scan <= 100ms per 1000 receipts.

    Args:
        receipts: List of receipts to scan
        patterns: Optional list of pattern types to check

    Returns:
        List of matches with scores
    """
    if patterns is None:
        patterns = list(ANOMALY_TYPES.keys())

    matches = []

    for pattern in patterns:
        if pattern == "ghost_vendor":
            matches.extend(_detect_ghost_vendor(receipts))
        elif pattern == "time_anomaly":
            matches.extend(_detect_time_anomaly(receipts))
        elif pattern == "cost_cascade":
            matches.extend(_detect_cost_cascade(receipts))
        elif pattern == "cert_fraud":
            matches.extend(_detect_cert_fraud(receipts))
        elif pattern == "inventory_ghost":
            matches.extend(_detect_inventory_ghost(receipts))

    return matches


def classify_anomaly(match: dict) -> str:
    """
    Return anomaly type per taxonomy.

    Args:
        match: Match dict from scan

    Returns:
        Anomaly type string
    """
    return match.get("anomaly_type", "unknown")


def temporal_cluster(receipts: list, window: str = "1d") -> dict:
    """
    Detect unusual timing patterns.
    Per spec: Temporal clustering analysis for time anomalies.

    Args:
        receipts: List of receipts to analyze
        window: Time window for clustering (e.g., "1d", "1h")

    Returns:
        Clustering analysis dict
    """
    if not receipts:
        return {"clusters": [], "anomalies": []}

    # Parse window
    if window.endswith("d"):
        delta = timedelta(days=int(window[:-1]))
    elif window.endswith("h"):
        delta = timedelta(hours=int(window[:-1]))
    else:
        delta = timedelta(days=1)

    # Group by time window
    clusters = defaultdict(list)
    for receipt in receipts:
        ts_str = receipt.get("ts", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                bucket = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                clusters[bucket.isoformat()].append(receipt)
            except ValueError:
                pass

    # Detect anomalous clusters (> 2 std dev from mean)
    cluster_sizes = [len(v) for v in clusters.values()]
    anomalies = []

    if len(cluster_sizes) >= 3:
        mean_size = statistics.mean(cluster_sizes)
        std_size = statistics.stdev(cluster_sizes)
        threshold = mean_size + (2 * std_size)

        for bucket, items in clusters.items():
            if len(items) > threshold:
                anomalies.append({
                    "bucket": bucket,
                    "count": len(items),
                    "threshold": threshold,
                    "anomaly_type": "time_anomaly",
                })

    return {
        "clusters": dict(clusters),
        "anomalies": anomalies,
        "mean_cluster_size": statistics.mean(cluster_sizes) if cluster_sizes else 0,
        "std_cluster_size": statistics.stdev(cluster_sizes) if len(cluster_sizes) >= 2 else 0,
    }


def cost_cascade_detect(program_receipts: list) -> dict:
    """
    Detect propagating cost overruns.
    Per spec: Graph analysis of cost_variance_receipts.

    Args:
        program_receipts: Receipts for a specific program

    Returns:
        Cascade analysis dict
    """
    # Filter to cost variance receipts
    variance_receipts = [
        r for r in program_receipts
        if r.get("receipt_type") == "cost_variance"
    ]

    if len(variance_receipts) < 2:
        return {"cascade_detected": False, "receipts": []}

    # Sort by timestamp
    variance_receipts.sort(key=lambda r: r.get("ts", ""))

    # Check for increasing variance pattern (cascade)
    variances = [r.get("variance_pct", 0) for r in variance_receipts]

    cascade_detected = False
    cascade_start = None

    for i in range(1, len(variances)):
        if variances[i] > variances[i-1]:
            if cascade_start is None:
                cascade_start = i - 1
            if i - cascade_start >= 2:  # 3+ increasing variances = cascade
                cascade_detected = True
                break
        else:
            cascade_start = None

    return {
        "cascade_detected": cascade_detected,
        "variance_trend": variances,
        "receipts": variance_receipts,
        "citation": get_citation("GAO_ZUMWALT"),
        "simulation_flag": DISCLAIMER,
    }


def emit_alert(anomaly: dict, severity: str) -> dict:
    """
    Create alert_receipt. Escalate if severity='critical'.
    Per spec: Alert generation <= 5s after detection.

    Args:
        anomaly: Anomaly details
        severity: Severity level

    Returns:
        alert_receipt dict
    """
    if severity not in SEVERITY_LEVELS:
        severity = "medium"

    alert = emit_receipt("alert", {
        "tenant_id": TENANT_ID,
        "anomaly_type": anomaly.get("anomaly_type", "unknown"),
        "severity": severity,
        "confidence": anomaly.get("confidence", 0.5),
        "affected_receipts": anomaly.get("affected_receipts", []),
        "description": anomaly.get("description", ""),
        "escalate": severity == "critical",
        "citation": anomaly.get("citation", get_citation("GAO_FRAUD_ESTIMATE")),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return alert


# === PATTERN DETECTORS ===

def _detect_ghost_vendor(receipts: list) -> list[dict]:
    """Detect ghost vendor patterns."""
    matches = []
    vendor_counts = defaultdict(int)
    vendor_amounts = defaultdict(float)

    # Count vendor occurrences
    for receipt in receipts:
        if receipt.get("receipt_type") == "warrant":
            vendor = receipt.get("vendor", receipt.get("description", ""))
            if vendor:
                vendor_counts[vendor] += 1
                vendor_amounts[vendor] += receipt.get("amount_usd", 0)

    # Flag vendors with single high-value transactions
    for vendor, count in vendor_counts.items():
        if count == 1 and vendor_amounts[vendor] > 1_000_000:
            matches.append({
                "anomaly_type": "ghost_vendor",
                "confidence": 0.7,
                "vendor": vendor,
                "amount": vendor_amounts[vendor],
                "pattern_matched": "single_high_value_vendor",
                "affected_receipts": [r.get("payload_hash") for r in receipts
                                     if r.get("vendor") == vendor or vendor in r.get("description", "")],
                "citation": get_citation("GAO_GHOST_VENDOR"),
                "simulation_flag": DISCLAIMER,
            })

    return matches


def _detect_time_anomaly(receipts: list) -> list[dict]:
    """Detect time-based anomalies."""
    matches = []
    cluster_result = temporal_cluster(receipts)

    for anomaly in cluster_result.get("anomalies", []):
        matches.append({
            "anomaly_type": "time_anomaly",
            "confidence": 0.8,
            "bucket": anomaly["bucket"],
            "count": anomaly["count"],
            "threshold": anomaly["threshold"],
            "pattern_matched": "temporal_spike",
            "affected_receipts": [],
            "citation": get_citation("GAO_FRAUD_ESTIMATE"),
            "simulation_flag": DISCLAIMER,
        })

    return matches


def _detect_cost_cascade(receipts: list) -> list[dict]:
    """Detect cost cascade patterns."""
    matches = []

    # Group by program
    programs = defaultdict(list)
    for receipt in receipts:
        program = receipt.get("program")
        if program:
            programs[program].append(receipt)

    # Check each program for cascades
    for program, program_receipts in programs.items():
        result = cost_cascade_detect(program_receipts)
        if result["cascade_detected"]:
            matches.append({
                "anomaly_type": "cost_cascade",
                "confidence": 0.85,
                "program": program,
                "variance_trend": result["variance_trend"],
                "pattern_matched": "increasing_overrun",
                "affected_receipts": [r.get("payload_hash") for r in result["receipts"]],
                "citation": get_citation("GAO_ZUMWALT"),
                "simulation_flag": DISCLAIMER,
            })

    return matches


def _detect_cert_fraud(receipts: list) -> list[dict]:
    """Detect certification fraud patterns."""
    matches = []

    # Look for quality attestations without proper lineage
    for receipt in receipts:
        if receipt.get("receipt_type") == "quality_attestation":
            lineage = receipt.get("decision_lineage", [])
            cert = receipt.get("certification", {})

            # Flag: high grade with no lineage
            if cert.get("grade") in ["A", "Pass"] and not lineage:
                matches.append({
                    "anomaly_type": "cert_fraud",
                    "confidence": 0.6,
                    "item": receipt.get("item"),
                    "inspector": receipt.get("inspector"),
                    "pattern_matched": "orphan_certification",
                    "affected_receipts": [receipt.get("payload_hash")],
                    "citation": get_citation("NEWPORT_NEWS_WELDING"),
                    "simulation_flag": DISCLAIMER,
                })

    return matches


def _detect_inventory_ghost(receipts: list) -> list[dict]:
    """Detect ghost inventory patterns."""
    matches = []

    # Look for delivery receipts without corresponding quality checks
    deliveries = [r for r in receipts if r.get("receipt_type") == "warrant"
                 and r.get("transaction_type") == "delivery"]
    attestations = {r.get("item"): r for r in receipts
                   if r.get("receipt_type") == "quality_attestation"}

    for delivery in deliveries:
        item = delivery.get("item")
        if item and item not in attestations:
            matches.append({
                "anomaly_type": "inventory_ghost",
                "confidence": 0.65,
                "item": item,
                "quantity": delivery.get("quantity"),
                "pattern_matched": "unverified_delivery",
                "affected_receipts": [delivery.get("payload_hash")],
                "citation": get_citation("DODIG_AMMO"),
                "simulation_flag": DISCLAIMER,
            })

    return matches


# === V2 AUTOCATALYTIC DETECTION ===

def autocatalytic_detect(
    receipts: list,
    mode: str = 'auto',
    existing_patterns: list = None
) -> list:
    """
    Detect fraud via autocatalytic pattern emergence.
    V2 replacement for hardcoded pattern_match.

    Steps:
    1. Build entropy_tree from receipts
    2. Query tree for high-entropy clusters
    3. Call autocatalytic.crystallize_pattern on clusters
    4. Validate patterns via pattern_coherence
    5. Return fraud detections

    Args:
        receipts: Receipts to analyze
        mode: 'auto' = pure v2 emergent, 'hybrid' = v2 + v1 fallback
        existing_patterns: Optional list of known patterns

    Returns:
        List of fraud detections
    """
    if not receipts:
        return []

    detections = []

    # Run v2 autocatalytic detection
    v2_detections = _autocatalytic_detect_internal(receipts, existing_patterns)

    for detection in v2_detections:
        detections.append({
            "anomaly_type": detection.get("anomaly_type", "autocatalytic"),
            "confidence": detection.get("confidence", 0.7),
            "pattern_id": detection.get("pattern_id", "emergent"),
            "receipt_id": detection.get("receipt_id", ""),
            "detection_mode": "v2_autocatalytic",
            "citation": get_citation("SHANNON_1948"),
            "simulation_flag": DISCLAIMER,
        })

    # If hybrid mode, also run v1 pattern matching as fallback
    if mode == 'hybrid':
        v1_matches = scan(receipts)
        for match in v1_matches:
            # Avoid duplicates
            if not any(
                d.get("receipt_id") == match.get("affected_receipts", [""])[0]
                for d in detections
            ):
                detections.append({
                    "anomaly_type": match.get("anomaly_type", "v1_pattern"),
                    "confidence": match.get("confidence", 0.6),
                    "pattern_id": "v1_hardcoded",
                    "receipt_id": match.get("affected_receipts", [""])[0] if match.get("affected_receipts") else "",
                    "detection_mode": "v1_fallback",
                    "citation": match.get("citation", get_citation("GAO_FRAUD_ESTIMATE")),
                    "simulation_flag": DISCLAIMER,
                })

    return detections


def pattern_match(receipt: dict, patterns: list) -> list:
    """
    Legacy v1 pattern matching - DEPRECATED in v2.
    Use autocatalytic_detect instead.

    Args:
        receipt: Receipt to check
        patterns: List of patterns to match against

    Returns:
        List of matching pattern IDs
    """
    # Delegate to scan for backward compatibility
    matches = scan([receipt])
    return [m.get("anomaly_type", "") for m in matches]


# === OMEGA v3: ZKP VERIFICATION GATE ===

def zkp_verification_gate(
    transaction: dict,
    state_prev: dict,
    state_next: dict,
    witness: dict = None
) -> dict:
    """
    OMEGA v3: ZKP verification gate for fraud detection.
    If proof invalid, override statistical fraud score with cryptographic certainty.

    This is the paradigm shift: Detection -> Prevention.
    Invalid transactions are rejected, not just flagged.

    Args:
        transaction: Transaction to verify
        state_prev: Previous ledger state
        state_next: Expected next state
        witness: Private witness data (optional, simulated)

    Returns:
        Verification result dict
    """
    if witness is None:
        # Extract witness from transaction
        witness = {
            "amount": transaction.get("amount_usd", 0),
            "vendor_signature": transaction.get("vendor_signature"),
            "sam_verified": transaction.get("sam_verified", True),
        }

    # Generate proof
    proof = generate_proof(state_prev, state_next, witness)

    # Verify proof
    is_valid = verify_proof(proof, state_prev, state_next)

    result = {
        "zkp_valid": is_valid,
        "proof_size_bytes": proof.size_bytes,
        "circuit_satisfied": proof.circuit_constraints_satisfied,
        "fraud_detected": not is_valid,
        "detection_method": "zkp_cryptographic",
        "confidence": 1.0 if not is_valid else 0.0,  # Cryptographic certainty
    }

    # If proof invalid, this is definitive fraud detection
    if not is_valid:
        result["fraud_reason"] = "ZKP proof verification failed - transaction mathematically impossible"
        result["action"] = "reject_transaction"

    return result


def detect_with_zkp_gate(
    receipts: list,
    mode: str = 'omega'
) -> list:
    """
    OMEGA v3: Fraud detection with ZKP verification gate.
    Combines Kolmogorov complexity, RAF detection, and ZKP verification.

    Args:
        receipts: Receipts to analyze
        mode: 'omega' = full OMEGA pipeline, 'hybrid' = omega + v2 fallback

    Returns:
        List of fraud detections
    """
    detections = []

    # Step 1: Kolmogorov complexity check
    K = calculate_kolmogorov(str(receipts))
    if K < KOLMOGOROV_THRESHOLD:
        generator_result = detect_generator_pattern(receipts)
        if generator_result.get("generator_detected"):
            detections.append({
                "anomaly_type": "kolmogorov_scripted",
                "confidence": 0.9,
                "K_complexity": K,
                "pattern": generator_result.get("pattern_description", ""),
                "detection_mode": "omega_kolmogorov",
                "citation": get_citation("SHANNON_1948"),
                "simulation_flag": DISCLAIMER,
            })

    # Step 2: For each transaction-like receipt, verify ZKP
    for i, receipt in enumerate(receipts):
        if receipt.get("receipt_type") == "warrant" and receipt.get("amount_usd"):
            # Simulate state transition
            state_prev = {"balance": 1000000000}  # Simulated
            state_next = {"balance": state_prev["balance"] - receipt.get("amount_usd", 0)}

            zkp_result = zkp_verification_gate(receipt, state_prev, state_next)

            if zkp_result.get("fraud_detected"):
                detections.append({
                    "anomaly_type": "zkp_invalid",
                    "confidence": 1.0,  # Cryptographic certainty
                    "receipt_index": i,
                    "zkp_valid": False,
                    "reason": zkp_result.get("fraud_reason", "ZKP verification failed"),
                    "detection_mode": "omega_zkp",
                    "action": "reject",
                    "simulation_flag": DISCLAIMER,
                })

    # Step 3: If hybrid mode, also run v2 autocatalytic detection
    if mode == 'hybrid':
        v2_detections = autocatalytic_detect(receipts, mode='auto')
        for d in v2_detections:
            d["detection_mode"] = "v2_autocatalytic_fallback"
            detections.append(d)

    return detections


# === DETECTION RECEIPT ===

def emit_detection_receipt(matches: list) -> dict:
    """
    Emit detection_receipt summarizing scan results.

    Args:
        matches: List of matches from scan

    Returns:
        detection_receipt dict
    """
    anomaly_counts = defaultdict(int)
    for match in matches:
        anomaly_counts[match.get("anomaly_type", "unknown")] += 1

    return emit_receipt("detection", {
        "tenant_id": TENANT_ID,
        "total_anomalies": len(matches),
        "anomaly_counts": dict(anomaly_counts),
        "patterns_checked": list(ANOMALY_TYPES.keys()),
        "matches": matches[:10],  # First 10 for summary
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import time

    print(f"# WarrantProof Detect Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Create test receipts
    test_receipts = [
        {"receipt_type": "warrant", "vendor": "OneTimeVendor", "amount_usd": 5_000_000,
         "ts": "2024-01-15T10:00:00Z", "description": "Test"},
        {"receipt_type": "cost_variance", "program": "TestProgram", "variance_pct": 5,
         "ts": "2024-01-01T00:00:00Z"},
        {"receipt_type": "cost_variance", "program": "TestProgram", "variance_pct": 12,
         "ts": "2024-02-01T00:00:00Z"},
        {"receipt_type": "cost_variance", "program": "TestProgram", "variance_pct": 23,
         "ts": "2024-03-01T00:00:00Z"},
        {"receipt_type": "quality_attestation", "item": "weld_1", "inspector": "SIM_001",
         "certification": {"grade": "A"}, "decision_lineage": []},
    ]

    # Test scan latency
    t0 = time.time()
    matches = scan(test_receipts * 200)  # 1000 receipts
    latency_ms = (time.time() - t0) * 1000
    print(f"# Scan latency for 1000 receipts: {latency_ms:.1f}ms", file=sys.stderr)
    assert latency_ms <= 100, f"Scan latency {latency_ms}ms > 100ms SLO"

    # Test pattern detection
    assert len(matches) > 0, "Should detect at least one anomaly"

    # Test classification
    for match in matches:
        anomaly_type = classify_anomaly(match)
        assert anomaly_type in ANOMALY_TYPES, f"Unknown anomaly type: {anomaly_type}"

    # Test cascade detection
    cascade = cost_cascade_detect(test_receipts)
    assert cascade["cascade_detected"] == True, "Should detect cost cascade"

    # Test alert emission
    alert = emit_alert(matches[0], "high")
    assert alert["receipt_type"] == "alert"
    assert "simulation_flag" in alert

    print(f"# PASS: detect module self-test ({len(matches)} anomalies found)", file=sys.stderr)
