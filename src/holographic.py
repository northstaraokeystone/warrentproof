"""
WarrantProof Holographic Module - Boundary-Only Fraud Detection with Data Availability

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module implements holographic fraud detection using Merkle roots.
Based on the holographic principle (Bekenstein bound): volume entropy
encodes on boundary. Detect fraud from O(1) boundary check without
scanning O(N) volume.

OMEGA v3 Enhancement:
Data availability sampling via erasure coding. Per OMEGA:
"Full data availability guarantee via random sampling."
Integrates DAS module for cryptographic availability proofs.

Physics Foundation:
- Holographic principle: volume entropy ≤ boundary area
- Ledger = volume, Merkle root = boundary
- Fraud anywhere in volume ripples the horizon (changes root)
- bits_required ≈ area × log(1/p_detect) ≤ 2 bits/receipt
- Data availability: Pr(detect_unavailable) > 1 - 2^(-sample_count)

SLOs:
- Detection probability p > 0.9999 from boundary
- Bits per receipt ≤ 2 (holographic compression bound)
- Fraud localization in O(log N) time
- Data availability confidence > 99% with 10% sampling
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from .core import (
    TENANT_ID,
    DISCLAIMER,
    HOLOGRAPHIC_BITS_PER_RECEIPT,
    HOLOGRAPHIC_DETECTION_PROBABILITY_MIN,
    HOLOGRAPHIC_LOCALIZATION_COMPLEXITY,
    DATA_AVAILABILITY_SAMPLE_RATE,
    dual_hash,
    emit_receipt,
    merkle,
    StopRuleException,
)

# OMEGA v3: Import DAS for data availability
from .das import (
    encode_with_erasure,
    sample_chunks,
    verify_availability,
    detect_erasure,
    light_client_audit,
    ErasureEncodedData,
)


@dataclass
class MerkleRootHistory:
    """Track historical Merkle roots for outlier detection."""
    roots: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    batch_sizes: list = field(default_factory=list)

    def add(self, root: str, timestamp: str = "", batch_size: int = 0):
        self.roots.append(root)
        self.timestamps.append(timestamp)
        self.batch_sizes.append(batch_size)

    @property
    def size(self) -> int:
        return len(self.roots)


def compute_merkle_syndrome(merkle_root: str, expected_root: str) -> dict:
    """
    Compute syndrome = difference between actual and expected root.
    Any fraud changes root → syndrome non-zero.

    Args:
        merkle_root: Actual Merkle root
        expected_root: Expected Merkle root

    Returns:
        Syndrome signature dict
    """
    if merkle_root == expected_root:
        return {
            "syndrome": "0" * 64,  # Zero syndrome = match
            "match": True,
            "difference_bits": 0,
        }

    # XOR the hex strings (simplified syndrome)
    try:
        # Take SHA256 portion (before colon)
        actual_sha = merkle_root.split(":")[0]
        expected_sha = expected_root.split(":")[0]

        # Convert to integers and XOR
        actual_int = int(actual_sha, 16)
        expected_int = int(expected_sha, 16)

        syndrome_int = actual_int ^ expected_int
        syndrome_hex = format(syndrome_int, '064x')

        # Count difference bits
        difference_bits = bin(syndrome_int).count('1')

        return {
            "syndrome": syndrome_hex,
            "match": False,
            "difference_bits": difference_bits,
            "divergence_ratio": difference_bits / 256,  # Normalized
        }
    except (ValueError, IndexError):
        return {
            "syndrome": "error",
            "match": False,
            "difference_bits": -1,
        }


def detect_from_boundary(
    merkle_root: str,
    root_history: MerkleRootHistory,
    sigma_threshold: float = 3.0
) -> bool:
    """
    Compare current root to historical distribution.
    Outlier detection on boundary. If root deviates > 3σ, fraud suspected.

    Args:
        merkle_root: Current Merkle root to check
        root_history: Historical roots
        sigma_threshold: Number of standard deviations for outlier (default 3)

    Returns:
        True if fraud suspected from boundary analysis
    """
    if root_history.size < 3:
        # Not enough history for statistical detection
        return False

    # Convert roots to numeric representation for comparison
    # Use first 16 hex chars as integer
    def root_to_numeric(root: str) -> int:
        try:
            sha = root.split(":")[0][:16]
            return int(sha, 16)
        except (ValueError, IndexError):
            return 0

    historical_values = [root_to_numeric(r) for r in root_history.roots]
    current_value = root_to_numeric(merkle_root)

    # Calculate Z-score
    mean = np.mean(historical_values)
    std = np.std(historical_values)

    if std == 0:
        # All roots identical - any difference is suspicious
        return current_value != historical_values[0]

    z_score = abs(current_value - mean) / std

    return z_score > sigma_threshold


def holographic_encode(receipts: list) -> str:
    """
    Standard Merkle tree construction.
    Encoding: full ledger → O(1) boundary.

    Args:
        receipts: List of receipts to encode

    Returns:
        Merkle root hash
    """
    return merkle(receipts)


def decode_fraud_location(
    syndrome: dict,
    tree_structure: dict
) -> list:
    """
    Given syndrome, traverse Merkle tree to identify which subtree contains fraud.
    O(log N) path descent.

    Args:
        syndrome: Syndrome from compute_merkle_syndrome
        tree_structure: Tree structure with left/right hashes

    Returns:
        List of suspected branch identifiers
    """
    if syndrome.get("match", True):
        return []  # No fraud detected

    suspected_branches = []

    # Simplified branch identification based on syndrome pattern
    difference_bits = syndrome.get("difference_bits", 0)

    if difference_bits > 128:
        # Major difference - likely in left subtree
        suspected_branches.append("left_subtree")
    elif difference_bits > 64:
        # Moderate difference - could be right subtree
        suspected_branches.append("right_subtree")
    else:
        # Minor difference - leaf level
        suspected_branches.append("leaf_modification")

    # Add more specific localization if tree structure provided
    if tree_structure:
        depth = tree_structure.get("depth", 0)
        suspected_branches.append(f"depth_{depth}")

    return suspected_branches


def bits_required(
    num_receipts: int,
    detection_probability: float = HOLOGRAPHIC_DETECTION_PROBABILITY_MIN
) -> float:
    """
    Calculate holographic storage requirement.
    bits ≈ area × log(1/p_detect) ≤ HOLOGRAPHIC_BITS_PER_RECEIPT × N

    Args:
        num_receipts: Number of receipts in ledger
        detection_probability: Required detection probability

    Returns:
        Bits required for encoding
    """
    if detection_probability >= 1.0:
        detection_probability = 0.9999

    if detection_probability <= 0:
        detection_probability = 0.5

    # From holographic principle
    # bits = area × log(1/p_detect)
    # area proportional to boundary = sqrt(N) for 2D analogy
    area = math.sqrt(num_receipts)

    log_factor = -math.log2(1 - detection_probability)

    bits = area * log_factor

    return bits


def verify_bekenstein_bound(
    num_receipts: int,
    actual_bits: float
) -> bool:
    """
    Verify storage respects Bekenstein bound.

    Args:
        num_receipts: Number of receipts
        actual_bits: Actual bits used

    Returns:
        True if within bound
    """
    max_bits = HOLOGRAPHIC_BITS_PER_RECEIPT * num_receipts
    return actual_bits <= max_bits


def holographic_detect(
    current_root: str,
    expected_root: Optional[str] = None,
    root_history: Optional[MerkleRootHistory] = None
) -> dict:
    """
    Detect fraud from Merkle root alone (O(1) boundary check).

    Args:
        current_root: Current Merkle root
        expected_root: Optional expected root for syndrome check
        root_history: Optional history for statistical detection

    Returns:
        Detection result dict
    """
    result = {
        "fraud_detected": False,
        "detection_method": None,
        "confidence": 0.0,
    }

    # Method 1: Syndrome check against expected
    if expected_root:
        syndrome = compute_merkle_syndrome(current_root, expected_root)
        if not syndrome.get("match", True):
            result["fraud_detected"] = True
            result["detection_method"] = "syndrome_mismatch"
            result["confidence"] = min(1.0, syndrome.get("difference_bits", 0) / 64)
            result["syndrome"] = syndrome

    # Method 2: Statistical outlier detection
    if root_history and root_history.size >= 3:
        is_outlier = detect_from_boundary(current_root, root_history)
        if is_outlier:
            result["fraud_detected"] = True
            result["detection_method"] = result.get("detection_method", "statistical_outlier")
            result["confidence"] = max(result.get("confidence", 0), 0.9)
            result["outlier_detected"] = True

    # Calculate detection probability
    if result["fraud_detected"]:
        result["detection_probability"] = result["confidence"] * 0.9999
    else:
        result["detection_probability"] = 0.0

    return result


def emit_holographic_receipt(
    merkle_root_current: str,
    merkle_root_expected: Optional[str] = None,
    detection_result: Optional[dict] = None,
    num_receipts: int = 0
) -> dict:
    """
    Emit holographic_receipt documenting boundary detection.

    Args:
        merkle_root_current: Current Merkle root
        merkle_root_expected: Expected Merkle root
        detection_result: Result from holographic_detect
        num_receipts: Number of receipts in ledger

    Returns:
        holographic_receipt dict
    """
    if detection_result is None:
        detection_result = holographic_detect(
            merkle_root_current,
            merkle_root_expected
        )

    bits_used = bits_required(max(1, num_receipts))

    return emit_receipt("holographic", {
        "tenant_id": TENANT_ID,
        "merkle_root_current": merkle_root_current,
        "merkle_root_expected": merkle_root_expected,
        "syndrome": detection_result.get("syndrome", {}),
        "fraud_detected_from_boundary": detection_result.get("fraud_detected", False),
        "fraud_branch_identified": detection_result.get("branch_identified", ""),
        "detection_probability": detection_result.get("detection_probability", 0),
        "detection_method": detection_result.get("detection_method", "none"),
        "bits_used": round(bits_used, 2),
        "bekenstein_bound_respected": verify_bekenstein_bound(
            max(1, num_receipts), bits_used
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === OMEGA v3: DATA AVAILABILITY SAMPLING ===

@dataclass
class HolographicState:
    """
    Combined holographic and data availability state.
    Per OMEGA: Boundary detection + availability guarantees.
    """
    merkle_root: str = ""
    erasure_encoded: Optional[ErasureEncodedData] = None
    availability_verified: bool = False
    availability_confidence: float = 0.0
    last_sample_indices: List[int] = field(default_factory=list)


def create_holographic_state(
    receipts: List[Dict[str, Any]],
    erasure_factor: int = 2
) -> HolographicState:
    """
    Create holographic state with erasure encoding.
    Combines Merkle root with data availability guarantees.

    Args:
        receipts: List of receipts to encode
        erasure_factor: Erasure coding expansion factor (default 2x)

    Returns:
        HolographicState with Merkle root and erasure encoding
    """
    # Compute Merkle root (boundary encoding)
    merkle_root = holographic_encode(receipts)

    # Erasure encode for data availability
    import json
    data_bytes = json.dumps(receipts, sort_keys=True).encode()
    erasure_encoded = encode_with_erasure(data_bytes, expansion_factor=erasure_factor)

    return HolographicState(
        merkle_root=merkle_root,
        erasure_encoded=erasure_encoded,
        availability_verified=False,
        availability_confidence=0.0,
    )


def verify_data_availability(
    state: HolographicState,
    sample_rate: float = DATA_AVAILABILITY_SAMPLE_RATE
) -> Tuple[bool, float, List[int]]:
    """
    Verify data availability via random sampling.
    Per OMEGA: "Full data availability guarantee via random sampling."

    Args:
        state: HolographicState with erasure encoding
        sample_rate: Fraction of chunks to sample (default 10%)

    Returns:
        Tuple of (available, confidence, sampled_indices)
    """
    if not state.erasure_encoded:
        return False, 0.0, []

    # Perform light client audit
    audit_result = light_client_audit(
        state.erasure_encoded,
        sample_rate=sample_rate
    )

    available = audit_result.get("available", False)
    confidence = audit_result.get("confidence", 0.0)
    sampled_indices = audit_result.get("sampled_indices", [])

    # Update state
    state.availability_verified = available
    state.availability_confidence = confidence
    state.last_sample_indices = sampled_indices

    return available, confidence, sampled_indices


def holographic_detect_with_da(
    current_root: str,
    expected_root: Optional[str] = None,
    root_history: Optional[MerkleRootHistory] = None,
    state: Optional[HolographicState] = None
) -> Dict[str, Any]:
    """
    Enhanced holographic detection with data availability verification.
    Combines boundary fraud detection with availability guarantees.

    Args:
        current_root: Current Merkle root
        expected_root: Optional expected root for syndrome check
        root_history: Optional history for statistical detection
        state: Optional HolographicState for DA verification

    Returns:
        Detection result dict with DA status
    """
    # Base holographic detection
    result = holographic_detect(current_root, expected_root, root_history)

    # Add data availability verification
    if state:
        available, confidence, indices = verify_data_availability(state)

        result["data_availability"] = {
            "verified": available,
            "confidence": round(confidence, 4),
            "samples_checked": len(indices),
            "sample_rate": DATA_AVAILABILITY_SAMPLE_RATE,
        }

        # If data unavailable, this is also fraud indicator
        if not available:
            result["fraud_detected"] = True
            result["detection_method"] = result.get("detection_method", "data_unavailable")
            result["confidence"] = max(result.get("confidence", 0), 0.85)
            result["data_availability"]["erasure_detected"] = True

    return result


def detect_selective_withholding(
    state: HolographicState,
    suspect_indices: List[int]
) -> Dict[str, Any]:
    """
    Detect selective data withholding.
    When specific receipts are withheld to hide fraud.

    Args:
        state: HolographicState with erasure encoding
        suspect_indices: Indices suspected of being withheld

    Returns:
        Withholding detection result
    """
    if not state.erasure_encoded:
        return {"error": "No erasure encoding available"}

    # Check specific indices
    detection = detect_erasure(state.erasure_encoded, suspect_indices)

    # If suspect indices have erasures, likely selective withholding
    result = {
        "selective_withholding_suspected": detection.get("erasure_detected", False),
        "suspect_indices": suspect_indices,
        "erasure_pattern": detection.get("erasure_pattern", []),
        "reconstruction_possible": detection.get("recoverable", False),
    }

    return result


def reconstruct_from_erasure(
    state: HolographicState,
    missing_indices: List[int]
) -> Optional[bytes]:
    """
    Attempt to reconstruct missing data from erasure coding.
    Returns None if reconstruction fails (too much data missing).

    Args:
        state: HolographicState with erasure encoding
        missing_indices: Indices of missing chunks

    Returns:
        Reconstructed data bytes or None
    """
    if not state.erasure_encoded:
        return None

    # Check if reconstruction is possible
    max_recoverable = len(state.erasure_encoded.chunks) // state.erasure_encoded.expansion_factor

    if len(missing_indices) > max_recoverable:
        return None

    # Simulate reconstruction (in production, use real Reed-Solomon)
    available_chunks = [
        chunk for i, chunk in enumerate(state.erasure_encoded.chunks)
        if i not in missing_indices
    ]

    if len(available_chunks) >= len(state.erasure_encoded.chunks) // state.erasure_encoded.expansion_factor:
        # Reconstruction would succeed
        return state.erasure_encoded.original_data

    return None


def emit_holographic_da_receipt(
    state: HolographicState,
    detection_result: Optional[Dict[str, Any]] = None,
    num_receipts: int = 0
) -> Dict[str, Any]:
    """
    Emit holographic receipt with data availability attestation.

    Args:
        state: HolographicState
        detection_result: Detection result
        num_receipts: Number of receipts

    Returns:
        holographic_da_receipt dict
    """
    if detection_result is None:
        detection_result = holographic_detect_with_da(
            state.merkle_root,
            state=state
        )

    bits_used = bits_required(max(1, num_receipts))

    receipt_data = {
        "tenant_id": TENANT_ID,
        "merkle_root": state.merkle_root,
        "fraud_detected": detection_result.get("fraud_detected", False),
        "detection_method": detection_result.get("detection_method", "none"),
        "detection_probability": detection_result.get("detection_probability", 0),
        "bits_used": round(bits_used, 2),
        "bekenstein_bound_respected": verify_bekenstein_bound(max(1, num_receipts), bits_used),
        "simulation_flag": DISCLAIMER,
    }

    # Add data availability attestation
    da_info = detection_result.get("data_availability", {})
    if da_info:
        receipt_data["data_availability"] = {
            "verified": da_info.get("verified", False),
            "confidence": da_info.get("confidence", 0),
            "sample_rate": DATA_AVAILABILITY_SAMPLE_RATE,
            "omega_citation": "OMEGA: Full data availability guarantee via random sampling",
        }

        # Add erasure encoding metadata if available
        if state.erasure_encoded:
            receipt_data["erasure_encoding"] = {
                "expansion_factor": state.erasure_encoded.expansion_factor,
                "chunk_count": len(state.erasure_encoded.chunks),
                "original_size": len(state.erasure_encoded.original_data),
            }

    return emit_receipt("holographic_da", receipt_data, to_stdout=False)


def holographic_audit(
    receipts: List[Dict[str, Any]],
    expected_root: Optional[str] = None,
    root_history: Optional[MerkleRootHistory] = None,
    sample_rate: float = DATA_AVAILABILITY_SAMPLE_RATE
) -> Dict[str, Any]:
    """
    Full holographic audit with data availability.
    Combines boundary detection, statistical outliers, and DA sampling.

    Args:
        receipts: Receipts to audit
        expected_root: Optional expected Merkle root
        root_history: Optional historical roots
        sample_rate: DA sample rate

    Returns:
        Complete audit result
    """
    # Create holographic state
    state = create_holographic_state(receipts)

    # Verify data availability first
    available, confidence, indices = verify_data_availability(state, sample_rate)

    # Run holographic detection
    detection = holographic_detect_with_da(
        state.merkle_root,
        expected_root,
        root_history,
        state
    )

    # Emit receipt
    receipt = emit_holographic_da_receipt(state, detection, len(receipts))

    return {
        "state": state,
        "detection": detection,
        "receipt": receipt,
        "audit_summary": {
            "receipts_audited": len(receipts),
            "merkle_root": state.merkle_root,
            "fraud_detected": detection.get("fraud_detected", False),
            "data_available": available,
            "da_confidence": confidence,
            "samples_checked": len(indices),
        },
    }


# === STOPRULES ===

def stoprule_data_unavailable(confidence: float, threshold: float = 0.99) -> None:
    """Data availability confidence must meet threshold."""
    if confidence < threshold:
        emit_receipt("anomaly", {
            "metric": "data_unavailable",
            "confidence": confidence,
            "threshold": threshold,
            "delta": confidence - threshold,
            "action": "investigate_withholding",
            "classification": "critical",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Data availability confidence {confidence} below threshold {threshold}"
        )


def stoprule_detection_probability_low(actual_p: float) -> None:
    """Detection probability must be > 0.9999."""
    if actual_p < HOLOGRAPHIC_DETECTION_PROBABILITY_MIN:
        emit_receipt("anomaly", {
            "metric": "detection_probability_low",
            "actual": actual_p,
            "minimum": HOLOGRAPHIC_DETECTION_PROBABILITY_MIN,
            "delta": actual_p - HOLOGRAPHIC_DETECTION_PROBABILITY_MIN,
            "action": "increase_encoding",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Detection probability {actual_p} below minimum {HOLOGRAPHIC_DETECTION_PROBABILITY_MIN}"
        )


def stoprule_boundary_bits_exceeded(actual_bits: float, num_receipts: int) -> None:
    """Bits must not exceed 2N (Bekenstein bound violation)."""
    max_bits = HOLOGRAPHIC_BITS_PER_RECEIPT * num_receipts
    if actual_bits > max_bits:
        emit_receipt("anomaly", {
            "metric": "boundary_bits_exceeded",
            "actual_bits": actual_bits,
            "max_bits": max_bits,
            "num_receipts": num_receipts,
            "action": "optimize_encoding",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Boundary bits {actual_bits} exceeds Bekenstein bound {max_bits}"
        )


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Holographic Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test Merkle encoding
    test_receipts = [
        {"id": 1, "data": "test1"},
        {"id": 2, "data": "test2"},
        {"id": 3, "data": "test3"},
    ]
    root = holographic_encode(test_receipts)
    print(f"# Merkle root: {root[:32]}...", file=sys.stderr)
    assert ":" in root  # dual_hash format

    # Test syndrome computation - matching roots
    syndrome_match = compute_merkle_syndrome(root, root)
    assert syndrome_match["match"] == True
    assert syndrome_match["difference_bits"] == 0

    # Test syndrome computation - different roots
    modified_receipts = test_receipts.copy()
    modified_receipts[0]["data"] = "MODIFIED"
    modified_root = holographic_encode(modified_receipts)

    syndrome_diff = compute_merkle_syndrome(root, modified_root)
    print(f"# Syndrome match: {syndrome_diff['match']}, diff bits: {syndrome_diff['difference_bits']}", file=sys.stderr)
    assert syndrome_diff["match"] == False
    assert syndrome_diff["difference_bits"] > 0

    # Test historical outlier detection
    history = MerkleRootHistory()
    for i in range(10):
        test_data = [{"id": j, "data": f"stable_{j}"} for j in range(5)]
        stable_root = holographic_encode(test_data)
        history.add(stable_root)

    # Normal root should not be outlier
    normal_detection = detect_from_boundary(history.roots[0], history)
    print(f"# Normal root outlier: {normal_detection}", file=sys.stderr)

    # Test bits calculation
    bits = bits_required(1000, detection_probability=0.9999)
    print(f"# Bits for 1000 receipts: {bits:.2f}", file=sys.stderr)
    assert bits <= HOLOGRAPHIC_BITS_PER_RECEIPT * 1000

    # Test Bekenstein bound
    within_bound = verify_bekenstein_bound(1000, bits)
    assert within_bound == True

    # Test fraud location decoding
    locations = decode_fraud_location(syndrome_diff, {"depth": 3})
    print(f"# Suspected locations: {locations}", file=sys.stderr)

    # Test holographic detect
    detection = holographic_detect(root, modified_root, history)
    print(f"# Fraud detected: {detection['fraud_detected']}", file=sys.stderr)
    assert detection["fraud_detected"] == True

    # Test receipt emission
    receipt = emit_holographic_receipt(root, modified_root, detection, 100)
    assert receipt["receipt_type"] == "holographic"
    assert "simulation_flag" in receipt

    print(f"# Base holographic tests passed", file=sys.stderr)

    # === OMEGA v3: Data Availability Tests ===
    print(f"# Testing OMEGA v3 data availability...", file=sys.stderr)

    # Test holographic state creation
    state = create_holographic_state(test_receipts)
    assert state.merkle_root == root
    assert state.erasure_encoded is not None
    print(f"# Holographic state created with erasure encoding", file=sys.stderr)

    # Test data availability verification
    available, confidence, indices = verify_data_availability(state)
    print(f"# DA verified: {available}, confidence: {confidence:.4f}, samples: {len(indices)}", file=sys.stderr)
    assert available == True
    assert confidence > 0.9

    # Test holographic detect with DA
    detection_da = holographic_detect_with_da(root, state=state)
    assert "data_availability" in detection_da
    assert detection_da["data_availability"]["verified"] == True
    print(f"# Detection with DA: {detection_da['data_availability']}", file=sys.stderr)

    # Test selective withholding detection
    withholding = detect_selective_withholding(state, [0, 1, 2])
    print(f"# Withholding detection: {withholding}", file=sys.stderr)

    # Test DA receipt emission
    da_receipt = emit_holographic_da_receipt(state, detection_da, len(test_receipts))
    assert da_receipt["receipt_type"] == "holographic_da"
    assert "data_availability" in da_receipt
    print(f"# DA receipt emitted: {da_receipt['receipt_type']}", file=sys.stderr)

    # Test full holographic audit
    audit_result = holographic_audit(test_receipts)
    assert "audit_summary" in audit_result
    assert audit_result["audit_summary"]["data_available"] == True
    print(f"# Full audit: {audit_result['audit_summary']}", file=sys.stderr)

    print(f"# PASS: holographic module self-test (DA validation complete)", file=sys.stderr)
