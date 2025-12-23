"""
Gov-OS Core Ledger - Immutable Merkle Ledger with Holographic Boundary

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Volume information encodes on boundary (Bekenstein bound).
Merkle root = boundary of transaction ledger.
Fraud perturbs boundary → O(1) detection without full scan.
"""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import (
    HOLOGRAPHIC_DETECTION_PROB,
    HOLOGRAPHIC_BITS_PER_RECEIPT,
    DISCLAIMER,
    TENANT_ID,
)
from .utils import dual_hash, merkle
from .receipt import emit_L0, emit_L2, StopRule


@dataclass
class MerkleRootHistory:
    """Track historical Merkle roots for outlier detection."""
    roots: List[str] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    def add(self, root: str, timestamp: str = "", batch_size: int = 0):
        self.roots.append(root)
        self.timestamps.append(timestamp or datetime.utcnow().isoformat())
        self.batch_sizes.append(batch_size)

    @property
    def size(self) -> int:
        return len(self.roots)


# Global root history
_ROOT_HISTORY = MerkleRootHistory()


def ingest(receipt: Dict[str, Any], domain: str) -> str:
    """
    Append receipt to ledger with domain tag. Return receipt_id.
    Emits ingest_receipt.

    Args:
        receipt: Receipt dict to ingest
        domain: Domain identifier (e.g., "defense", "medicaid")

    Returns:
        Receipt ID (payload_hash)
    """
    receipt_id = dual_hash(json.dumps(receipt, sort_keys=True))

    emit_L0("ingest_receipt", {
        "receipt_id": receipt_id,
        "domain": domain,
        "payload_hash": receipt_id,
        "tenant_id": receipt.get("tenant_id", TENANT_ID),
        "simulation_flag": DISCLAIMER,
    })

    return receipt_id


def anchor(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute merkle root + holographic boundary. Emit anchor_receipt.

    Args:
        receipts: List of receipts to anchor

    Returns:
        Anchor receipt with merkle_root and holographic_boundary
    """
    global _ROOT_HISTORY

    merkle_root = merkle(receipts)
    boundary = boundary_encode(receipts)

    # Add to history
    _ROOT_HISTORY.add(merkle_root, batch_size=len(receipts))

    anchor_receipt = emit_L2("anchor_receipt", {
        "merkle_root": merkle_root,
        "holographic_boundary": boundary,
        "batch_size": len(receipts),
        "bits_required": bits_required(len(receipts)),
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })

    return anchor_receipt


def holographic_detect(
    current_root: str,
    expected_root: Optional[str] = None,
    root_history: Optional[MerkleRootHistory] = None,
) -> bool:
    """
    O(1) fraud detection via boundary perturbation.
    Return True if anomaly detected.

    Args:
        current_root: Current Merkle root
        expected_root: Optional expected root for comparison
        root_history: Optional history for statistical detection

    Returns:
        True if fraud detected from boundary analysis
    """
    global _ROOT_HISTORY

    if root_history is None:
        root_history = _ROOT_HISTORY

    # Method 1: Direct comparison with expected
    if expected_root is not None:
        perturbation = boundary_perturbation(current_root, expected_root)
        if perturbation > 0:
            return True

    # Method 2: Statistical outlier detection
    if root_history.size >= 3:
        return _detect_from_boundary_stats(current_root, root_history)

    return False


def boundary_encode(receipts: List[Dict[str, Any]]) -> str:
    """
    Encode transaction set as boundary hash.
    Uses holographic principle: volume → boundary compression.

    Args:
        receipts: List of receipts to encode

    Returns:
        Boundary hash string
    """
    if not receipts:
        return dual_hash(b"empty_boundary")

    # Concatenate key fields to create boundary signature
    boundary_data = []
    for r in receipts:
        key_fields = [
            r.get("receipt_type", ""),
            r.get("tenant_id", ""),
            str(r.get("amount_usd", 0))[:6],  # First 6 digits
            r.get("domain", ""),
        ]
        boundary_data.append("|".join(key_fields))

    return dual_hash("||".join(boundary_data))


def boundary_perturbation(boundary: str, baseline: str) -> float:
    """
    Measure deviation from expected boundary.
    Returns 0 if match, >0 if different.

    Args:
        boundary: Current boundary hash
        baseline: Expected baseline hash

    Returns:
        Perturbation measure (0 = match)
    """
    if boundary == baseline:
        return 0.0

    try:
        # XOR the SHA256 portions
        actual_sha = boundary.split(":")[0]
        expected_sha = baseline.split(":")[0]

        actual_int = int(actual_sha, 16)
        expected_int = int(expected_sha, 16)

        xor_result = actual_int ^ expected_int
        # Count difference bits
        difference_bits = bin(xor_result).count('1')

        return difference_bits / 256  # Normalized to [0, 1]
    except (ValueError, IndexError):
        return 1.0  # Maximum perturbation on error


def verify_chain(receipts: List[Dict[str, Any]]) -> bool:
    """
    Verify hash continuity. Stoprule on mismatch.

    Args:
        receipts: List of receipts with hash chain

    Returns:
        True if chain valid

    Raises:
        StopRule: On hash chain violation
    """
    if not receipts:
        return True

    for i in range(1, len(receipts)):
        prev_hash = receipts[i - 1].get("payload_hash")
        referenced_hash = receipts[i].get("prev_hash")

        if referenced_hash and prev_hash and referenced_hash != prev_hash:
            emit_L2("anomaly", {
                "metric": "chain_violation",
                "index": i,
                "expected": prev_hash,
                "actual": referenced_hash,
                "action": "halt",
                "simulation_flag": DISCLAIMER,
            })
            raise StopRule(f"Chain violation at index {i}")

    return True


def compact(before: str, domain: str) -> Dict[str, Any]:
    """
    Compact old receipts with invariants preserved.
    Returns compaction summary receipt.

    Args:
        before: ISO timestamp cutoff
        domain: Domain to compact

    Returns:
        Compaction receipt
    """
    # In a real implementation, this would:
    # 1. Read receipts before cutoff
    # 2. Create summary receipt preserving merkle root
    # 3. Remove detailed receipts
    # 4. Emit compaction receipt

    return emit_L2("compact_receipt", {
        "before": before,
        "domain": domain,
        "status": "simulated",
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })


def bits_required(
    num_receipts: int,
    detection_probability: float = HOLOGRAPHIC_DETECTION_PROB,
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
    area = math.sqrt(num_receipts) if num_receipts > 0 else 0
    log_factor = -math.log2(1 - detection_probability) if detection_probability < 1 else 10

    return area * log_factor


def _detect_from_boundary_stats(
    merkle_root: str,
    root_history: MerkleRootHistory,
    sigma_threshold: float = 3.0,
) -> bool:
    """
    Statistical outlier detection on boundary.
    If root deviates > 3σ from historical distribution, fraud suspected.
    """
    if root_history.size < 3:
        return False

    def root_to_numeric(root: str) -> int:
        try:
            sha = root.split(":")[0][:16]
            return int(sha, 16)
        except (ValueError, IndexError):
            return 0

    historical_values = [root_to_numeric(r) for r in root_history.roots]
    current_value = root_to_numeric(merkle_root)

    mean = np.mean(historical_values)
    std = np.std(historical_values)

    if std == 0:
        return current_value != historical_values[0]

    z_score = abs(current_value - mean) / std
    return z_score > sigma_threshold


def get_root_history() -> MerkleRootHistory:
    """Get the global root history."""
    global _ROOT_HISTORY
    return _ROOT_HISTORY


def reset_root_history() -> None:
    """Reset the root history. Use for testing."""
    global _ROOT_HISTORY
    _ROOT_HISTORY = MerkleRootHistory()
