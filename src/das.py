"""
WarrantProof DAS Module - Data Availability Sampling

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements Data Availability Sampling (DAS) to detect "erasure fraud"
where data is published (invoice header) but the body is withheld (itemized receipt).

Key Insight:
Data Availability Attack: Fraudster publishes header but withholds body.
Fraud happens in the redacted space (404, "Proprietary", "Classified").

Solution: Ethereum-style erasure coding with random chunk sampling.
Auditors sample 10% of chunks. If chunks missing, reject transaction.

OMEGA Citation:
"The fraudster publishes the header (invoice summary) but withholds the
block body (itemized receipt). The fraud happens in the redacted space."
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .core import (
    TENANT_ID,
    DISCLAIMER,
    DATA_AVAILABILITY_SAMPLE_RATE,
    DATA_AVAILABILITY_THRESHOLD,
    dual_hash,
    emit_receipt,
    merkle,
    StopRuleException,
)


@dataclass
class ErasureCodedData:
    """Data encoded with Reed-Solomon erasure coding."""
    original_size: int
    chunk_count: int
    chunks: List[bytes] = field(default_factory=list)
    chunk_hashes: List[str] = field(default_factory=list)
    merkle_root: str = ""
    redundancy_factor: float = 2.0


def encode_with_erasure(
    data: bytes,
    redundancy: float = 2.0,
    chunk_size: int = 256
) -> ErasureCodedData:
    """
    Reed-Solomon style erasure coding. Add redundancy for recovery.

    Note: This is a simulation. Real erasure coding uses GF(2^8) math.

    Args:
        data: Original data bytes
        redundancy: Redundancy factor (default 2x)
        chunk_size: Size of each chunk

    Returns:
        ErasureCodedData with chunks and hashes
    """
    if not data:
        return ErasureCodedData(original_size=0, chunk_count=0)

    # Split into chunks
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        chunks.append(chunk)

    # Add redundancy chunks (simulated parity)
    original_count = len(chunks)
    redundancy_count = int(original_count * (redundancy - 1))

    for i in range(redundancy_count):
        # Simulated parity: XOR of chunks
        parity = bytes(chunk_size)
        for j in range(len(chunks)):
            parity = bytes(a ^ b for a, b in zip(parity, chunks[j]))
        chunks.append(parity)

    # Hash each chunk
    chunk_hashes = [dual_hash(chunk) for chunk in chunks]

    # Compute Merkle root of chunks
    root = merkle(chunk_hashes)

    return ErasureCodedData(
        original_size=len(data),
        chunk_count=len(chunks),
        chunks=chunks,
        chunk_hashes=chunk_hashes,
        merkle_root=root,
        redundancy_factor=redundancy,
    )


def sample_chunks(
    encoded_data: ErasureCodedData,
    sample_rate: float = DATA_AVAILABILITY_SAMPLE_RATE
) -> Tuple[List[bytes], List[int]]:
    """
    Randomly sample chunks at sample_rate.

    Args:
        encoded_data: ErasureCodedData to sample
        sample_rate: Fraction to sample (default 0.10)

    Returns:
        (sampled_chunks, chunk_indices)
    """
    if not encoded_data.chunks:
        return [], []

    # Calculate sample size
    sample_count = max(1, int(encoded_data.chunk_count * sample_rate))

    # Random sample without replacement
    indices = random.sample(range(encoded_data.chunk_count), sample_count)
    indices.sort()

    sampled_chunks = [encoded_data.chunks[i] for i in indices]

    return sampled_chunks, indices


def verify_availability(
    sampled_chunks: List[bytes],
    chunk_indices: List[int],
    encoded_data: ErasureCodedData
) -> bool:
    """
    Verify sampled chunks match expected hashes.

    Args:
        sampled_chunks: Chunks that were retrieved
        chunk_indices: Indices of sampled chunks
        encoded_data: Original encoded data

    Returns:
        True if availability verified
    """
    if not sampled_chunks or not chunk_indices:
        return False

    for chunk, idx in zip(sampled_chunks, chunk_indices):
        expected_hash = encoded_data.chunk_hashes[idx]
        actual_hash = dual_hash(chunk)

        if expected_hash != actual_hash:
            return False

    return True


def detect_erasure(
    transaction: dict,
    required_fields: List[str] = None
) -> dict:
    """
    Check if required fields present. Missing critical fields = erasure attack.

    Args:
        transaction: Transaction dict to check
        required_fields: Fields that must be present

    Returns:
        Detection result dict
    """
    if required_fields is None:
        required_fields = [
            "amount_usd", "vendor", "description", "approver",
            "itemized_receipt", "proof_of_delivery", "invoice_number"
        ]

    missing = []
    redacted = []

    for field in required_fields:
        value = transaction.get(field)

        if value is None:
            missing.append(field)
        elif isinstance(value, str):
            # Check for redaction indicators
            lower = value.lower()
            if any(indicator in lower for indicator in [
                "redacted", "proprietary", "classified", "withheld",
                "n/a", "not available", "404", "missing"
            ]):
                redacted.append(field)

    erasure_detected = len(missing) > 0 or len(redacted) > 0

    return {
        "erasure_detected": erasure_detected,
        "missing_fields": missing,
        "redacted_fields": redacted,
        "fields_checked": len(required_fields),
        "availability_score": 1.0 - (len(missing) + len(redacted)) / len(required_fields),
    }


def light_client_audit(
    transactions: List[dict],
    auditor_id: str,
    sample_rate: float = DATA_AVAILABILITY_SAMPLE_RATE
) -> dict:
    """
    Auditor acts as "light client." Randomly samples chunks from transactions.
    Reports availability score.

    Args:
        transactions: List of transactions to audit
        auditor_id: Unique auditor identifier
        sample_rate: Fraction of data to sample

    Returns:
        Audit result dict
    """
    if not transactions:
        return {
            "auditor_id": auditor_id,
            "transactions_audited": 0,
            "availability_score": 1.0,
            "erasure_detected": False,
        }

    # Sample transactions
    sample_count = max(1, int(len(transactions) * sample_rate))
    sampled_transactions = random.sample(transactions, sample_count)

    # Check each sampled transaction
    erasure_results = []
    for tx in sampled_transactions:
        result = detect_erasure(tx)
        erasure_results.append(result)

    # Calculate overall availability
    scores = [r["availability_score"] for r in erasure_results]
    avg_score = sum(scores) / len(scores) if scores else 1.0

    erasure_detected = avg_score < DATA_AVAILABILITY_THRESHOLD

    return {
        "auditor_id": auditor_id,
        "transactions_audited": len(sampled_transactions),
        "transactions_total": len(transactions),
        "sample_rate": sample_rate,
        "availability_score": round(avg_score, 4),
        "erasure_detected": erasure_detected,
        "threshold": DATA_AVAILABILITY_THRESHOLD,
        "missing_field_count": sum(len(r["missing_fields"]) for r in erasure_results),
        "redacted_field_count": sum(len(r["redacted_fields"]) for r in erasure_results),
    }


def emit_das_receipt(
    transaction_id: str,
    audit_result: dict = None,
    erasure_result: dict = None
) -> dict:
    """
    Emit das_receipt documenting availability sampling.

    Args:
        transaction_id: Transaction being checked
        audit_result: From light_client_audit
        erasure_result: From detect_erasure

    Returns:
        das_receipt dict
    """
    if audit_result:
        return emit_receipt("das", {
            "tenant_id": TENANT_ID,
            "transaction_id": transaction_id,
            "auditor_id": audit_result.get("auditor_id", "unknown"),
            "chunks_sampled": audit_result.get("transactions_audited", 0),
            "chunks_available": int(audit_result.get("availability_score", 1.0) * audit_result.get("transactions_audited", 0)),
            "chunks_missing": audit_result.get("missing_field_count", 0),
            "erasure_detected": audit_result.get("erasure_detected", False),
            "availability_score": audit_result.get("availability_score", 1.0),
            "sample_rate": DATA_AVAILABILITY_SAMPLE_RATE,
            "threshold": DATA_AVAILABILITY_THRESHOLD,
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)
    elif erasure_result:
        return emit_receipt("das", {
            "tenant_id": TENANT_ID,
            "transaction_id": transaction_id,
            "auditor_id": "single_tx_check",
            "chunks_sampled": erasure_result.get("fields_checked", 0),
            "chunks_available": int(erasure_result.get("availability_score", 1.0) * erasure_result.get("fields_checked", 1)),
            "chunks_missing": len(erasure_result.get("missing_fields", [])),
            "erasure_detected": erasure_result.get("erasure_detected", False),
            "availability_score": erasure_result.get("availability_score", 1.0),
            "missing_fields": erasure_result.get("missing_fields", []),
            "redacted_fields": erasure_result.get("redacted_fields", []),
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)
    else:
        return emit_receipt("das", {
            "tenant_id": TENANT_ID,
            "transaction_id": transaction_id,
            "auditor_id": "unknown",
            "erasure_detected": False,
            "availability_score": 1.0,
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)


# === STOPRULES ===

def stoprule_availability_low(score: float) -> None:
    """If availability < 0.90, reject transaction."""
    if score < DATA_AVAILABILITY_THRESHOLD:
        emit_receipt("anomaly", {
            "metric": "availability_low",
            "score": score,
            "threshold": DATA_AVAILABILITY_THRESHOLD,
            "delta": score - DATA_AVAILABILITY_THRESHOLD,
            "action": "reject_transaction",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Data availability {score:.2f} below threshold {DATA_AVAILABILITY_THRESHOLD}"
        )


def stoprule_erasure_detected(missing_fields: List[str]) -> None:
    """If critical fields missing, flag as fraud."""
    critical_fields = {"amount_usd", "vendor", "approver"}
    critical_missing = set(missing_fields) & critical_fields

    if critical_missing:
        emit_receipt("anomaly", {
            "metric": "erasure_detected",
            "missing_fields": list(critical_missing),
            "action": "flag_fraud",
            "classification": "violation",
            "simulation_flag": DISCLAIMER,
        })
        raise StopRuleException(
            f"Critical fields missing (erasure attack): {critical_missing}"
        )


def stoprule_sample_size_insufficient(sample_rate: float) -> None:
    """If sample rate < 0.10, cannot guarantee detection."""
    if sample_rate < DATA_AVAILABILITY_SAMPLE_RATE:
        emit_receipt("anomaly", {
            "metric": "sample_size_insufficient",
            "sample_rate": sample_rate,
            "minimum": DATA_AVAILABILITY_SAMPLE_RATE,
            "action": "increase_sampling",
            "classification": "deviation",
            "simulation_flag": DISCLAIMER,
        })


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof DAS Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Erasure coding
    test_data = b"Test transaction data for erasure coding validation" * 100
    encoded = encode_with_erasure(test_data, redundancy=2.0)
    print(f"# Original size: {encoded.original_size}, Chunks: {encoded.chunk_count}", file=sys.stderr)
    assert encoded.chunk_count > 0
    assert len(encoded.chunks) > 0

    # Test 2: Sample chunks
    sampled, indices = sample_chunks(encoded, sample_rate=0.10)
    print(f"# Sampled {len(sampled)} chunks at indices {indices}", file=sys.stderr)
    assert len(sampled) >= 1

    # Test 3: Verify availability
    is_available = verify_availability(sampled, indices, encoded)
    print(f"# Availability verified: {is_available}", file=sys.stderr)
    assert is_available == True

    # Test 4: Detect erasure on complete transaction
    complete_tx = {
        "amount_usd": 1000000,
        "vendor": "Test Corp",
        "description": "Software development",
        "approver": "John Smith",
        "itemized_receipt": "Line items...",
        "proof_of_delivery": "Signed receipt",
        "invoice_number": "INV-001",
    }
    erasure = detect_erasure(complete_tx)
    print(f"# Complete transaction - erasure: {erasure['erasure_detected']}", file=sys.stderr)
    assert erasure["erasure_detected"] == False
    assert erasure["availability_score"] == 1.0

    # Test 5: Detect erasure on incomplete transaction
    incomplete_tx = {
        "amount_usd": 1000000,
        "vendor": "Test Corp",
        "description": "REDACTED - Proprietary",
        # Missing: approver, itemized_receipt, proof_of_delivery, invoice_number
    }
    erasure_incomplete = detect_erasure(incomplete_tx)
    print(f"# Incomplete transaction - erasure: {erasure_incomplete['erasure_detected']}", file=sys.stderr)
    print(f"# Missing: {erasure_incomplete['missing_fields']}", file=sys.stderr)
    print(f"# Redacted: {erasure_incomplete['redacted_fields']}", file=sys.stderr)
    assert erasure_incomplete["erasure_detected"] == True

    # Test 6: Light client audit
    transactions = [complete_tx] * 8 + [incomplete_tx] * 2
    audit = light_client_audit(transactions, "auditor_001", sample_rate=0.30)
    print(f"# Audit score: {audit['availability_score']:.2f}", file=sys.stderr)
    assert 0 <= audit["availability_score"] <= 1.0

    # Test 7: Receipt emission
    receipt = emit_das_receipt("TX_001", erasure_result=erasure_incomplete)
    assert receipt["receipt_type"] == "das"
    assert receipt["erasure_detected"] == True

    print(f"# PASS: das module self-test", file=sys.stderr)
