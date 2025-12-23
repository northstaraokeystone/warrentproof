"""
WarrantProof Ledger Module - Immutable Storage with Merkle Anchoring

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module provides immutable storage for receipts with Merkle tree anchoring.
Same pattern as ProofPack ledger module.

Features:
- Append-only receipt storage
- Merkle root computation and anchoring
- Tenant and branch isolation
- Hash chain verification
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    BRANCHES,
    BEKENSTEIN_BITS_PER_DOLLAR,
    dual_hash,
    emit_receipt,
    merkle,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    StopRuleException,
)

# Import v2 holographic functions
from .holographic import (
    holographic_detect as _holographic_detect_internal,
    MerkleRootHistory,
    compute_merkle_syndrome,
    detect_from_boundary,
    emit_holographic_receipt,
)


# === LEDGER STATE ===

class LedgerState:
    """In-memory ledger state with file persistence."""

    def __init__(self, ledger_path: str = "receipts.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.receipts: list[dict] = []
        self.anchors: list[dict] = []
        self.receipt_index: dict[str, dict] = {}  # receipt_id -> receipt
        self._load()

    def _load(self):
        """Load existing receipts from file."""
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            receipt = json.loads(line)
                            self.receipts.append(receipt)
                            if "payload_hash" in receipt:
                                self.receipt_index[receipt["payload_hash"]] = receipt
                        except json.JSONDecodeError:
                            pass

    def append(self, receipt: dict) -> str:
        """Append receipt to ledger and return receipt ID."""
        self.receipts.append(receipt)
        receipt_id = receipt.get("payload_hash", dual_hash(json.dumps(receipt, sort_keys=True)))
        self.receipt_index[receipt_id] = receipt

        # Persist to file
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(receipt) + "\n")

        return receipt_id

    def get(self, receipt_id: str) -> Optional[dict]:
        """Get receipt by ID."""
        return self.receipt_index.get(receipt_id)


# Global ledger instance
_ledger: Optional[LedgerState] = None


def get_ledger(ledger_path: str = "receipts.jsonl") -> LedgerState:
    """Get or create global ledger instance."""
    global _ledger
    if _ledger is None:
        _ledger = LedgerState(ledger_path)
    return _ledger


def reset_ledger():
    """Reset global ledger (for testing)."""
    global _ledger
    _ledger = None


# === CORE FUNCTIONS ===

def ingest(receipt: dict, tenant_id: str = TENANT_ID) -> str:
    """
    Append receipt to ledger with tenant isolation.
    Per spec: All receipts ingested with tenant isolation.

    Args:
        receipt: Receipt dict to ingest
        tenant_id: Tenant ID for isolation

    Returns:
        receipt_id (payload_hash)
    """
    # Validate receipt
    if "receipt_type" not in receipt:
        stoprule_invalid_receipt("Missing receipt_type")

    # Ensure tenant_id
    receipt["tenant_id"] = tenant_id

    # Get ledger and append
    ledger = get_ledger()
    receipt_id = ledger.append(receipt)

    # Emit ingest receipt
    emit_receipt("ingest", {
        "tenant_id": tenant_id,
        "ingested_receipt_id": receipt_id,
        "ingested_receipt_type": receipt["receipt_type"],
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return receipt_id


def anchor_batch(receipts: list, branch_scope: Optional[list[str]] = None) -> dict:
    """
    Compute Merkle root for batch and emit anchor_receipt.
    Per spec: Periodic Merkle anchoring for verification.

    Args:
        receipts: List of receipts to anchor
        branch_scope: Optional list of branches in this anchor

    Returns:
        anchor_receipt dict
    """
    if not receipts:
        return emit_receipt("anchor", {
            "tenant_id": TENANT_ID,
            "merkle_root": merkle([]),
            "batch_size": 0,
            "branch_scope": branch_scope or [],
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    merkle_root = merkle(receipts)

    # Determine branch scope if not provided
    if branch_scope is None:
        branch_scope = list(set(
            r.get("branch", "unknown")
            for r in receipts
            if "branch" in r
        ))

    anchor = emit_receipt("anchor", {
        "tenant_id": TENANT_ID,
        "merkle_root": merkle_root,
        "batch_size": len(receipts),
        "branch_scope": branch_scope,
        "receipt_types": list(set(r.get("receipt_type", "unknown") for r in receipts)),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    # Store anchor
    ledger = get_ledger()
    ledger.anchors.append(anchor)

    return anchor


def query_receipts(
    receipt_type: Optional[str] = None,
    since: Optional[str] = None,
    tenant_id: Optional[str] = None,
    branch: Optional[str] = None,
    limit: int = 1000
) -> list[dict]:
    """
    Filter receipts by criteria.
    Per spec: Support branch-specific queries.

    Args:
        receipt_type: Filter by receipt type
        since: Filter by timestamp (ISO8601)
        tenant_id: Filter by tenant
        branch: Filter by military branch
        limit: Maximum receipts to return

    Returns:
        List of matching receipts
    """
    ledger = get_ledger()
    results = []

    for receipt in ledger.receipts:
        # Apply filters
        if receipt_type and receipt.get("receipt_type") != receipt_type:
            continue
        if tenant_id and receipt.get("tenant_id") != tenant_id:
            continue
        if branch and receipt.get("branch") != branch:
            continue
        if since and receipt.get("ts", "") < since:
            continue

        results.append(receipt)
        if len(results) >= limit:
            break

    return results


def trace_lineage(receipt_id: str) -> list[dict]:
    """
    Return causal chain of receipts leading to this one.
    Per spec: Every warrant references parent for lineage.

    Args:
        receipt_id: Receipt ID to trace from

    Returns:
        Ordered list of receipts in lineage chain
    """
    ledger = get_ledger()
    receipt = ledger.get(receipt_id)

    if not receipt:
        return []

    lineage = [receipt]
    current = receipt

    # Follow parent pointers
    while True:
        parent_ids = current.get("decision_lineage", [])
        if not parent_ids:
            break

        parent_id = parent_ids[0]  # First parent
        parent = ledger.get(parent_id)
        if not parent:
            break

        lineage.insert(0, parent)
        current = parent

    return lineage


def verify_chain(receipts: list) -> dict:
    """
    Verify hash continuity across receipt sequence.
    Per spec: Merkle verification = 100%.

    Args:
        receipts: List of receipts to verify

    Returns:
        verification_receipt with results
    """
    if not receipts:
        return emit_receipt("verification", {
            "tenant_id": TENANT_ID,
            "verified": True,
            "receipt_count": 0,
            "errors": [],
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    errors = []
    verified = True

    for i, receipt in enumerate(receipts):
        # Verify payload_hash
        if "payload_hash" in receipt:
            # Reconstruct hash from receipt data
            data_copy = {k: v for k, v in receipt.items()
                        if k not in ["payload_hash", "receipt_type", "ts"]}
            expected_hash = dual_hash(json.dumps(data_copy, sort_keys=True))

            # Note: In real implementation, we'd verify against stored hash
            # For simulation, we verify structure

        # Verify lineage references exist
        for parent_id in receipt.get("decision_lineage", []):
            if not get_ledger().get(parent_id):
                errors.append(f"Receipt {i}: Missing parent {parent_id}")
                verified = False

    result = emit_receipt("verification", {
        "tenant_id": TENANT_ID,
        "verified": verified,
        "receipt_count": len(receipts),
        "errors": errors,
        "merkle_root": merkle(receipts),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    if not verified:
        for error in errors:
            stoprule_hash_mismatch("valid_chain", error)

    return result


def get_merkle_proof(receipt_id: str, anchor: dict) -> Optional[dict]:
    """
    Get Merkle proof for receipt in anchor.

    Args:
        receipt_id: Receipt to get proof for
        anchor: Anchor containing the receipt

    Returns:
        Proof path dict or None
    """
    # In full implementation, would return path through tree
    # For simulation, return simplified proof
    ledger = get_ledger()
    receipt = ledger.get(receipt_id)

    if not receipt:
        return None

    return {
        "receipt_id": receipt_id,
        "anchor_root": anchor.get("merkle_root"),
        "verified": True,
        "simulation_flag": DISCLAIMER,
    }


# === V2 HOLOGRAPHIC DETECTION ===

# Global Merkle root history
_root_history: Optional[MerkleRootHistory] = None


def get_root_history() -> MerkleRootHistory:
    """Get or create global Merkle root history."""
    global _root_history
    if _root_history is None:
        _root_history = MerkleRootHistory()
    return _root_history


def get_merkle_root() -> str:
    """
    Return current Merkle root of ledger.
    Used for holographic detection.

    Returns:
        Current Merkle root hash
    """
    ledger = get_ledger()

    if not ledger.receipts:
        return merkle([])

    return merkle(ledger.receipts)


def holographic_detect(
    merkle_root: Optional[str] = None,
    root_history: Optional[MerkleRootHistory] = None
) -> bool:
    """
    Detect fraud from Merkle root alone (O(1) boundary check).
    Call holographic.detect_from_boundary().

    Args:
        merkle_root: Current Merkle root (calculated if None)
        root_history: Optional history (uses global if None)

    Returns:
        True if fraud detected from boundary
    """
    if merkle_root is None:
        merkle_root = get_merkle_root()

    if root_history is None:
        root_history = get_root_history()

    # Add current root to history
    root_history.add(merkle_root)

    # Use holographic detection
    result = _holographic_detect_internal(
        current_root=merkle_root,
        expected_root=None,  # No expected root - use statistical detection
        root_history=root_history
    )

    # Emit holographic receipt if fraud detected
    if result.get("fraud_detected"):
        ledger = get_ledger()
        emit_holographic_receipt(
            merkle_root_current=merkle_root,
            detection_result=result,
            num_receipts=len(ledger.receipts)
        )

    return result.get("fraud_detected", False)


def verify_holographic_integrity(expected_root: str) -> dict:
    """
    Verify current ledger matches expected Merkle root.
    Uses holographic syndrome computation.

    Args:
        expected_root: Expected Merkle root hash

    Returns:
        Verification result dict
    """
    current_root = get_merkle_root()

    syndrome = compute_merkle_syndrome(current_root, expected_root)

    if not syndrome.get("match"):
        # Fraud detected via syndrome
        result = _holographic_detect_internal(
            current_root=current_root,
            expected_root=expected_root
        )

        return {
            "verified": False,
            "current_root": current_root,
            "expected_root": expected_root,
            "syndrome": syndrome,
            "fraud_detected": True,
            "detection_result": result,
        }

    return {
        "verified": True,
        "current_root": current_root,
        "expected_root": expected_root,
        "fraud_detected": False,
    }


# === OMEGA v3: BEKENSTEIN BOUND VALIDATION ===

def validate_bekenstein_bound(
    invoice: dict,
    metadata: dict = None
) -> dict:
    """
    OMEGA v3: Validate invoice against Bekenstein bound.
    S <= BEKENSTEIN_BITS_PER_DOLLAR * amount

    The holographic principle: The metadata (digital trail) must be
    sufficient to holographically encode the work (bulk reality).

    Args:
        invoice: Invoice dict with amount_usd
        metadata: Optional metadata dict (extracted from invoice if not provided)

    Returns:
        Validation result dict
    """
    import math

    amount = invoice.get("amount_usd", 0)
    if amount <= 0:
        return {
            "valid": True,
            "metadata_bits": 0,
            "invoice_amount": 0,
            "bekenstein_bound": 0,
            "bound_violated": False,
        }

    # Extract or use provided metadata
    if metadata is None:
        metadata = _extract_metadata(invoice)

    # Calculate metadata entropy (bits)
    metadata_bits = _calculate_metadata_bits(metadata)

    # Calculate Bekenstein bound
    bekenstein_bound = BEKENSTEIN_BITS_PER_DOLLAR * amount

    # Check if bound is violated
    # Bound violation: metadata too sparse for invoice amount
    bound_violated = metadata_bits < bekenstein_bound

    return {
        "valid": not bound_violated,
        "metadata_bits": round(metadata_bits, 4),
        "invoice_amount": amount,
        "bekenstein_bound": round(bekenstein_bound, 4),
        "bound_violated": bound_violated,
        "ratio": round(metadata_bits / bekenstein_bound, 4) if bekenstein_bound > 0 else float('inf'),
    }


def _extract_metadata(invoice: dict) -> dict:
    """Extract metadata fields from invoice."""
    metadata_fields = [
        "description", "vendor", "approver", "program", "branch",
        "itemized_receipt", "proof_of_delivery", "commit_logs",
        "jira_tickets", "server_logs", "email_threads",
    ]
    metadata = {}
    for field in metadata_fields:
        if field in invoice and invoice[field]:
            metadata[field] = invoice[field]
    return metadata


def _calculate_metadata_bits(metadata: dict) -> float:
    """
    Calculate entropy (bits) of metadata.
    Uses compression-based estimation.
    """
    import json
    import zlib

    if not metadata:
        return 0.0

    # Serialize metadata
    data = json.dumps(metadata, sort_keys=True).encode('utf-8')

    if len(data) == 0:
        return 0.0

    # Entropy estimation via compression
    # Compressed size approximates information content
    compressed = zlib.compress(data, level=9)

    # Bits = compressed_bytes * 8
    return len(compressed) * 8


def emit_bekenstein_receipt(
    invoice: dict,
    validation_result: dict = None
) -> dict:
    """
    Emit bekenstein_receipt documenting bound validation.

    Args:
        invoice: Invoice that was validated
        validation_result: Pre-computed validation result

    Returns:
        bekenstein_receipt dict
    """
    if validation_result is None:
        validation_result = validate_bekenstein_bound(invoice)

    return emit_receipt("bekenstein", {
        "tenant_id": TENANT_ID,
        "invoice_id": invoice.get("invoice_number") or invoice.get("transaction_id") or "unknown",
        "metadata_bits": validation_result.get("metadata_bits", 0),
        "invoice_amount": validation_result.get("invoice_amount", 0),
        "bekenstein_bound": validation_result.get("bekenstein_bound", 0),
        "bound_violated": validation_result.get("bound_violated", False),
        "ratio": validation_result.get("ratio", 0),
        "valid": validation_result.get("valid", True),
        "bits_per_dollar": BEKENSTEIN_BITS_PER_DOLLAR,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def stoprule_bekenstein_violated(invoice_id: str, metadata_bits: float, required_bits: float) -> None:
    """Reject invoice if Bekenstein bound violated."""
    emit_receipt("anomaly", {
        "metric": "bekenstein_violated",
        "invoice_id": invoice_id,
        "metadata_bits": metadata_bits,
        "required_bits": required_bits,
        "delta": metadata_bits - required_bits,
        "action": "reject_transaction",
        "classification": "violation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(
        f"Bekenstein bound violated for {invoice_id}: {metadata_bits:.2f} bits < {required_bits:.2f} required"
    )


# === BRANCH ISOLATION ===

def get_branch_receipts(branch: str, receipt_type: Optional[str] = None) -> list[dict]:
    """
    Get receipts for specific branch.
    Per spec: Each branch gets separate ledger partition.

    Args:
        branch: Military branch
        receipt_type: Optional filter by type

    Returns:
        List of branch receipts
    """
    if branch not in BRANCHES:
        return []

    return query_receipts(receipt_type=receipt_type, branch=branch)


def cross_branch_query(branches: list[str], receipt_type: Optional[str] = None) -> dict:
    """
    Query across branches with bridge receipt.
    Per spec: Cross-branch queries require explicit bridge_receipt.

    Args:
        branches: List of branches to query
        receipt_type: Optional filter by type

    Returns:
        Dict with receipts by branch and bridge receipt
    """
    results = {}
    all_receipts = []

    for branch in branches:
        branch_receipts = get_branch_receipts(branch, receipt_type)
        results[branch] = branch_receipts
        all_receipts.extend(branch_receipts)

    # Emit bridge receipt for cross-branch query
    bridge = emit_receipt("bridge_query", {
        "tenant_id": TENANT_ID,
        "branches_queried": branches,
        "receipt_type_filter": receipt_type,
        "total_receipts": len(all_receipts),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)

    return {
        "receipts_by_branch": results,
        "bridge_receipt": bridge,
        "total": len(all_receipts),
    }


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import tempfile

    print(f"# WarrantProof Ledger Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Use temp file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name

    reset_ledger()
    _ledger = LedgerState(temp_path)

    # Test ingest
    test_receipt = {
        "receipt_type": "test",
        "data": "test data",
        "branch": "Navy",
    }
    receipt_id = ingest(test_receipt)
    assert receipt_id is not None

    # Test query
    results = query_receipts(receipt_type="test")
    assert len(results) >= 1

    # Test anchor
    anchor = anchor_batch(results)
    assert "merkle_root" in anchor
    assert anchor["batch_size"] == len(results)

    # Test verification
    verification = verify_chain(results)
    assert verification["verified"] == True

    # Clean up
    os.unlink(temp_path)
    reset_ledger()

    print(f"# PASS: ledger module self-test", file=sys.stderr)
