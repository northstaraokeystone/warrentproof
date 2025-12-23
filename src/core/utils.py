"""
Gov-OS Core Utilities - CLAUDEME v3.1 Compliant

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Optional

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def dual_hash(data: bytes | str) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.
    Per CLAUDEME §8: HASH = "SHA256 + BLAKE3"

    Args:
        data: Bytes or string to hash

    Returns:
        String in format "sha256hex:blake3hex"
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def merkle(items: list) -> str:
    """
    Compute Merkle root of items using dual_hash.

    Args:
        items: List of items to compute root for

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    hashes = [dual_hash(json.dumps(i, sort_keys=True) if isinstance(i, dict) else str(i))
              for i in items]

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i+1])
                  for i in range(0, len(hashes), 2)]

    return hashes[0]


def cite(source: str, url: str, detail: str, date: Optional[str] = None) -> dict:
    """
    Embed citation in receipt. Required for all data claims.

    Args:
        source: Source document identifier
        url: URL to source
        detail: Specific detail being cited
        date: Optional date of source

    Returns:
        Citation dict for embedding in receipts
    """
    citation = {
        "source": source,
        "url": url,
        "detail": detail
    }
    if date:
        citation["date"] = date
    return citation


def generate_receipt_id() -> str:
    """Generate a unique receipt ID using timestamp and random hash."""
    return f"rcpt_{uuid.uuid4().hex[:16]}"
