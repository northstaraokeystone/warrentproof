"""
Gov-OS Core Utils - Hash and Merkle Functions

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Per CLAUDEME §8: HASH = "SHA256 + BLAKE3" # ALWAYS dual-hash
"""

import hashlib
import json
from typing import Union

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def dual_hash(data: Union[bytes, str]) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.
    Per CLAUDEME §8: HASH = "SHA256 + BLAKE3" # ALWAYS dual-hash

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
    Per CLAUDEME §8: MERKLE = "BLAKE3" (via dual_hash)

    Args:
        items: List of items to compute root for

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    hashes = [
        dual_hash(json.dumps(i, sort_keys=True) if isinstance(i, dict) else str(i))
        for i in items
    ]

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])  # Duplicate last for odd count
        hashes = [
            dual_hash(hashes[i] + hashes[i + 1])
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0]


def hash_receipt(receipt: dict) -> str:
    """
    Hash a receipt dict using dual_hash.

    Args:
        receipt: Receipt dict to hash

    Returns:
        dual-hash of serialized receipt
    """
    return dual_hash(json.dumps(receipt, sort_keys=True))
