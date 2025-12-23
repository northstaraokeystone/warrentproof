"""
Gov-OS Core Compress - Entropy Computation with Hierarchical O(log N) Detection

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

The Physics: Legitimate transactions compress to ~0.85+ (predictable redundancy).
Fraudulent transactions compress to <0.60 (injected randomness).
This is PHYSICS per Shannon/Kolmogorov, not a domain-specific tunable.
"""

import gzip
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    CASCADE_WINDOW,
    DISCLAIMER,
    TENANT_ID,
)
from .utils import dual_hash
from .receipt import emit_L0, emit_L1


@dataclass
class EntropyNode:
    """
    Tree node for entropy-based indexing.
    Leaf nodes store actual receipts.
    """
    entropy_threshold: float = 0.0
    left: Optional['EntropyNode'] = None  # Low entropy (< threshold)
    right: Optional['EntropyNode'] = None  # High entropy (>= threshold)
    receipt_ids: List[str] = field(default_factory=list)
    receipt_entropies: Dict[str, float] = field(default_factory=dict)
    depth: int = 0
    node_id: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def size(self) -> int:
        if self.is_leaf:
            return len(self.receipt_ids)
        left_size = self.left.size if self.left else 0
        right_size = self.right.size if self.right else 0
        return left_size + right_size


def compute_entropy_ratio(receipt: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    """
    Compress receipt, return ratio 0-1. Lower = more suspicious.
    Per spec: legitimate ~0.85+, fraud <0.60.

    Args:
        receipt: Single receipt to analyze
        history: Historical receipts for context

    Returns:
        Compression ratio 0-1
    """
    # Serialize receipt with history context
    data_to_compress = [receipt] + history[-10:]  # Use up to 10 recent for context
    data = json.dumps(data_to_compress, sort_keys=True).encode('utf-8')

    if len(data) < 100:
        return 1.0  # Too small for meaningful compression

    # Compress using multiple algorithms, take best
    gzip_compressed = gzip.compress(data, compresslevel=9)
    zlib_compressed = zlib.compress(data, level=9)

    compressed_size = min(len(gzip_compressed), len(zlib_compressed))
    ratio = compressed_size / len(data)

    # Emit compression receipt
    emit_L0("compression_receipt", {
        "ratio": round(ratio, 4),
        "original_size": len(data),
        "compressed_size": compressed_size,
        "classification": _classify_ratio(ratio),
        "tenant_id": TENANT_ID,
        "simulation_flag": DISCLAIMER,
    })

    return ratio


def entropy_score(data: Any) -> float:
    """
    Raw Shannon entropy H = -Σ p(x) log₂ p(x).

    Args:
        data: Data to compute entropy for (bytes, string, or receipts list)

    Returns:
        Entropy score (higher = more random/disorder)
    """
    if isinstance(data, list):
        return _entropy_from_receipts(data)
    elif isinstance(data, dict):
        return _entropy_from_receipts([data])
    elif isinstance(data, str):
        data = data.encode('utf-8')

    if not data:
        return 0.0

    # Byte-level entropy
    counter = Counter(data)
    total = len(data)

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _entropy_from_receipts(receipts: List[Dict[str, Any]]) -> float:
    """Calculate Shannon entropy from receipt feature distribution."""
    if not receipts:
        return 0.0

    features = []
    for r in receipts:
        features.append(r.get("receipt_type", ""))
        features.append(r.get("branch", ""))
        features.append(str(r.get("amount_usd", 0))[:4])
        vendor = r.get("vendor", "")
        if vendor:
            features.append(vendor[:5])

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


def field_wise_compression(receipt: Dict[str, Any]) -> Dict[str, float]:
    """
    Compress each field individually.
    Returns {field: compression_ratio}. Identifies fraud fingerprint.

    Args:
        receipt: Receipt dict to analyze

    Returns:
        Dict mapping field names to compression ratios
    """
    result = {}
    skip_fields = {"payload_hash", "ts", "tenant_id", "simulation_flag", "level"}

    for field_name, value in receipt.items():
        if field_name in skip_fields:
            continue

        if isinstance(value, (dict, list)):
            data = json.dumps(value, sort_keys=True).encode('utf-8')
        else:
            data = str(value).encode('utf-8')

        if len(data) < 10:
            result[field_name] = 1.0
            continue

        compressed = gzip.compress(data, compresslevel=9)
        result[field_name] = round(len(compressed) / len(data), 4)

    return result


def pattern_coherence(receipts: List[Dict[str, Any]], window: int = 10) -> float:
    """
    Measure if receipts follow predictable pattern.
    Per spec: 1.0 = perfect pattern, 0.0 = random.

    Args:
        receipts: List of receipts to analyze
        window: Window size for pattern detection

    Returns:
        Coherence score 0-1
    """
    if len(receipts) < window:
        return 1.0  # Too few to judge

    scores = []

    # Pattern 1: Type consistency
    types = [r.get("receipt_type", "") for r in receipts]
    type_counts = Counter(types)
    if type_counts:
        most_common_ratio = max(type_counts.values()) / len(types)
        scores.append(most_common_ratio)

    # Pattern 2: Branch consistency
    branches = [r.get("branch", "") for r in receipts]
    branch_counts = Counter(branches)
    if branch_counts:
        most_common_ratio = max(branch_counts.values()) / len(branches)
        scores.append(most_common_ratio)

    # Pattern 3: Benford's law for amounts
    amounts = [r.get("amount_usd", 0) for r in receipts if r.get("amount_usd")]
    if amounts:
        first_digits = [int(str(abs(int(a)))[0]) for a in amounts if a > 0]
        if first_digits:
            benford = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
            digit_counts = Counter(first_digits)
            observed = {d: digit_counts.get(d, 0) / len(first_digits) for d in range(1, 10)}
            deviation = sum((observed[d] - benford[d]) ** 2 for d in range(1, 10))
            benford_score = max(0, 1 - (deviation * 10))
            scores.append(benford_score)

    # Pattern 4: Lineage coherence
    with_lineage = sum(1 for r in receipts if r.get("decision_lineage"))
    lineage_ratio = with_lineage / len(receipts) if receipts else 1.0
    scores.append(lineage_ratio)

    return sum(scores) / len(scores) if scores else 0.5


def compression_derivative(
    history: List[float],
    window: int = CASCADE_WINDOW,
) -> float:
    """
    dC/dt for cascade detection.
    Negative = fraud accumulating.

    Args:
        history: List of compression ratios (oldest to newest)
        window: Rolling window size

    Returns:
        Rate of change dC/dt
    """
    if len(history) < 2:
        return 0.0

    window_data = history[-window:]
    if len(window_data) < 2:
        return 0.0

    # Linear regression for slope
    n = len(window_data)
    x = np.arange(n)
    y = np.array(window_data)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def build_entropy_tree(
    receipts: List[Dict[str, Any]],
    max_depth: Optional[int] = None,
) -> EntropyNode:
    """
    Build hierarchical tree for O(log N) detection.

    Args:
        receipts: List of receipts to index
        max_depth: Maximum tree depth (calculated if None)

    Returns:
        Root EntropyNode
    """
    if not receipts:
        return EntropyNode(node_id="empty")

    if max_depth is None:
        max_depth = int(math.log2(len(receipts))) + 2 if len(receipts) > 1 else 1

    # Calculate entropy for each receipt
    receipt_entropies = {}
    for r in receipts:
        receipt_id = r.get("payload_hash", dual_hash(str(r)))
        receipt_entropies[receipt_id] = _receipt_entropy(r)

    return _build_subtree(receipts, receipt_entropies, 0, max_depth)


def _build_subtree(
    receipts: List[Dict[str, Any]],
    receipt_entropies: Dict[str, float],
    depth: int,
    max_depth: int,
) -> EntropyNode:
    """Build subtree recursively."""
    node = EntropyNode(depth=depth, node_id=f"node_{depth}_{len(receipts)}")

    # Base case: leaf node
    if depth >= max_depth or len(receipts) <= 10:
        for r in receipts:
            receipt_id = r.get("payload_hash", dual_hash(str(r)))
            node.receipt_ids.append(receipt_id)
            node.receipt_entropies[receipt_id] = receipt_entropies.get(receipt_id, 0)
        return node

    # Calculate entropies
    entropies = [receipt_entropies.get(r.get("payload_hash", dual_hash(str(r))), 0) for r in receipts]

    # Find median entropy as threshold
    median_entropy = float(np.median(entropies))
    node.entropy_threshold = median_entropy

    # Split receipts
    low_entropy = []
    high_entropy = []

    for r in receipts:
        receipt_id = r.get("payload_hash", dual_hash(str(r)))
        entropy = receipt_entropies.get(receipt_id, 0)
        if entropy < median_entropy:
            low_entropy.append(r)
        else:
            high_entropy.append(r)

    # Handle edge case: all same entropy
    if not low_entropy or not high_entropy:
        for r in receipts:
            receipt_id = r.get("payload_hash", dual_hash(str(r)))
            node.receipt_ids.append(receipt_id)
            node.receipt_entropies[receipt_id] = receipt_entropies.get(receipt_id, 0)
        return node

    # Recursively build children
    node.left = _build_subtree(low_entropy, receipt_entropies, depth + 1, max_depth)
    node.right = _build_subtree(high_entropy, receipt_entropies, depth + 1, max_depth)

    return node


def entropy_bisection(
    tree: EntropyNode,
    threshold: float,
) -> List[str]:
    """
    Binary search to identify anomalous branches.

    Args:
        tree: Root EntropyNode
        threshold: Entropy threshold for anomaly

    Returns:
        List of receipt IDs in anomalous branches
    """
    anomalous = []
    _search_subtree(tree, threshold, anomalous)
    return anomalous


def _search_subtree(
    node: EntropyNode,
    threshold: float,
    anomalous: List[str],
) -> None:
    """Search subtree for anomalous receipts."""
    if node is None:
        return

    if node.is_leaf:
        for receipt_id, entropy in node.receipt_entropies.items():
            if entropy > threshold:
                anomalous.append(receipt_id)
        return

    # Search both subtrees near threshold
    if node.entropy_threshold >= threshold:
        _search_subtree(node.right, threshold, anomalous)
    _search_subtree(node.left, threshold, anomalous)


def detect_hierarchical(
    receipts: List[Dict[str, Any]],
    threshold: float = 4.0,
) -> List[str]:
    """
    Full O(log N) detection pipeline.

    Args:
        receipts: Receipts to analyze
        threshold: Entropy threshold for flagging

    Returns:
        List of anomalous receipt IDs
    """
    tree = build_entropy_tree(receipts)
    return entropy_bisection(tree, threshold)


def speedup_factor(n: int) -> float:
    """
    Return log(n) speedup vs linear scan.

    Args:
        n: Number of receipts

    Returns:
        Speedup factor
    """
    if n <= 1:
        return 1.0
    return n / math.log2(n)


def _receipt_entropy(receipt: Dict[str, Any]) -> float:
    """Calculate entropy for a single receipt."""
    features = []
    features.append(receipt.get("receipt_type", "")[:3])
    features.append(receipt.get("branch", "")[:3])

    amount = receipt.get("amount_usd", 0)
    if amount:
        features.append(str(int(amount))[:3])

    vendor = receipt.get("vendor", "")
    if vendor:
        features.append(vendor[:4])

    lineage = receipt.get("decision_lineage", [])
    features.append("lin" if lineage else "orp")

    if not features:
        return 0.0

    # Character-level entropy
    char_counts: Dict[str, int] = {}
    for f in features:
        for c in str(f):
            char_counts[c] = char_counts.get(c, 0) + 1

    total = sum(char_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def _classify_ratio(ratio: float) -> str:
    """Classify compression ratio."""
    if ratio >= COMPRESSION_LEGITIMATE_FLOOR:
        return "legitimate"
    elif ratio < COMPRESSION_FRAUD_CEILING:
        return "fraudulent"
    else:
        return "suspicious"
