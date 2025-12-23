"""
WarrantProof Layout Entropy Module - PDF Visual Structure Analysis

SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY

This module implements layout entropy analysis to distinguish human-generated
PDF invoices from script-generated fraudulent documents.

Key Insight:
- Script-generated PDFs have perfect alignment, uniform fonts, pixel-perfect grids
- Human scans have warping, compression artifacts, slight misalignments
- Layout entropy measures this: low entropy = suspicious, high entropy = legitimate

OMEGA Citation:
"Layout Entropy of PDFs distinguishes human scans from script-generated invoices"
"""

import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    LAYOUT_ENTROPY_THRESHOLD,
    LAYOUT_HUMAN_SCAN_MIN,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


@dataclass
class BoundingBox:
    """Bounding box for a PDF element."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class LayoutFeatures:
    """Extracted layout features from a PDF."""
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    fonts: List[str] = field(default_factory=list)
    font_sizes: List[float] = field(default_factory=list)
    alignments: List[str] = field(default_factory=list)  # left, center, right
    text_blocks: int = 0
    image_count: int = 0
    page_count: int = 1


@dataclass
class ScanArtifacts:
    """Detected scan artifacts in a PDF."""
    jpeg_artifacts: float = 0.0  # 0-1 score
    scan_lines: bool = False
    page_warping: float = 0.0  # Degrees of rotation/skew
    noise_level: float = 0.0  # 0-1 score


def extract_layout_features(pdf_bytes: bytes) -> LayoutFeatures:
    """
    Extract layout features from PDF.
    Note: This is a simulation. Real implementation uses LayoutLM or pdfminer.

    Args:
        pdf_bytes: Raw PDF bytes

    Returns:
        LayoutFeatures extracted from the document
    """
    # Simulate feature extraction based on PDF hash
    # In production: Use pdfminer, pypdf2, or LayoutLM
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Deterministic simulation based on hash
    seed = int(pdf_hash[:8], 16)
    import random
    rng = random.Random(seed)

    # Generate simulated bounding boxes
    bboxes = []
    num_elements = rng.randint(10, 50)
    for _ in range(num_elements):
        bboxes.append(BoundingBox(
            x=rng.uniform(0, 500),
            y=rng.uniform(0, 700),
            width=rng.uniform(50, 200),
            height=rng.uniform(10, 30),
        ))

    # Generate simulated fonts
    font_pool = ["Arial", "Times New Roman", "Helvetica", "Courier", "Calibri"]
    fonts = [rng.choice(font_pool) for _ in range(num_elements)]

    # Generate simulated font sizes
    font_sizes = [rng.choice([10, 11, 12, 14, 16, 18, 24]) for _ in range(num_elements)]

    # Generate alignments
    alignment_pool = ["left", "center", "right"]
    alignments = [rng.choice(alignment_pool) for _ in range(num_elements)]

    return LayoutFeatures(
        bounding_boxes=bboxes,
        fonts=fonts,
        font_sizes=font_sizes,
        alignments=alignments,
        text_blocks=num_elements,
        image_count=rng.randint(0, 5),
        page_count=rng.randint(1, 3),
    )


def calculate_layout_entropy(layout_features: LayoutFeatures) -> float:
    """
    Compute entropy of layout structure.
    High entropy = human-generated (varied).
    Low entropy = script-generated (uniform).

    Args:
        layout_features: Extracted layout features

    Returns:
        Layout entropy in bits
    """
    if not layout_features.bounding_boxes:
        return 0.0

    entropies = []

    # 1. Bounding box position entropy
    if layout_features.bounding_boxes:
        # Discretize positions into grid cells
        x_bins = [int(bb.x / 50) for bb in layout_features.bounding_boxes]
        y_bins = [int(bb.y / 50) for bb in layout_features.bounding_boxes]

        x_entropy = _shannon_entropy(x_bins)
        y_entropy = _shannon_entropy(y_bins)
        entropies.extend([x_entropy, y_entropy])

    # 2. Bounding box size entropy
    if layout_features.bounding_boxes:
        widths = [int(bb.width / 10) for bb in layout_features.bounding_boxes]
        heights = [int(bb.height / 5) for bb in layout_features.bounding_boxes]

        width_entropy = _shannon_entropy(widths)
        height_entropy = _shannon_entropy(heights)
        entropies.extend([width_entropy, height_entropy])

    # 3. Font variety entropy
    if layout_features.fonts:
        font_entropy = _shannon_entropy(layout_features.fonts)
        entropies.append(font_entropy)

    # 4. Font size entropy
    if layout_features.font_sizes:
        size_entropy = _shannon_entropy(layout_features.font_sizes)
        entropies.append(size_entropy)

    # 5. Alignment entropy
    if layout_features.alignments:
        align_entropy = _shannon_entropy(layout_features.alignments)
        entropies.append(align_entropy)

    # Average entropy across all dimensions
    if not entropies:
        return 0.0

    return sum(entropies) / len(entropies)


def _shannon_entropy(values: List) -> float:
    """Calculate Shannon entropy of a list of values."""
    if not values:
        return 0.0

    counter = Counter(values)
    total = len(values)

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def detect_scan_artifacts(pdf_bytes: bytes) -> ScanArtifacts:
    """
    Detect scan artifacts: compression, warping, noise.
    Human scans have artifacts. Script PDFs are clean.

    Args:
        pdf_bytes: Raw PDF bytes

    Returns:
        ScanArtifacts detection results
    """
    # Simulate artifact detection
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    seed = int(pdf_hash[:8], 16)
    import random
    rng = random.Random(seed)

    # In production: Use image processing (PIL, OpenCV)
    return ScanArtifacts(
        jpeg_artifacts=rng.uniform(0, 0.5),
        scan_lines=rng.random() > 0.7,
        page_warping=rng.uniform(0, 2.0),
        noise_level=rng.uniform(0, 0.3),
    )


def detect_perfect_alignment(layout_features: LayoutFeatures) -> bool:
    """
    Check if all elements aligned to pixel-perfect grid.
    Perfect alignment = suspicious (script-generated).

    Args:
        layout_features: Extracted layout features

    Returns:
        True if suspiciously perfect alignment detected
    """
    if len(layout_features.bounding_boxes) < 3:
        return False

    # Check if x positions are all multiples of a common grid
    x_positions = [bb.x for bb in layout_features.bounding_boxes]
    y_positions = [bb.y for bb in layout_features.bounding_boxes]

    # Check for grid alignment
    def check_grid_alignment(positions: List[float], grid_size: int = 10) -> bool:
        """Check if positions align to grid."""
        remainders = [pos % grid_size for pos in positions]
        # If >90% align to grid, suspicious
        aligned = sum(1 for r in remainders if r < 0.5 or r > grid_size - 0.5)
        return aligned / len(positions) > 0.9

    x_aligned = check_grid_alignment(x_positions)
    y_aligned = check_grid_alignment(y_positions)

    return x_aligned and y_aligned


def calculate_human_warping_score(
    layout_features: LayoutFeatures,
    scan_artifacts: ScanArtifacts
) -> float:
    """
    Combined score indicating human-generated document.
    Higher = more likely human-generated.

    Args:
        layout_features: Layout features
        scan_artifacts: Scan artifacts

    Returns:
        Human warping score 0-1
    """
    scores = []

    # Entropy contribution
    entropy = calculate_layout_entropy(layout_features)
    entropy_score = min(1.0, entropy / 4.0)  # Normalize to 0-1
    scores.append(entropy_score)

    # Artifact contributions
    scores.append(scan_artifacts.jpeg_artifacts)
    scores.append(1.0 if scan_artifacts.scan_lines else 0.0)
    scores.append(min(1.0, scan_artifacts.page_warping / 2.0))
    scores.append(scan_artifacts.noise_level)

    # Perfect alignment penalty
    if detect_perfect_alignment(layout_features):
        scores.append(0.0)  # Suspicious
    else:
        scores.append(1.0)  # Normal variation

    return sum(scores) / len(scores)


def emit_layout_receipt(
    document_id: str,
    pdf_bytes: Optional[bytes] = None,
    layout_features: Optional[LayoutFeatures] = None,
    scan_artifacts: Optional[ScanArtifacts] = None
) -> dict:
    """
    Emit layout_receipt documenting PDF analysis.

    Args:
        document_id: Document identifier
        pdf_bytes: Optional PDF bytes to analyze
        layout_features: Pre-extracted layout features
        scan_artifacts: Pre-detected scan artifacts

    Returns:
        layout_receipt dict
    """
    if pdf_bytes and not layout_features:
        layout_features = extract_layout_features(pdf_bytes)
    if pdf_bytes and not scan_artifacts:
        scan_artifacts = detect_scan_artifacts(pdf_bytes)

    if not layout_features:
        layout_features = LayoutFeatures()
    if not scan_artifacts:
        scan_artifacts = ScanArtifacts()

    layout_entropy = calculate_layout_entropy(layout_features)
    perfect_alignment = detect_perfect_alignment(layout_features)
    human_score = calculate_human_warping_score(layout_features, scan_artifacts)

    # Fraud indicator: low entropy + perfect alignment
    fraud_indicator = layout_entropy < LAYOUT_ENTROPY_THRESHOLD and perfect_alignment

    return emit_receipt("layout", {
        "tenant_id": TENANT_ID,
        "document_id": document_id,
        "layout_entropy": round(layout_entropy, 4),
        "scan_artifacts": round(scan_artifacts.jpeg_artifacts, 4),
        "perfect_alignment_flag": perfect_alignment,
        "human_warping_score": round(human_score, 4),
        "fraud_indicator": fraud_indicator,
        "text_blocks": layout_features.text_blocks,
        "image_count": layout_features.image_count,
        "entropy_threshold": LAYOUT_ENTROPY_THRESHOLD,
        "human_scan_min": LAYOUT_HUMAN_SCAN_MIN,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === STOPRULES ===

def stoprule_ocr_failed(reason: str) -> None:
    """If PDF extraction fails, cannot analyze."""
    emit_receipt("anomaly", {
        "metric": "ocr_failed",
        "reason": reason,
        "action": "manual_review",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })
    raise StopRuleException(f"OCR/PDF extraction failed: {reason}")


def stoprule_entropy_calculation_invalid(reason: str) -> None:
    """If layout features insufficient for entropy calculation."""
    emit_receipt("anomaly", {
        "metric": "entropy_calculation_invalid",
        "reason": reason,
        "action": "request_better_scan",
        "classification": "deviation",
        "simulation_flag": DISCLAIMER,
    })


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Layout Entropy Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test 1: Extract layout features from simulated PDF
    pdf_bytes = b"Simulated PDF content for testing layout entropy analysis"
    features = extract_layout_features(pdf_bytes)
    print(f"# Text blocks: {features.text_blocks}, Images: {features.image_count}", file=sys.stderr)
    assert features.text_blocks > 0

    # Test 2: Calculate layout entropy
    entropy = calculate_layout_entropy(features)
    print(f"# Layout entropy: {entropy:.4f} bits", file=sys.stderr)
    assert entropy >= 0

    # Test 3: Detect scan artifacts
    artifacts = detect_scan_artifacts(pdf_bytes)
    print(f"# JPEG artifacts: {artifacts.jpeg_artifacts:.3f}", file=sys.stderr)
    print(f"# Page warping: {artifacts.page_warping:.3f} degrees", file=sys.stderr)

    # Test 4: Check perfect alignment
    perfect = detect_perfect_alignment(features)
    print(f"# Perfect alignment detected: {perfect}", file=sys.stderr)

    # Test 5: Human warping score
    human_score = calculate_human_warping_score(features, artifacts)
    print(f"# Human warping score: {human_score:.4f}", file=sys.stderr)
    assert 0 <= human_score <= 1

    # Test 6: Emit receipt
    receipt = emit_layout_receipt("DOC-001", pdf_bytes)
    assert receipt["receipt_type"] == "layout"
    assert "layout_entropy" in receipt
    assert "fraud_indicator" in receipt

    # Test 7: Test with different PDFs for different entropies
    pdf1 = b"Uniform content " * 100  # Low entropy
    pdf2 = b"Varied content with different elements and random data xyz 123 !@#"  # Higher entropy

    features1 = extract_layout_features(pdf1)
    features2 = extract_layout_features(pdf2)

    entropy1 = calculate_layout_entropy(features1)
    entropy2 = calculate_layout_entropy(features2)

    print(f"# PDF1 entropy: {entropy1:.4f}, PDF2 entropy: {entropy2:.4f}", file=sys.stderr)

    print(f"# PASS: layout_entropy module self-test", file=sys.stderr)
