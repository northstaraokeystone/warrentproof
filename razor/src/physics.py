"""
RAZOR Physics Module - Kolmogorov Complexity via Compression

Kolmogorov complexity measurement using lossless compression as proxy.
K(x) = len(compressed) / len(original)

THE PHYSICS:
  - Honest market = high-entropy gas: Chaotic, diverse, incompressible
  - Corrupt market = ordered crystal: Coordinated, repetitive, compressible
  - K(x) = compression ratio: Kolmogorov complexity proxy
  - No proof system needed: The compression ratio IS the proof

COMPRESSION ALGORITHMS:
  - zlib (DEFLATE): Detects local repetitions (copy-paste invoices)
  - lzma (LZMA2): Finds long-range correlations (templated billing)
  - bz2 (Burrows-Wheeler): Identifies structural reorderings (permuted boilerplate)

Multi-modal spectrometer: All three used because different fraud types
create different compression signatures.
"""

import zlib
import lzma
import bz2
import math
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .core import (
    emit_receipt,
    StopRule,
    stoprule_compression_invalid,
    CR_THRESHOLD_LOW,
    CR_THRESHOLD_HIGH,
)

# ============================================================================
# KOLMOGOROV METRIC CLASS
# ============================================================================

class KolmogorovMetric:
    """
    Kolmogorov complexity measurement via lossless compression.

    Methods:
      - canonicalize: Normalize text for consistent measurement
      - measure_complexity: Full multi-modal compression analysis
      - compress_zlib: DEFLATE compression ratio
      - compress_lzma: LZMA2 compression ratio
      - compress_bz2: Burrows-Wheeler compression ratio
      - calculate_shannon_entropy: Character-level entropy
      - analyze_record: Compress a single record
      - analyze_cohort: Compress all records in DataFrame

    SLOs:
      - Compression time <= 10ms per record
      - Shannon entropy bounds: 0.0 <= H <= log2(charset_size)
    """

    def __init__(self):
        """Initialize the Kolmogorov metric calculator."""
        self.min_text_length = 10  # Minimum chars for valid compression

    def canonicalize(self, text: str) -> str:
        """
        Normalize text for consistent complexity measurement.

        Removes formatting artifacts that don't represent semantic content:
        - Excessive whitespace
        - JSON artifacts
        - Case normalization

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Normalize whitespace (collapse multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)

        # Remove common JSON/formatting artifacts
        text = re.sub(r'[{}\[\]"\':]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def compress_zlib(self, data: bytes) -> float:
        """
        Calculate zlib (DEFLATE) compression ratio.

        Detects local repetitions - catches copy-paste fraud.

        Args:
            data: Bytes to compress

        Returns:
            Compression ratio (compressed_len / original_len)
        """
        if len(data) == 0:
            return 1.0

        try:
            compressed = zlib.compress(data, level=9)
            return len(compressed) / len(data)
        except Exception:
            return 1.0

    def compress_lzma(self, data: bytes) -> float:
        """
        Calculate lzma (LZMA2) compression ratio.

        Finds long-range correlations - catches templated billing.

        Args:
            data: Bytes to compress

        Returns:
            Compression ratio (compressed_len / original_len)
        """
        if len(data) == 0:
            return 1.0

        try:
            compressed = lzma.compress(data)
            return len(compressed) / len(data)
        except Exception:
            return 1.0

    def compress_bz2(self, data: bytes) -> float:
        """
        Calculate bz2 (Burrows-Wheeler) compression ratio.

        Identifies structural reorderings - catches permuted boilerplate.

        Args:
            data: Bytes to compress

        Returns:
            Compression ratio (compressed_len / original_len)
        """
        if len(data) == 0:
            return 1.0

        try:
            compressed = bz2.compress(data)
            return len(compressed) / len(data)
        except Exception:
            return 1.0

    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of character distribution.

        H = -SUM p(x) * log2(p(x))

        Provides baseline entropy measurement independent of
        compression algorithm specifics.

        Args:
            text: Text to analyze

        Returns:
            Shannon entropy in bits per character
        """
        if not text:
            return 0.0

        # Count character frequencies
        counter = Counter(text)
        total = len(text)

        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def measure_complexity(self, text: str) -> Dict[str, float]:
        """
        Full multi-modal complexity analysis.

        Returns compression ratios from all three algorithms plus
        Shannon entropy and original length.

        Args:
            text: Text to analyze

        Returns:
            Dict with cr_zlib, cr_lzma, cr_bz2, shannon_entropy, len_original
        """
        # Canonicalize text
        canonical = self.canonicalize(text)

        if len(canonical) < self.min_text_length:
            # Text too short for meaningful compression
            return {
                "cr_zlib": 1.0,
                "cr_lzma": 1.0,
                "cr_bz2": 1.0,
                "shannon_entropy": 0.0,
                "len_original": len(text),
                "len_canonical": len(canonical),
                "valid": False,
            }

        # Convert to bytes for compression
        data = canonical.encode('utf-8')

        return {
            "cr_zlib": self.compress_zlib(data),
            "cr_lzma": self.compress_lzma(data),
            "cr_bz2": self.compress_bz2(data),
            "shannon_entropy": self.calculate_shannon_entropy(canonical),
            "len_original": len(text),
            "len_canonical": len(canonical),
            "valid": True,
        }

    def analyze_record(
        self,
        row: "pd.Series",
        description_column: str = "description",
    ) -> Dict[str, float]:
        """
        Analyze a single record from DataFrame.

        Compresses description field and returns all metrics.

        Args:
            row: DataFrame row (Series)
            description_column: Column name for description text

        Returns:
            Dict with complexity metrics
        """
        description = str(row.get(description_column, ""))
        metrics = self.measure_complexity(description)

        # Add record identifiers
        metrics["award_id"] = row.get("award_id", "unknown")

        # Calculate price-complexity ratio if amount available
        amount = row.get("total_obligation", 0)
        if amount and amount > 0 and metrics["cr_zlib"] > 0:
            metrics["price_complexity_ratio"] = math.log10(max(amount, 1)) / metrics["cr_zlib"]
        else:
            metrics["price_complexity_ratio"] = 0.0

        return metrics

    def analyze_cohort(
        self,
        df: "pd.DataFrame",
        description_column: str = "description",
    ) -> "pd.DataFrame":
        """
        Analyze all records in a cohort DataFrame.

        Applies analyze_record to each row and adds complexity columns.

        Args:
            df: DataFrame with records
            description_column: Column name for description text

        Returns:
            DataFrame with complexity columns added
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for cohort analysis")

        if df.empty:
            return df

        # Apply analysis to each row
        complexity_data = []
        for idx, row in df.iterrows():
            metrics = self.analyze_record(row, description_column)
            complexity_data.append(metrics)

        # Convert to DataFrame and merge
        complexity_df = pd.DataFrame(complexity_data)

        # Add complexity columns to original DataFrame
        result = df.copy()
        for col in ["cr_zlib", "cr_lzma", "cr_bz2", "shannon_entropy",
                    "len_canonical", "valid", "price_complexity_ratio"]:
            if col in complexity_df.columns:
                result[col] = complexity_df[col].values

        # Emit receipt for cohort analysis
        valid_count = complexity_df["valid"].sum() if "valid" in complexity_df else 0
        emit_receipt("complexity_cohort", {
            "total_records": len(df),
            "valid_records": int(valid_count),
            "mean_cr_zlib": float(complexity_df["cr_zlib"].mean()) if not complexity_df.empty else 0.0,
            "mean_cr_lzma": float(complexity_df["cr_lzma"].mean()) if not complexity_df.empty else 0.0,
            "mean_cr_bz2": float(complexity_df["cr_bz2"].mean()) if not complexity_df.empty else 0.0,
            "mean_shannon_entropy": float(complexity_df["shannon_entropy"].mean()) if not complexity_df.empty else 0.0,
        }, to_stdout=False)

        return result

    def emit_complexity_receipt(
        self,
        award_id: str,
        metrics: Dict[str, float],
    ) -> dict:
        """
        Emit a receipt for a single complexity measurement.

        Args:
            award_id: Award identifier
            metrics: Complexity metrics dict

        Returns:
            Receipt dict
        """
        return emit_receipt("complexity", {
            "award_id": award_id,
            "cr_zlib": metrics.get("cr_zlib", 0.0),
            "cr_lzma": metrics.get("cr_lzma", 0.0),
            "cr_bz2": metrics.get("cr_bz2", 0.0),
            "shannon_entropy": metrics.get("shannon_entropy", 0.0),
            "len_original": metrics.get("len_original", 0),
            "valid": metrics.get("valid", False),
        }, to_stdout=False)

    def classify_complexity(self, cr_zlib: float) -> str:
        """
        Classify compression ratio into risk category.

        Args:
            cr_zlib: Zlib compression ratio

        Returns:
            Classification string: "suspicious", "normal", "high_entropy"
        """
        if cr_zlib < CR_THRESHOLD_LOW:
            return "suspicious"  # Too compressible
        elif cr_zlib > CR_THRESHOLD_HIGH:
            return "high_entropy"  # Normal market behavior
        else:
            return "normal"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_compression_test(text: str) -> Dict[str, float]:
    """
    Quick compression test for a single string.

    Args:
        text: Text to analyze

    Returns:
        Dict with compression ratios
    """
    km = KolmogorovMetric()
    return km.measure_complexity(text)


def compare_compression(text1: str, text2: str) -> Dict[str, float]:
    """
    Compare compression ratios of two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dict with both results and delta
    """
    km = KolmogorovMetric()
    m1 = km.measure_complexity(text1)
    m2 = km.measure_complexity(text2)

    return {
        "text1_cr_zlib": m1["cr_zlib"],
        "text2_cr_zlib": m2["cr_zlib"],
        "delta_cr_zlib": m2["cr_zlib"] - m1["cr_zlib"],
        "text1_more_compressible": m1["cr_zlib"] < m2["cr_zlib"],
    }


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("# RAZOR Physics Module", file=sys.stderr)

    km = KolmogorovMetric()

    # Test 1: Repetitive text (fraud-like) should compress better
    repetitive = "husbanding services for ship " * 100
    random_text = "qp8jf2kd9s7hgb3cx5nz1w4m" * 100

    r1 = km.measure_complexity(repetitive)
    r2 = km.measure_complexity(random_text)

    print(f"# Repetitive CR: {r1['cr_zlib']:.3f}", file=sys.stderr)
    print(f"# Random CR: {r2['cr_zlib']:.3f}", file=sys.stderr)

    assert r1["cr_zlib"] < r2["cr_zlib"], "Repetitive text should compress better"
    assert r1["cr_zlib"] < 0.10, "Highly repetitive should be very compressible"
    assert r2["cr_zlib"] > 0.50, "Random-ish text should not compress well"

    # Test 2: Shannon entropy
    low_entropy = "aaaaaaaaaaaa"
    high_entropy = "abcdefghijkl"

    e1 = km.calculate_shannon_entropy(low_entropy)
    e2 = km.calculate_shannon_entropy(high_entropy)

    print(f"# Low entropy H: {e1:.3f}", file=sys.stderr)
    print(f"# High entropy H: {e2:.3f}", file=sys.stderr)

    assert e1 < e2, "Uniform chars should have higher entropy than single char"

    # Test 3: Canonicalization
    raw = "  HUSBANDING   Services\n\t  FOR Ship  "
    canonical = km.canonicalize(raw)
    assert canonical == "husbanding services for ship", f"Got: {canonical}"

    print("# PASS: RAZOR physics module self-test", file=sys.stderr)
