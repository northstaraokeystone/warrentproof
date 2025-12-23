"""
WarrantProof Compress Module Tests

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️
"""

import pytest
import random
import string
import time

from src.compress import (
    compress_receipt_stream,
    entropy_score,
    pattern_coherence,
    fraud_likelihood,
    compress_vendor_metadata,
    compress_certification_chain,
    detect_via_compression,
    THRESHOLDS,
)
from src.core import DISCLAIMER


class TestCompressReceiptStream:
    """Tests for compress_receipt_stream function."""

    def test_empty_receipts(self):
        """Test compression with empty receipt list."""
        result = compress_receipt_stream([])

        assert result["receipts_analyzed"] == 0
        assert result["compression_ratio"] == 1.0
        assert result["classification"] == "legitimate"

    def test_compression_returns_receipt(self):
        """Test compression returns valid receipt."""
        receipts = [{"receipt_type": "warrant", "amount_usd": 1000000}]
        result = compress_receipt_stream(receipts)

        assert result["receipt_type"] == "compression"
        assert "compression_ratio" in result
        assert "entropy_score" in result
        assert "simulation_flag" in result

    def test_compression_latency_10k_slo(self):
        """SLO: Compression analysis <= 10s per 10,000 receipts."""
        receipts = [
            {"receipt_type": "warrant", "amount_usd": i * 1000}
            for i in range(10000)
        ]

        t0 = time.time()
        compress_receipt_stream(receipts)
        latency_s = time.time() - t0

        assert latency_s <= 10, f"Compression latency {latency_s}s > 10s SLO"


class TestEntropyScore:
    """Tests for entropy_score function."""

    def test_empty_receipts(self):
        """Test entropy with empty receipts."""
        result = entropy_score([])
        assert result == 0.0

    def test_entropy_latency_slo(self):
        """SLO: Entropy calculation <= 100ms."""
        receipts = [
            {"receipt_type": "warrant", "branch": "Navy", "amount_usd": 1000000}
            for _ in range(100)
        ]

        t0 = time.time()
        entropy_score(receipts)
        latency_ms = (time.time() - t0) * 1000

        assert latency_ms <= 100, f"Entropy latency {latency_ms}ms > 100ms SLO"

    def test_low_entropy_uniform_data(self):
        """Test low entropy for uniform data."""
        receipts = [
            {"receipt_type": "warrant", "branch": "Navy", "amount_usd": 1000000}
            for _ in range(100)
        ]

        result = entropy_score(receipts)
        # Uniform data should have low entropy
        assert result < 3.0

    def test_high_entropy_random_data(self):
        """Test higher entropy for random data."""
        receipts = [
            {
                "receipt_type": random.choice(["warrant", "milestone", "cost_variance"]),
                "branch": random.choice(["Navy", "Army", "AirForce", "Marines", "SpaceForce"]),
                "amount_usd": random.randint(1000, 9999999),
                "vendor": ''.join(random.choices(string.ascii_uppercase, k=5))
            }
            for _ in range(100)
        ]

        result = entropy_score(receipts)
        # Random data should have higher entropy
        assert result > 1.0


class TestPatternCoherence:
    """Tests for pattern_coherence function."""

    def test_few_receipts(self):
        """Test coherence with few receipts."""
        receipts = [{"receipt_type": "warrant"}]
        result = pattern_coherence(receipts)
        assert result == 1.0  # Too few to judge

    def test_high_coherence_uniform(self):
        """Test high coherence for uniform patterns."""
        receipts = [
            {
                "receipt_type": "warrant",
                "branch": "Navy",
                "amount_usd": 1000000,
                "decision_lineage": ["parent"]
            }
            for _ in range(20)
        ]

        result = pattern_coherence(receipts)
        assert result >= 0.5  # Should show some coherence


class TestFraudLikelihood:
    """Tests for fraud_likelihood function."""

    def test_legitimate_scores_low(self):
        """Test legitimate pattern scores low likelihood."""
        # Good compression, low entropy, high coherence
        likelihood = fraud_likelihood(
            compression_ratio=0.85,
            entropy=2.0,
            coherence=0.8
        )

        assert likelihood < 0.5

    def test_fraudulent_scores_high(self):
        """Test fraudulent pattern scores high likelihood."""
        # Poor compression, high entropy, low coherence
        likelihood = fraud_likelihood(
            compression_ratio=0.35,
            entropy=6.0,
            coherence=0.2
        )

        assert likelihood > 0.5


class TestVendorCompression:
    """Tests for compress_vendor_metadata function."""

    def test_empty_receipts(self):
        """Test with empty receipts."""
        result = compress_vendor_metadata([])
        assert result["vendor_compression_ratio"] == 1.0
        assert result["ghost_vendor_likelihood"] == 0.0

    def test_vendor_compression_structure(self):
        """Test vendor compression returns proper structure."""
        receipts = [
            {"vendor": "Vendor1", "amount_usd": 1000000, "branch": "Navy"}
        ]

        result = compress_vendor_metadata(receipts)

        assert "vendor_compression_ratio" in result
        assert "citation" in result
        assert "simulation_flag" in result


class TestCertificationCompression:
    """Tests for compress_certification_chain function."""

    def test_no_certifications(self):
        """Test with no certification receipts."""
        result = compress_certification_chain([])
        assert result["cert_compression_ratio"] == 1.0
        assert result["cert_fraud_likelihood"] == 0.0

    def test_certification_compression_structure(self):
        """Test certification compression structure."""
        receipts = [
            {
                "receipt_type": "quality_attestation",
                "inspector": "Inspector1",
                "certification": {"grade": "A"}
            }
        ]

        result = compress_certification_chain(receipts)

        assert "cert_compression_ratio" in result
        assert "citation" in result


class TestDetectViaCompression:
    """Tests for detect_via_compression function."""

    def test_empty_receipts(self):
        """Test with empty receipts."""
        anomalies = detect_via_compression([])
        assert isinstance(anomalies, list)

    def test_returns_anomalies_list(self):
        """Test returns list of anomalies."""
        receipts = [
            {"receipt_type": "warrant", "amount_usd": 1000000}
        ]

        anomalies = detect_via_compression(receipts)
        assert isinstance(anomalies, list)


class TestThresholds:
    """Tests for threshold constants."""

    def test_thresholds_exist(self):
        """Test all threshold categories exist."""
        assert "legitimate" in THRESHOLDS
        assert "suspicious" in THRESHOLDS
        assert "fraudulent" in THRESHOLDS

    def test_legitimate_thresholds(self):
        """Test legitimate thresholds are defined."""
        assert THRESHOLDS["legitimate"]["compression_min"] >= 0.80
        assert THRESHOLDS["legitimate"]["entropy_max"] <= 3.5
        assert THRESHOLDS["legitimate"]["coherence_min"] >= 0.70

    def test_fraudulent_thresholds(self):
        """Test fraudulent thresholds are defined."""
        assert THRESHOLDS["fraudulent"]["compression_max"] <= 0.50
        assert THRESHOLDS["fraudulent"]["entropy_min"] >= 5.0
        assert THRESHOLDS["fraudulent"]["coherence_max"] <= 0.40
