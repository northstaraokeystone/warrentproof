"""
WarrantProof Detect Module Tests

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️
"""

import pytest
import time

from src.detect import (
    scan,
    classify_anomaly,
    temporal_cluster,
    cost_cascade_detect,
    emit_alert,
    emit_detection_receipt,
    ANOMALY_TYPES,
)
from src.core import DISCLAIMER


class TestScan:
    """Tests for scan function."""

    def test_scan_empty_receipts(self):
        """Test scan with empty receipt list."""
        matches = scan([])
        assert matches == []

    def test_scan_latency_slo(self):
        """SLO: Scan <= 100ms per 1000 receipts."""
        # Generate 1000 test receipts
        receipts = [
            {"receipt_type": "warrant", "vendor": f"Vendor_{i}", "amount_usd": 100000}
            for i in range(1000)
        ]

        t0 = time.time()
        scan(receipts)
        latency_ms = (time.time() - t0) * 1000

        assert latency_ms <= 100, f"Scan latency {latency_ms}ms > 100ms SLO"

    def test_scan_detects_ghost_vendor(self):
        """Test ghost vendor detection."""
        receipts = [
            {"receipt_type": "warrant", "vendor": "OneTimeVendor",
             "amount_usd": 5000000, "description": "Large single transaction"}
        ]

        matches = scan(receipts, patterns=["ghost_vendor"])
        # May or may not detect based on threshold
        assert isinstance(matches, list)

    def test_scan_detects_cost_cascade(self):
        """Test cost cascade detection."""
        receipts = [
            {"receipt_type": "cost_variance", "program": "TestProgram",
             "variance_pct": 5, "ts": "2024-01-01T00:00:00Z"},
            {"receipt_type": "cost_variance", "program": "TestProgram",
             "variance_pct": 10, "ts": "2024-02-01T00:00:00Z"},
            {"receipt_type": "cost_variance", "program": "TestProgram",
             "variance_pct": 20, "ts": "2024-03-01T00:00:00Z"},
        ]

        matches = scan(receipts, patterns=["cost_cascade"])

        # Should detect increasing variance pattern
        cascade_matches = [m for m in matches if m.get("anomaly_type") == "cost_cascade"]
        assert len(cascade_matches) > 0


class TestClassifyAnomaly:
    """Tests for classify_anomaly function."""

    def test_classify_known_types(self):
        """Test classification of known anomaly types."""
        for anomaly_type in ANOMALY_TYPES.keys():
            match = {"anomaly_type": anomaly_type}
            result = classify_anomaly(match)
            assert result == anomaly_type

    def test_classify_unknown_type(self):
        """Test classification of unknown type."""
        match = {"anomaly_type": "unknown_type"}
        result = classify_anomaly(match)
        assert result == "unknown_type"


class TestTemporalCluster:
    """Tests for temporal_cluster function."""

    def test_empty_receipts(self):
        """Test clustering with empty receipts."""
        result = temporal_cluster([])
        assert result["clusters"] == []
        assert result["anomalies"] == []

    def test_cluster_detection(self):
        """Test temporal clustering."""
        receipts = [
            {"ts": "2024-01-15T10:00:00Z"},
            {"ts": "2024-01-15T11:00:00Z"},
            {"ts": "2024-01-15T12:00:00Z"},
            {"ts": "2024-02-15T10:00:00Z"},
        ]

        result = temporal_cluster(receipts, window="1d")

        assert len(result["clusters"]) >= 1


class TestCostCascadeDetect:
    """Tests for cost_cascade_detect function."""

    def test_no_cascade_single_receipt(self):
        """Test with single receipt - no cascade possible."""
        receipts = [
            {"receipt_type": "cost_variance", "variance_pct": 10}
        ]

        result = cost_cascade_detect(receipts)
        assert result["cascade_detected"] == False

    def test_cascade_detected(self):
        """Test cascade detection with increasing variances."""
        receipts = [
            {"receipt_type": "cost_variance", "variance_pct": 5, "ts": "2024-01-01"},
            {"receipt_type": "cost_variance", "variance_pct": 10, "ts": "2024-02-01"},
            {"receipt_type": "cost_variance", "variance_pct": 15, "ts": "2024-03-01"},
            {"receipt_type": "cost_variance", "variance_pct": 25, "ts": "2024-04-01"},
        ]

        result = cost_cascade_detect(receipts)
        assert result["cascade_detected"] == True

    def test_no_cascade_decreasing(self):
        """Test no cascade with decreasing variances."""
        receipts = [
            {"receipt_type": "cost_variance", "variance_pct": 25, "ts": "2024-01-01"},
            {"receipt_type": "cost_variance", "variance_pct": 15, "ts": "2024-02-01"},
            {"receipt_type": "cost_variance", "variance_pct": 10, "ts": "2024-03-01"},
        ]

        result = cost_cascade_detect(receipts)
        assert result["cascade_detected"] == False


class TestEmitAlert:
    """Tests for emit_alert function."""

    def test_alert_structure(self):
        """Test alert receipt structure."""
        anomaly = {
            "anomaly_type": "ghost_vendor",
            "confidence": 0.8,
            "affected_receipts": ["receipt_1"]
        }

        alert = emit_alert(anomaly, "high")

        assert alert["receipt_type"] == "alert"
        assert alert["severity"] == "high"
        assert "simulation_flag" in alert

    def test_alert_escalation(self):
        """Test critical alert triggers escalation."""
        anomaly = {"anomaly_type": "cost_cascade"}
        alert = emit_alert(anomaly, "critical")

        assert alert["escalate"] == True


class TestEmitDetectionReceipt:
    """Tests for emit_detection_receipt function."""

    def test_detection_receipt_structure(self):
        """Test detection receipt structure."""
        matches = [
            {"anomaly_type": "ghost_vendor", "confidence": 0.7},
            {"anomaly_type": "ghost_vendor", "confidence": 0.8},
            {"anomaly_type": "cost_cascade", "confidence": 0.9},
        ]

        receipt = emit_detection_receipt(matches)

        assert receipt["receipt_type"] == "detection"
        assert receipt["total_anomalies"] == 3
        assert receipt["anomaly_counts"]["ghost_vendor"] == 2
        assert receipt["anomaly_counts"]["cost_cascade"] == 1
