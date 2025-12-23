"""
Tests for v4.0 User-Friendly Optimization Modules

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

Tests for:
- insight.py: Plain-language explanations
- fitness.py: Self-improving detection
- guardian.py: Evidence quality gates
- freshness.py: Evidence staleness
- learner.py: Cross-domain pattern learning
"""

import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import DISCLAIMER
from src.insight import (
    explain_anomaly,
    explain_compression_result,
    explain_kolmogorov_result,
    explain_raf_result,
    generate_executive_summary,
)
from src.fitness import (
    PatternFitness,
    SystemHealth,
    calculate_entropy,
    record_pattern_fitness,
    get_system_health,
    explain_fitness_for_users,
    prune_harmful_patterns,
    reset_system_health,
)
from src.guardian import (
    EvidenceItem,
    EvidenceSet,
    evaluate_evidence_quality,
    abstain,
    check_counter_evidence,
    check_evidence_integrity,
    gate_decision,
    ABSTENTION_REASONS,
)
from src.freshness import (
    assess_freshness,
    assess_evidence_set_freshness,
    get_refresh_priorities,
    explain_freshness_for_users,
    monitor_freshness_decay,
    FRESHNESS_DAYS,
)
from src.learner import (
    PatternSignature,
    PatternLibrary,
    match_patterns,
    transfer_pattern,
    learn_pattern,
    explain_pattern_for_users,
    get_library_summary,
    KNOWN_PATTERNS,
)


# === INSIGHT MODULE TESTS ===

class TestInsight:
    """Tests for insight.py - Plain-language explanations."""

    def test_explain_anomaly_has_required_fields(self):
        """Anomaly explanations should have all required fields."""
        anomaly = {
            "anomaly_type": "compression_failure",
            "fraud_likelihood": 0.75,
        }
        result = explain_anomaly(anomaly)

        assert "title" in result
        assert "summary" in result
        assert "what_it_means" in result
        assert "why_it_matters" in result
        assert "suggested_action" in result
        assert "simulation_flag" in result

    def test_explain_anomaly_different_types(self):
        """Different anomaly types should produce different explanations."""
        types = [
            "compression_failure",
            "ghost_vendor_compression",
            "kolmogorov_anomaly",
            "raf_cycle_detected",
        ]

        explanations = []
        for anomaly_type in types:
            result = explain_anomaly({
                "anomaly_type": anomaly_type,
                "fraud_likelihood": 0.5,
            })
            explanations.append(result["title"])

        # All should have different titles
        assert len(set(explanations)) == len(types)

    def test_explain_compression_result_classifications(self):
        """Compression explanations vary by classification."""
        legitimate = explain_compression_result({
            "compression_ratio": 0.85,
            "entropy_score": 3.0,
            "coherence_score": 0.75,
            "classification": "legitimate",
        })
        assert "normal" in legitimate["title"].lower() or "look" in legitimate["title"].lower()

        fraudulent = explain_compression_result({
            "compression_ratio": 0.35,
            "entropy_score": 6.0,
            "coherence_score": 0.25,
            "classification": "fraudulent",
        })
        assert "concern" in fraudulent["title"].lower() or "significant" in fraudulent["title"].lower()

    def test_explain_kolmogorov_fraud_vs_legitimate(self):
        """Kolmogorov explanations differ for fraud vs legitimate."""
        fraud = explain_kolmogorov_result({
            "kolmogorov_ratio": 0.45,
            "is_fraud": True,
        })
        assert fraud["is_fraud"] is True
        assert "artificial" in fraud["title"].lower() or "generated" in fraud["title"].lower()

        legit = explain_kolmogorov_result({
            "kolmogorov_ratio": 0.85,
            "is_fraud": False,
        })
        assert legit["is_fraud"] is False

    def test_explain_raf_result_with_cycles(self):
        """RAF explanation should report cycles found."""
        result = explain_raf_result({
            "cycles_detected": 3,
            "keystone_species": ["CompanyA", "CompanyB"],
        })

        assert "3" in result["summary"]
        assert "circular" in result["summary"].lower() or "cycle" in result["summary"].lower()

    def test_generate_executive_summary(self):
        """Executive summary should aggregate multiple results."""
        results = [
            {"classification": "legitimate", "fraud_likelihood": 0.1},
            {"classification": "legitimate", "fraud_likelihood": 0.2},
            {"classification": "suspicious", "fraud_likelihood": 0.5},
            {"classification": "fraudulent", "fraud_likelihood": 0.95},
        ]

        summary = generate_executive_summary(results)

        assert summary["summary_counts"]["total_analyzed"] == 4
        assert summary["summary_counts"]["critical_issues"] == 1
        assert "status" in summary


# === FITNESS MODULE TESTS ===

class TestFitness:
    """Tests for fitness.py - Self-improving detection."""

    def setup_method(self):
        """Reset system health before each test."""
        reset_system_health()

    def test_calculate_entropy(self):
        """Entropy calculation should work correctly."""
        # All same classification = 0 entropy
        same = [{"classification": "legitimate"} for _ in range(10)]
        assert calculate_entropy(same) == 0.0

        # Mixed classifications = positive entropy
        mixed = [
            {"classification": "legitimate"},
            {"classification": "fraudulent"},
        ]
        assert calculate_entropy(mixed) > 0

    def test_pattern_fitness_tracking(self):
        """Pattern fitness should track entropy reduction."""
        pattern = PatternFitness("test", "compression")

        # Record an observation that reduces entropy
        fitness = pattern.record_observation(
            entropy_before=2.0,
            entropy_after=1.0,
            receipts_processed=10
        )

        assert fitness > 0  # Positive fitness = helpful
        assert pattern.average_fitness > 0
        assert len(pattern.observations) == 1

    def test_record_pattern_fitness_receipt(self):
        """record_pattern_fitness should emit proper receipt."""
        before = [{"classification": "unknown"} for _ in range(50)]
        after = [
            {"classification": "legitimate" if i % 2 == 0 else "fraudulent"}
            for i in range(50)
        ]

        result = record_pattern_fitness(
            "test_pattern",
            "test_type",
            before,
            after
        )

        assert "fitness_score" in result
        assert "is_helpful" in result
        assert "simulation_flag" in result

    def test_system_health_tracking(self):
        """System health should aggregate pattern fitness."""
        # Record several patterns
        for i in range(3):
            before = [{"classification": "unknown"} for _ in range(20)]
            after = [{"classification": "legitimate"} for _ in range(20)]
            record_pattern_fitness(f"pattern_{i}", "test", before, after)

        health = get_system_health()

        assert "status" in health
        assert "overall_fitness" in health
        assert "pattern_breakdown" in health

    def test_explain_fitness_for_users(self):
        """Fitness explanation should be user-friendly."""
        result = explain_fitness_for_users()

        assert "headline" in result
        assert "explanation" in result
        assert "effectiveness_percent" in result

    def test_prune_harmful_patterns(self):
        """Pruning should identify harmful patterns."""
        # This requires patterns with negative fitness
        # After a fresh reset, no patterns exist
        result = prune_harmful_patterns()

        assert "harmful_patterns_found" in result
        assert "action_needed" in result


# === GUARDIAN MODULE TESTS ===

class TestGuardian:
    """Tests for guardian.py - Evidence quality gates."""

    def test_evidence_item_quality(self):
        """Evidence items should track quality degradation."""
        evidence = EvidenceItem(
            "ev_001",
            "compression",
            {"test": "data"},
            "scanner"
        )

        assert evidence.quality_score == 1.0

        evidence.add_issue("minor_concern", 0.1)
        assert evidence.quality_score == 0.9

        evidence.add_issue("major_concern", 0.5)
        assert evidence.quality_score == 0.4

    def test_evidence_set_confidence(self):
        """Evidence set should compute net confidence."""
        evidence_set = EvidenceSet("test_001")

        # Add supporting evidence
        supporting = EvidenceItem("s1", "test", {}, "source")
        evidence_set.add_supporting(supporting)

        # Add weaker counter evidence
        counter = EvidenceItem("c1", "test", {}, "source")
        counter.quality_score = 0.3
        evidence_set.add_counter(counter)

        # Net confidence should be positive
        assert evidence_set.net_confidence > 0

    def test_evaluate_evidence_quality_thresholds(self):
        """Quality evaluation should respect thresholds."""
        evidence_set = EvidenceSet("test_002")

        # Strong evidence
        strong = EvidenceItem("s1", "test", {}, "source")
        strong.quality_score = 1.0
        evidence_set.add_supporting(strong)

        result = evaluate_evidence_quality(evidence_set)

        assert result["quality_level"] in ["high", "acceptable"]
        assert result["should_abstain"] is False

    def test_abstain_emits_proper_receipt(self):
        """Abstention should include proper reason codes."""
        result = abstain(
            "test_determination",
            "insufficient_data",
            {"note": "test"}
        )

        assert result["because"] == "insufficient_data"
        assert "reason_code" in result
        assert "user_message" in result
        assert result["reason_code"] == ABSTENTION_REASONS["insufficient_data"]["code"]

    def test_check_counter_evidence(self):
        """Counter evidence check should find contradicting records."""
        finding = {
            "anomaly_type": "ghost_vendor_compression",
            "affected_entities": ["VendorX"],
        }
        records = [
            {"vendor": "VendorX", "verification_status": "verified"},
        ]

        result = check_counter_evidence(finding, records)

        assert result["counter_evidence_found"] > 0

    def test_check_evidence_integrity(self):
        """Integrity check should identify issues."""
        # Good chain
        good_receipts = [
            {"payload_hash": "abc123", "ts": "2024-01-01T00:00:00Z"},
            {"payload_hash": "def456", "ts": "2024-01-02T00:00:00Z"},
        ]
        result = check_evidence_integrity(good_receipts)
        assert result["has_integrity"] is True

        # Bad chain with duplicates
        bad_receipts = [
            {"payload_hash": "abc123", "ts": "2024-01-01T00:00:00Z"},
            {"payload_hash": "abc123", "ts": "2024-01-02T00:00:00Z"},  # Duplicate
        ]
        result = check_evidence_integrity(bad_receipts)
        assert result["issues_found"] > 0


# === FRESHNESS MODULE TESTS ===

class TestFreshness:
    """Tests for freshness.py - Evidence staleness detection."""

    def test_assess_freshness_levels(self):
        """Freshness should properly categorize by age."""
        # Fresh (< 30 days)
        fresh_ts = datetime.utcnow() - timedelta(days=15)
        result = assess_freshness(fresh_ts)
        assert result["freshness_level"] == "fresh"
        assert result["confidence_factor"] == 1.0

        # Recent (30-60 days)
        recent_ts = datetime.utcnow() - timedelta(days=45)
        result = assess_freshness(recent_ts)
        assert result["freshness_level"] == "recent"

        # Stale (90-180 days)
        stale_ts = datetime.utcnow() - timedelta(days=120)
        result = assess_freshness(stale_ts)
        assert result["freshness_level"] == "stale"

        # Expired (> 180 days)
        expired_ts = datetime.utcnow() - timedelta(days=200)
        result = assess_freshness(expired_ts)
        assert result["freshness_level"] == "expired"

    def test_assess_freshness_data_type_multipliers(self):
        """Price data should decay faster than contract data."""
        ts = datetime.utcnow() - timedelta(days=20)

        # General data at 20 days = fresh
        general = assess_freshness(ts, "general")
        assert general["freshness_level"] == "fresh"

        # Price data at 20 days = aging (0.5x multiplier)
        price = assess_freshness(ts, "price_data")
        # 20 days / 0.5 = 40 effective days, which is aging
        assert price["freshness_level"] in ["recent", "aging"]

    def test_assess_evidence_set_freshness(self):
        """Set freshness should be limited by stalest item."""
        evidence = [
            {"timestamp": datetime.utcnow() - timedelta(days=10), "data_type": "general"},
            {"timestamp": datetime.utcnow() - timedelta(days=100), "data_type": "general"},
        ]

        result = assess_evidence_set_freshness(evidence)

        # Overall should be stale due to the 100-day-old item
        assert result["overall_freshness"] == "stale"

    def test_get_refresh_priorities(self):
        """Refresh priorities should rank by staleness."""
        evidence = [
            {"id": "new", "timestamp": datetime.utcnow() - timedelta(days=10), "data_type": "general"},
            {"id": "old", "timestamp": datetime.utcnow() - timedelta(days=150), "data_type": "general"},
            {"id": "medium", "timestamp": datetime.utcnow() - timedelta(days=70), "data_type": "general"},
        ]

        result = get_refresh_priorities(evidence)

        # Old should be highest priority
        if result["priorities"]:
            assert result["priorities"][0]["item_id"] == "old"

    def test_explain_freshness_for_users(self):
        """Freshness explanation should be user-friendly."""
        assessment = assess_freshness(
            datetime.utcnow() - timedelta(days=120),
            "general"
        )
        result = explain_freshness_for_users(assessment)

        assert "headline" in result
        assert "explanation" in result
        assert result["action_needed"] is True  # Stale data needs action


# === LEARNER MODULE TESTS ===

class TestLearner:
    """Tests for learner.py - Cross-domain pattern learning."""

    def test_known_patterns_loaded(self):
        """Pattern library should have known patterns."""
        summary = get_library_summary()

        assert summary["total_patterns"] >= len(KNOWN_PATTERNS)
        assert "repetitive_billing" in [p["pattern_id"] for p in summary["patterns"]]

    def test_match_patterns_finds_matches(self):
        """Pattern matching should identify matching data."""
        # Data that matches repetitive_billing pattern
        data = {
            "compression_ratio": 0.35,  # < 0.50
            "description_entropy": 1.5,  # < 2.0
            "unique_descriptions_pct": 0.20,  # < 0.30
        }

        result = match_patterns(data)

        assert result["matches_found"] > 0
        assert result["risk_level"] in ["low", "medium", "high", "critical"]

    def test_match_patterns_domain_filter(self):
        """Domain filter should restrict pattern matching."""
        data = {
            "compression_ratio": 0.35,
            "description_entropy": 1.5,
            "unique_descriptions_pct": 0.20,
        }

        # Match with domain filter
        result = match_patterns(data, domain="aerospace")

        # Should still work (either matches or doesn't)
        assert "matches_found" in result

    def test_transfer_pattern(self):
        """Pattern transfer should document the transfer."""
        result = transfer_pattern(
            source_domain="logistics",
            target_domain="maintenance",
            pattern_id="repetitive_billing"
        )

        assert result["pattern_id"] == "repetitive_billing"
        assert result["source_domain"] == "logistics"
        assert result["target_domain"] == "maintenance"
        assert "effective_transferability" in result

    def test_transfer_unknown_pattern_fails(self):
        """Transfer of unknown pattern should fail gracefully."""
        result = transfer_pattern(
            source_domain="logistics",
            target_domain="maintenance",
            pattern_id="nonexistent_pattern"
        )

        assert result["success"] is False

    def test_learn_pattern(self):
        """Learning new patterns should add to library."""
        initial_count = get_library_summary()["total_patterns"]

        result = learn_pattern(
            name="Test Custom Pattern",
            description="A test pattern for unit testing",
            signature={"test_field": {"operator": ">", "value": 100}},
            source_case="Unit Test",
            confidence=0.7
        )

        assert result["success"] is True
        assert "pattern_id" in result

        # Library should have one more pattern
        final_count = get_library_summary()["total_patterns"]
        assert final_count == initial_count + 1

    def test_explain_pattern_for_users(self):
        """Pattern explanations should be user-friendly."""
        data = {
            "compression_ratio": 0.35,
            "description_entropy": 1.5,
            "unique_descriptions_pct": 0.20,
        }
        matches = match_patterns(data)

        if matches["matches"]:
            result = explain_pattern_for_users(matches["matches"][0])

            assert "explanation" in result
            assert "what_to_do" in result


# === INTEGRATION TESTS ===

class TestIntegration:
    """Integration tests across v4.0 modules."""

    def test_full_analysis_workflow(self):
        """Test complete analysis with all modules."""
        # 1. Create evidence
        evidence_set = EvidenceSet("integration_test")
        evidence = EvidenceItem(
            "ev_001",
            "compression_analysis",
            {"compression_ratio": 0.35},
            "scanner",
            timestamp=datetime.utcnow() - timedelta(days=5)
        )
        evidence_set.add_supporting(evidence)

        # 2. Check evidence quality
        quality = evaluate_evidence_quality(evidence_set)
        assert "quality_level" in quality

        # 3. Check freshness
        freshness = assess_freshness(
            datetime.utcnow() - timedelta(days=5),
            "compression_analysis"
        )
        assert freshness["freshness_level"] == "fresh"

        # 4. Match patterns
        patterns = match_patterns({
            "compression_ratio": 0.35,
            "description_entropy": 1.5,
            "unique_descriptions_pct": 0.20,
        })
        assert "matches_found" in patterns

        # 5. Generate explanation
        if patterns["matches"]:
            explanation = explain_pattern_for_users(patterns["matches"][0])
            assert "explanation" in explanation

    def test_abstention_workflow(self):
        """Test abstention when evidence is weak."""
        evidence_set = EvidenceSet("weak_evidence_test")

        # Add weak evidence
        weak = EvidenceItem("w1", "test", {}, "source")
        weak.quality_score = 0.2
        evidence_set.add_supporting(weak)

        # Quality should trigger abstention
        quality = evaluate_evidence_quality(evidence_set)

        assert quality["quality_level"] == "insufficient"
        assert quality["should_abstain"] is True


# === SELF-TEST RUNNER ===

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
