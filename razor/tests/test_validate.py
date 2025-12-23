"""
Tests for RAZOR Validate Module - Statistical Signal Detection
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.validate import (
    calculate_baseline,
    calculate_z_scores,
    run_t_test,
    run_mann_whitney,
    calculate_cohens_d,
    detect_signal,
    interpret_cohens_d,
    generate_report,
)
from src.core import Z_SCORE_THRESHOLD, MIN_CONTROL_SIZE


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestCalculateBaseline:
    """Tests for calculate_baseline function."""

    def test_baseline_calculation(self):
        """Test basic baseline calculation."""
        df = pd.DataFrame({"cr_zlib": [0.5, 0.6, 0.7, 0.5, 0.6] * 30})
        baseline = calculate_baseline(df, "cr_zlib")

        assert baseline["valid"] is True
        assert 0.5 < baseline["mean"] < 0.7
        assert baseline["std"] > 0
        assert baseline["n"] == 150

    def test_baseline_insufficient_data(self):
        """Test baseline with insufficient data."""
        df = pd.DataFrame({"cr_zlib": [0.5, 0.6]})
        baseline = calculate_baseline(df, "cr_zlib")

        assert baseline["valid"] is False
        assert baseline["n"] == 2

    def test_baseline_missing_column(self):
        """Test baseline with missing column."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        baseline = calculate_baseline(df, "cr_zlib")

        assert baseline["valid"] is False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestCalculateZScores:
    """Tests for Z-score calculation."""

    def test_z_scores_calculation(self):
        """Test Z-score calculation."""
        fraud_df = pd.DataFrame({"cr_zlib": [0.3, 0.35, 0.4]})
        baseline = {"mean": 0.6, "std": 0.1, "valid": True}

        z_scores = calculate_z_scores(fraud_df, baseline, "cr_zlib")

        assert len(z_scores) == 3
        assert all(z < 0 for z in z_scores)  # All below mean
        assert z_scores.iloc[0] < -2.0  # 0.3 is 3 std below 0.6

    def test_z_scores_zero_std(self):
        """Test Z-score with zero std (degenerate case)."""
        fraud_df = pd.DataFrame({"cr_zlib": [0.5, 0.5]})
        baseline = {"mean": 0.5, "std": 0.0, "valid": True}

        z_scores = calculate_z_scores(fraud_df, baseline, "cr_zlib")

        assert all(z == 0 for z in z_scores)


class TestTTest:
    """Tests for T-test function."""

    def test_t_test_significant_difference(self, fraud_cr_values, control_cr_values):
        """Test T-test detects significant difference."""
        result = run_t_test(fraud_cr_values, control_cr_values, "less")

        if HAS_SCIPY:
            assert result["scipy_available"] is True
            assert result["significant"] == True  # Use == for numpy bool comparison
            assert result["p_value"] < 0.05
            assert result["t_statistic"] < 0  # fraud mean < control mean

    def test_t_test_no_difference(self):
        """Test T-test with no significant difference."""
        values = [0.5, 0.51, 0.49, 0.52, 0.48] * 20
        result = run_t_test(values, values, "less")

        if HAS_SCIPY:
            # Same distribution should not be significant
            assert result["p_value"] > 0.05

    def test_t_test_insufficient_samples(self):
        """Test T-test with insufficient samples."""
        result = run_t_test([0.5], [0.6])

        assert result["significant"] is False
        assert result.get("insufficient_samples", True)


class TestMannWhitney:
    """Tests for Mann-Whitney U test."""

    def test_mann_whitney_significant(self, fraud_cr_values, control_cr_values):
        """Test Mann-Whitney detects significant difference."""
        result = run_mann_whitney(fraud_cr_values, control_cr_values, "less")

        if HAS_SCIPY:
            assert result["scipy_available"] is True
            assert result["significant"] == True  # Use == for numpy bool comparison
            assert result["p_value"] < 0.05


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_large_effect(self, fraud_cr_values, control_cr_values):
        """Test Cohen's d calculation for large effect."""
        d = calculate_cohens_d(fraud_cr_values, control_cr_values)

        # Fraud (mean ~0.35) vs Control (mean ~0.65) = large effect
        assert abs(d) > 0.8

    def test_cohens_d_no_effect(self):
        """Test Cohen's d with identical distributions."""
        values = [0.5] * 50
        d = calculate_cohens_d(values, values)

        assert d == 0.0

    def test_cohens_d_insufficient_samples(self):
        """Test Cohen's d with insufficient samples."""
        d = calculate_cohens_d([0.5], [0.6])

        assert d == 0.0


class TestInterpretCohensD:
    """Tests for Cohen's d interpretation."""

    def test_interpret_large(self):
        assert interpret_cohens_d(0.9) == "large"
        assert interpret_cohens_d(-0.9) == "large"

    def test_interpret_medium(self):
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(-0.6) == "medium"

    def test_interpret_small(self):
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(-0.3) == "small"

    def test_interpret_negligible(self):
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.0) == "negligible"


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestDetectSignal:
    """Tests for full signal detection pipeline."""

    def test_detect_signal_with_synthetic_data(self):
        """Test signal detection with synthetic fraud/control data."""
        import random
        random.seed(42)

        control_df = pd.DataFrame({
            "cr_zlib": [random.gauss(0.65, 0.10) for _ in range(150)],
        })
        fraud_df = pd.DataFrame({
            "cr_zlib": [random.gauss(0.35, 0.08) for _ in range(75)],
        })

        results = detect_signal(fraud_df, control_df, "cr_zlib")

        assert results["verdict"]["signal_detected"] is True
        assert results["z_scores"]["mean"] < Z_SCORE_THRESHOLD
        assert results["verdict"]["signal_strength"] in ["weak", "moderate", "strong"]

    def test_detect_signal_no_difference(self):
        """Test signal detection with no real difference."""
        import random
        random.seed(42)

        # Same distribution for both
        control_df = pd.DataFrame({
            "cr_zlib": [random.gauss(0.50, 0.10) for _ in range(150)],
        })
        fraud_df = pd.DataFrame({
            "cr_zlib": [random.gauss(0.50, 0.10) for _ in range(75)],
        })

        results = detect_signal(fraud_df, control_df, "cr_zlib")

        # Should not detect significant signal
        assert results["z_scores"]["mean"] > -1.0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report_signal_detected(self):
        """Test report generation when signal is detected."""
        results = {
            "verdict": {
                "signal_detected": True,
                "signal_strength": "strong",
                "threshold_met": True,
                "statistically_significant": True,
            },
            "baseline": {"mean": 0.65, "std": 0.10, "n": 150},
            "z_scores": {"mean": -3.0, "min": -4.5, "max": -1.5, "count": 75},
            "t_test": {"t_statistic": -15.2, "p_value": 0.00001, "significant": True},
            "effect_size": {"cohens_d": -3.0, "interpretation": "large"},
        }

        report = generate_report(results, "fat_leonard", "control")

        assert "SIGNAL DETECTED" in report
        assert "fat_leonard" in report
        assert "STRONG" in report

    def test_generate_report_no_signal(self):
        """Test report generation when no signal detected."""
        results = {
            "verdict": {
                "signal_detected": False,
                "signal_strength": "none",
                "threshold_met": False,
                "statistically_significant": False,
            },
            "baseline": {"mean": 0.50, "std": 0.10, "n": 150},
            "z_scores": {"mean": -0.5, "min": -1.5, "max": 0.5, "count": 75},
            "t_test": {"t_statistic": -1.2, "p_value": 0.15, "significant": False},
            "effect_size": {"cohens_d": -0.3, "interpretation": "small"},
        }

        report = generate_report(results, "test_cohort", "control")

        assert "NO SIGNAL" in report
        assert "hypothesis does not hold" in report or "No statistically" in report
