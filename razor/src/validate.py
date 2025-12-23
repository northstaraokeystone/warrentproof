"""
RAZOR Validate Module - Statistical Signal Detection

Statistical validation that fraud cohorts are distinguishable from controls.
Uses Z-scores and T-tests to detect whether fraud records exhibit
significantly lower Kolmogorov complexity than control records.

THE HYPOTHESIS:
  H0: mu_fraud = mu_control (no difference in compression)
  Ha: mu_fraud < mu_control (fraud is MORE compressible)

  If p < 0.05 AND mean(Z_fraud) < -2.0 => "SIGNAL DETECTED"

STATISTICAL LOGIC:
  1. Group control cohort by NAICS/PSC matching fraud cohort
  2. Calculate control baseline: mu_control, sigma_control for CR_zlib
  3. Calculate Z-scores for fraud records: Z = (CR_fraud - mu_control) / sigma_control
  4. T-test: one-tailed, alpha = 0.05
  5. Effect size: Cohen's d >= 0.5 for "meaningful" signal
"""

import math
from typing import Dict, List, Optional, Tuple

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

from .core import (
    emit_receipt,
    StopRule,
    stoprule_no_signal,
    stoprule_insufficient_control,
    stoprule_degenerate_baseline,
    Z_SCORE_THRESHOLD,
    ALPHA_LEVEL,
    MIN_CONTROL_SIZE,
    MIN_POWER,
    MIN_EFFECT_SIZE,
)

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def group_by_category(
    df: "pd.DataFrame",
    category: str = "naics_code",
) -> Dict[str, "pd.DataFrame"]:
    """
    Group records by NAICS/PSC code for apples-to-apples comparison.

    Args:
        df: DataFrame with records
        category: Column to group by (naics_code or psc_code)

    Returns:
        Dict mapping category value to DataFrame subset
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for grouping")

    if category not in df.columns:
        return {"all": df}

    groups = {}
    for value, group_df in df.groupby(category):
        if len(group_df) > 0:
            groups[str(value)] = group_df

    return groups


def calculate_baseline(
    control_df: "pd.DataFrame",
    metric: str = "cr_zlib",
) -> Dict[str, float]:
    """
    Calculate baseline statistics for control cohort.

    Args:
        control_df: Control cohort DataFrame
        metric: Column name for compression metric

    Returns:
        Dict with mean, std, median, q25, q75
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for baseline calculation")

    if metric not in control_df.columns:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "n": 0,
            "valid": False,
        }

    values = control_df[metric].dropna()

    if len(values) < MIN_CONTROL_SIZE:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "n": len(values),
            "valid": False,
        }

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "n": len(values),
        "valid": True,
    }


def calculate_z_scores(
    fraud_df: "pd.DataFrame",
    baseline: Dict[str, float],
    metric: str = "cr_zlib",
) -> "pd.Series":
    """
    Calculate Z-scores for fraud cohort against control baseline.

    Z = (x - mu) / sigma

    Args:
        fraud_df: Fraud cohort DataFrame
        baseline: Control baseline statistics
        metric: Column name for compression metric

    Returns:
        Series of Z-scores
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for Z-score calculation")

    if metric not in fraud_df.columns:
        return pd.Series([0.0] * len(fraud_df))

    if baseline["std"] == 0 or not baseline["valid"]:
        # Degenerate baseline - all values identical
        return pd.Series([0.0] * len(fraud_df))

    values = fraud_df[metric]
    z_scores = (values - baseline["mean"]) / baseline["std"]

    return z_scores


def run_t_test(
    fraud_values: List[float],
    control_values: List[float],
    alternative: str = "less",
) -> Dict[str, float]:
    """
    Run independent samples T-test.

    H0: mu_fraud = mu_control
    Ha: mu_fraud < mu_control (fraud more compressible)

    Args:
        fraud_values: List of fraud compression ratios
        control_values: List of control compression ratios
        alternative: 'less' (one-tailed) or 'two-sided'

    Returns:
        Dict with t_statistic, p_value, significant
    """
    if not HAS_SCIPY:
        # Fallback: simple comparison without formal test
        fraud_mean = sum(fraud_values) / len(fraud_values) if fraud_values else 0
        control_mean = sum(control_values) / len(control_values) if control_values else 0
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "fraud_mean": fraud_mean,
            "control_mean": control_mean,
            "scipy_available": False,
        }

    if len(fraud_values) < 2 or len(control_values) < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "insufficient_samples": True,
        }

    result = stats.ttest_ind(
        fraud_values,
        control_values,
        alternative=alternative,
    )

    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < ALPHA_LEVEL,
        "fraud_mean": float(sum(fraud_values) / len(fraud_values)),
        "control_mean": float(sum(control_values) / len(control_values)),
        "scipy_available": True,
    }


def run_mann_whitney(
    fraud_values: List[float],
    control_values: List[float],
    alternative: str = "less",
) -> Dict[str, float]:
    """
    Run Mann-Whitney U test (non-parametric alternative).

    Use when normality assumption violated.

    Args:
        fraud_values: List of fraud compression ratios
        control_values: List of control compression ratios
        alternative: 'less' (one-tailed) or 'two-sided'

    Returns:
        Dict with u_statistic, p_value, significant
    """
    if not HAS_SCIPY:
        return {
            "u_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "scipy_available": False,
        }

    if len(fraud_values) < 2 or len(control_values) < 2:
        return {
            "u_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "insufficient_samples": True,
        }

    result = stats.mannwhitneyu(
        fraud_values,
        control_values,
        alternative=alternative,
    )

    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < ALPHA_LEVEL,
        "scipy_available": True,
    }


def calculate_cohens_d(
    fraud_values: List[float],
    control_values: List[float],
) -> float:
    """
    Calculate Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std

    Args:
        fraud_values: Fraud compression ratios
        control_values: Control compression ratios

    Returns:
        Cohen's d effect size
    """
    if len(fraud_values) < 2 or len(control_values) < 2:
        return 0.0

    n1, n2 = len(fraud_values), len(control_values)
    mean1 = sum(fraud_values) / n1
    mean2 = sum(control_values) / n2

    var1 = sum((x - mean1) ** 2 for x in fraud_values) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in control_values) / (n2 - 1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def detect_signal(
    fraud_df: "pd.DataFrame",
    control_df: "pd.DataFrame",
    metric: str = "cr_zlib",
) -> Dict:
    """
    Full signal detection pipeline.

    Steps:
      1. Calculate control baseline
      2. Calculate Z-scores for fraud cohort
      3. Run T-test
      4. Calculate effect size
      5. Determine verdict

    Args:
        fraud_df: Fraud cohort DataFrame with complexity metrics
        control_df: Control cohort DataFrame with complexity metrics
        metric: Column name for compression metric

    Returns:
        Dict with full analysis results and verdict
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for signal detection")

    # Calculate baseline
    baseline = calculate_baseline(control_df, metric)

    if not baseline["valid"]:
        stoprule_insufficient_control("control", baseline["n"])

    if baseline["std"] == 0:
        stoprule_degenerate_baseline("control")

    # Calculate Z-scores
    z_scores = calculate_z_scores(fraud_df, baseline, metric)
    mean_z = float(z_scores.mean()) if len(z_scores) > 0 else 0.0

    # Extract values for statistical tests
    fraud_values = fraud_df[metric].dropna().tolist()
    control_values = control_df[metric].dropna().tolist()

    # Run T-test
    t_test = run_t_test(fraud_values, control_values, alternative="less")

    # Run Mann-Whitney as backup
    mann_whitney = run_mann_whitney(fraud_values, control_values, alternative="less")

    # Calculate effect size
    cohens_d = calculate_cohens_d(fraud_values, control_values)

    # Determine verdict
    signal_detected = (
        t_test["significant"]
        and mean_z < Z_SCORE_THRESHOLD
    )

    signal_strength = "none"
    if signal_detected:
        if abs(cohens_d) >= 0.8:
            signal_strength = "strong"
        elif abs(cohens_d) >= MIN_EFFECT_SIZE:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"

    result = {
        "baseline": baseline,
        "z_scores": {
            "mean": mean_z,
            "min": float(z_scores.min()) if len(z_scores) > 0 else 0.0,
            "max": float(z_scores.max()) if len(z_scores) > 0 else 0.0,
            "count": len(z_scores),
        },
        "t_test": t_test,
        "mann_whitney": mann_whitney,
        "effect_size": {
            "cohens_d": cohens_d,
            "interpretation": interpret_cohens_d(cohens_d),
        },
        "verdict": {
            "signal_detected": signal_detected,
            "signal_strength": signal_strength,
            "threshold_met": mean_z < Z_SCORE_THRESHOLD,
            "statistically_significant": t_test["significant"],
        },
    }

    # Emit signal receipt (convert numpy types to native Python)
    emit_receipt("signal", {
        "signal_detected": bool(signal_detected),
        "signal_strength": signal_strength,
        "mean_z_score": float(mean_z),
        "t_statistic": float(t_test["t_statistic"]),
        "p_value": float(t_test["p_value"]),
        "cohens_d": float(cohens_d),
        "fraud_n": len(fraud_values),
        "control_n": len(control_values),
    }, to_stdout=False)

    # Check for no signal (informational, not fatal)
    if not signal_detected:
        stoprule_no_signal(mean_z, Z_SCORE_THRESHOLD)

    return result


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "large"
    elif abs_d >= 0.5:
        return "medium"
    elif abs_d >= 0.2:
        return "small"
    else:
        return "negligible"


def generate_report(
    results: Dict,
    fraud_cohort_name: str = "unknown",
    control_cohort_name: str = "control",
) -> str:
    """
    Generate human-readable summary report.

    Args:
        results: Detection results dict from detect_signal
        fraud_cohort_name: Name of fraud cohort
        control_cohort_name: Name of control cohort

    Returns:
        Formatted report string
    """
    verdict = results["verdict"]
    baseline = results["baseline"]
    z_scores = results["z_scores"]
    t_test = results["t_test"]
    effect = results["effect_size"]

    status = "SIGNAL DETECTED" if verdict["signal_detected"] else "NO SIGNAL"

    report = f"""
================================================================================
RAZOR VALIDATION REPORT: {fraud_cohort_name}
================================================================================

VERDICT: {status}
Signal Strength: {verdict['signal_strength'].upper()}

HYPOTHESIS:
  H0: mu_fraud = mu_control (no difference in compression)
  Ha: mu_fraud < mu_control (fraud is MORE compressible)

BASELINE (Control Cohort: {control_cohort_name}):
  Mean CR: {baseline['mean']:.4f}
  Std CR:  {baseline['std']:.4f}
  N:       {baseline['n']}

FRAUD COHORT Z-SCORES:
  Mean Z:  {z_scores['mean']:.2f}
  Min Z:   {z_scores['min']:.2f}
  Max Z:   {z_scores['max']:.2f}
  N:       {z_scores['count']}

STATISTICAL TESTS:
  T-test:
    t-statistic: {t_test['t_statistic']:.4f}
    p-value:     {t_test['p_value']:.6f}
    Significant: {t_test['significant']} (alpha = {ALPHA_LEVEL})

  Effect Size:
    Cohen's d:   {effect['cohens_d']:.4f} ({effect['interpretation']})

THRESHOLDS:
  Z-score threshold: {Z_SCORE_THRESHOLD}
  Threshold met:     {verdict['threshold_met']}

INTERPRETATION:
"""

    if verdict["signal_detected"]:
        report += f"""
  The fraud cohort exhibits LOWER Kolmogorov complexity than the control
  cohort, consistent with the hypothesis that fraudulent procurement data
  is more compressible due to coordination and templated patterns.

  Mean Z-score of {z_scores['mean']:.2f} indicates fraud records are
  approximately {abs(z_scores['mean']):.1f} standard deviations more
  compressible than control records.
"""
    else:
        report += f"""
  No statistically significant difference detected between fraud and
  control cohorts. This may indicate:
    - The hypothesis does not hold for this fraud type
    - Insufficient sample size
    - Control cohort contamination
    - Fraud pattern too sophisticated to detect via compression
"""

    report += """
================================================================================
"""

    return report


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("# RAZOR Validate Module", file=sys.stderr)

    if not HAS_PANDAS:
        print("# SKIP: pandas not installed", file=sys.stderr)
        sys.exit(0)

    if not HAS_SCIPY:
        print("# WARNING: scipy not installed, statistical tests limited", file=sys.stderr)

    # Create synthetic test data
    import random

    # Control cohort: higher compression ratios (less compressible)
    control_data = {
        "cr_zlib": [random.gauss(0.65, 0.10) for _ in range(100)],
        "naics_code": ["123456"] * 100,
    }
    control_df = pd.DataFrame(control_data)

    # Fraud cohort: lower compression ratios (more compressible)
    fraud_data = {
        "cr_zlib": [random.gauss(0.35, 0.08) for _ in range(50)],
        "naics_code": ["123456"] * 50,
    }
    fraud_df = pd.DataFrame(fraud_data)

    # Run detection
    results = detect_signal(fraud_df, control_df, "cr_zlib")

    print(f"# Signal detected: {results['verdict']['signal_detected']}", file=sys.stderr)
    print(f"# Signal strength: {results['verdict']['signal_strength']}", file=sys.stderr)
    print(f"# Mean Z-score: {results['z_scores']['mean']:.2f}", file=sys.stderr)
    print(f"# Cohen's d: {results['effect_size']['cohens_d']:.2f}", file=sys.stderr)

    # Should detect signal in synthetic data
    assert results["verdict"]["signal_detected"], "Should detect signal in synthetic fraud data"

    print("# PASS: RAZOR validate module self-test", file=sys.stderr)
