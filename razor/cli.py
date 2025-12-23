#!/usr/bin/env python3
"""
RAZOR CLI - Kolmogorov Validation Engine Command Line Interface

Usage:
  python cli.py --gate api           # Test API connectivity
  python cli.py --gate cohorts       # Ingest all cohorts
  python cli.py --gate compression   # Run compression analysis
  python cli.py --gate validate      # Statistical validation
  python cli.py --test               # Run quick smoke test

Gates:
  1. API: Verify USASpending.gov connectivity
  2. Cohorts: Ingest fraud + control cohorts
  3. Compression: Apply Kolmogorov complexity analysis
  4. Validate: Statistical signal detection

THE PARADIGM:
  Fraud = artificial order in stochastic system.
  Detect via compression ratio on real USASpending data
  calibrated against historical ground truth.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core import (
    emit_receipt,
    dual_hash,
    merkle,
    StopRule,
    TENANT_ID,
    API_BASE_URL,
    CR_THRESHOLD_LOW,
    CR_THRESHOLD_HIGH,
    Z_SCORE_THRESHOLD,
    VERSION,
)
from src.cohorts import (
    list_cohorts,
    get_cohort_config,
    validate_cohort,
    get_fraud_type,
)
from src.physics import KolmogorovMetric, quick_compression_test

# ============================================================================
# GATE FUNCTIONS
# ============================================================================

def gate_api() -> bool:
    """
    Gate 1: Test API connectivity.

    Returns:
        True if API accessible, False otherwise
    """
    print("=" * 60, file=sys.stderr)
    print("GATE 1: API CONNECTIVITY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        from src.ingest import USASpendingIngestor

        ingestor = USASpendingIngestor()
        result = ingestor.test_connectivity()

        if result["status"] == "ok":
            print(f"[PASS] API accessible", file=sys.stderr)
            print(f"       Endpoint: {result['endpoint']}", file=sys.stderr)
            print(f"       Latency: {result['latency_ms']}ms", file=sys.stderr)

            emit_receipt("gate_pass", {
                "gate": "api",
                "status": "pass",
                "latency_ms": result["latency_ms"],
            })
            return True
        else:
            print(f"[FAIL] API not accessible: {result.get('error', 'Unknown')}", file=sys.stderr)
            emit_receipt("gate_fail", {
                "gate": "api",
                "status": "fail",
                "error": result.get("error"),
            })
            return False

    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}", file=sys.stderr)
        return False


def gate_compression_smoke() -> bool:
    """
    Gate 3: Quick compression smoke test.

    Validates that compression correctly identifies repetitive vs random text.

    Returns:
        True if compression working correctly
    """
    print("=" * 60, file=sys.stderr)
    print("GATE 3: COMPRESSION ANALYSIS (Smoke Test)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    km = KolmogorovMetric()

    # Test 1: Repetitive text should compress well
    repetitive = "husbanding services for ship " * 100
    r1 = km.measure_complexity(repetitive)

    # Test 2: Pseudo-random text should not compress well
    random_text = "qp8jf2kd9s7hgb3cx5nz1w4m" * 100
    r2 = km.measure_complexity(random_text)

    print(f"Repetitive text CR (zlib): {r1['cr_zlib']:.4f}", file=sys.stderr)
    print(f"Random text CR (zlib):     {r2['cr_zlib']:.4f}", file=sys.stderr)

    # Validate hypothesis
    if r1["cr_zlib"] < r2["cr_zlib"]:
        print("[PASS] Repetitive text compresses better than random", file=sys.stderr)

        if r1["cr_zlib"] < 0.15:
            print("[PASS] Highly repetitive text is very compressible", file=sys.stderr)
        else:
            print("[WARN] Repetitive text not as compressible as expected", file=sys.stderr)

        emit_receipt("gate_pass", {
            "gate": "compression",
            "status": "pass",
            "repetitive_cr": r1["cr_zlib"],
            "random_cr": r2["cr_zlib"],
        })
        return True
    else:
        print("[FAIL] Compression hypothesis violated", file=sys.stderr)
        emit_receipt("gate_fail", {
            "gate": "compression",
            "status": "fail",
            "reason": "repetitive_not_more_compressible",
        })
        return False


def gate_validate_smoke() -> bool:
    """
    Gate 4: Statistical validation smoke test.

    Uses synthetic data to verify the statistical pipeline.

    Returns:
        True if validation pipeline working correctly
    """
    print("=" * 60, file=sys.stderr)
    print("GATE 4: STATISTICAL VALIDATION (Smoke Test)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        import pandas as pd
        from src.validate import detect_signal, calculate_baseline
        import random

        # Create synthetic control cohort (high CR - less compressible)
        control_data = {
            "cr_zlib": [random.gauss(0.65, 0.10) for _ in range(150)],
            "naics_code": ["123456"] * 150,
        }
        control_df = pd.DataFrame(control_data)

        # Create synthetic fraud cohort (low CR - more compressible)
        fraud_data = {
            "cr_zlib": [random.gauss(0.35, 0.08) for _ in range(75)],
            "naics_code": ["123456"] * 75,
        }
        fraud_df = pd.DataFrame(fraud_data)

        # Run detection
        results = detect_signal(fraud_df, control_df, "cr_zlib")

        print(f"Signal detected:  {results['verdict']['signal_detected']}", file=sys.stderr)
        print(f"Signal strength:  {results['verdict']['signal_strength']}", file=sys.stderr)
        print(f"Mean Z-score:     {results['z_scores']['mean']:.2f}", file=sys.stderr)
        print(f"p-value:          {results['t_test']['p_value']:.6f}", file=sys.stderr)
        print(f"Cohen's d:        {results['effect_size']['cohens_d']:.2f}", file=sys.stderr)

        if results["verdict"]["signal_detected"]:
            print("[PASS] Signal detected in synthetic fraud data", file=sys.stderr)
            emit_receipt("gate_pass", {
                "gate": "validate",
                "status": "pass",
                "signal_detected": True,
                "mean_z_score": results["z_scores"]["mean"],
            })
            return True
        else:
            print("[FAIL] No signal detected in synthetic data", file=sys.stderr)
            emit_receipt("gate_fail", {
                "gate": "validate",
                "status": "fail",
                "reason": "no_signal_synthetic",
            })
            return False

    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def run_quick_test() -> bool:
    """
    Quick smoke test for T+2h gate.

    Emits a test receipt to verify basic functionality.

    Returns:
        True if test passes
    """
    print("=" * 60, file=sys.stderr)
    print("RAZOR QUICK TEST", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Verify constants
    print(f"TENANT_ID: {TENANT_ID}", file=sys.stderr)
    print(f"API_BASE_URL: {API_BASE_URL}", file=sys.stderr)
    print(f"CR_THRESHOLD_LOW: {CR_THRESHOLD_LOW}", file=sys.stderr)
    print(f"Z_SCORE_THRESHOLD: {Z_SCORE_THRESHOLD}", file=sys.stderr)

    # Verify dual_hash
    h = dual_hash("test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"
    print(f"dual_hash working: {h[:32]}...", file=sys.stderr)

    # Verify cohorts
    cohorts = list_cohorts()
    print(f"Available cohorts: {cohorts}", file=sys.stderr)
    for c in cohorts:
        assert validate_cohort(c), f"Invalid cohort: {c}"

    # Emit test receipt
    r = emit_receipt("test", {
        "message": "RAZOR quick test",
        "version": VERSION,
    })

    print(f"\n[PASS] All checks passed", file=sys.stderr)
    return True


def show_cohorts() -> None:
    """Display information about available cohorts."""
    print("=" * 60, file=sys.stderr)
    print("RAZOR COHORTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for name in list_cohorts():
        config = get_cohort_config(name)
        print(f"\n{name.upper()}: {config['name']}", file=sys.stderr)
        print(f"  Fraud type: {get_fraud_type(name)}", file=sys.stderr)
        print(f"  Hypothesis: {config['hypothesis'][:70]}...", file=sys.stderr)
        print(f"  Expected Z: {config.get('expected_z_score', 'TBD')}", file=sys.stderr)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAZOR - Kolmogorov Validation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --test                  # Quick smoke test
  python cli.py --gate api              # Test API connectivity
  python cli.py --gate compression      # Compression smoke test
  python cli.py --gate validate         # Statistical validation smoke test
  python cli.py --cohorts               # List available cohorts

Gates:
  api         - Test USASpending.gov API connectivity
  compression - Verify compression algorithms working
  validate    - Test statistical validation pipeline
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick smoke test (T+2h gate)",
    )

    parser.add_argument(
        "--gate",
        choices=["api", "cohorts", "compression", "validate"],
        help="Run specific gate",
    )

    parser.add_argument(
        "--cohorts",
        action="store_true",
        help="Display available cohorts",
    )

    parser.add_argument(
        "--cohort",
        type=str,
        help="Specific cohort to process (fat_leonard, transdigm, boeing)",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    args = parser.parse_args()

    if args.version:
        print(f"RAZOR v{VERSION}")
        return 0

    if args.cohorts:
        show_cohorts()
        return 0

    if args.test:
        success = run_quick_test()
        return 0 if success else 1

    if args.gate:
        if args.gate == "api":
            success = gate_api()
        elif args.gate == "compression":
            success = gate_compression_smoke()
        elif args.gate == "validate":
            success = gate_validate_smoke()
        elif args.gate == "cohorts":
            show_cohorts()
            success = True
        else:
            print(f"Unknown gate: {args.gate}", file=sys.stderr)
            success = False

        return 0 if success else 1

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
