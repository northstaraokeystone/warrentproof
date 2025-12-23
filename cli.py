#!/usr/bin/env python3
"""
WarrantProof CLI - Command Line Interface

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This CLI provides access to WarrantProof simulation capabilities.

Usage:
    python cli.py --test                    # Emit test receipt
    python cli.py scenario --run BASELINE   # Run scenario
    python cli.py export --scenario BASELINE --format json

v4.0 User-Friendly Commands:
    python cli.py explain --file data.json  # Plain-language explanations
    python cli.py health                    # System health check
    python cli.py patterns --list           # View known fraud patterns
    python cli.py freshness --check data/   # Check evidence freshness
"""

import argparse
import json
import sys
import time

# Add src to path
sys.path.insert(0, '.')

from src.core import (
    TENANT_ID,
    DISCLAIMER,
    CITATIONS,
    VERSION,
    dual_hash,
    emit_receipt,
    get_citation,
)


def main():
    parser = argparse.ArgumentParser(
        description="WarrantProof CLI - Military Accountability Simulation",
        epilog=f"⚠️ {DISCLAIMER}"
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Emit a test receipt to verify system'
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )

    # Scenario subcommand
    subparsers = parser.add_subparsers(dest='command')

    scenario_parser = subparsers.add_parser('scenario', help='Run simulation scenario')
    scenario_parser.add_argument('--run', type=str, required=True,
                                 choices=['BASELINE', 'SHIPYARD_STRESS', 'CROSS_BRANCH_INTEGRATION',
                                         'FRAUD_DISCOVERY', 'REAL_TIME_OVERSIGHT', 'GODEL'],
                                 help='Scenario to run')
    scenario_parser.add_argument('--cycles', type=int, default=10,
                                 help='Number of cycles (default: 10)')
    scenario_parser.add_argument('--verbose', action='store_true',
                                 help='Verbose output')

    export_parser = subparsers.add_parser('export', help='Export simulation results')
    export_parser.add_argument('--scenario', type=str, required=True,
                              help='Scenario to export')
    export_parser.add_argument('--format', type=str, default='json',
                              choices=['json', 'summary'],
                              help='Export format')
    export_parser.add_argument('--include-citations', action='store_true',
                              help='Include all citations')

    # v4.0 User-Friendly Commands
    explain_parser = subparsers.add_parser('explain', help='Get plain-language explanations')
    explain_parser.add_argument('--file', type=str,
                               help='JSON file with analysis results to explain')
    explain_parser.add_argument('--demo', action='store_true',
                               help='Run demo with sample data')

    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.add_argument('--detailed', action='store_true',
                              help='Show detailed pattern breakdown')

    patterns_parser = subparsers.add_parser('patterns', help='Manage fraud patterns')
    patterns_parser.add_argument('--list', action='store_true',
                                help='List all known patterns')
    patterns_parser.add_argument('--check', type=str,
                                help='Check data file against patterns')
    patterns_parser.add_argument('--domain', type=str,
                                help='Filter patterns by domain')

    freshness_parser = subparsers.add_parser('freshness', help='Check evidence freshness')
    freshness_parser.add_argument('--check', type=str,
                                 help='Check freshness of evidence file')
    freshness_parser.add_argument('--demo', action='store_true',
                                 help='Run demo with sample dates')

    args = parser.parse_args()

    # Handle commands
    if args.version:
        print(f"WarrantProof v{VERSION}")
        print(f"Tenant: {TENANT_ID}")
        print(f"Citations: {len(CITATIONS)}")
        print(f"\n⚠️ {DISCLAIMER}")
        return 0

    if args.test:
        return run_test()

    if args.command == 'scenario':
        return run_scenario(args.run, args.cycles, args.verbose)

    if args.command == 'export':
        return run_export(args.scenario, args.format, args.include_citations)

    if args.command == 'explain':
        return run_explain(args.file, args.demo)

    if args.command == 'health':
        return run_health(args.detailed)

    if args.command == 'patterns':
        return run_patterns(args.list, args.check, args.domain)

    if args.command == 'freshness':
        return run_freshness(args.check, args.demo)

    # Default: show help
    parser.print_help()
    return 0


def run_test():
    """Emit a test receipt to verify system."""
    print(f"# WarrantProof CLI Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test dual_hash
    h = dual_hash("test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"

    # Test citation
    citation = get_citation("SHANNON_1948")
    assert "url" in citation, "Citation must include URL"

    # Emit test receipt (goes to stdout)
    receipt = emit_receipt("test", {
        "tenant_id": TENANT_ID,
        "message": "CLI test receipt",
        "citation": citation,
        "simulation_flag": DISCLAIMER,
    })

    print(f"\n# Test receipt emitted successfully", file=sys.stderr)
    print(f"# Receipt type: {receipt['receipt_type']}", file=sys.stderr)
    print(f"# Payload hash: {receipt['payload_hash'][:32]}...", file=sys.stderr)

    return 0


def run_scenario(scenario: str, cycles: int, verbose: bool):
    """Run a simulation scenario."""
    from src.sim import run_simulation, SimConfig, export_results

    print(f"# Running scenario: {scenario}", file=sys.stderr)
    print(f"# Cycles: {cycles}", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    config = SimConfig(
        n_cycles=cycles,
        n_transactions_per_cycle=100,  # Reduced for CLI
        scenario=scenario
    )

    t0 = time.time()
    result = run_simulation(config)
    elapsed = time.time() - t0

    # Print summary to stderr
    print(f"\n# Scenario completed in {elapsed:.2f}s", file=sys.stderr)
    print(f"# Receipts: {len(result.receipts)}", file=sys.stderr)
    print(f"# Detections: {len(result.detections)}", file=sys.stderr)
    print(f"# Violations: {len(result.violations)}", file=sys.stderr)

    if result.scenario_results:
        passed = result.scenario_results.get("passed", False)
        print(f"# Passed: {passed}", file=sys.stderr)

    if verbose:
        # Print detailed results
        export = export_results(result)
        print(json.dumps(export, indent=2))

    return 0 if result.scenario_results.get("passed", False) else 1


def run_export(scenario: str, format: str, include_citations: bool):
    """Export simulation results."""
    from src.sim import run_simulation, SimConfig, export_results

    print(f"# Exporting scenario: {scenario}", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    config = SimConfig(
        n_cycles=10,
        n_transactions_per_cycle=100,
        scenario=scenario
    )

    result = run_simulation(config)
    export = export_results(result)

    if include_citations:
        export["citations"] = CITATIONS

    if format == 'json':
        print(json.dumps(export, indent=2))
    else:
        # Summary format
        print(f"Scenario: {export.get('scenario', 'unknown')}")
        print(f"Passed: {export.get('passed', False)}")
        print(f"Total Receipts: {export.get('summary', {}).get('total_receipts', 0)}")
        print(f"Detections: {export.get('summary', {}).get('detections', 0)}")
        print(f"Simulated Spend: ${export.get('summary', {}).get('total_simulated_spend_usd', 0):,.2f}")
        print(f"\n⚠️ {DISCLAIMER}")

    return 0


def run_explain(file_path: str, demo: bool):
    """Generate plain-language explanations."""
    from src.insight import (
        explain_anomaly,
        explain_compression_result,
        generate_executive_summary,
    )

    print(f"# WarrantProof Explain - Plain Language Analysis", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    if demo:
        # Demo with sample results
        sample_results = [
            {
                "anomaly_type": "compression_failure",
                "fraud_likelihood": 0.75,
                "compression_ratio": 0.42,
            },
            {
                "classification": "suspicious",
                "compression_ratio": 0.55,
                "entropy_score": 4.5,
                "coherence_score": 0.45,
                "fraud_likelihood": 0.6,
            },
        ]

        print("\n--- Demo: Explaining Sample Anomaly ---\n")
        explanation = explain_anomaly(sample_results[0])
        print(f"Title: {explanation['title']}")
        print(f"\nSummary: {explanation['summary']}")
        print(f"\nWhat This Means:\n{explanation['what_it_means']}")
        print(f"\nSuggested Action:\n{explanation['suggested_action']}")
        print(f"\nConfidence: {explanation['confidence_level']}")

        print("\n\n--- Demo: Executive Summary ---\n")
        summary = generate_executive_summary(sample_results)
        print(f"Status: {summary['status'].upper()}")
        print(f"\n{summary['status_message']}")
        print(f"\nRecommendation:\n{summary['recommendation']}")

        return 0

    if file_path:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            summary = generate_executive_summary(data)
            print(json.dumps(summary, indent=2))
        else:
            explanation = explain_anomaly(data)
            print(json.dumps(explanation, indent=2))

        return 0

    print("Use --demo for demo or --file <path> for file analysis", file=sys.stderr)
    return 1


def run_health(detailed: bool):
    """Check system health."""
    from src.fitness import (
        get_system_health,
        explain_fitness_for_users,
        prune_harmful_patterns,
    )

    print(f"# WarrantProof System Health Check", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Get user-friendly explanation
    explanation = explain_fitness_for_users()
    print(f"\n{explanation['headline']}")
    print(f"\n{explanation['explanation']}")
    print(f"\nEffectiveness: {explanation['effectiveness_percent']}%")

    if detailed:
        print("\n--- Detailed Health Report ---")
        health = get_system_health()
        print(f"\nStatus: {health['status'].upper()}")
        print(f"Overall Fitness: {health['overall_fitness']:.4f}")

        breakdown = health.get('pattern_breakdown', {})
        print(f"\nPattern Breakdown:")
        print(f"  Total: {breakdown.get('total', 0)}")
        print(f"  Excellent: {breakdown.get('excellent', 0)}")
        print(f"  Good: {breakdown.get('good', 0)}")
        print(f"  Harmful: {breakdown.get('harmful', 0)}")

        prune = prune_harmful_patterns()
        if prune['action_needed']:
            print(f"\n⚠️ {prune['summary']}")

    return 0


def run_patterns(list_patterns: bool, check_file: str, domain: str):
    """Manage fraud patterns."""
    from src.learner import (
        get_library_summary,
        match_patterns,
        explain_pattern_for_users,
    )

    print(f"# WarrantProof Pattern Library", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    if list_patterns:
        summary = get_library_summary()
        print(f"\nKnown Fraud Patterns: {summary['total_patterns']}")
        print(f"Domains Covered: {', '.join(summary['domains_covered'])}")
        print(f"\n--- Pattern List ---")

        for p in summary['patterns']:
            print(f"\n• {p['name']} ({p['pattern_id']})")
            print(f"  Source: {p['source']}")
            print(f"  Domains: {', '.join(p['domains'])}")
            print(f"  Transferability: {p['transferability']:.0%}")

        return 0

    if check_file:
        with open(check_file, 'r') as f:
            data = json.load(f)

        result = match_patterns(data, domain=domain)

        print(f"\nPatterns Checked: {result['patterns_checked']}")
        print(f"Matches Found: {result['matches_found']}")
        print(f"Risk Level: {result['risk_level'].upper()}")

        if result['matches']:
            print("\n--- Matching Patterns ---")
            for match in result['matches'][:5]:
                print(f"\n• {match['pattern_name']}")
                print(f"  Confidence: {match['confidence']:.0%}")
                print(f"  Source Case: {match['source_case']}")

                # Get user-friendly explanation
                explanation = explain_pattern_for_users(match)
                print(f"  {explanation['explanation'][:200]}...")

        return 0

    print("Use --list to view patterns or --check <file> to analyze data", file=sys.stderr)
    return 1


def run_freshness(check_file: str, demo: bool):
    """Check evidence freshness."""
    from datetime import datetime, timedelta
    from src.freshness import (
        assess_freshness,
        assess_evidence_set_freshness,
        explain_freshness_for_users,
        get_refresh_priorities,
    )

    print(f"# WarrantProof Evidence Freshness Check", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    if demo:
        print("\n--- Demo: Evidence Freshness ---\n")

        # Demo with various ages
        demo_dates = [
            ("15 days old", datetime.utcnow() - timedelta(days=15), "general"),
            ("45 days old", datetime.utcnow() - timedelta(days=45), "general"),
            ("100 days old", datetime.utcnow() - timedelta(days=100), "general"),
            ("20 days old (price data)", datetime.utcnow() - timedelta(days=20), "price_data"),
        ]

        for label, ts, dtype in demo_dates:
            result = assess_freshness(ts, dtype)
            explanation = explain_freshness_for_users(result)
            print(f"• {label}: {explanation['headline']}")
            print(f"  Confidence: {result['confidence_factor']:.0%}")
            print(f"  {explanation['explanation'][:100]}...")
            print()

        return 0

    if check_file:
        with open(check_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            result = assess_evidence_set_freshness(data)
            print(f"\nEvidence Items: {result['evidence_count']}")
            print(f"Overall Freshness: {result['overall_freshness'].upper()}")
            print(f"Confidence Factor: {result['confidence_factor']:.0%}")
            print(f"\n{result['recommendation']}")

            priorities = get_refresh_priorities(data)
            if priorities['items_needing_refresh'] > 0:
                print(f"\n⚠️ {priorities['items_needing_refresh']} items need refresh")

        return 0

    print("Use --demo for demo or --check <file> for file analysis", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
