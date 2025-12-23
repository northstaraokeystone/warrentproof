#!/usr/bin/env python3
"""
WarrantProof CLI - Command Line Interface

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This CLI provides access to WarrantProof simulation capabilities.

Usage:
    python cli.py --test                    # Emit test receipt
    python cli.py scenario --run BASELINE   # Run scenario
    python cli.py export --scenario BASELINE --format json
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


if __name__ == "__main__":
    sys.exit(main())
