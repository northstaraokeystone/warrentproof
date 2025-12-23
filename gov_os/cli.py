#!/usr/bin/env python3
"""
Gov-OS CLI - Unified Command Line Interface for Federal Fraud Detection

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Usage:
    gov-os defense simulate --cycles 1000 --seed 42
    gov-os medicaid scenario PROVIDER_RING
    gov-os validate --domain all
"""

import argparse
import json
import sys
from typing import Optional

from src.core.constants import DISCLAIMER
from src.core.harness import run_simulation, run_all_scenarios, ScenarioResult
from src.core.domain import load_domain, list_domains


def print_disclaimer():
    """Print simulation disclaimer."""
    print("=" * 70)
    print(DISCLAIMER)
    print("=" * 70)
    print()


def cmd_simulate(args):
    """Run Monte Carlo simulation for a domain."""
    print_disclaimer()
    print(f"Running simulation for domain: {args.domain}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Seed: {args.seed}")
    print()

    state = run_simulation(
        domain=args.domain,
        n_cycles=args.cycles,
        seed=args.seed,
    )

    print("Simulation Results:")
    print(f"  Total receipts: {len(state.receipts)}")
    print(f"  Violations detected: {len(state.violations)}")
    print(f"  Cycles completed: {state.cycle}")

    if args.output:
        results = {
            "domain": args.domain,
            "cycles": args.cycles,
            "seed": args.seed,
            "total_receipts": len(state.receipts),
            "violations": len(state.violations),
            "simulation_flag": DISCLAIMER,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results written to: {args.output}")

    return 0


def cmd_scenario(args):
    """Run a specific scenario for a domain."""
    print_disclaimer()
    print(f"Running scenario: {args.scenario}")
    print(f"  Domain: {args.domain}")
    print()

    # Import domain-specific scenarios
    if args.domain == "defense":
        from src.modules.defense.scenarios import run_defense_scenario
        result = run_defense_scenario(args.scenario)
    elif args.domain == "medicaid":
        from src.modules.medicaid.scenarios import run_medicaid_scenario
        result = run_medicaid_scenario(args.scenario)
    else:
        print(f"Unknown domain: {args.domain}")
        return 1

    print("Scenario Results:")
    print(f"  Name: {result.name}")
    print(f"  Passed: {result.passed}")
    print(f"  Message: {result.message}")
    if result.metrics:
        print("  Metrics:")
        for key, value in result.metrics.items():
            print(f"    {key}: {value}")

    return 0 if result.passed else 1


def cmd_scenarios(args):
    """Run all scenarios for a domain."""
    print_disclaimer()
    print(f"Running all scenarios for domain: {args.domain}")
    print()

    results = run_all_scenarios(args.domain)

    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print("=" * 50)
    print(f"Results: {passed}/{total} scenarios passed")
    print("=" * 50)

    for name, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  [{status}] {name}: {result.message}")

    if args.output:
        output_data = {
            "domain": args.domain,
            "passed": passed,
            "total": total,
            "scenarios": {
                name: {
                    "passed": r.passed,
                    "message": r.message,
                    "metrics": r.metrics,
                }
                for name, r in results.items()
            },
            "simulation_flag": DISCLAIMER,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {args.output}")

    return 0 if passed == total else 1


def cmd_validate(args):
    """Validate domain configuration and imports."""
    print_disclaimer()

    domains = [args.domain] if args.domain != "all" else list_domains()
    errors = []

    for domain in domains:
        print(f"Validating domain: {domain}")
        try:
            config = load_domain(domain)
            print(f"  Config loaded: {config.name}")
            print(f"  Node key: {config.node_key}")
            print(f"  Edge key: {config.edge_key}")

            # Try to get volatility
            vol = config.volatility()
            print(f"  Volatility index: {vol.current():.4f}")
            print(f"  ✓ Domain {domain} validated successfully")
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            errors.append((domain, str(e)))
        print()

    if errors:
        print("Validation Errors:")
        for domain, error in errors:
            print(f"  {domain}: {error}")
        return 1

    print("All domains validated successfully!")
    return 0


def cmd_list(args):
    """List available domains and scenarios."""
    print_disclaimer()

    if args.what == "domains":
        print("Available domains:")
        for domain in list_domains():
            config = load_domain(domain)
            print(f"  - {domain}: {config.description}")

    elif args.what == "scenarios":
        domain = args.domain or "defense"
        print(f"Available scenarios for {domain}:")

        if domain == "defense":
            from src.modules.defense.scenarios import DEFENSE_SCENARIOS
            for scenario in DEFENSE_SCENARIOS:
                print(f"  - {scenario}")
        elif domain == "medicaid":
            from src.modules.medicaid.scenarios import MEDICAID_SCENARIOS
            for scenario in MEDICAID_SCENARIOS:
                print(f"  - {scenario}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="gov-os",
        description="Gov-OS: Universal Federal Fraud Detection Operating System",
        epilog=DISCLAIMER,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Defense subcommand
    defense_parser = subparsers.add_parser("defense", help="Defense domain operations")
    defense_sub = defense_parser.add_subparsers(dest="action")

    defense_sim = defense_sub.add_parser("simulate", help="Run simulation")
    defense_sim.add_argument("--cycles", type=int, default=100, help="Number of cycles")
    defense_sim.add_argument("--seed", type=int, default=42, help="Random seed")
    defense_sim.add_argument("--output", "-o", type=str, help="Output JSON file")
    defense_sim.set_defaults(func=lambda a: cmd_simulate(argparse.Namespace(domain="defense", **vars(a))))

    defense_scenario = defense_sub.add_parser("scenario", help="Run specific scenario")
    defense_scenario.add_argument("scenario", type=str, help="Scenario name")
    defense_scenario.set_defaults(func=lambda a: cmd_scenario(argparse.Namespace(domain="defense", **vars(a))))

    defense_all = defense_sub.add_parser("scenarios", help="Run all scenarios")
    defense_all.add_argument("--output", "-o", type=str, help="Output JSON file")
    defense_all.set_defaults(func=lambda a: cmd_scenarios(argparse.Namespace(domain="defense", **vars(a))))

    # Medicaid subcommand
    medicaid_parser = subparsers.add_parser("medicaid", help="Medicaid domain operations")
    medicaid_sub = medicaid_parser.add_subparsers(dest="action")

    medicaid_sim = medicaid_sub.add_parser("simulate", help="Run simulation")
    medicaid_sim.add_argument("--cycles", type=int, default=100, help="Number of cycles")
    medicaid_sim.add_argument("--seed", type=int, default=42, help="Random seed")
    medicaid_sim.add_argument("--output", "-o", type=str, help="Output JSON file")
    medicaid_sim.set_defaults(func=lambda a: cmd_simulate(argparse.Namespace(domain="medicaid", **vars(a))))

    medicaid_scenario = medicaid_sub.add_parser("scenario", help="Run specific scenario")
    medicaid_scenario.add_argument("scenario", type=str, help="Scenario name")
    medicaid_scenario.set_defaults(func=lambda a: cmd_scenario(argparse.Namespace(domain="medicaid", **vars(a))))

    medicaid_all = medicaid_sub.add_parser("scenarios", help="Run all scenarios")
    medicaid_all.add_argument("--output", "-o", type=str, help="Output JSON file")
    medicaid_all.set_defaults(func=lambda a: cmd_scenarios(argparse.Namespace(domain="medicaid", **vars(a))))

    # Validate subcommand
    validate_parser = subparsers.add_parser("validate", help="Validate domain configuration")
    validate_parser.add_argument("--domain", "-d", type=str, default="all", help="Domain to validate")
    validate_parser.set_defaults(func=cmd_validate)

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List domains or scenarios")
    list_parser.add_argument("what", choices=["domains", "scenarios"], help="What to list")
    list_parser.add_argument("--domain", "-d", type=str, help="Domain for scenario listing")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
