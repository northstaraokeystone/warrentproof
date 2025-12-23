#!/usr/bin/env python3
"""
ShieldProof v2.0 CLI

Minimal command-line interface for ShieldProof operations.
Per CLAUDEME: cli.py must emit valid receipt JSON to stdout.

"One receipt. One milestone. One truth."
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.shieldproof.core import (
    dual_hash,
    emit_receipt,
    merkle,
    TENANT_ID,
    RECEIPT_TYPES,
    VERSION,
)


def cmd_test(args):
    """Run self-test and emit test receipt."""
    # Test dual_hash
    h = dual_hash("test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"

    # Emit test receipt
    receipt = emit_receipt("test", {
        "message": "ShieldProof v2.0 self-test",
        "test_hash": h,
    }, to_ledger=False)

    return 0


def cmd_hash(args):
    """Compute dual-hash of input."""
    if args.input == "-":
        data = sys.stdin.read()
    else:
        data = args.input
    print(dual_hash(data))
    return 0


def cmd_receipt(args):
    """Emit a custom receipt."""
    try:
        data = json.loads(args.data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1

    emit_receipt(args.type, data, to_ledger=not args.no_ledger)
    return 0


def cmd_contract(args):
    """Register a new contract."""
    from src.shieldproof.contract import register_contract

    try:
        milestones = json.loads(args.milestones)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid milestones JSON: {e}", file=sys.stderr)
        return 1

    terms = {}
    if args.terms:
        try:
            terms = json.loads(args.terms)
        except json.JSONDecodeError:
            terms = {"raw": args.terms}

    try:
        receipt = register_contract(
            contractor=args.contractor,
            amount=args.amount,
            milestones=milestones,
            terms=terms,
            contract_id=args.id,
        )
        print(f"Contract registered: {receipt['contract_id']}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_submit(args):
    """Submit a milestone deliverable."""
    from src.shieldproof.milestone import submit_deliverable

    if args.file:
        with open(args.file, "rb") as f:
            deliverable = f.read()
    else:
        deliverable = args.content or ""

    try:
        receipt = submit_deliverable(args.contract_id, args.milestone_id, deliverable)
        print(f"Deliverable submitted: {receipt['milestone_id']} -> {receipt['status']}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_verify(args):
    """Verify a milestone."""
    from src.shieldproof.milestone import verify_milestone

    try:
        receipt = verify_milestone(
            args.contract_id,
            args.milestone_id,
            args.verifier_id,
            passed=not args.reject,
        )
        print(f"Milestone {receipt['milestone_id']} -> {receipt['status']}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_pay(args):
    """Release payment for a verified milestone."""
    from src.shieldproof.payment import release_payment

    try:
        receipt = release_payment(args.contract_id, args.milestone_id)
        print(f"Payment released: ${receipt['amount']}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_reconcile(args):
    """Reconcile contracts."""
    from src.shieldproof.reconcile import reconcile_contract, reconcile_all, get_waste_summary

    if args.contract_id:
        report = reconcile_contract(args.contract_id)
        print(json.dumps(report, indent=2))
    else:
        reports = reconcile_all()
        summary = get_waste_summary()
        output = {"summary": summary, "contracts": reports}
        print(json.dumps(output, indent=2))

    return 0


def cmd_dashboard(args):
    """Show or serve dashboard."""
    from src.shieldproof.dashboard import generate_summary, print_dashboard, serve, check, export_csv, export_json

    if args.check:
        if check():
            print("Dashboard: OK")
            return 0
        else:
            print("Dashboard: FAIL")
            return 1

    if args.serve:
        serve(args.port)
        return 0

    if args.export_csv:
        export_csv(args.export_csv)
        print(f"Exported to {args.export_csv}", file=sys.stderr)
        return 0

    if args.export_json:
        export_json(args.export_json)
        print(f"Exported to {args.export_json}", file=sys.stderr)
        return 0

    if args.json:
        summary = generate_summary()
        print(json.dumps(summary, indent=2))
    else:
        print_dashboard()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ShieldProof v2.0 CLI - Minimal Viable Truth for Defense Accountability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='"One receipt. One milestone. One truth."',
    )
    parser.add_argument("--version", action="version", version=f"ShieldProof {VERSION}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run self-test")
    test_parser.add_argument("--test", action="store_true", help="(ignored, for compatibility)")

    # Hash command
    hash_parser = subparsers.add_parser("hash", help="Compute dual-hash")
    hash_parser.add_argument("input", nargs="?", default="-", help="Input to hash (- for stdin)")

    # Receipt command
    receipt_parser = subparsers.add_parser("receipt", help="Emit a receipt")
    receipt_parser.add_argument("type", help="Receipt type")
    receipt_parser.add_argument("data", help="Receipt data as JSON")
    receipt_parser.add_argument("--no-ledger", action="store_true", help="Don't write to ledger")

    # Contract command
    contract_parser = subparsers.add_parser("contract", help="Register a contract")
    contract_parser.add_argument("--contractor", required=True, help="Contractor name")
    contract_parser.add_argument("--amount", type=float, required=True, help="Total amount")
    contract_parser.add_argument("--milestones", required=True, help="Milestones as JSON array")
    contract_parser.add_argument("--terms", help="Contract terms")
    contract_parser.add_argument("--id", help="Contract ID (generated if not provided)")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a deliverable")
    submit_parser.add_argument("contract_id", help="Contract ID")
    submit_parser.add_argument("milestone_id", help="Milestone ID")
    submit_parser.add_argument("--file", help="Deliverable file")
    submit_parser.add_argument("--content", help="Deliverable content")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a milestone")
    verify_parser.add_argument("contract_id", help="Contract ID")
    verify_parser.add_argument("milestone_id", help="Milestone ID")
    verify_parser.add_argument("--verifier-id", required=True, help="Verifier ID")
    verify_parser.add_argument("--reject", action="store_true", help="Reject instead of approve")

    # Pay command
    pay_parser = subparsers.add_parser("pay", help="Release payment")
    pay_parser.add_argument("contract_id", help="Contract ID")
    pay_parser.add_argument("milestone_id", help="Milestone ID")

    # Reconcile command
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile contracts")
    reconcile_parser.add_argument("--contract-id", help="Single contract to reconcile")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show dashboard")
    dashboard_parser.add_argument("--check", action="store_true", help="Health check only")
    dashboard_parser.add_argument("--serve", action="store_true", help="Serve as HTTP")
    dashboard_parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    dashboard_parser.add_argument("--json", action="store_true", help="Output as JSON")
    dashboard_parser.add_argument("--export-csv", help="Export to CSV file")
    dashboard_parser.add_argument("--export-json", help="Export to JSON file")

    args = parser.parse_args()

    # Handle --test flag on root for backward compatibility
    if hasattr(args, "test") and args.test or args.command == "test":
        return cmd_test(args)

    if args.command == "hash":
        return cmd_hash(args)
    elif args.command == "receipt":
        return cmd_receipt(args)
    elif args.command == "contract":
        return cmd_contract(args)
    elif args.command == "submit":
        return cmd_submit(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "pay":
        return cmd_pay(args)
    elif args.command == "reconcile":
        return cmd_reconcile(args)
    elif args.command == "dashboard":
        return cmd_dashboard(args)
    else:
        # Default: run test
        return cmd_test(args)


if __name__ == "__main__":
    sys.exit(main())
