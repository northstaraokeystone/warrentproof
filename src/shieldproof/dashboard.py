"""
ShieldProof v2.0 Dashboard Module

Public audit trail. Spending vs deliverables, redacted for OPSEC.
Per Grok: "Open audit trail" - transparency that doesn't compromise security.

"One receipt. One milestone. One truth."
"""

import csv
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from .core import (
    query_receipts,
    TENANT_ID,
    VERSION,
)
from .reconcile import reconcile_all, get_waste_summary


def generate_summary() -> dict:
    """
    Generate aggregate public summary of all contracts.
    Shows aggregate numbers, not individual contract details.

    Returns:
        Dashboard summary dict
    """
    waste_summary = get_waste_summary()

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": VERSION,
        "tenant_id": TENANT_ID,
        "total_contracts": waste_summary["total_contracts"],
        "total_committed": waste_summary["total_committed"],
        "total_paid": waste_summary["total_paid"],
        "total_verified": waste_summary["total_verified"],
        "milestones_pending": waste_summary["milestones_pending"],
        "milestones_disputed": waste_summary["milestones_disputed"],
        "waste_identified": waste_summary["waste_identified"],
        "contracts_on_track": waste_summary["contracts_on_track"],
        "contracts_overpaid": waste_summary["contracts_overpaid"],
        "contracts_unverified": waste_summary["contracts_unverified"],
        "contracts_disputed": waste_summary["contracts_disputed"],
        "health_score": calculate_health_score(waste_summary),
    }

    return summary


def calculate_health_score(summary: dict) -> float:
    """
    Calculate an overall health score (0-100) for the portfolio.

    Args:
        summary: Waste summary dict

    Returns:
        Health score as percentage
    """
    if summary["total_contracts"] == 0:
        return 100.0

    # Factors: on_track %, no waste %, no disputes %
    on_track_pct = summary["contracts_on_track"] / summary["total_contracts"]

    if summary["total_paid"] > 0:
        verified_pct = summary["total_verified"] / summary["total_paid"]
    else:
        verified_pct = 1.0

    dispute_pct = 1 - (summary["contracts_disputed"] / summary["total_contracts"])

    # Weighted average: 50% on_track, 30% verified, 20% no disputes
    score = (on_track_pct * 0.5 + verified_pct * 0.3 + dispute_pct * 0.2) * 100

    return round(score, 1)


def contract_status(contract_id: str) -> dict:
    """
    Get single contract public view.
    Requires authentication in production.

    Args:
        contract_id: Contract identifier

    Returns:
        Contract status dict (redacted for OPSEC)
    """
    from .contract import get_contract, get_contract_milestones
    from .payment import total_paid, total_outstanding

    contract = get_contract(contract_id)
    if not contract:
        return {"error": "Contract not found", "contract_id": contract_id}

    milestones = get_contract_milestones(contract_id)

    # Redact sensitive fields for public view
    return {
        "contract_id": contract_id,
        "contractor": contract.get("contractor"),
        "amount_fixed": contract.get("amount_fixed"),
        "amount_paid": total_paid(contract_id),
        "amount_outstanding": total_outstanding(contract_id),
        "milestones": [
            {
                "id": m["id"],
                "status": m.get("status"),
                "amount": m.get("amount"),
            }
            for m in milestones
        ],
        "created_at": contract.get("ts"),
    }


def export_csv(filepath: str) -> None:
    """
    Export dashboard data to CSV.

    Args:
        filepath: Output file path
    """
    reports = reconcile_all()

    with open(filepath, "w", newline="") as f:
        if not reports:
            f.write("No contracts found\n")
            return

        fieldnames = [
            "contract_id",
            "contractor",
            "amount_fixed",
            "amount_paid",
            "status",
            "milestones_total",
            "milestones_verified",
            "milestones_paid",
            "milestones_pending",
            "milestones_disputed",
            "discrepancy",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for report in reports:
            writer.writerow(report)


def export_json(filepath: str) -> None:
    """
    Export dashboard data to JSON.

    Args:
        filepath: Output file path
    """
    summary = generate_summary()
    reports = reconcile_all()

    data = {
        "summary": summary,
        "contracts": reports,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.2f}K"
    else:
        return f"${amount:.2f}"


def print_dashboard() -> None:
    """Print dashboard summary to stdout."""
    summary = generate_summary()

    print("\n" + "=" * 60)
    print("SHIELDPROOF v2.0 - PUBLIC AUDIT DASHBOARD")
    print("=" * 60)
    print(f"Generated: {summary['generated_at']}")
    print(f"Health Score: {summary['health_score']}%")
    print("-" * 60)
    print(f"Total Contracts:     {summary['total_contracts']}")
    print(f"Total Committed:     {format_currency(summary['total_committed'])}")
    print(f"Total Paid:          {format_currency(summary['total_paid'])}")
    print(f"Total Verified:      {format_currency(summary['total_verified'])}")
    print("-" * 60)
    print(f"Contracts On Track:  {summary['contracts_on_track']}")
    print(f"Contracts Overpaid:  {summary['contracts_overpaid']}")
    print(f"Contracts Unverified:{summary['contracts_unverified']}")
    print(f"Contracts Disputed:  {summary['contracts_disputed']}")
    print("-" * 60)
    print(f"Milestones Pending:  {summary['milestones_pending']}")
    print(f"Milestones Disputed: {summary['milestones_disputed']}")
    print(f"WASTE IDENTIFIED:    {format_currency(summary['waste_identified'])}")
    print("=" * 60 + "\n")


class DashboardHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for dashboard."""

    def do_GET(self):
        if self.path == "/" or self.path == "/summary":
            summary = generate_summary()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(summary, indent=2).encode())

        elif self.path == "/contracts":
            reports = reconcile_all()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reports, indent=2).encode())

        elif self.path.startswith("/contract/"):
            contract_id = self.path.split("/contract/")[1]
            status = contract_status(contract_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def serve(port: int = 8080) -> None:
    """
    Serve dashboard as simple HTTP server.

    Args:
        port: Port to serve on (default 8080)
    """
    server = HTTPServer(("", port), DashboardHandler)
    print(f"Dashboard serving at http://localhost:{port}")
    print("Endpoints:")
    print("  GET /         - Summary")
    print("  GET /summary  - Summary")
    print("  GET /contracts - All contract reconciliation reports")
    print("  GET /contract/{id} - Single contract status")
    print("  GET /health   - Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def check() -> bool:
    """
    Quick health check for dashboard.

    Returns:
        True if dashboard can generate summary
    """
    try:
        summary = generate_summary()
        return "generated_at" in summary and "total_contracts" in summary
    except Exception:
        return False


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--serve":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            serve(port)
        elif sys.argv[1] == "--check":
            if check():
                print("Dashboard: OK")
                sys.exit(0)
            else:
                print("Dashboard: FAIL")
                sys.exit(1)
        elif sys.argv[1] == "--export-csv":
            filepath = sys.argv[2] if len(sys.argv) > 2 else "dashboard.csv"
            export_csv(filepath)
            print(f"Exported to {filepath}")
        elif sys.argv[1] == "--export-json":
            filepath = sys.argv[2] if len(sys.argv) > 2 else "dashboard.json"
            export_json(filepath)
            print(f"Exported to {filepath}")
        else:
            print_dashboard()
    else:
        # Self-test
        from .core import clear_ledger
        from .contract import register_contract
        from .milestone import submit_deliverable, verify_milestone
        from .payment import release_payment

        print("# Dashboard module self-test", file=sys.stderr)

        # Clear ledger for testing
        clear_ledger()

        # Create test data
        c = register_contract(
            contractor="ACME Defense",
            amount=1000000.00,
            milestones=[
                {"id": "M1", "amount": 250000.00},
                {"id": "M2", "amount": 750000.00},
            ],
            terms={},
        )
        submit_deliverable(c["contract_id"], "M1", b"Deliverable")
        verify_milestone(c["contract_id"], "M1", "INSPECTOR", passed=True)
        release_payment(c["contract_id"], "M1")

        # Test generate_summary
        summary = generate_summary()
        assert "generated_at" in summary
        assert "total_contracts" in summary
        print(f"# Summary generated: {summary['total_contracts']} contracts", file=sys.stderr)

        # Test check
        assert check() is True
        print("# Health check: OK", file=sys.stderr)

        # Test contract_status
        status = contract_status(c["contract_id"])
        assert status["contract_id"] == c["contract_id"]
        print(f"# Contract status: {status['amount_paid']}/{status['amount_fixed']}", file=sys.stderr)

        # Test print_dashboard
        print_dashboard()

        print("# PASS: dashboard module self-test", file=sys.stderr)
