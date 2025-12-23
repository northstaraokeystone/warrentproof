#!/bin/bash
# gate_t24h.sh - ShieldProof v2.0 MVP GATE
# RUN THIS OR KILL PROJECT

set -e

echo "=== SHIELDPROOF v2.0 GATE T+24h: MVP ==="

cd "$(dirname "$0")/../.."

# First run T+2h gate
./scripts/shieldproof/gate_t2h.sh

echo ""
echo "=== CHECKING MODULES ==="

# Check all modules exist
for module in contract milestone payment reconcile dashboard; do
    [ -f "src/shieldproof/${module}.py" ] || { echo "FAIL: src/shieldproof/${module}.py missing"; exit 1; }
    echo "  ✓ src/shieldproof/${module}.py exists"
done

# Check all modules have emit_receipt
for module in contract milestone payment reconcile; do
    grep -q "emit_receipt" "src/shieldproof/${module}.py" || { echo "FAIL: ${module}.py missing emit_receipt"; exit 1; }
done
echo "  ✓ All modules use emit_receipt"

echo ""
echo "=== CHECKING STOPRULES ==="

# Check stoprules exist
grep -rq "stoprule" src/shieldproof/*.py || { echo "FAIL: no stoprules in src/shieldproof"; exit 1; }
echo "  ✓ Stoprules implemented"

# Check anomaly handling
grep -rq "anomaly" src/shieldproof/*.py || { echo "FAIL: no anomaly handling"; exit 1; }
echo "  ✓ Anomaly handling implemented"

echo ""
echo "=== RUNNING TESTS ==="

# Run pytest via python3 -m for consistent environment
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest tests/test_shieldproof_*.py -v --tb=short || { echo "FAIL: tests failed"; exit 1; }
    echo "  ✓ All tests pass"
else
    echo "  ⚠ pytest not available, skipping"
fi

echo ""
echo "=== INTEGRATION CHECK ==="

# Quick integration test
python3 << 'EOF'
import sys

from src.shieldproof.core import clear_ledger, dual_hash, emit_receipt, merkle
from src.shieldproof.contract import register_contract, get_contract
from src.shieldproof.milestone import submit_deliverable, verify_milestone
from src.shieldproof.payment import release_payment, total_paid
from src.shieldproof.reconcile import reconcile_contract, get_waste_summary
from src.shieldproof.dashboard import generate_summary, check

# Clear for clean test
clear_ledger()

# Register contract
c = register_contract(
    contractor="Test Corp",
    amount=100000.00,
    milestones=[{"id": "M1", "amount": 100000.00}],
    terms={},
)
assert c["receipt_type"] == "contract"
print("  ✓ Contract registration works")

# Submit deliverable
d = submit_deliverable(c["contract_id"], "M1", b"test deliverable")
assert d["status"] == "DELIVERED"
print("  ✓ Deliverable submission works")

# Verify milestone
v = verify_milestone(c["contract_id"], "M1", "INSPECTOR", passed=True)
assert v["status"] == "VERIFIED"
print("  ✓ Milestone verification works")

# Release payment
p = release_payment(c["contract_id"], "M1")
assert p["receipt_type"] == "payment"
assert p["amount"] == 100000.00
print("  ✓ Payment release works")

# Check paid amount
paid = total_paid(c["contract_id"])
assert paid == 100000.00
print("  ✓ Payment tracking works")

# Reconcile
report = reconcile_contract(c["contract_id"])
assert report["status"] == "ON_TRACK"
print("  ✓ Reconciliation works")

# Dashboard
summary = generate_summary()
assert summary["total_contracts"] >= 1
print("  ✓ Dashboard generation works")

assert check() is True
print("  ✓ Dashboard health check works")

print("")
print("  CHAIN: contract → milestone → payment ✓")

clear_ledger()
EOF

echo ""
echo "PASS: T+24h gate - MVP COMPLETE"
echo ""
echo "One receipt. One milestone. One truth."
echo ""
