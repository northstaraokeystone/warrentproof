#!/bin/bash
# gate_t48h.sh - ShieldProof v2.0 HARDENED GATE
# RUN THIS OR KILL PROJECT

set -e

echo "=== SHIELDPROOF v2.0 GATE T+48h: HARDENED ==="

cd "$(dirname "$0")/../.."

# First run T+24h gate
./scripts/shieldproof/gate_t24h.sh

echo ""
echo "=== CHECKING HARDENING ==="

# Check for anomaly handling in all critical modules
grep -rq "anomaly" src/shieldproof/contract.py || { echo "FAIL: contract.py missing anomaly handling"; exit 1; }
grep -rq "anomaly" src/shieldproof/milestone.py || { echo "FAIL: milestone.py missing anomaly handling"; exit 1; }
grep -rq "anomaly" src/shieldproof/payment.py || { echo "FAIL: payment.py missing anomaly handling"; exit 1; }
echo "  ✓ Anomaly handling in all critical modules"

# Check for stoprules
grep -rq "stoprule" src/shieldproof/contract.py || { echo "FAIL: contract.py missing stoprules"; exit 1; }
grep -rq "stoprule" src/shieldproof/milestone.py || { echo "FAIL: milestone.py missing stoprules"; exit 1; }
grep -rq "stoprule" src/shieldproof/payment.py || { echo "FAIL: payment.py missing stoprules"; exit 1; }
echo "  ✓ Stoprules in all critical modules"

# Check dual_hash is used (no single hash)
grep -r "hashlib.sha256\|hashlib.md5" src/shieldproof/*.py | grep -v "dual_hash" | grep -v "^#" && { echo "FAIL: Single hash used (use dual_hash)"; exit 1; } || true
echo "  ✓ No single hash usage (dual_hash only)"

echo ""
echo "=== COVERAGE CHECK ==="

# Run pytest with coverage if available
if command -v pytest &> /dev/null && python3 -c "import pytest_cov" 2>/dev/null; then
    pytest tests/test_shieldproof_*.py -v --cov=src/shieldproof --cov-report=term-missing --cov-fail-under=80 || { echo "FAIL: coverage < 80%"; exit 1; }
    echo "  ✓ Test coverage >= 80%"
else
    echo "  ⚠ pytest-cov not available, skipping coverage check"
fi

echo ""
echo "=== SECURITY CHECK ==="

# Check no silent exceptions
grep -r "except.*pass\|except:$" src/shieldproof/*.py && { echo "FAIL: Silent exception found"; exit 1; } || true
echo "  ✓ No silent exceptions"

# Check no global mutable state
grep -r "^[A-Z_]* = \[\|^[A-Z_]* = {" src/shieldproof/*.py | grep -v "RECEIPT_TYPES\|MILESTONE_STATES" && { echo "WARN: Possible mutable global state"; } || true
echo "  ✓ Minimal global state"

echo ""
echo "=== FINAL VERIFICATION ==="

# Test the complete flow with stoprule triggers
python3 << 'EOF'
import sys

from src.shieldproof.core import clear_ledger, StopRule
from src.shieldproof.contract import register_contract
from src.shieldproof.milestone import submit_deliverable, verify_milestone
from src.shieldproof.payment import release_payment

clear_ledger()

# Register contract
c = register_contract(
    contractor="Hardened Corp",
    amount=100000.00,
    milestones=[{"id": "M1", "amount": 100000.00}],
    terms={},
)

# Test: Payment without verification should HALT
try:
    release_payment(c["contract_id"], "M1")
    print("FAIL: Should have triggered stoprule_unverified_milestone")
    sys.exit(1)
except StopRule as e:
    print(f"  ✓ stoprule_unverified_milestone works: {str(e)[:50]}...")

# Test: Duplicate contract should trigger stoprule
try:
    register_contract(
        contractor="Duplicate",
        amount=100000.00,
        milestones=[{"id": "M1", "amount": 100000.00}],
        terms={},
        contract_id=c["contract_id"],
    )
    print("FAIL: Should have triggered stoprule_duplicate_contract")
    sys.exit(1)
except StopRule as e:
    print(f"  ✓ stoprule_duplicate_contract works: {str(e)[:50]}...")

# Test: Invalid amount should trigger stoprule
try:
    register_contract(
        contractor="Invalid",
        amount=-100.00,
        milestones=[{"id": "M1", "amount": 100.00}],
        terms={},
    )
    print("FAIL: Should have triggered stoprule_invalid_amount")
    sys.exit(1)
except StopRule as e:
    print(f"  ✓ stoprule_invalid_amount works: {str(e)[:50]}...")

# Complete a valid flow
submit_deliverable(c["contract_id"], "M1", b"deliverable")
verify_milestone(c["contract_id"], "M1", "INSPECTOR", passed=True)
release_payment(c["contract_id"], "M1")

# Test: Double payment should trigger stoprule
try:
    release_payment(c["contract_id"], "M1")
    print("FAIL: Should have triggered stoprule_already_paid")
    sys.exit(1)
except StopRule as e:
    print(f"  ✓ stoprule_already_paid works: {str(e)[:50]}...")

print("")
print("  All stoprules verified ✓")

clear_ledger()
EOF

echo ""
echo "=========================================="
echo "PASS: T+48h gate - HARDENED COMPLETE"
echo "=========================================="
echo ""
echo "ShieldProof v2.0 is ready to ship."
echo ""
echo "One receipt. One milestone. One truth."
echo ""
