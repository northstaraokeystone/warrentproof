#!/bin/bash
# gate_t24h.sh - T+24h Gate: MVP
# Per CLAUDEME §3: RUN THIS OR KILL PROJECT
#
# ⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

set -e

echo "=== WarrantProof T+24h Gate ==="
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
echo ""

# First, verify T+2h gate passes
echo "Running T+2h gate first..."
./gate_t2h.sh || { echo "FAIL: T+2h gate failed"; exit 1; }

echo ""
echo "=== T+24h Specific Checks ==="
echo ""

# Run pytest
echo "Running tests..."
python -m pytest tests/ -q || { echo "FAIL: tests failed"; exit 1; }
echo "✓ All tests pass"

# Check emit_receipt in all src files
echo ""
echo "Checking receipt emission in modules..."
for f in src/*.py; do
    if [[ "$f" != "src/__init__.py" ]]; then
        grep -q "emit_receipt" "$f" || { echo "FAIL: $f missing emit_receipt"; exit 1; }
        echo "✓ $f has emit_receipt"
    fi
done

# Check assertions in tests
echo ""
echo "Checking test assertions..."
for f in tests/test_*.py; do
    grep -q "assert" "$f" || { echo "FAIL: $f missing assertions"; exit 1; }
    echo "✓ $f has assertions"
done

# Run 10-cycle smoke test
echo ""
echo "Running 10-cycle smoke test..."
python -c "
from src.sim import run_simulation, SimConfig
r = run_simulation(SimConfig(n_cycles=10, n_transactions_per_cycle=100))
print(f'✓ 10 cycles: {len(r.violations)} violations, {len(r.receipts)} receipts')
assert len(r.receipts) > 0, 'No receipts generated'
"

# Verify ALL receipts have citations or simulation flags
echo ""
echo "Verifying receipt coverage..."
python -c "
from src.sim import run_simulation, SimConfig
r = run_simulation(SimConfig(n_cycles=10))
for rec in r.receipts:
    has_citation = 'citation' in rec or 'citations' in rec
    has_flag = 'simulation_flag' in rec
    assert has_citation or has_flag, f'Receipt missing citation/flag: {rec.get(\"receipt_type\")}'
print('✓ All receipts have citations or simulation flags')
"

echo ""
echo "=== PASS: T+24h Gate ==="
echo ""
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
