#!/bin/bash
# gate_t2h.sh - T+2h Gate: SKELETON
# Per CLAUDEME §3: RUN THIS OR KILL PROJECT
#
# ⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

set -e

echo "=== WarrantProof T+2h Gate ==="
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
echo ""

# Check required files exist
echo "Checking required files..."

[ -f spec.md ] || { echo "FAIL: no spec.md"; exit 1; }
echo "✓ spec.md exists"

[ -f ledger_schema.json ] || { echo "FAIL: no ledger_schema.json"; exit 1; }
echo "✓ ledger_schema.json exists"

[ -f cli.py ] || { echo "FAIL: no cli.py"; exit 1; }
echo "✓ cli.py exists"

[ -f DISCLAIMER.md ] || { echo "FAIL: no DISCLAIMER.md"; exit 1; }
echo "✓ DISCLAIMER.md exists"

[ -f CITATIONS.md ] || { echo "FAIL: no CITATIONS.md"; exit 1; }
echo "✓ CITATIONS.md exists"

# Check CLI emits valid receipt JSON
echo ""
echo "Testing CLI receipt emission..."
python cli.py --test 2>&1 | grep -q '"receipt_type"' || { echo "FAIL: CLI does not emit receipt"; exit 1; }
echo "✓ CLI emits valid receipt JSON"

# Verify simulation disclaimers present
echo ""
echo "Checking simulation disclaimers..."
grep -rq "SIMULATION" src/*.py || { echo "FAIL: Missing SIMULATION disclaimers in src/"; exit 1; }
echo "✓ SIMULATION disclaimers present in src/"

grep -rq "NOT REAL DoD DATA" src/*.py || { echo "FAIL: Missing 'NOT REAL DoD DATA' disclaimers"; exit 1; }
echo "✓ 'NOT REAL DoD DATA' disclaimers present"

# Verify citations embedded
echo ""
echo "Checking citations..."
python -c "from src.core import CITATIONS; assert len(CITATIONS) >= 20; print(f'✓ Citations loaded: {len(CITATIONS)}')" || { echo "FAIL: Insufficient citations"; exit 1; }

# Verify core functions exist
echo ""
echo "Testing core functions..."
python -c "
from src.core import dual_hash, emit_receipt, cite
h = dual_hash('test')
assert ':' in h, 'dual_hash must return SHA256:BLAKE3 format'
c = cite('TEST', 'http://test.com', 'test')
assert 'url' in c, 'cite must include url'
print('✓ Core functions working')
"

echo ""
echo "=== PASS: T+2h Gate ==="
echo ""
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
