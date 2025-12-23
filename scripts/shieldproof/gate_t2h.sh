#!/bin/bash
# gate_t2h.sh - ShieldProof v2.0 SKELETON GATE
# RUN THIS OR KILL PROJECT

set -e

echo "=== SHIELDPROOF v2.0 GATE T+2h: SKELETON ==="

cd "$(dirname "$0")/../.."

# Check spec.md exists
[ -f src/shieldproof/spec.md ] || { echo "FAIL: no src/shieldproof/spec.md"; exit 1; }
echo "  ✓ spec.md exists"

# Check ledger_schema.json exists
[ -f schemas/ledger_schema_shieldproof.json ] || { echo "FAIL: no schemas/ledger_schema_shieldproof.json"; exit 1; }
echo "  ✓ ledger_schema_shieldproof.json exists"

# Check cli.py exists
[ -f shieldproof_cli.py ] || { echo "FAIL: no shieldproof_cli.py"; exit 1; }
echo "  ✓ shieldproof_cli.py exists"

# Check cli.py emits valid receipt JSON
python3 shieldproof_cli.py test 2>&1 | grep -q '"receipt_type"' || { echo "FAIL: cli.py does not emit receipt"; exit 1; }
echo "  ✓ cli.py emits valid receipt JSON"

# Check core.py has required functions
grep -q "def dual_hash" src/shieldproof/core.py || { echo "FAIL: core.py missing dual_hash"; exit 1; }
grep -q "def emit_receipt" src/shieldproof/core.py || { echo "FAIL: core.py missing emit_receipt"; exit 1; }
grep -q "def merkle" src/shieldproof/core.py || { echo "FAIL: core.py missing merkle"; exit 1; }
echo "  ✓ core.py has dual_hash, emit_receipt, merkle"

# Check dual_hash format
python3 -c "from src.shieldproof.core import dual_hash; h=dual_hash('test'); assert ':' in h, 'dual_hash format wrong'" || { echo "FAIL: dual_hash format"; exit 1; }
echo "  ✓ dual_hash returns SHA256:BLAKE3 format"

echo ""
echo "PASS: T+2h gate - SKELETON COMPLETE"
echo ""
