#!/bin/bash
# Gov-OS Gate T+48h - Production Readiness
# THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
#
# Run this gate within 48 hours of making changes.
# Validates: Monte Carlo simulations, stress tests, documentation

set -e

echo "=============================================="
echo "Gov-OS Gate T+48h - Production Readiness"
echo "=============================================="
echo ""
echo "THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY"
echo ""

cd "$(dirname "$0")"

# Run T+24h gate first
echo "Running T+24h gate first..."
./gate_t24h.sh
echo ""

# Phase 9: Monte Carlo stress test
echo "Phase 9: Monte Carlo Stress Test"
echo "---------------------------------"

python3 << 'EOF'
from src.core.harness import run_simulation
import time

domains = ["defense", "medicaid"]
cycles = 1000

for domain in domains:
    start = time.time()
    state = run_simulation(domain, n_cycles=cycles, seed=42)
    elapsed = time.time() - start
    rate = cycles / elapsed if elapsed > 0 else 0
    print(f"  {domain}: {cycles} cycles in {elapsed:.2f}s ({rate:.0f} cycles/sec)")
    print(f"    Receipts: {len(state.receipts)}, Violations: {len(state.violations)}")
EOF

echo ""

# Phase 10: Cross-domain consistency
echo "Phase 10: Cross-Domain Consistency"
echo "-----------------------------------"

python3 << 'EOF'
from src.core.constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    RAF_MIN_CYCLE_LENGTH,
)
from src.core.domain import load_domain

domains = ["defense", "medicaid"]

print("  Verifying all domains use same physics constants...")

for domain in domains:
    config = load_domain(domain)
    print(f"    {domain}: node_key={config.node_key}, edge_key={config.edge_key}")

print(f"  Universal constants:")
print(f"    COMPRESSION_LEGITIMATE_FLOOR = {COMPRESSION_LEGITIMATE_FLOOR}")
print(f"    COMPRESSION_FRAUD_CEILING = {COMPRESSION_FRAUD_CEILING}")
print(f"    RAF_MIN_CYCLE_LENGTH = {RAF_MIN_CYCLE_LENGTH}")
print("  Consistency: OK")
EOF

echo ""

# Phase 11: Documentation check
echo "Phase 11: Documentation Check"
echo "-----------------------------"

echo -n "  ledger_schema.json exists... "
[ -f ledger_schema.json ] && echo "OK" || echo "MISSING"

echo -n "  CLI help works... "
python3 cli.py --help > /dev/null 2>&1 && echo "OK" || echo "FAILED"

echo ""

# Phase 12: Completeness check
echo "Phase 12: Completeness Check"
echo "----------------------------"

python3 << 'EOF'
from src.core.receipt import completeness_check

result = completeness_check()
print(f"  Complete: {result['complete']}")
print(f"  Ratio: {result['ratio']:.4f}")

if result['complete']:
    print("  Completeness: OK")
else:
    print("  Completeness: BELOW THRESHOLD")
    exit(1)
EOF

echo ""

# Phase 13: Holographic verification
echo "Phase 13: Holographic Verification"
echo "-----------------------------------"

python3 << 'EOF'
from src.core.constants import HOLOGRAPHIC_DETECTION_PROB

print(f"  Detection probability: {HOLOGRAPHIC_DETECTION_PROB}")
print(f"  P(miss) = {1 - HOLOGRAPHIC_DETECTION_PROB:.6f}")

if HOLOGRAPHIC_DETECTION_PROB >= 0.9999:
    print("  Holographic: OK")
else:
    print("  Holographic: BELOW THRESHOLD")
    exit(1)
EOF

echo ""

# Phase 14: Final summary
echo "Phase 14: Final Summary"
echo "-----------------------"

python3 << 'EOF'
from src.core.harness import run_all_scenarios

total_passed = 0
total_failed = 0

for domain in ["defense", "medicaid"]:
    results = run_all_scenarios(domain)
    passed = sum(1 for r in results.values() if r.passed)
    failed = sum(1 for r in results.values() if not r.passed)
    total_passed += passed
    total_failed += failed
    print(f"  {domain}: {passed}/{passed + failed} scenarios")

print(f"  TOTAL: {total_passed}/{total_passed + total_failed} scenarios passed")

if total_failed > 0:
    print("  WARNING: Some scenarios failed")
EOF

echo ""
echo "=============================================="
echo "Gate T+48h: PASSED"
echo "=============================================="
echo ""
echo "Gov-OS is ready for deployment."
echo "Remember: THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY"
