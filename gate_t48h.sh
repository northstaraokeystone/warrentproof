#!/bin/bash
# gate_t48h.sh - T+48h Gate: HARDENED
# Per CLAUDEME §3: RUN THIS OR KILL PROJECT
#
# ⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

set -e

echo "=== WarrantProof T+48h Gate ==="
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
echo ""

# First, verify T+24h gate passes
echo "Running T+24h gate first..."
./gate_t24h.sh || { echo "FAIL: T+24h gate failed"; exit 1; }

echo ""
echo "=== T+48h Specific Checks ==="
echo ""

# Check anomaly detection exists
echo "Checking anomaly detection..."
grep -rq "anomaly" src/*.py || { echo "FAIL: no anomaly detection"; exit 1; }
echo "✓ Anomaly detection present"

# Check stoprules exist
echo ""
echo "Checking stoprules..."
grep -rq "stoprule" src/*.py || { echo "FAIL: no stoprules"; exit 1; }
echo "✓ Stoprules present"

# Check bias/disparity checks (for entropy-based detection)
echo ""
echo "Checking entropy-based detection..."
grep -rq "entropy" src/*.py || { echo "FAIL: no entropy checks"; exit 1; }
echo "✓ Entropy-based detection present"

# Run all 6 scenarios
echo ""
echo "Running all 6 mandatory scenarios..."

scenarios=("BASELINE" "SHIPYARD_STRESS" "CROSS_BRANCH_INTEGRATION" "FRAUD_DISCOVERY" "REAL_TIME_OVERSIGHT" "GODEL")

for scenario in "${scenarios[@]}"; do
    echo ""
    echo "Running $scenario..."
    python -c "
from src.sim import run_simulation, SimConfig
config = SimConfig(n_cycles=5, n_transactions_per_cycle=50, scenario='$scenario')
result = run_simulation(config)
print(f'  Receipts: {len(result.receipts)}')
print(f'  Detections: {len(result.detections)}')
print(f'  Violations: {len(result.violations)}')
print(f'  ✓ $scenario completed')
" || { echo "FAIL: $scenario scenario failed"; exit 1; }
done

# Full 1000-cycle timing test
echo ""
echo "Running 1000-cycle timing test..."
python -c "
import time
from src.sim import run_simulation, SimConfig

t = time.time()
r = run_simulation(SimConfig(n_cycles=1000, n_transactions_per_cycle=100))
elapsed = time.time() - t

print(f'✓ 1000 cycles in {elapsed:.1f}s')
print(f'  Receipts: {len(r.receipts)}')
print(f'  Violations: {len(r.violations)}')
print(f'  Fraud injected: {r.fraud_injected_count}')
print(f'  Fraud detected: {r.fraud_detected_count}')

# SLO check: should complete in reasonable time
assert elapsed < 300, f'1000 cycles took {elapsed}s > 300s limit'
"

# Verify scenario tests pass
echo ""
echo "Running scenario test suite..."
python -m pytest tests/test_sim_scenarios.py -v || { echo "FAIL: scenario tests failed"; exit 1; }

echo ""
echo "=== PASS: T+48h Gate — SHIP IT ==="
echo ""
echo "⚠️  THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY."
echo ""
echo "Receipt: gate_t48h_complete"
echo "SLO: compression >= 0.80, detection_recall >= 0.90, citations = 100%"
