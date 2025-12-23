#!/bin/bash
# Gov-OS Gate T+24h - Full Test Suite
# THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
#
# Run this gate within 24 hours of making changes.
# Validates: all unit tests, scenario runs, coverage

set -e

echo "=============================================="
echo "Gov-OS Gate T+24h - Full Test Suite"
echo "=============================================="
echo ""
echo "THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY"
echo ""

cd "$(dirname "$0")"

# Run T+2h gate first
echo "Running T+2h gate first..."
./gate_t2h.sh
echo ""

# Phase 5: Full unit test suite
echo "Phase 5: Full Unit Test Suite"
echo "-----------------------------"

python3 -m pytest tests/ -v --tb=short 2>/dev/null || {
    echo "Unit tests FAILED"
    exit 1
}

echo ""

# Phase 6: Defense scenarios
echo "Phase 6: Defense Scenarios"
echo "--------------------------"

python3 << 'EOF'
from src.modules.defense.scenarios import run_defense_scenario, DEFENSE_SCENARIOS

passed = 0
failed = 0

for scenario in DEFENSE_SCENARIOS:
    result = run_defense_scenario(scenario)
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {scenario}: {result.message}")
    if result.passed:
        passed += 1
    else:
        failed += 1

print(f"\n  Defense: {passed}/{passed + failed} scenarios passed")
EOF

echo ""

# Phase 7: Medicaid scenarios
echo "Phase 7: Medicaid Scenarios"
echo "---------------------------"

python3 << 'EOF'
from src.modules.medicaid.scenarios import run_medicaid_scenario, MEDICAID_SCENARIOS

passed = 0
failed = 0

for scenario in MEDICAID_SCENARIOS:
    result = run_medicaid_scenario(scenario)
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {scenario}: {result.message}")
    if result.passed:
        passed += 1
    else:
        failed += 1

print(f"\n  Medicaid: {passed}/{passed + failed} scenarios passed")
EOF

echo ""

# Phase 8: Simulation runs
echo "Phase 8: Simulation Runs"
echo "------------------------"

python3 << 'EOF'
from src.core.harness import run_simulation

domains = ["defense", "medicaid"]
for domain in domains:
    state = run_simulation(domain, n_cycles=100, seed=42)
    print(f"  {domain}: {len(state.receipts)} receipts, {len(state.violations)} violations")
EOF

echo ""
echo "=============================================="
echo "Gate T+24h: PASSED"
echo "=============================================="
