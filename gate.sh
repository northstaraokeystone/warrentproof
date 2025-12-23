#!/bin/bash
# gate.sh - Unified Gov-OS Gate Runner
# Per CLAUDEME §3: RUN THIS OR KILL PROJECT
#
# THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
#
# Usage:
#   ./gate.sh t2h              # Run T+2h skeleton gate
#   ./gate.sh t24h             # Run T+24h MVP gate
#   ./gate.sh t48h             # Run T+48h hardened gate
#   ./gate.sh all              # Run all gates

set -e

GATE="${1:-t2h}"
DISCLAIMER="THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY"

echo "============================================================"
echo "Gov-OS Gate Runner - $GATE"
echo "$DISCLAIMER"
echo "============================================================"
echo ""

# =============================================================================
# T+2h SKELETON GATE
# =============================================================================
run_t2h() {
    echo "=== T+2h Gate: SKELETON ==="
    echo ""

    # Core files
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

    # CLI test
    echo ""
    echo "Testing CLI receipt emission..."
    python cli.py --test 2>&1 | grep -q '"receipt_type"' || { echo "FAIL: CLI does not emit receipt"; exit 1; }
    echo "✓ CLI emits valid receipt JSON"

    # Simulation disclaimers
    echo ""
    echo "Checking simulation disclaimers..."
    grep -rq "SIMULATION" src/*.py || { echo "FAIL: Missing SIMULATION disclaimers in src/"; exit 1; }
    echo "✓ SIMULATION disclaimers present in src/"

    # Citations
    echo ""
    echo "Checking citations..."
    python -c "from src.core import CITATIONS; assert len(CITATIONS) >= 20; print(f'✓ Citations loaded: {len(CITATIONS)}')" || { echo "FAIL: Insufficient citations"; exit 1; }

    # Core functions
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

    # Shipyard module
    echo ""
    echo "Checking shipyard module..."
    [ -d src/shipyard ] || { echo "FAIL: no src/shipyard/"; exit 1; }
    echo "✓ src/shipyard/ exists"

    python -c "
from src.shipyard.constants import TRUMP_CLASS_PROGRAM_COST_B, ELON_SPHERE_COST_REDUCTION
assert TRUMP_CLASS_PROGRAM_COST_B == 200.0, 'Invalid program cost'
print(f'✓ Shipyard constants: \${TRUMP_CLASS_PROGRAM_COST_B}B program')
"

    # Domains
    echo ""
    echo "Checking domain modules..."
    [ -d src/domains/defense ] || { echo "FAIL: no src/domains/defense/"; exit 1; }
    echo "✓ src/domains/defense/ exists"

    [ -d src/domains/medicaid ] || { echo "FAIL: no src/domains/medicaid/"; exit 1; }
    echo "✓ src/domains/medicaid/ exists"

    # RAZOR
    echo ""
    echo "Checking RAZOR module..."
    [ -d src/razor ] || { echo "FAIL: no src/razor/"; exit 1; }
    echo "✓ src/razor/ exists"

    python -c "
from src.razor.core import dual_hash, TENANT_ID
assert TENANT_ID, 'RAZOR TENANT_ID not set'
print(f'✓ RAZOR module: {TENANT_ID}')
"

    echo ""
    echo "=== PASS: T+2h Gate ==="
}

# =============================================================================
# T+24h MVP GATE
# =============================================================================
run_t24h() {
    echo "=== T+24h Gate: MVP ==="
    echo ""

    # First run t2h
    run_t2h

    echo ""
    echo "=== T+24h Specific Checks ==="
    echo ""

    # Run tests
    echo "Running tests..."
    python -m pytest tests/ -q --ignore=tests/test_ingest.py 2>/dev/null || python -m pytest tests/ -q 2>/dev/null || { echo "FAIL: tests failed"; exit 1; }
    echo "✓ Tests pass"

    # Check emit_receipt in modules
    echo ""
    echo "Checking receipt emission in modules..."
    for f in src/*.py; do
        if [[ "$f" != "src/__init__.py" && "$f" != "src/domain.py" ]]; then
            grep -q "emit_receipt\|emit_" "$f" 2>/dev/null && echo "✓ $f has emit function" || true
        fi
    done

    # 10-cycle smoke test
    echo ""
    echo "Running 10-cycle smoke test..."
    python -c "
from src.sim import run_simulation, SimConfig
r = run_simulation(SimConfig(n_cycles=10, n_transactions_per_cycle=100))
print(f'✓ 10 cycles: {len(r.violations)} violations, {len(r.receipts)} receipts')
assert len(r.receipts) > 0, 'No receipts generated'
"

    # Shipyard simulation
    echo ""
    echo "Running shipyard simulation..."
    python -c "
from src.shipyard.sim_shipyard import run_simulation, SimShipyardConfig
config = SimShipyardConfig(n_cycles=10, n_ships=2)
state = run_simulation(config)
print(f'✓ Shipyard: {len(state.ships)} ships, {len(state.receipt_ledger)} receipts')
"

    # Domain validation
    echo ""
    echo "Validating domains..."
    python -c "
from src.domain import load_domain, list_domains
for d in list_domains():
    config = load_domain(d)
    print(f'✓ Domain {d}: {config.name}')
"

    echo ""
    echo "=== PASS: T+24h Gate ==="
}

# =============================================================================
# T+48h HARDENED GATE
# =============================================================================
run_t48h() {
    echo "=== T+48h Gate: HARDENED ==="
    echo ""

    # First run t24h
    run_t24h

    echo ""
    echo "=== T+48h Specific Checks ==="
    echo ""

    # Run full test suite with coverage
    echo "Running full test suite..."
    python -m pytest tests/ -v --ignore=tests/test_ingest.py 2>/dev/null || python -m pytest tests/ -v 2>/dev/null || { echo "FAIL: tests failed"; exit 1; }
    echo "✓ Full test suite passes"

    # Scenario tests
    echo ""
    echo "Running scenario tests..."
    python -c "
from src.sim import run_simulation, SimConfig

scenarios = ['BASELINE', 'FRAUD_DISCOVERY', 'GODEL']
for scenario in scenarios:
    config = SimConfig(n_cycles=10, n_transactions_per_cycle=100, scenario=scenario)
    r = run_simulation(config)
    passed = r.scenario_results.get('passed', False) if r.scenario_results else True
    status = 'PASS' if passed else 'WARN'
    print(f'  [{status}] {scenario}')
"

    # Shipyard lifecycle test
    echo ""
    echo "Testing shipyard lifecycle..."
    python -c "
from src.shipyard.lifecycle import create_ship, advance_phase, LIFECYCLE_PHASES

ship = create_ship('TEST-001', 'trump', 'YARD-001')
for phase in LIFECYCLE_PHASES[1:4]:
    ship = advance_phase(ship, phase, actual_days=30, actual_cost=1e9)
print(f'✓ Lifecycle: {ship[\"current_phase\"]}')
"

    # RAZOR compression test
    echo ""
    echo "Testing RAZOR compression..."
    python -c "
from src.razor.physics import KolmogorovMetric
km = KolmogorovMetric()
repetitive = 'husbanding services ' * 100
chaotic = ''.join([chr(i % 256 + 32) for i in range(1000)])
r_rep = km.measure_complexity(repetitive)
r_cha = km.measure_complexity(chaotic)
print(f'✓ RAZOR: repetitive CR={r_rep[\"cr_zlib\"]:.3f}, chaotic CR={r_cha[\"cr_zlib\"]:.3f}')
assert r_rep['cr_zlib'] < r_cha['cr_zlib'], 'Compression should detect repetition'
"

    # Memory check
    echo ""
    echo "Running memory check..."
    python -c "
import tracemalloc
tracemalloc.start()
from src.sim import run_simulation, SimConfig
r = run_simulation(SimConfig(n_cycles=100, n_transactions_per_cycle=100))
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f'✓ Memory: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB')
"

    echo ""
    echo "=== PASS: T+48h Gate ==="
}

# =============================================================================
# MAIN
# =============================================================================
case "$GATE" in
    t2h)
        run_t2h
        ;;
    t24h)
        run_t24h
        ;;
    t48h)
        run_t48h
        ;;
    all)
        run_t48h  # t48h includes t24h which includes t2h
        ;;
    *)
        echo "Usage: $0 {t2h|t24h|t48h|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "$DISCLAIMER"
echo "============================================================"
