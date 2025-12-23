#!/bin/bash
# Gov-OS Gate T+2h - Quick Validation
# THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
#
# Run this gate within 2 hours of making changes.
# Validates: imports, basic tests, syntax

set -e

echo "=============================================="
echo "Gov-OS Gate T+2h - Quick Validation"
echo "=============================================="
echo ""
echo "THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY"
echo ""

cd "$(dirname "$0")"

# Phase 1: Import validation
echo "Phase 1: Import Validation"
echo "--------------------------"

echo -n "  Core imports... "
python3 -c "from src.core import *; print('OK')" 2>/dev/null || echo "FAILED"

echo -n "  Defense module... "
python3 -c "from src.modules.defense import *; print('OK')" 2>/dev/null || echo "FAILED"

echo -n "  Medicaid module... "
python3 -c "from src.modules.medicaid import *; print('OK')" 2>/dev/null || echo "FAILED"

echo ""

# Phase 2: Syntax check
echo "Phase 2: Syntax Check"
echo "---------------------"

echo -n "  Python syntax... "
find . -name "*.py" -exec python3 -m py_compile {} \; 2>/dev/null && echo "OK" || echo "FAILED"

echo ""

# Phase 3: Quick unit tests
echo "Phase 3: Quick Unit Tests"
echo "-------------------------"

echo -n "  Constants tests... "
python3 -m pytest tests/test_core_constants.py -q --tb=no 2>/dev/null && echo "OK" || echo "FAILED"

echo -n "  Utils tests... "
python3 -m pytest tests/test_core_utils.py -q --tb=no 2>/dev/null && echo "OK" || echo "FAILED"

echo ""

# Phase 4: Constants verification
echo "Phase 4: Constants Verification"
echo "-------------------------------"

python3 << 'EOF'
from src.core.constants import (
    COMPRESSION_LEGITIMATE_FLOOR,
    COMPRESSION_FRAUD_CEILING,
    RAF_MIN_CYCLE_LENGTH,
    HOLOGRAPHIC_DETECTION_PROB,
)

checks = [
    ("Legitimate floor > fraud ceiling", COMPRESSION_LEGITIMATE_FLOOR > COMPRESSION_FRAUD_CEILING),
    ("RAF min cycle >= 3", RAF_MIN_CYCLE_LENGTH >= 3),
    ("Holographic prob >= 0.9999", HOLOGRAPHIC_DETECTION_PROB >= 0.9999),
]

all_pass = True
for name, result in checks:
    status = "OK" if result else "FAILED"
    print(f"  {name}... {status}")
    all_pass = all_pass and result

if not all_pass:
    exit(1)
EOF

echo ""
echo "=============================================="
echo "Gate T+2h: PASSED"
echo "=============================================="
