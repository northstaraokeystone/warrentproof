#!/bin/bash
# RAZOR Gate 4: Statistical Validation
# RUN THIS OR KILL PROJECT

set -e

echo "============================================================"
echo "GATE 4: STATISTICAL VALIDATION"
echo "============================================================"

cd "$(dirname "$0")"

python cli.py --gate validate || {
    echo "FAIL: Statistical validation gate failed"
    exit 1
}

echo "PASS: Gate 4 (Stats)"
