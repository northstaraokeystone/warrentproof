#!/bin/bash
# RAZOR Gate 3: Compression Analysis
# RUN THIS OR KILL PROJECT

set -e

echo "============================================================"
echo "GATE 3: COMPRESSION ANALYSIS"
echo "============================================================"

cd "$(dirname "$0")"

python cli.py --gate compression || {
    echo "FAIL: Compression gate failed"
    exit 1
}

echo "PASS: Gate 3 (Compression)"
