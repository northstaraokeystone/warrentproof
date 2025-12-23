#!/bin/bash
# RAZOR Gate 1: API Connectivity
# RUN THIS OR KILL PROJECT

set -e

echo "============================================================"
echo "GATE 1: API CONNECTIVITY"
echo "============================================================"

cd "$(dirname "$0")"

python cli.py --gate api || {
    echo "FAIL: API connectivity gate failed"
    exit 1
}

echo "PASS: Gate 1 (API)"
