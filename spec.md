# WarrantProof Specification v1.0

## ⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This document specifies the inputs, outputs, receipts, SLOs, stoprules,
and rollback procedures for the WarrantProof simulation system.

---

## Overview

WarrantProof is a simulation and research system that models how receipts-native
accountability infrastructure could theoretically apply to defense procurement
and operations, based solely on publicly available data.

**Core Paradigm:** Fraud creates entropy. Compression detects disorder.
Every decision becomes a cryptographic warrant. Receipt chains prove accountability lineage.

---

## Inputs

### Transaction Data (Simulated)
- Contract actions (awards, modifications, deliveries)
- Quality attestations (inspections, certifications)
- Milestone events (program checkpoints)
- Cost variance reports (budget vs actual)

### Configuration
- `SimConfig`: Simulation parameters
  - `n_cycles`: Number of Monte Carlo iterations (default: 1000)
  - `n_transactions_per_cycle`: Transactions per cycle (default: 10000)
  - `fraud_injection_rate`: Percentage of fraudulent transactions (default: 0.05)
  - `branch_distribution`: Budget share per branch
  - `random_seed`: For reproducibility (default: 42)
  - `scenario`: One of 6 mandatory scenarios

### Source Data
- All data derived from public sources (see CITATIONS.md)
- No proprietary, classified, or non-public information
- All dollar figures are hypothetical simulations

---

## Outputs

### Receipts
Every operation emits a receipt. Receipt types:

| Type | Module | Frequency | Key Fields |
|------|--------|-----------|------------|
| warrant | warrant.py | Per transaction | approver, lineage, amount |
| quality_attestation | warrant.py | Per inspection | inspector, certification |
| milestone | warrant.py | Per checkpoint | program, status |
| cost_variance | warrant.py | When variance ≥5% | baseline, actual, variance_pct |
| anchor | ledger.py | Per batch | merkle_root, batch_size |
| detection | detect.py | Per anomaly | anomaly_type, confidence |
| compression | compress.py | Per analysis | compression_ratio, entropy |
| lineage | trace.py | Per query | approval_chain, gaps |
| bridge | bridge.py | Per translation | source_system, target_system |
| simulation | sim.py | Per scenario | scenario, pass/fail, metrics |

### Simulation Results
- `SimState`: Final simulation state
  - `receipts`: All generated receipts
  - `detections`: Detected anomalies
  - `compressions`: Compression analysis results
  - `violations`: Constraint violations
  - `merkle_roots`: Anchor hashes
  - `scenario_results`: Pass/fail with metrics

### Dashboard Export
- JSON format suitable for visualization
- Includes all citations and disclaimers
- X/Twitter thread format for executive summary

---

## SLO Thresholds

| SLO | Threshold | Test Assertion | Stoprule Action |
|-----|-----------|----------------|-----------------|
| Warrant generation | ≤ 50ms | `assert time <= 50` | emit violation |
| Citation coverage | 100% | `assert all_cited` | HALT |
| Compression (legitimate) | ≥ 0.80 | `assert ratio >= 0.80` | emit violation |
| Compression (fraud) | ≤ 0.50 | `assert ratio <= 0.50` | log discovery |
| Detection recall | ≥ 0.90 | `assert recall >= 0.90` | emit violation |
| False positive rate | ≤ 0.05 | `assert fp_rate <= 0.05` | emit violation |
| Merkle verification | 100% | `assert all_verify` | HALT |
| Lineage completeness | ≥ 0.95 | `assert completeness >= 0.95` | emit violation |
| Scan latency | ≤ 100ms/1000 receipts | `assert time <= 100` | emit violation |
| Translation latency | ≤ 200ms | `assert time <= 200` | emit violation |

---

## Stoprules

Stoprules are CLAUDEME-mandated exception handlers that emit anomaly receipts
and halt execution on critical failures.

### stoprule_hash_mismatch
- **Trigger:** Merkle root verification fails
- **Action:** Emit anomaly receipt, HALT execution
- **Classification:** violation

### stoprule_invalid_receipt
- **Trigger:** Receipt missing required fields
- **Action:** Emit anomaly receipt, HALT execution
- **Classification:** violation

### stoprule_uncited_data
- **Trigger:** Data claim without citation
- **Action:** Emit violation receipt, HALT execution
- **Classification:** violation

### stoprule_missing_approver
- **Trigger:** Warrant without approver field
- **Action:** Emit violation receipt, HALT execution
- **Classification:** violation

### stoprule_missing_lineage
- **Trigger:** Receipt without parent reference (when required)
- **Action:** Emit violation receipt, HALT execution
- **Classification:** violation

### stoprule_budget_exceeded
- **Trigger:** Operation exceeds budget constraints
- **Action:** Emit anomaly receipt, reject operation
- **Classification:** violation

---

## Rollback Procedures

### On Stoprule Trigger
1. Emit anomaly/violation receipt
2. Log state to receipts.jsonl
3. Halt current operation
4. Return error to caller
5. Do NOT continue processing

### On Scenario Failure
1. Log all receipts and state
2. Emit simulation receipt with failure details
3. Report which criteria failed
4. Allow retry with modified parameters

### On System Error
1. Catch exception
2. Emit anomaly receipt with error details
3. Preserve state for debugging
4. Graceful shutdown

---

## The 6 Mandatory Scenarios

### 1. BASELINE
Standard military procurement simulation.
- 10,000 synthetic contract actions across 5 branches
- 95% "legitimate" patterns, 5% injected anomalies
- **Pass:** compression ≥ 0.85, detection recall ≥ 0.90, FP ≤ 5%

### 2. SHIPYARD_STRESS
Trump-class battleship program simulation.
- 20-ship program, $10B per hull
- Inject welding fraud, cost cascade, schedule slip
- **Pass:** Detect welding fraud by ship 10, predict overrun within 15%

### 3. CROSS_BRANCH_INTEGRATION
Unified receipt layer across incompatible systems.
- Navy ERP, Army GFEBS, Air Force DEAMS
- Require proof chain connecting all three
- **Pass:** Zero proof failures, traceable lineage

### 4. FRAUD_DISCOVERY
Compression-based fraud detection without training.
- Inject 3 novel fraud patterns
- **Pass:** compression ≥ 0.80 on legitimate, ≤ 0.40 on fraud

### 5. REAL_TIME_OVERSIGHT
Congressional/GAO dashboard simulation.
- Streaming at 1000/second
- **Pass:** Latency ≤ 100ms, alert ≤ 5s, zero dropped

### 6. GODEL
Edge cases and pathological inputs.
- Zero-dollar, $1T, circular chains, hash mismatch
- **Pass:** No crashes, graceful degradation, stoprules trigger

---

## Data Flow

```
INPUT                    PROCESSING                    OUTPUT
─────                    ──────────                    ──────
Transaction     ──►     warrant.py      ──►     warrant_receipt
                              │
                              ▼
                        ledger.py       ──►     anchor_receipt
                              │
                              ▼
                        detect.py       ──►     detection_receipt
                              │
                              ▼
                        compress.py     ──►     compression_receipt
                              │
                              ▼
                        trace.py        ──►     lineage_receipt
                              │
                              ▼
                        sim.py          ──►     simulation_receipt
```

---

## Security Considerations

1. **No Real Data:** System processes only synthetic/simulated data
2. **No Network Access:** No external API calls or data retrieval
3. **No Persistence:** Ledger is file-based for simulation only
4. **No Authentication:** Approvers are simulated identifiers
5. **No Authorization:** All operations are research-mode only

---

## Legal Requirements

1. Every output includes SIMULATION disclaimer
2. Every data claim includes citation
3. No assertions about actual DoD operations
4. No identification of specific vendors
5. No fraud accusations - pattern detection only

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial specification |

---

**⚠️ THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY. ⚠️**
