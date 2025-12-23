# WarrantProof Specification v2.0

## ⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This document specifies the inputs, outputs, receipts, SLOs, stoprules,
and rollback procedures for the WarrantProof simulation system.

---

## Overview

WarrantProof is a simulation and research system that models how receipts-native
accountability infrastructure could theoretically apply to defense procurement
and operations, based solely on publicly available data.

**v1 Core Paradigm:** Fraud creates entropy. Compression detects disorder.
Every decision becomes a cryptographic warrant. Receipt chains prove accountability lineage.

---

## v2 PARADIGM: Thermodynamic Autocatalytic Pattern Emergence

v2 represents a fundamental shift from hardcoded rule matching to physics-based
pattern emergence. The system discovers fraud patterns through entropy dynamics
rather than explicit programming.

### Core Physics Principles

| Principle | Application | Module |
|-----------|-------------|--------|
| Shannon Entropy | Legitimate receipts compress (low H), fraud resists (high H) | autocatalytic.py |
| Kolmogorov Complexity | Compression ratio ≈ minimum description length | compress.py |
| Autocatalytic RAF Sets | Patterns crystallize when catalytic coverage > threshold | autocatalytic.py |
| Thompson Sampling | Bayesian thresholds collapse contextually per branch | thompson.py |
| Holographic Principle | Volume entropy encodes on boundary (Merkle root) | holographic.py |
| SIR Epidemic Model | R₀ = density × volume / latency for vendor spread | epidemic.py |
| Entropy Trees | O(log N) hierarchical detection vs O(N) scan | entropy_tree.py |
| dC/dt Cascade Detection | Compression derivative as leading indicator | cascade.py |

### Key Insight

Instead of programming "if transaction > $10M and vendor in watchlist, flag fraud",
the v2 system observes that fraudulent transactions have measurably higher entropy
than legitimate ones. When enough receipts accumulate, patterns **crystallize
spontaneously** - they emerge from the data rather than being imposed on it.

### N_critical: The Phase Transition

The critical insight is the N_critical formula:
```
N_critical = log₂(1/ΔH) × (H_legit / ΔH)
```

Where:
- ΔH = H_fraud - H_legit (entropy gap)
- H_legit = average entropy of legitimate receipts
- H_fraud = average entropy of fraudulent receipts

When N > N_critical, patterns become statistically distinguishable. Below this
threshold, detection is unreliable. The system tracks this phase transition
in real-time.

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

## v2 Module Architecture

### New Modules (v2)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| thompson.py | Bayesian threshold sampling | `sample_threshold()`, `update_posterior()`, `contextual_collapse()` |
| autocatalytic.py | Pattern emergence without hardcoding | `autocatalytic_detect()`, `crystallize_pattern()`, `compute_entropy_gap()` |
| entropy_tree.py | O(log N) hierarchical detection | `build_entropy_tree()`, `entropy_bisect()`, `search_tree()` |
| cascade.py | dC/dt monitoring for early alerts | `calculate_compression_derivative()`, `detect_cascade_onset()`, `alert_early_warning()` |
| epidemic.py | R₀ vendor spread modeling | `calculate_R0()`, `SIR_model_step()`, `predict_spread()`, `recommend_quarantine()` |
| holographic.py | Boundary-only fraud detection | `holographic_detect()`, `compute_merkle_syndrome()`, `detect_from_boundary()` |
| meta_receipt.py | Receipts about receipts | `emit_meta_receipt()`, `validate_self_reference()`, `test_autocatalytic_closure()` |

### Modified Modules (v2 enhancements)

| Module | v2 Additions |
|--------|--------------|
| core.py | Physics-derived constants (N_CRITICAL_FORMULA, ENTROPY_GAP_MIN, etc.) |
| compress.py | `compression_derivative()`, `field_wise_compression()`, `compress_receipt_with_entropy()` |
| detect.py | `autocatalytic_detect()` mode, v1 patterns deprecated to fallback |
| ledger.py | `holographic_detect()`, `get_merkle_root()`, `verify_holographic_integrity()` |
| bridge.py | `mutual_information()`, `transfer_pattern()`, `cross_branch_learning()` |

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

### v2 Receipt Types

| Type | Module | Frequency | Key Fields |
|------|--------|-----------|------------|
| threshold | thompson.py | Per collapse | mean, variance, branch_context |
| pattern_emergence | autocatalytic.py | Per crystallization | entropy_gap, coherence, N_critical |
| entropy_tree | entropy_tree.py | Per tree update | depth, node_count, rebalanced |
| cascade_alert | cascade.py | Per onset detection | dC_dt, time_to_cascade, severity |
| epidemic_warning | epidemic.py | Per R₀ threshold | R0, infected_vendors, quarantine_rec |
| holographic | holographic.py | Per boundary check | syndrome, fraud_detected, bits_used |
| meta_receipt | meta_receipt.py | Per closure test | closure_test, self_reference_valid |
| mutual_info | bridge.py | Per transfer | source_branch, target_branch, MI_score |

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

### v1 SLOs (maintained)

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

### v2 SLOs (physics-derived)

| SLO | Threshold | Physics Basis | Module |
|-----|-----------|---------------|--------|
| N_critical | < 10,000 receipts | Phase transition bound | autocatalytic.py |
| Entropy gap | ΔH ≥ 0.15 bits | Pattern distinguishability | autocatalytic.py |
| Pattern coherence | ≥ 0.80 | Autocatalytic closure | autocatalytic.py |
| Thompson FP rate | ≤ 2% | Bayesian regret bound | thompson.py |
| Variance convergence | Yes | Posterior concentration | thompson.py |
| Cascade speedup | ≥ 5× vs v1 | dC/dt leading indicator | cascade.py |
| Epidemic R₀ | < 1.0 (contained) | SIR stability | epidemic.py |
| Detection probability | p > 0.9999 | Holographic bound | holographic.py |
| Bits per receipt | ≤ 2 | Bekenstein bound | holographic.py |
| Mutual info threshold | ≥ 0.30 | Transfer criterion | bridge.py |
| Cross-branch accuracy | ≥ 85% | Transfer fidelity | bridge.py |
| Tree depth | O(log N) | Hierarchical bound | entropy_tree.py |

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

### v2 Stoprules

#### stoprule_entropy_gap_insufficient
- **Trigger:** ΔH < ENTROPY_GAP_MIN (0.15)
- **Action:** Emit anomaly receipt, fall back to v1 detection
- **Classification:** deviation

#### stoprule_threshold_divergent
- **Trigger:** Thompson variance > 10× prior variance
- **Action:** Emit anomaly receipt, recalibrate prior
- **Classification:** deviation

#### stoprule_false_positive_exceeded
- **Trigger:** FP rate > THOMPSON_FP_TARGET (2%)
- **Action:** Emit anomaly receipt, tighten threshold
- **Classification:** deviation

#### stoprule_cascade_imminent
- **Trigger:** dC/dt > CASCADE_DERIVATIVE_THRESHOLD
- **Action:** Emit cascade_alert receipt, trigger early warning
- **Classification:** alert

#### stoprule_epidemic_spreading
- **Trigger:** R₀ > EPIDEMIC_R0_THRESHOLD (1.0)
- **Action:** Emit epidemic_warning receipt, recommend quarantine
- **Classification:** alert

#### stoprule_detection_probability_low
- **Trigger:** p < 0.9999 (holographic detection)
- **Action:** Emit anomaly receipt, increase encoding
- **Classification:** deviation

#### stoprule_boundary_bits_exceeded
- **Trigger:** bits > 2N (Bekenstein bound violation)
- **Action:** Emit anomaly receipt, optimize encoding
- **Classification:** violation

#### stoprule_pattern_incoherent
- **Trigger:** Coherence < PATTERN_COHERENCE_MIN (0.80)
- **Action:** Emit anomaly receipt, wait for more receipts
- **Classification:** deviation

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

## The 6 Mandatory Scenarios (v1)

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

## v2 Scenarios (Physics-Based)

### 7. AUTOCATALYTIC
Pattern emergence without hardcoded rules.
- Generate N receipts, measure entropy gap, wait for crystallization
- Track N_critical phase transition
- **Pass:** N_critical < 10,000, pattern_coherence ≥ 0.80

### 8. THOMPSON
Bayesian threshold calibration.
- Start with prior distribution, sample thresholds, update posterior
- Measure false positive rate convergence
- **Pass:** FP_rate ≤ 2%, variance_converged = True

### 9. HOLOGRAPHIC
Boundary-only fraud detection.
- Detect fraud from Merkle root alone without scanning receipts
- Verify Bekenstein bound (bits ≤ 2N)
- **Pass:** detection_probability > 0.9999, bits_per_receipt ≤ 2

---

## Data Flow

### v1 Flow
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

### v2 Flow (Physics Layer)
```
INPUT                    v2 PROCESSING                 OUTPUT
─────                    ─────────────                 ──────
Receipt Stream  ──►     entropy_tree.py  ──►     tree structure (O(log N))
                              │
                              ▼
                        autocatalytic.py ──►     pattern_emergence_receipt
                              │                  (crystallization)
                              ▼
                        thompson.py      ──►     threshold_receipt
                              │                  (Bayesian collapse)
                              ▼
                        cascade.py       ──►     cascade_alert_receipt
                              │                  (dC/dt monitoring)
                              ▼
                        epidemic.py      ──►     epidemic_warning_receipt
                              │                  (R₀ spread modeling)
                              ▼
                        holographic.py   ──►     holographic_receipt
                              │                  (boundary detection)
                              ▼
                        meta_receipt.py  ──►     meta_receipt
                                                 (autocatalytic closure)

Ledger          ──►     holographic_detect()    Fraud detected from
(Merkle root)           (O(1) boundary check)   boundary alone
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
| 2.0.0 | 2024-12-23 | v2 paradigm: thermodynamic autocatalytic pattern emergence, Thompson sampling, holographic detection |

### v2.0.0 Changelog

**New Modules:**
- thompson.py: Bayesian threshold sampling with contextual collapse
- autocatalytic.py: Pattern emergence without hardcoded rules
- entropy_tree.py: O(log N) hierarchical detection
- cascade.py: dC/dt compression derivative monitoring
- epidemic.py: R₀ vendor spread modeling with SIR
- holographic.py: Boundary-only fraud detection (Bekenstein bound)
- meta_receipt.py: Receipts-about-receipts for autocatalytic closure

**Enhanced Modules:**
- core.py: Physics-derived constants (N_critical formula, entropy gaps, etc.)
- compress.py: Compression derivative, field-wise entropy analysis
- detect.py: Autocatalytic detection mode (v1 patterns as fallback)
- ledger.py: Holographic detection from Merkle root
- bridge.py: Mutual information for cross-branch pattern transfer

**New Scenarios:**
- AUTOCATALYTIC: Pattern crystallization without hardcoding
- THOMPSON: Bayesian threshold calibration
- HOLOGRAPHIC: Boundary-only fraud detection

**Key Insight:**
The v2 paradigm inverts the detection model. Instead of "if amount > threshold, flag fraud",
patterns emerge from entropy dynamics. Fraud has higher Shannon entropy than legitimate
transactions. When N > N_critical, patterns crystallize spontaneously.

---

**⚠️ THIS IS A SIMULATION. NOT REAL DoD DATA. FOR RESEARCH ONLY. ⚠️**
