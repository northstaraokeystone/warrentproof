# WarrantProof Specification v3.0 — Project OMEGA

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

## v3 OMEGA PARADIGM: Deterministic Zero-Knowledge Proof Architecture

v3 (Project OMEGA) transforms WarrantProof from probabilistic detection to
cryptographic certainty. Key innovations:

| Component | v2 (Probabilistic) | v3 OMEGA (Deterministic) |
|-----------|-------------------|--------------------------|
| Complexity | Shannon entropy | Kolmogorov complexity |
| Proofs | Statistical confidence | ZK-SNARKs (Mina IVC) |
| Networks | Ad-hoc graph analysis | RAF autocatalytic cycles |
| Data integrity | Merkle verification | Data Availability Sampling |
| Robustness | Single-pass detection | Adversarial training (PGD) |
| External data | Manual integration | USASpending ETL + SAM.gov CA |
| Documents | Text-only analysis | Layout entropy (PDF structure) |

### OMEGA Physics Constants

```python
KOLMOGOROV_THRESHOLD = 0.65        # K(x) < 0.65 → templated/generated
BEKENSTEIN_BITS_PER_DOLLAR = 1e-6  # S ≤ B × amount (holographic bound)
RAF_CYCLE_MIN_LENGTH = 3           # Minimum corruption cycle
RAF_CYCLE_MAX_LENGTH = 5           # Maximum corruption cycle
ZKP_PROOF_SIZE_BYTES = 22000       # Mina-style 22KB IVC proof
DATA_AVAILABILITY_SAMPLE_RATE = 0.10  # 10% sampling for 99% confidence
```

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

## v3 OMEGA Module Architecture

### New Modules (OMEGA v3)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| kolmogorov.py | Algorithmic complexity via compression | `calculate_kolmogorov()`, `is_kolmogorov_fraud()`, `detect_generator_pattern()` |
| zkp.py | Recursive ZK-SNARKs (Mina IVC) | `generate_proof()`, `verify_proof()`, `recursive_compose()`, `verify_recursive_chain()` |
| raf.py | RAF network cycle detection | `build_transaction_graph()`, `detect_cycles()`, `identify_keystone_species()` |
| das.py | Data Availability Sampling | `encode_with_erasure()`, `sample_chunks()`, `verify_availability()`, `light_client_audit()` |
| adversarial.py | PGD attack generator | `pgd_attack()`, `fgsm_attack()`, `generate_adversarial_dataset()`, `evaluate_robustness()` |
| usaspending_etl.py | USASpending.gov ETL | `fetch_awards()`, `fetch_transactions()`, `handle_pagination()`, `detect_missing_fields()` |
| layout_entropy.py | PDF visual structure analysis | `extract_layout_features()`, `calculate_layout_entropy()`, `detect_scan_artifacts()` |
| sam_validator.py | SAM.gov CA validation | `fetch_entity()`, `validate_signature()`, `reject_na_fields()`, `validate_entity()` |

### Modified Modules (OMEGA v3 enhancements)

| Module | OMEGA Additions |
|--------|-----------------|
| core.py | OMEGA constants (KOLMOGOROV_THRESHOLD, BEKENSTEIN_BITS_PER_DOLLAR, etc.) |
| compress.py | `compress_receipt_kolmogorov()` - Kolmogorov complexity integration |
| thompson.py | `thompson_audit_selection()` - Multi-armed bandit for audit allocation |
| ledger.py | `validate_bekenstein_bound()` - Holographic information bound |
| detect.py | `zkp_verification_gate()` - ZKP proof as fraud verification |
| bridge.py | `detect_all_catalytic_links()` - Non-financial relationship detection |
| holographic.py | `holographic_detect_with_da()` - Data availability sampling |

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

### v3 OMEGA Receipt Types

| Type | Module | Frequency | Key Fields |
|------|--------|-----------|------------|
| kolmogorov | kolmogorov.py | Per analysis | kolmogorov_ratio, is_fraud, compression_algorithm |
| zkp | zkp.py | Per proof | proof_valid, proof_size_bytes, circuit_type |
| raf | raf.py | Per network scan | cycles_detected, keystone_species, cycle_lengths |
| das | das.py | Per audit | available, confidence, samples_checked |
| adversarial | adversarial.py | Per attack test | attack_type, epsilon, robust |
| usaspending | usaspending_etl.py | Per ETL batch | records_fetched, endpoint, fiscal_year |
| layout_entropy | layout_entropy.py | Per document | layout_entropy, is_suspicious, scan_artifacts |
| sam_validation | sam_validator.py | Per entity | entity_valid, signature_verified, ca_trust_score |
| catalytic | bridge.py | Per detection | total_catalytic_links, links_by_type |
| holographic_da | holographic.py | Per DA check | merkle_root, data_availability, erasure_encoding |
| thompson_audit | thompson.py | Per selection | selected_count, budget, exploration_exploitation_ratio |
| bekenstein | ledger.py | Per bound check | amount_usd, max_bits, bound_respected |

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

### v3 OMEGA SLOs

| SLO | Threshold | Physics Basis | Module |
|-----|-----------|---------------|--------|
| Kolmogorov threshold | K(x) < 0.65 → fraud | Algorithmic complexity | kolmogorov.py |
| Kolmogorov legitimate | K(x) ≥ 0.75 | Natural data complexity | kolmogorov.py |
| ZKP proof size | ≤ 22KB | Mina IVC constant | zkp.py |
| ZKP verification time | ≤ 100ms | O(1) verification | zkp.py |
| RAF cycle length | 3-5 entities | Corruption topology | raf.py |
| Data availability | > 99% confidence | Erasure coding bound | das.py |
| Adversarial epsilon | 0.01 L∞ | PGD robustness | adversarial.py |
| Layout entropy | < 1.0 → suspicious | Visual structure | layout_entropy.py |
| SAM.gov trust | ≥ 0.50 | CA validation | sam_validator.py |
| Bekenstein bound | S ≤ 1e-6 × $ | Holographic principle | ledger.py |
| Catalytic F1 | ≥ 0.80 | Detection accuracy | bridge.py |
| Thompson audit budget | 5% of contractors | Multi-armed bandit | thompson.py |

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

### v3 OMEGA Stoprules

#### stoprule_kolmogorov_anomaly
- **Trigger:** K(x) < KOLMOGOROV_THRESHOLD (0.65)
- **Action:** Emit kolmogorov anomaly receipt, flag for ZKP verification
- **Classification:** alert

#### stoprule_zkp_verification_failed
- **Trigger:** ZKP proof verification returns False
- **Action:** Emit anomaly receipt, reject transaction
- **Classification:** violation

#### stoprule_raf_cycle_detected
- **Trigger:** RAF cycle of length 3-5 detected
- **Action:** Emit raf_cycle receipt, escalate to investigation
- **Classification:** critical

#### stoprule_data_unavailable
- **Trigger:** DA confidence < 99%
- **Action:** Emit anomaly receipt, investigate withholding
- **Classification:** critical

#### stoprule_adversarial_attack
- **Trigger:** Adversarial perturbation detected
- **Action:** Emit adversarial receipt, harden model
- **Classification:** alert

#### stoprule_layout_entropy_anomaly
- **Trigger:** Layout entropy < LAYOUT_ENTROPY_THRESHOLD (1.0)
- **Action:** Emit layout_entropy receipt, flag document
- **Classification:** alert

#### stoprule_sam_validation_failed
- **Trigger:** Entity validation fails (N/A fields, expired, excluded)
- **Action:** Emit sam_validation receipt, reject entity
- **Classification:** violation

#### stoprule_bekenstein_violated
- **Trigger:** Data entropy exceeds Bekenstein bound for amount
- **Action:** Emit bekenstein receipt, flag transaction
- **Classification:** violation

#### stoprule_catalytic_cycle_detected
- **Trigger:** Catalytic links form cycle (shared address + board + temporal)
- **Action:** Emit catalytic receipt, escalate investigation
- **Classification:** critical

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
| 3.0.0 | 2024-12-23 | v3 OMEGA: deterministic ZKP architecture, Kolmogorov complexity, RAF networks, data availability |

### v3.0.0 OMEGA Changelog

**New Modules (8):**
- kolmogorov.py: Algorithmic complexity via compression (replaces Shannon entropy)
- zkp.py: Recursive ZK-SNARKs (Mina-style IVC, 22KB constant proofs)
- raf.py: RAF autocatalytic network cycle detection (keystone species)
- das.py: Data Availability Sampling via erasure coding
- adversarial.py: PGD attack generator for adversarial robustness
- usaspending_etl.py: USASpending.gov API ETL pipeline
- layout_entropy.py: PDF visual structure analysis (scan artifacts, alignment)
- sam_validator.py: SAM.gov entity validation as Certificate Authority

**Enhanced Modules (7):**
- core.py: OMEGA constants (KOLMOGOROV_THRESHOLD, BEKENSTEIN_BITS_PER_DOLLAR, etc.)
- compress.py: Kolmogorov complexity integration
- thompson.py: Multi-armed bandit audit selection (70/30 exploit/explore)
- ledger.py: Bekenstein bound validation
- detect.py: ZKP verification gate
- bridge.py: Catalytic link detection (shared addresses, board connections, IP proximity)
- holographic.py: Data availability sampling integration

**New Receipt Types (12):**
- kolmogorov, zkp, raf, das, adversarial, usaspending
- layout_entropy, sam_validation, catalytic, holographic_da
- thompson_audit, bekenstein

**Key OMEGA Insight:**
The v3 paradigm transforms probabilistic detection into cryptographic certainty.
Instead of "compression ratio suggests fraud with 95% confidence", OMEGA provides:
- Deterministic: K(x) < 0.65 → provably templated
- Verifiable: ZKP proof validates fraud claim
- Network-aware: RAF cycles identify corruption topology
- Data-available: Erasure coding prevents hiding evidence

**Physics Derivation:**
```
KOLMOGOROV_THRESHOLD = 0.65      # From Grok Q2 empirical analysis
BEKENSTEIN_BITS_PER_DOLLAR = 1e-6  # Holographic information bound
RAF_CYCLE_BOUNDS = [3, 5]        # Corruption network topology
ZKP_PROOF_SIZE = 22KB            # Mina constant-size IVC
```

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
