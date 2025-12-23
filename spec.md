# Gov-OS Specification v5.0

**THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY**

---

## System Purpose

Gov-OS is a universal federal fraud detection operating system that detects procurement fraud by measuring data complexity. Legitimate markets are chaotic (high entropy). Fraud requires coordination, which creates patterns (low entropy). Low entropy data compresses better than legitimate data.

**Core Principle:** Compression ratio reveals fraud.

---

## Architecture Overview

Gov-OS unifies multiple fraud detection subsystems into a single cohesive platform:

| Subsystem | Purpose | Location |
|-----------|---------|----------|
| Core Detection | Compression-based fraud detection | `src/` |
| Domain Modules | Extensible domain adapters | `src/domains/` |
| Shipyard | Trump-class battleship tracking | `src/shipyard/` |
| RAZOR | Kolmogorov validation engine | `src/razor/` |

### Version History

| Version | Focus | Key Innovation |
|---------|-------|----------------|
| v1.0 | Detection | Receipt chains + Merkle anchors |
| v2.0 | Physics | Entropy-based pattern emergence |
| v3.0 OMEGA | Certainty | Kolmogorov complexity + ZK proofs |
| v4.0 | Usability | Plain-language explanations + self-improvement |
| v5.0 | Unification | Single cohesive platform with domain modules |

---

## Unified Directory Structure

```
gov-os/
├── cli.py                      # Unified command-line interface
├── spec.md                     # This specification
├── ledger_schema.json          # Core receipt type definitions
├── CLAUDEME.md                 # Execution standard
├── CITATIONS.md                # Source references
├── DISCLAIMER.md               # Legal disclaimers
│
├── src/                        # Source code
│   ├── __init__.py             # Package exports
│   ├── core.py                 # Foundation: hash, receipts, citations
│   ├── domain.py               # Domain loader and registry
│   │
│   ├── # Core Detection
│   ├── compress.py             # Entropy compression analysis
│   ├── detect.py               # Multi-stage anomaly detection
│   ├── kolmogorov.py           # Algorithmic complexity
│   ├── zkp.py                  # Zero-knowledge proofs
│   ├── raf.py                  # Network cycle detection
│   ├── holographic.py          # Boundary-only detection
│   ├── thompson.py             # Bayesian audit sampling
│   ├── ledger.py               # Merkle ledger + Bekenstein bounds
│   ├── bridge.py               # Cross-branch translation
│   ├── sim.py                  # Scenario simulation engine
│   │
│   ├── # v4.0 User-Friendly
│   ├── insight.py              # Plain-language explanations
│   ├── fitness.py              # Self-improving pattern tracking
│   ├── guardian.py             # Evidence quality gates
│   ├── freshness.py            # Evidence staleness detection
│   ├── learner.py              # Cross-domain pattern transfer
│   │
│   ├── # Integration
│   ├── usaspending_etl.py      # USASpending.gov integration
│   ├── sam_validator.py        # SAM.gov vendor validation
│   │
│   ├── domains/                # Domain-specific modules
│   │   ├── __init__.py
│   │   ├── defense/            # Defense spending domain
│   │   │   ├── config.yaml
│   │   │   ├── data.py
│   │   │   ├── receipts.py
│   │   │   ├── scenarios.py
│   │   │   ├── schema.py
│   │   │   └── volatility.py
│   │   └── medicaid/           # Medicaid spending domain
│   │       ├── config.yaml
│   │       ├── data.py
│   │       ├── receipts.py
│   │       ├── scenarios.py
│   │       ├── schema.py
│   │       └── volatility.py
│   │
│   ├── shipyard/               # Trump-class battleship module
│   │   ├── constants.py        # Verified values with citations
│   │   ├── receipts.py         # 8 shipbuilding receipt types
│   │   ├── lifecycle.py        # Keel-to-delivery state machine
│   │   ├── assembly.py         # Block welding + robotics
│   │   ├── additive.py         # 3D printing validation
│   │   ├── iterate.py          # SpaceX-style rapid iteration
│   │   ├── nuclear.py          # SMR reactor installation
│   │   ├── procurement.py      # Contract management
│   │   └── sim_shipyard.py     # Monte Carlo simulation
│   │
│   └── razor/                  # Kolmogorov validation engine
│       ├── core.py             # RAZOR constants and receipts
│       ├── cohorts.py          # Historical fraud cohorts
│       ├── ingest.py           # USASpending API client
│       ├── physics.py          # Compression metrics
│       └── validate.py         # Statistical validation
│
├── schemas/                    # Ledger schema definitions
│   ├── ledger_schema_domains.json
│   ├── ledger_schema_razor.json
│   └── ledger_schema_shipyard.json
│
├── tests/                      # Unified test suite
│   ├── conftest.py
│   ├── test_compress.py
│   ├── test_detect.py
│   ├── test_v4_modules.py
│   ├── test_shipyard_*.py
│   ├── test_core_*.py
│   ├── test_modules_*.py
│   └── test_*.py
│
├── data/                       # Data and citations
│   └── citations/
│
└── gate.sh                     # Unified execution gate runner
```

---

## CLI Commands

### Core Commands

```bash
gov-os --test                           # System test
gov-os --version                        # Version info
gov-os scenario --run BASELINE          # Run simulation scenario
gov-os export --scenario BASELINE       # Export results
```

### v4.0 User-Friendly Commands

```bash
gov-os explain --demo                   # Plain-language demo
gov-os health                           # System health check
gov-os patterns --list                  # View fraud patterns
gov-os freshness --demo                 # Evidence freshness demo
```

### Domain Commands

```bash
gov-os defense simulate --cycles 100    # Defense domain simulation
gov-os defense scenario BASELINE        # Run defense scenario
gov-os defense scenarios                # Run all defense scenarios

gov-os medicaid simulate --cycles 100   # Medicaid domain simulation
gov-os medicaid scenario PROVIDER_RING  # Run medicaid scenario

gov-os validate --domain all            # Validate all domains
gov-os list domains                     # List available domains
```

### RAZOR Commands

```bash
gov-os razor --test                     # RAZOR quick test
gov-os razor --gate api                 # API connectivity gate
gov-os razor --gate compression         # Compression analysis gate
gov-os razor --cohorts                  # List fraud cohorts
```

### Shipyard Commands

```bash
gov-os shipyard --status                # Program status
gov-os shipyard --simulate              # Run simulation
```

---

## Detection Thresholds

| Metric | Fraud Threshold | Legitimate Threshold | Source |
|--------|-----------------|---------------------|--------|
| Kolmogorov Complexity | K(x) < 0.65 | K(x) >= 0.75 | Compression physics |
| Compression Ratio | < 0.50 | >= 0.80 | Shannon 1948 |
| RAF Cycle Length | 3-5 entities | N/A | Network topology |
| ZKP Proof Size | 22 KB constant | N/A | Mina IVC |
| Evidence Freshness | > 90 days = stale | < 30 days = fresh | v4.0 |
| Pattern Confidence | > 0.50 match | N/A | v4.0 learner |

---

## Module Inventory

### Core Detection (src/)

| Module | Purpose |
|--------|---------|
| core.py | Hash, receipts, citations, constants |
| compress.py | Entropy compression analysis |
| detect.py | Multi-stage anomaly detection |
| kolmogorov.py | Algorithmic complexity |
| zkp.py | Zero-knowledge proofs |
| raf.py | Network cycle detection |
| holographic.py | Boundary-only detection |
| thompson.py | Bayesian audit sampling |

### v4.0 User-Friendly (src/)

| Module | Purpose |
|--------|---------|
| insight.py | Plain-language explanations |
| fitness.py | Self-improving pattern tracking |
| guardian.py | Evidence quality gates |
| freshness.py | Evidence staleness detection |
| learner.py | Cross-domain pattern transfer |

### Domains (src/domains/)

| Domain | Purpose |
|--------|---------|
| defense | Defense spending, military contracts |
| medicaid | Healthcare spending, provider payments |

### Shipyard (src/shipyard/)

| Module | Purpose |
|--------|---------|
| constants.py | Verified values with citations |
| receipts.py | 8 shipbuilding receipt types |
| lifecycle.py | Keel-to-delivery state machine |
| assembly.py | Block welding + robotics |
| additive.py | 3D printing validation |
| iterate.py | SpaceX-style rapid iteration |
| nuclear.py | SMR reactor installation |
| procurement.py | Contract management |
| sim_shipyard.py | Monte Carlo simulation |

### RAZOR (src/razor/)

| Module | Purpose |
|--------|---------|
| core.py | Constants, receipts, hashing |
| cohorts.py | Historical fraud cohort definitions |
| ingest.py | USASpending.gov API client |
| physics.py | Kolmogorov complexity measurement |
| validate.py | Statistical signal detection |

---

## Domain Architecture

Domains are plug-in modules that adapt the core physics engine to specific spending domains.

### Domain Structure

Each domain provides:
- `config.yaml`: Domain configuration
- `volatility.py`: Domain-specific volatility index
- `schema.py`: Data schemas
- `receipts.py`: Domain receipt types
- `scenarios.py`: Domain scenarios
- `data.py`: Simulated data generators

### Adding a New Domain

1. Create `src/domains/{name}/config.yaml`
2. Implement `volatility.py` with `get_primary_volatility()`
3. Implement `schema.py` with data schemas
4. Implement `receipts.py` with receipt types
5. Implement `scenarios.py` with test scenarios
6. Register in `src/domains/__init__.py`

---

## Shipyard Module: Trump-Class Program

The Shipyard module tracks the announced $200B, 22-ship battleship program.

### Program Constants (Cited)

| Constant | Value | Citation |
|----------|-------|----------|
| TRUMP_CLASS_PROGRAM_COST_B | $200B | TRUMP_2025 |
| TRUMP_CLASS_SHIP_COUNT | 22 ships | TRUMP_2025 |
| TRUMP_CLASS_PER_SHIP_B | $9.09B/ship | Derived |
| FORD_CVN78_OVERRUN_PCT | 23% | GAO_2022 |
| ZUMWALT_COST_INCREASE_PCT | 81% | GAO_2018 |

### Shipyard Receipt Types (8)

| Receipt | Trigger | Key Fields |
|---------|---------|------------|
| keel | Ship construction start | ship_id, hull_number |
| block | Hull section assembly | block_id, welds, inspection |
| additive | 3D printed section | material, layer_hash |
| iteration | Design cycle complete | iteration_count, delta |
| milestone | Phase complete | phase, cost_to_date |
| procurement | Contract action | contract_type, amount |
| propulsion | Reactor installation | reactor_type, power_mwe |
| delivery | Ship handoff | final_cost, variance_pct |

---

## RAZOR Validation Engine

RAZOR validates fraud detection against real USASpending.gov data using compression-based analysis.

### The Paradigm

```
Corrupt markets are ordered. Order compresses.
Honest markets are chaotic. Chaos resists compression.

K(x) = len(compressed) / len(original)

The compression ratio IS the proof.
```

### Historical Fraud Cohorts

| Cohort | Years | Pattern | Expected Signal |
|--------|-------|---------|-----------------|
| Fat Leonard (GDMA) | 2006-2013 | Repetitive billing | Z < -2.0 |
| TransDigm | 2015-2019 | Price gouging | Price/estimate > 2x |
| Boeing/Druyun | 2000-2003 | Conflict of interest | Approval anomaly |

---

## v4.0 User-Friendly Features

### Plain-Language Explanations (insight.py)

```
Technical: compression_ratio=0.42, kolmogorov=0.38, entropy=5.2
Plain: "This contract appears to involve copied or templated billing.
       The billing records show unusually repetitive patterns."
```

### Evidence Freshness (freshness.py)

| Level | Age | Confidence |
|-------|-----|------------|
| Fresh | < 30 days | 100% |
| Recent | 30-60 days | 90% |
| Aging | 60-90 days | 70% |
| Stale | 90-180 days | 40% |
| Expired | > 180 days | 10% |

### Cross-Domain Pattern Transfer (learner.py)

| Pattern | Source Case | Transferability |
|---------|-------------|-----------------|
| Repetitive Billing | Fat Leonard | 85% |
| Price Gouging | TransDigm | 90% |
| Shell Company | General | 95% |
| Conflict of Interest | Boeing/Druyun | 75% |
| Cost Escalation | General | 80% |

---

## Receipt Types (67 Total)

### Core (12)
warrant, quality_attestation, milestone, cost_variance, anchor, detection, compression, lineage, bridge, simulation, anomaly, violation

### v2 Physics (8)
threshold, pattern_emergence, entropy_tree, cascade_alert, epidemic_warning, holographic, meta_receipt, mutual_info

### v3 OMEGA (12)
kolmogorov, zkp, raf, das, adversarial, usaspending, layout_entropy, sam_validation, catalytic, holographic_da, thompson_audit, bekenstein

### v4 User-Friendly (17)
insight, fitness, health, quality, abstain, counter_evidence, integrity, gate, freshness, refresh_priority, monitoring, pattern_match, transfer, learn, library_summary, prune

### Shipyard (8)
keel, block, additive, iteration, milestone, procurement, propulsion, delivery

### RAZOR (5)
ingest, cohort, compression, validation, signal

### Domain (5)
domain_receipt, domain_simulation, domain_scenario, domain_validation, domain_volatility

---

## Scenarios

### Core Scenarios

| Scenario | Purpose | Pass Criteria |
|----------|---------|---------------|
| BASELINE | Standard procurement | compression >= 0.85, recall >= 0.90 |
| SHIPYARD_STRESS | Trump-class simulation | detect fraud by ship 10 |
| CROSS_BRANCH_INTEGRATION | Multi-system | zero proof failures |
| FRAUD_DISCOVERY | Novel patterns | legitimate >= 0.80, fraud <= 0.40 |
| GODEL | Edge cases | no crashes, stoprules trigger |
| AUTOCATALYTIC | Pattern emergence | N_critical < 10,000 |
| THOMPSON | Bayesian calibration | FP <= 2% |
| HOLOGRAPHIC | Boundary detection | p > 0.9999 |

### Domain Scenarios

Each domain provides:
- BASELINE: Standard operation
- STRESS: High volume
- FRAUD_INJECTION: Synthetic fraud

---

## Execution Gates

All gates are run via the unified `gate.sh` script:

```bash
./gate.sh t2h      # T+2h skeleton gate
./gate.sh t24h     # T+24h MVP gate (includes t2h)
./gate.sh t48h     # T+48h hardened gate (includes t24h)
./gate.sh all      # Run all gates
```

| Gate | Requirement | Checks |
|------|-------------|--------|
| T+2h | Skeleton working | Core files, CLI, disclaimers, citations, all modules |
| T+24h | MVP complete | Tests, smoke tests, simulations, domain validation |
| T+48h | Hardened | Full test suite, scenarios, memory check |

---

## Key Formulas

### Kolmogorov Complexity
```
K(x) = compressed_size / original_size
K(x) < 0.65 -> likely generated/templated
K(x) >= 0.75 -> likely legitimate
```

### Entropy Fitness
```
fitness = (H_before - H_after) / receipts_processed
fitness > 0 -> pattern reduces uncertainty
fitness < 0 -> pattern adds noise
```

### N_critical (Phase Transition)
```
N_critical = log2(1/dH) x (H_legit / dH)
When N > N_critical, patterns become distinguishable
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial: receipts, compression, detection |
| 2.0.0 | 2024-12-23 | Physics: entropy, autocatalytic, Thompson |
| 3.0.0 | 2024-12-23 | OMEGA: Kolmogorov, ZKP, RAF, DA sampling |
| 3.1.0 | 2024-12-23 | Shipyard: Trump-class, 8 receipt types |
| 4.0.0 | 2024-12-23 | User-friendly: insight, fitness, guardian, freshness, learner |
| 5.0.0 | 2024-12-23 | Unification: Single cohesive platform with domain modules |

---

**THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY**
