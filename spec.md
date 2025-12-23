# Gov-OS Specification v6.0

**THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY**

---

## System Purpose

Gov-OS is a universal federal fraud detection operating system that detects procurement fraud by measuring data complexity. Legitimate markets are chaotic (high entropy). Fraud requires coordination, which creates patterns (low entropy). Low entropy data compresses better than legitimate data.

**Core Principle:** Compression ratio reveals fraud.

### v6.0 ProofChain Integration

v6.0 integrates 12 standalone proof builds into Gov-OS as modular domain plugins:

| ID | Module | Target Domain | Annual Exposure | Priority |
|----|--------|---------------|-----------------|----------|
| D1 | doge | Efficiency claims verification | $214B claimed | P0 |
| S1 | spend | Federal budget verification | $34T budget | P0 |
| G1 | green | Climate/ESG accountability | $40T ESG market | P0 |
| B1 | benefit | Government benefits (Medicaid) | $100B fraud | P0 |
| V1 | vote | Election integrity | 160M votes | P1 |
| C1 | claim | Government claims | $500B+ claims | P1 |
| F1 | safety | Pharmacovigilance | Drug safety | P1 |
| K1 | coin | Crypto ownership verification | $3B+ unverified | P1 |
| O1 | origin | Supply chain verification | $500B counterfeit | P1 |
| R1 | graft | FBI public corruption tracking | Corruption cases | P1 |
| W1 | warrant | Military efficiency | $850B budget | P1 |
| L1 | lab | Scientific reproducibility | Research crisis | P2 |

**The Paradigm Inversion:**
- ❌ OLD: Build standalone proofs for each domain. Each proof has its own infrastructure.
- ✅ NEW: Build modular domain plugins for unified Gov-OS. Share infrastructure. Detect cross-domain fraud via super-graph contagion.

**The Core Insight:**
> "When a shell entity links DOGEProof efficiency claims to SpendProof budget flows to BenefitProof disbursements, a fraud detection in any single domain propagates temporal rigidity to all linked domains—flagging preemptive fraud before local evidence exists."

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GOV-OS v6.0 ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        CORE INFRASTRUCTURE                              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │    │
│  │  │  ledger  │ │  anchor  │ │  detect  │ │ temporal │ │   loop   │      │    │
│  │  │ receipts │ │ dual-hash│ │ entropy  │ │  decay   │ │ self-imp │      │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                   │    │
│  │  │   raf    │ │  insight │ │   gate   │ │ contagion│                   │    │
│  │  │  graphs  │ │plain-text│ │ T+2h/48h │ │super-grph│                   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                            │
│                    ┌───────────────┴───────────────┐                            │
│                    ▼                               ▼                            │
│  ┌─────────────────────────────────┐ ┌─────────────────────────────────┐        │
│  │       PRIORITY 0 MODULES        │ │       PRIORITY 1 MODULES        │        │
│  │  ┌──────────┐ ┌──────────┐      │ │  ┌──────────┐ ┌──────────┐      │        │
│  │  │   doge   │ │  spend   │      │ │  │   vote   │ │  claim   │      │        │
│  │  │efficiency│ │  budget  │      │ │  │ election │ │gov claims│      │        │
│  │  └──────────┘ └──────────┘      │ │  └──────────┘ └──────────┘      │        │
│  │  ┌──────────┐ ┌──────────┐      │ │  ┌──────────┐ ┌──────────┐      │        │
│  │  │  green   │ │ benefit  │      │ │  │  safety  │ │   coin   │      │        │
│  │  │ ESG/clim │ │ medicaid │      │ │  │pharma-vig│ │  crypto  │      │        │
│  │  └──────────┘ └──────────┘      │ │  └──────────┘ └──────────┘      │        │
│  └─────────────────────────────────┘ │  ┌──────────┐ ┌──────────┐      │        │
│                                      │  │  origin  │ │  graft   │      │        │
│                                      │  │supply-chn│ │corruption│      │        │
│                                      │  └──────────┘ └──────────┘      │        │
│                                      │  ┌──────────┐                   │        │
│                                      │  │ warrant  │                   │        │
│                                      │  │ military │                   │        │
│                                      │  └──────────┘                   │        │
│                                      └─────────────────────────────────┘        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      PRIORITY 2 MODULES                                 │    │
│  │  ┌──────────┐                                                           │    │
│  │  │   lab    │                                                           │    │
│  │  │ science  │                                                           │    │
│  │  └──────────┘                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Version History

| Version | Focus | Key Innovation |
|---------|-------|----------------|
| v1.0 | Detection | Receipt chains + Merkle anchors |
| v2.0 | Physics | Entropy-based pattern emergence |
| v3.0 OMEGA | Certainty | Kolmogorov complexity + ZK proofs |
| v4.0 | Usability | Plain-language explanations + self-improvement |
| v5.0 | Unification | Single cohesive platform with domain modules |
| v5.1 | Temporal | Cross-domain contagion via temporal decay physics |
| v6.0 | ProofChain | 12 modules unified + cross-domain fraud detection |

---

## Unified Directory Structure

```
gov-os/
├── cli.py                              # Unified CLI with module commands
├── spec.md                             # This specification
├── ledger_schema.json                  # Core receipt type definitions
├── CLAUDEME.md                         # Execution standard
├── CITATIONS.md                        # Source references
├── DISCLAIMER.md                       # Legal disclaimers
├── gate.sh                             # Unified execution gate runner
│
├── src/                                # Source code
│   ├── __init__.py                     # Package exports
│   ├── core.py                         # Foundation: hash, receipts, citations
│   ├── domain.py                       # Domain loader and registry
│   │
│   ├── core/                           # SHARED INFRASTRUCTURE
│   │   ├── __init__.py                 # Export all core modules
│   │   ├── constants.py                # Universal constants (all modules)
│   │   ├── receipt.py                  # All receipt types
│   │   ├── utils.py                    # Utility functions (hash, merkle)
│   │   ├── ledger.py                   # Receipts storage, Merkle batching
│   │   ├── anchor.py                   # Dual-hash (SHA256:BLAKE3)
│   │   ├── compress.py                 # QED compression
│   │   ├── detect.py                   # Entropy-based detection
│   │   ├── raf.py                      # RAF graphs + super-graph builder
│   │   ├── temporal.py                 # Temporal decay (v5.1)
│   │   ├── contagion.py                # Cross-domain propagation (v6.0)
│   │   ├── insight.py                  # Plain-English explanations
│   │   ├── gate.py                     # Universal T+2h/24h/48h gates (v6.0)
│   │   ├── loop.py                     # LOOP self-improvement (v6.0)
│   │   ├── zk.py                       # ZK privacy layer (v6.0)
│   │   ├── harness.py                  # Simulation harness
│   │   └── volatility.py               # Volatility index
│   │
│   ├── modules/                        # DOMAIN MODULES (v6.0)
│   │   ├── __init__.py
│   │   ├── doge/                       # DOGEProof → Module (P0)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Domain constants
│   │   │   ├── ingest.py               # Efficiency claim ingestion
│   │   │   ├── verify.py               # Pre+post causality pairs
│   │   │   ├── receipts.py             # doge_proof, qed_claim receipts
│   │   │   ├── data.py                 # Sample data generators
│   │   │   └── scenario.py             # DOGE-specific scenarios
│   │   │
│   │   ├── spend/                      # SpendProof → Module (P0)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Budget verification constants
│   │   │   ├── ingest.py               # Disbursement ingestion
│   │   │   ├── verify.py               # ZK conservation proofs
│   │   │   ├── receipts.py             # disbursement_proof, audit_compress
│   │   │   ├── zk_circuits.py          # Groth16 over Pedersen
│   │   │   ├── data.py                 # Budget simulation data
│   │   │   └── scenario.py             # Budget-specific scenarios
│   │   │
│   │   ├── green/                      # GreenProof → Module (P0)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # ESG/climate constants
│   │   │   ├── ingest.py               # Emissions data ingestion
│   │   │   ├── verify.py               # Satellite timeline verification
│   │   │   ├── receipts.py             # green_proof, emissions_anchor
│   │   │   ├── data.py                 # ESG simulation data
│   │   │   └── scenario.py             # Climate-specific scenarios
│   │   │
│   │   ├── benefit/                    # BenefitProof → Module (P0)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Benefits constants
│   │   │   ├── ingest.py               # Enrollment/disbursement ingestion
│   │   │   ├── verify.py               # Qualification verification
│   │   │   ├── receipts.py             # benefit_disburse, fraud_compress
│   │   │   ├── data.py                 # Medicaid simulation data
│   │   │   └── scenario.py             # Benefits-specific scenarios
│   │   │
│   │   ├── vote/                       # VoteProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Election integrity constants
│   │   │   ├── ingest.py               # Vote record ingestion
│   │   │   ├── verify.py               # Adversarial ML hardening
│   │   │   ├── receipts.py             # vote_proof, tally_anchor
│   │   │   ├── data.py                 # Election simulation data
│   │   │   └── scenario.py             # Election-specific scenarios
│   │   │
│   │   ├── claim/                      # ClaimProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Government claims constants
│   │   │   ├── ingest.py               # Claim ingestion
│   │   │   ├── verify.py               # Political pressure modeling
│   │   │   ├── receipts.py             # claim_proof receipts
│   │   │   ├── data.py                 # Claims simulation data
│   │   │   └── scenario.py             # Claims-specific scenarios
│   │   │
│   │   ├── safety/                     # SafetyProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Pharmacovigilance constants
│   │   │   ├── ingest.py               # VAERS/FAERS ingestion
│   │   │   ├── verify.py               # ZK patient privacy
│   │   │   ├── receipts.py             # safety_proof, adverse_event
│   │   │   ├── data.py                 # Pharma simulation data
│   │   │   └── scenario.py             # Pharma-specific scenarios
│   │   │
│   │   ├── coin/                       # CoinProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Crypto verification constants
│   │   │   ├── ingest.py               # Wallet transaction ingestion
│   │   │   ├── verify.py               # Ownership clustering via compression
│   │   │   ├── receipts.py             # wallet_cluster, revenue_share
│   │   │   ├── data.py                 # Crypto simulation data
│   │   │   └── scenario.py             # Crypto-specific scenarios
│   │   │
│   │   ├── origin/                     # OriginProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Supply chain constants
│   │   │   ├── ingest.py               # Tier boundary ingestion
│   │   │   ├── verify.py               # Multi-tier authentication
│   │   │   ├── receipts.py             # tier_auth, origin_chain
│   │   │   ├── data.py                 # Supply chain simulation data
│   │   │   └── scenario.py             # Supply chain-specific scenarios
│   │   │
│   │   ├── graft/                      # GraftProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Corruption tracking constants
│   │   │   ├── ingest.py               # Case lifecycle ingestion
│   │   │   ├── verify.py               # Whistleblower privacy
│   │   │   ├── receipts.py             # graft_proof, case_chain
│   │   │   ├── data.py                 # Corruption simulation data
│   │   │   └── scenario.py             # Corruption-specific scenarios
│   │   │
│   │   ├── warrant/                    # WarrantProof → Module (P1)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Military efficiency constants
│   │   │   ├── ingest.py               # Defense contract ingestion
│   │   │   ├── verify.py               # Lifecycle accountability
│   │   │   ├── receipts.py             # warrant_proof, cost_savings
│   │   │   ├── data.py                 # Military simulation data
│   │   │   └── scenario.py             # Military-specific scenarios
│   │   │
│   │   ├── lab/                        # LabProof → Module (P2)
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Scientific reproducibility constants
│   │   │   ├── ingest.py               # Protocol ingestion
│   │   │   ├── verify.py               # Federated baseline calibration
│   │   │   ├── receipts.py             # lab_proof, replication_chain
│   │   │   ├── data.py                 # Research simulation data
│   │   │   └── scenario.py             # Research-specific scenarios
│   │   │
│   │   ├── defense/                    # EXISTING (from v1.0)
│   │   │   └── ...
│   │   │
│   │   └── medicaid/                   # EXISTING (from v1.0)
│   │       └── ...
│   │
│   ├── scenarios/                      # CROSS-MODULE SCENARIOS
│   │   ├── __init__.py
│   │   ├── contagion.py                # v5.1 contagion scenario
│   │   ├── cross_domain.py             # Multi-module contagion (v6.0)
│   │   ├── shell_cascade.py            # Shell entity propagation (v6.0)
│   │   ├── temporal_stress.py          # Temporal decay at scale (v6.0)
│   │   └── loop_learning.py            # LOOP effectiveness test (v6.0)
│   │
│   ├── # Core Detection
│   ├── compress.py                     # Entropy compression analysis
│   ├── detect.py                       # Multi-stage anomaly detection
│   ├── kolmogorov.py                   # Algorithmic complexity
│   ├── zkp.py                          # Zero-knowledge proofs
│   ├── raf.py                          # Network cycle detection
│   ├── holographic.py                  # Boundary-only detection
│   ├── thompson.py                     # Bayesian audit sampling
│   ├── ledger.py                       # Merkle ledger + Bekenstein bounds
│   ├── bridge.py                       # Cross-branch translation
│   ├── sim.py                          # Scenario simulation engine
│   │
│   ├── # v4.0 User-Friendly
│   ├── insight.py                      # Plain-language explanations
│   ├── fitness.py                      # Self-improving pattern tracking
│   ├── guardian.py                     # Evidence quality gates
│   ├── freshness.py                    # Evidence staleness detection
│   ├── learner.py                      # Cross-domain pattern transfer
│   │
│   ├── # Integration
│   ├── usaspending_etl.py              # USASpending.gov integration
│   ├── sam_validator.py                # SAM.gov vendor validation
│   │
│   ├── domains/                        # Legacy domain-specific modules
│   │   ├── __init__.py
│   │   ├── defense/                    # Defense spending domain
│   │   └── medicaid/                   # Medicaid spending domain
│   │
│   ├── shipyard/                       # Trump-class battleship module
│   │   └── ...
│   │
│   └── razor/                          # Kolmogorov validation engine
│       └── ...
│
├── schemas/                            # Ledger schema definitions
│   ├── ledger_schema_domains.json
│   ├── ledger_schema_razor.json
│   └── ledger_schema_shipyard.json
│
├── tests/                              # Unified test suite
│   ├── conftest.py
│   ├── test_core_contagion.py          # Contagion tests (v6.0)
│   ├── test_core_gate.py               # Gate tests (v6.0)
│   ├── test_core_loop.py               # LOOP tests (v6.0)
│   ├── test_core_zk.py                 # ZK tests (v6.0)
│   ├── test_modules_doge.py            # DOGE module tests
│   ├── test_modules_spend.py           # Spend module tests
│   ├── test_modules_green.py           # Green module tests
│   ├── test_modules_benefit.py         # Benefit module tests
│   ├── test_modules_vote.py            # Vote module tests
│   ├── test_modules_claim.py           # Claim module tests
│   ├── test_modules_safety.py          # Safety module tests
│   ├── test_modules_coin.py            # Coin module tests
│   ├── test_modules_origin.py          # Origin module tests
│   ├── test_modules_graft.py           # Graft module tests
│   ├── test_modules_warrant.py         # Warrant module tests
│   ├── test_modules_lab.py             # Lab module tests
│   ├── test_scenarios_cross_domain.py  # Cross-domain tests
│   └── ...
│
└── data/                               # Data and citations
    └── citations/
```

---

## Core Infrastructure (v6.0)

### core/constants.py

Universal constants for all modules:

| Constant | Value | Description |
|----------|-------|-------------|
| LAMBDA_NATURAL | 0.005 | Monthly decay rate |
| RESISTANCE_THRESHOLD | 1.5 | Flag if resistance > threshold |
| ZOMBIE_DAYS | 180 | Days without activity = zombie |
| GATE_T2H_SECONDS | 7200 | 2-hour challenge window |
| GATE_T24H_SECONDS | 86400 | 24-hour validation |
| GATE_T48H_SECONDS | 172800 | 48-hour finalization |
| ENTROPY_FLOOR | 0.1 | Below = suspicious uniformity |
| ENTROPY_CEILING | 0.85 | Above = suspicious chaos |
| COMPRESSION_THRESHOLD | 0.75 | Minimum viable compression |
| ZK_CURVE | "BN254" | Groth16 curve |
| ZK_SECURITY_BITS | 128 | Security level |
| ANCHOR_BATCH_SIZE | 1000 | Receipts per Merkle batch |
| LOOP_CYCLE_SECONDS | 60 | LOOP cycle frequency |
| LOOP_WOUND_THRESHOLD | 5 | Occurrences before automation |
| CONTAGION_PROPAGATION_DEPTH | 3 | Max hop depth |
| CONTAGION_AMPLIFICATION | 1.22 | 22% sensitivity boost |
| SHELL_OVERLAP_THRESHOLD | 0.05 | 5% overlap = shared shell |

### Module Identifiers

```python
ALL_MODULES = [
    "doge", "spend", "green", "benefit",
    "vote", "claim", "safety", "coin",
    "origin", "graft", "warrant", "lab",
    "defense", "medicaid"
]
```

---

## New Core Modules (v6.0)

### core/contagion.py

Cross-module fraud propagation via super-graph.

| Function | Purpose |
|----------|---------|
| `build_super_graph(graphs)` | Merge module RAF graphs into unified super-graph |
| `identify_shared_entities(super_graph)` | Find entities appearing in multiple modules |
| `propagate_contagion(super_graph, source_node, source_module)` | Propagate fraud signal via shared entities |
| `calculate_propagation_path(super_graph, source, target)` | Shortest path across module boundaries |
| `amplify_by_resistance(base_entropy, resistance)` | Multiply by (1 + resistance × 1.22) |
| `preemptive_flag(super_graph, flagged_entity)` | Flag linked entities BEFORE local evidence |

**Contagion Physics:**
1. Build super-graph from all active module graphs
2. Identify shared entities (appear in ≥2 modules)
3. When fraud detected in module A:
   - Calculate resistance to temporal decay
   - Propagate amplified signal to linked entities
   - Emit contagion_receipt with propagation path
4. Preemptive flagging enables 2-4x earlier detection

### core/gate.py

Universal T+2h/24h/48h gate structure for all modules.

| Function | Purpose |
|----------|---------|
| `open_gate(gate_type, receipt_id, module)` | Open a gate for a receipt |
| `challenge_gate(gate_id, challenger, evidence)` | Challenge before T+2h |
| `resolve_challenge(gate_id, resolution)` | Resolve a challenged gate |
| `finalize_gate(gate_id)` | Finalize after T+48h |
| `get_gate_status(gate_id)` | Return current status and timing |
| `batch_finalize(gate_ids)` | Batch finalize multiple gates |

**Gate States:**

| State | Timing | Description |
|-------|--------|-------------|
| pending | T+0 → T+2h | Open for challenges |
| challenged | Any time during pending | Challenge submitted |
| resolved | After challenge review | Challenge accepted/rejected |
| finalized | T+48h if no valid challenges | Receipt is immutable |
| rejected | After failed challenge | Receipt invalidated |

### core/loop.py

LOOP self-improvement layer.

| Function | Purpose |
|----------|---------|
| `run_loop_cycle()` | Execute full SENSE→EMIT cycle |
| `sense()` | Query receipt stream for L0-L3 receipts |
| `analyze(receipts)` | Run anomaly detection, pattern extraction |
| `harvest_wounds(days)` | Collect manual intervention patterns |
| `synthesize_helper(pattern)` | Create helper blueprint from wound |
| `calculate_risk(action)` | Return 0-1 risk score |
| `request_approval(helper, risk)` | Submit for HITL approval if risk > 0.2 |
| `actuate(approved_actions)` | Execute approved actions |
| `measure_completeness()` | Return L0-L4 coverage scores |

**LOOP Cycle (every 60 seconds):**
1. SENSE: Query receipts from all modules
2. ANALYZE: Run pattern detection
3. HARVEST: Collect wound patterns
4. HYPOTHESIZE: Synthesize helper blueprints
5. GATE: Request approval for high-risk actions
6. ACTUATE: Execute approved actions
7. EMIT: Emit loop_cycle_receipt

**Receipt Levels:**

| Level | Name | Description |
|-------|------|-------------|
| L0 | Telemetry | Raw ingestion receipts |
| L1 | Agent | Verification and detection receipts |
| L2 | Decision | Brief and packet receipts |
| L3 | Quality | Effectiveness and health receipts |
| L4 | Meta | Loop cycle and completeness receipts |

### core/zk.py

Zero-knowledge privacy layer.

| Function | Purpose |
|----------|---------|
| `generate_pedersen_commitment(value, blinding)` | Create Pedersen commitment |
| `verify_pedersen_commitment(commitment, value, blinding)` | Verify commitment |
| `create_range_proof(value, commitment, range_bits)` | Prove value in range |
| `verify_range_proof(commitment, proof, range_bits)` | Verify range proof |
| `create_conservation_proof(inputs, outputs)` | Prove sum equality |
| `verify_conservation_proof(inputs, outputs, proof)` | Verify conservation |
| `create_membership_proof(element, merkle_root, path)` | Prove set membership |
| `verify_membership_proof(merkle_root, proof)` | Verify membership |

**ZK Applications by Module:**

| Module | ZK Application | Privacy Protected |
|--------|----------------|-------------------|
| spend | Conservation proofs | Actual disbursement amounts |
| safety | Patient membership | Patient identifiers |
| benefit | Recipient membership | Recipient identifiers |
| graft | Whistleblower attestation | Whistleblower identity |
| coin | Conflict attestation | Wallet ownership |

---

## Module Interface

Every module in `modules/` implements this interface:

**config.py:**
- `MODULE_ID: str` - Unique module identifier
- `MODULE_PRIORITY: int` - 0=critical, 1=important, 2=enhancement
- `RECEIPT_TYPES: list[str]` - Receipt types this module emits
- Domain-specific constants

**ingest.py:**
- `ingest(data: dict) → dict` - Ingest domain data, emit ingestion receipt
- `batch_ingest(data_list: list[dict]) → list[dict]` - Batch ingestion

**verify.py:**
- `verify(claim: dict) → dict` - Verify domain claim, emit verification receipt
- `batch_verify(claims: list[dict]) → list[dict]` - Batch verification

**receipts.py:**
- Receipt type dataclasses with all fields
- `emit_MODULE_receipt(data: dict) → dict` - Emit domain receipt

**data.py:**
- `sample_MODULE_receipts(n: int, scenario: str) → list[dict]` - Generate sample data
- `generate_fraud_scenario(scenario: str) → list[dict]` - Generate fraud test data

**scenario.py:**
- `run_MODULE_scenario(scenario: str) → dict` - Run domain-specific scenario
- Returns: `{"passed": bool, "metrics": dict, "receipts": list}`

---

## Module Specifications

### modules/doge/ (DOGEProof)

**Purpose:** Verify government efficiency claims with pre+post causality pairs.

| Constant | Value | Description |
|----------|-------|-------------|
| MODULE_ID | "doge" | Module identifier |
| MODULE_PRIORITY | 0 | Critical |
| CLAIM_THRESHOLD_USD | 1,000,000 | Extra verification threshold |
| CHALLENGE_WINDOW_HOURS | 2 | T+2h challenge window |

**Key Functions:**
- `ingest_claim(claim, pre_state)` - Ingest with pre-state snapshot
- `verify_causality(claim_id)` - Verify pre+post form valid pair
- `verify_savings(claim_id, claimed_savings)` - Verify claimed vs actual

### modules/spend/ (SpendProof)

**Purpose:** Federal budget verification with ZK privacy.

| Constant | Value | Description |
|----------|-------|-------------|
| MODULE_ID | "spend" | Module identifier |
| MODULE_PRIORITY | 0 | Critical |
| ENTROPY_THRESHOLD | 0.007 | Entropy delta trigger |

**Key Functions:**
- `verify_conservation(inputs, outputs)` - ZK proof sum equality
- `verify_entropy_delta(before, after)` - Validate entropy bounds
- `create_audit_compress(disbursements)` - 99% compression, 99.9% recall

### modules/green/ (GreenProof)

**Purpose:** Climate/ESG accountability verification.

| Constant | Value | Description |
|----------|-------|-------------|
| MODULE_ID | "green" | Module identifier |
| MODULE_PRIORITY | 0 | Critical |
| EMISSIONS_VARIANCE_THRESHOLD | 0.15 | 15% variance from satellite |

**Key Functions:**
- `verify_emissions(claimed, satellite)` - Compare to satellite data
- `verify_timeline(claim_chain)` - Verify temporal consistency
- `detect_greenwashing(entity_id, claims)` - Detect via compression

### modules/benefit/ (BenefitProof)

**Purpose:** Government benefits verification (Medicaid focus).

**Key Functions:**
- `verify_qualification(recipient_id, criteria)` - Verify recipient qualifies
- `verify_disbursement(disbursement, recipient)` - Verify to qualified recipient
- `detect_fraud_pattern(entity_id, history)` - Detect via compression ratio

### modules/coin/ (CoinProof)

**Purpose:** Crypto wallet ownership verification.

**Key Functions:**
- `cluster_wallets(transactions)` - Cluster by transaction compression
- `verify_separation(wallet_a, wallet_b)` - Verify truly separate
- `detect_revenue_share(flow)` - Detect hidden sharing via compression
- `zk_conflict_attestation(wallet, entity)` - ZK proof of conflict

### modules/graft/ (GraftProof)

**Purpose:** FBI public corruption case tracking.

**Key Functions:**
- `track_case(case_id, events)` - Track lifecycle with crypto chain
- `verify_disposition(case_id)` - Verify proper disposition
- `protect_whistleblower(report)` - ZK attestation protecting identity
- `measure_staffing_impact(case_id)` - Measure staffing changes impact

---

## Receipt Types (82+ Total)

### Core (12)
warrant, quality_attestation, milestone, cost_variance, anchor, detection, compression, lineage, bridge, simulation, anomaly, violation

### v2 Physics (8)
threshold, pattern_emergence, entropy_tree, cascade_alert, epidemic_warning, holographic, meta_receipt, mutual_info

### v3 OMEGA (12)
kolmogorov, zkp, raf, das, adversarial, usaspending, layout_entropy, sam_validation, catalytic, holographic_da, thompson_audit, bekenstein

### v4 User-Friendly (17)
insight, fitness, health, quality, abstain, counter_evidence, integrity, gate, freshness, refresh_priority, monitoring, pattern_match, transfer, learn, library_summary, prune

### v5.1 Temporal (5)
temporal_anomaly_receipt, zombie_receipt, contagion_receipt, super_graph_receipt, insight_receipt

### v6.0 Module Receipts (20+)
| Receipt | Module | Purpose |
|---------|--------|---------|
| doge_proof | doge | Verified efficiency claim |
| qed_claim | doge | QED compression claim |
| disbursement_proof | spend | ZK-verified disbursement |
| audit_compress | spend | Compressed audit |
| green_proof | green | Verified emissions |
| emissions_anchor | green | Satellite-anchored emissions |
| benefit_disburse | benefit | Verified benefit disbursement |
| fraud_compress | benefit | Compressed fraud pattern |
| vote_proof | vote | Verified ballot |
| tally_anchor | vote | Anchored tally |
| claim_proof | claim | Verified government claim |
| safety_proof | safety | ZK patient safety event |
| adverse_event | safety | Adverse event anchor |
| wallet_cluster | coin | Wallet cluster detection |
| revenue_share | coin | Revenue sharing detection |
| tier_auth | origin | Tier authentication |
| origin_chain | origin | Provenance chain |
| graft_proof | graft | Corruption case proof |
| case_chain | graft | Case lifecycle chain |
| warrant_proof | warrant | Military contract proof |
| lab_proof | lab | Replication proof |

### v6.0 Core Receipts (4)
| Receipt | Purpose |
|---------|---------|
| contagion_receipt | Cross-domain fraud propagation |
| loop_cycle_receipt | LOOP cycle completion |
| gate_receipt | Gate status change |
| zk_proof_receipt | ZK verification result |

### Shipyard (8)
keel, block, additive, iteration, milestone, procurement, propulsion, delivery

### RAZOR (5)
ingest, cohort, compression, validation, signal

### Domain (5)
domain_receipt, domain_simulation, domain_scenario, domain_validation, domain_volatility

---

## Cross-Module Scenarios (v6.0)

### scenarios/cross_domain.py

| Scenario | Modules | Test Case |
|----------|---------|-----------|
| SHELL_CASCADE | doge, spend, benefit | Shell company in all three. Detection in benefit triggers preemptive flags. |
| CRYPTO_CORRUPTION | coin, graft | Wallet cluster linked to corruption. Detection in graft flags wallets. |
| ESG_SUPPLY_CHAIN | green, origin | Emissions linked to fraudulent supply chain. Detection invalidates green claim. |
| MILITARY_BENEFIT | warrant, benefit | Defense contractor = Medicaid provider. Cross-domain anomaly. |

**run_cross_domain_scenario returns:**
```python
{
    "passed": bool,
    "contagion_detected": bool,
    "preemptive_flags": int,
    "detection_speedup_factor": float,  # 2-4x earlier
    "modules_affected": list[str],
    "receipts": list[dict]
}
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

### Module Commands (v6.0)

```bash
gov-os doge ingest --claim FILE         # Ingest efficiency claim
gov-os doge verify --claim-id ID        # Verify claim causality
gov-os doge scenario BASELINE           # Run DOGE scenario

gov-os spend verify --conservation      # Verify ZK conservation
gov-os spend audit --compress           # Create compressed audit

gov-os green verify --emissions FILE    # Verify emissions claim
gov-os green detect-greenwashing ID     # Detect greenwashing

gov-os contagion --run SHELL_CASCADE    # Run contagion scenario
gov-os loop --cycle                     # Run LOOP cycle
gov-os gate --status GATE_ID            # Check gate status
```

### Domain Commands

```bash
gov-os defense simulate --cycles 100    # Defense domain simulation
gov-os medicaid scenario PROVIDER_RING  # Run medicaid scenario
gov-os validate --domain all            # Validate all domains
gov-os list domains                     # List available domains
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
| Decay Resistance | > 1.5 = anomaly | <= 1.0 | v6.0 temporal |
| Zombie Days | > 180 days dormant | N/A | v6.0 temporal |
| Contagion Overlap | >= 5% shared entities | N/A | v6.0 contagion |
| LOOP Completeness | >= 99.9% L0-L4 | N/A | v6.0 LOOP |

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
| T+2h | Skeleton working | Core files, CLI, all modules importable |
| T+24h | MVP complete | Tests pass, smoke tests, simulations |
| T+48h | Hardened | Full test suite, scenarios, coverage ≥80% |

### v6.0 Gate Phases

| Phase | Gate | Requirements |
|-------|------|--------------|
| Phase 1 | 1.1 | Core skeleton (contagion, gate, loop, zk) |
| Phase 1 | 1.2 | Core functions pass tests |
| Phase 2 | 2.1-2.4 | P0 modules (doge, spend, green, benefit) |
| Phase 3 | 3.1-3.3 | P1 modules (vote, claim, safety, coin, origin, graft, warrant) |
| Phase 4 | 4.1-4.3 | Cross-domain integration, LOOP, hardened |

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
```

### Temporal Decay (v5.1/v6.0)
```
Wt = W₀ × e^(-λt)

λ = LAMBDA_NATURAL = 0.005 (per month)

Resistance = max(0, (observed_weight / expected_weight) - 1.0)
Resistance > 1.5 -> Anomaly
```

### Cross-Domain Contagion (v6.0)
```
When fraud detected in module A:
  → Identify shared entities across modules
  → Amplify by resistance: signal × (1 + resistance × 1.22)
  → Propagate to linked entities in modules B, C, ...
  → Preemptive flag BEFORE local evidence

Detection improvement: 2-4x earlier via contagion propagation
```

### LOOP Completeness (v6.0)
```
Completeness = coverage(L0, L1, L2, L3, L4) / 5

When L4 >= 99.9% and L4 feeds back to L0:
  → Computational sovereignty achieved
  → System is self-auditing
```

---

## Version History (Detailed)

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial: receipts, compression, detection |
| 2.0.0 | 2024-12-23 | Physics: entropy, autocatalytic, Thompson |
| 3.0.0 | 2024-12-23 | OMEGA: Kolmogorov, ZKP, RAF, DA sampling |
| 3.1.0 | 2024-12-23 | Shipyard: Trump-class, 8 receipt types |
| 4.0.0 | 2024-12-23 | User-friendly: insight, fitness, guardian, freshness, learner |
| 5.0.0 | 2024-12-23 | Unification: Single cohesive platform with domain modules |
| 5.1.0 | 2024-12-23 | Temporal: Decay physics, resistance detection, cross-domain contagion |
| 6.0.0 | 2024-12-23 | ProofChain: 12 modules unified, contagion, gate, loop, zk |

---

**THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY**
