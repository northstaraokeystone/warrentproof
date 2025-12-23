# ShieldProof v2.0 Specification

**"One receipt. One milestone. One truth."**

## Overview

ShieldProof v2.0 is the minimal viable truth for defense accountability. It strips away physics theater and focuses on what matters: receipts that prove payment follows verification.

## Components (3, not 20)

1. **Immutable Receipts** - contract, milestone, payment
2. **Automated Reconciliation** - spend vs deliverable matching
3. **Public Audit Trail** - aggregate dashboard

## Receipt Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `contract` | Register fixed-price contract | contract_id, contractor, amount_fixed, milestones[], terms_hash |
| `milestone` | Track deliverable verification | contract_id, milestone_id, deliverable_hash, status, verifier_id |
| `payment` | Release payment on verification | contract_id, milestone_id, amount, payment_hash, released_at |

## Milestone States

```
PENDING → DELIVERED → VERIFIED → PAID
                  ↘ DISPUTED
```

## SLOs (Realistic, not microsecond)

| Operation | Target |
|-----------|--------|
| Receipt emission | ≤ 10ms |
| Verification | ≤ 50ms |
| Dashboard refresh | ≤ 60s |

## Stoprules

| Rule | Trigger | Action |
|------|---------|--------|
| stoprule_duplicate_contract | contract_id exists | Emit anomaly |
| stoprule_invalid_amount | amount ≤ 0 or milestones don't sum | Emit anomaly |
| stoprule_unknown_contract | contract_id not found | Emit anomaly |
| stoprule_unknown_milestone | milestone_id not in contract | Emit anomaly |
| stoprule_already_verified | milestone already VERIFIED/PAID | Emit anomaly |
| stoprule_unverified_milestone | payment on non-VERIFIED | HALT |
| stoprule_already_paid | milestone already PAID | Emit anomaly |
| stoprule_amount_mismatch | payment ≠ milestone amount | Emit anomaly |
| stoprule_overpayment | paid > verified amount | Emit anomaly |
| stoprule_unverified_payment | payment without verification | Emit anomaly |

## What We KILLED

- ZK-SNARKs (latency suicide)
- PQC signatures (overkill)
- Entropy detection (physics theater)
- RAF cycles (no predictive power)
- Holographic bounds (Bekenstein applies to horizons, not budgets)
- 5-layer receipt chain (simplified to 3)
- Microsecond latency targets (unrealistic)

## Hash Strategy

```
SHA256:BLAKE3 (dual-hash per CLAUDEME §8)
```

## The SpaceX Model

Fixed-price contracts where payment follows verification. A public dashboard showing taxpayers exactly where their money went.

*No receipt → not real. Ship at T+24h or kill.*
