# RAZOR Specification v1.0

> Kolmogorov Validation Engine for Federal Procurement Fraud Detection

## Overview

RAZOR validates the Kolmogorov Compression Hypothesis on real USASpending.gov
data to detect fraud via algorithmic complexity measurement.

**Core Hypothesis:** Fraudulent procurement data exhibits lower Kolmogorov
complexity than legitimate data because fraud requires coordination, and
coordination implies rules that reduce entropy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT RAZOR ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [USASpending API] ──▶ [Ingestor] ──▶ [Cohort DataFrames]       │
│                                              │                   │
│                                              ▼                   │
│  [KolmogorovMetric] ◀── Compression ──▶ [Physics Engine]        │
│         │                                                        │
│         ▼                                                        │
│  [Statistical Validator] ──▶ Z-scores, T-tests ──▶ [Verdict]    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Inputs

| Input | Source | Format |
|-------|--------|--------|
| Procurement records | USASpending.gov API | JSON |
| Cohort configurations | cohorts.py | Python dict |
| Historical windows | core.py constants | Date strings |

## Outputs

| Output | Format | Purpose |
|--------|--------|---------|
| Cohort DataFrames | pandas DataFrame | Ingested records |
| Complexity metrics | Dict per record | CR_zlib, CR_lzma, CR_bz2, Shannon H |
| Z-scores | Float per record | Statistical deviation from baseline |
| Signal verdict | Dict | SIGNAL_DETECTED or NO_SIGNAL |
| Receipts | JSONL | Audit trail per CLAUDEME |

## Receipts

| Receipt Type | Module | Frequency | Key Fields |
|--------------|--------|-----------|------------|
| ingest | ingest.py | Per API call | url, status_code, records_fetched |
| cohort | ingest.py | Per cohort | cohort_name, total_records |
| complexity | physics.py | Per record | award_id, cr_zlib, cr_lzma, cr_bz2 |
| complexity_cohort | physics.py | Per cohort | mean_cr_zlib, valid_records |
| baseline | validate.py | Per control | mean_cr, std_cr, n_records |
| signal | validate.py | Per analysis | signal_detected, signal_strength |
| gate_pass | cli.py | Per gate | gate, status |
| gate_fail | cli.py | Per gate | gate, status, error |

## SLOs

| SLO | Threshold | Stoprule Action |
|-----|-----------|-----------------|
| API response time | ≤ 5s p95 | Exponential backoff |
| Cohort size | ≥ 500 records | stoprule_insufficient_data |
| Control cohort size | ≥ 100 records | stoprule_insufficient_control |
| Compression time | ≤ 10ms per record | Log warning |
| Statistical power | ≥ 0.80 | Log warning |
| Type I error (α) | ≤ 0.05 | Standard significance |
| Z-score threshold | < -2.0 | Required for signal |
| Effect size | ≥ 0.5 | Required for "meaningful" |

## Stoprules

| Stoprule | Trigger | Action |
|----------|---------|--------|
| stoprule_api_failure | HTTP != 200 after MAX_RETRIES | Halt |
| stoprule_insufficient_data | cohort size < 500 | Halt |
| stoprule_insufficient_control | control size < 100 | Halt |
| stoprule_compression_invalid | Compression engine failure | Halt |
| stoprule_degenerate_baseline | Control variance = 0 | Halt |
| stoprule_no_signal | Z > -2.0 | Alert (not halt) |

## Cohorts

### Fat Leonard (GDMA)
- **Period:** 2008-01-01 to 2013-09-01
- **Hypothesis:** Repetitive husbanding invoices compress better
- **Fraud type:** copy_paste
- **Expected Z-score:** < -2.5

### TransDigm
- **Period:** 2015-01-01 to 2019-12-31
- **Hypothesis:** Simple part descriptions with high prices
- **Fraud type:** value_decoupling
- **Expected Z-score:** < -1.5

### Boeing KC-767
- **Period:** 2001-01-01 to 2003-12-31
- **Hypothesis:** Boilerplate sole-source justifications
- **Fraud type:** template_fraud
- **Expected Z-score:** < -1.8

## Constants

```python
TENANT_ID = "project-razor"
API_BASE_URL = "https://api.usaspending.gov/api/v2"
API_RATE_LIMIT_DELAY = 1.0
API_MAX_RETRIES = 5
CR_THRESHOLD_LOW = 0.30
CR_THRESHOLD_HIGH = 0.70
Z_SCORE_THRESHOLD = -2.0
MIN_COHORT_SIZE = 500
MIN_CONTROL_SIZE = 100
ALPHA_LEVEL = 0.05
```

## Rollback

If validation fails:
1. Check receipts.jsonl for failure point
2. Identify cohort with insufficient signal
3. Review control cohort contamination
4. Adjust time windows or exclusion criteria
5. Re-run with modified parameters

## References

- Shannon, C.E. (1948). A Mathematical Theory of Communication
- Kolmogorov, A.N. (1965). Three Approaches to the Quantitative Definition of Information
- USASpending.gov API Documentation: https://api.usaspending.gov/
