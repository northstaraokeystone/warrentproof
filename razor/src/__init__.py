"""
Project RAZOR - Kolmogorov Validation Engine

THE PARADIGM INVERSION:
  v3 OMEGA was "physics-theater" - ZK-SNARKs and RAF networks
  obscuring the signal. RAZOR strips to minimalist brutalism:

  Corrupt markets are ordered. Order compresses.
  Honest markets are chaotic. Chaos resists compression.

  K(x) = len(compressed) / len(original)

  The compression ratio IS the proof.

Modules:
  - core: CLAUDEME-compliant foundation with RAZOR constants
  - cohorts: Historical fraud cohort definitions (Fat Leonard, TransDigm, Boeing)
  - ingest: USASpending.gov API client with robust pagination
  - physics: Kolmogorov complexity measurement via compression
  - validate: Statistical signal detection (Z-scores, T-tests)
"""

__version__ = "1.0.0"
__author__ = "Project RAZOR"

from .core import (
    dual_hash,
    emit_receipt,
    merkle,
    StopRule,
    TENANT_ID,
    API_BASE_URL,
    CR_THRESHOLD_LOW,
    CR_THRESHOLD_HIGH,
    Z_SCORE_THRESHOLD,
)
