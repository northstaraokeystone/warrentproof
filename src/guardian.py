"""
WarrantProof Guardian Module - Evidence Quality & Abstention

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module protects the system from making decisions based on
unreliable evidence. When evidence is weak or conflicting, the
system abstains rather than guessing.

The Philosophy:
"I don't know" is better than "I think, but I'm probably wrong."

Key Features:
1. Evidence Scoring: Rate confidence in each piece of evidence
2. Abstention Logic: Know when NOT to make a decision
3. Counter-Evidence: Track evidence that contradicts findings
4. Quality Gates: Prevent low-quality decisions from propagating

For Non-Technical Users:
This module is like a quality control inspector. Before any
finding is reported, it checks:
- Is the evidence strong enough?
- Is there conflicting evidence?
- Are there gaps in the data?

If the answer to any of these is concerning, the system says
"we need more information" rather than making a shaky call.

Abstention Reasons:
- "leak": Evidence may have been compromised
- "no_counter_evidence": Can't rule out alternative explanations
- "insufficient_data": Not enough information to decide
- "conflicting_signals": Evidence points in multiple directions
- "stale_evidence": Data is too old to trust
"""

from datetime import datetime, timedelta
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    dual_hash,
    emit_receipt,
    get_citation,
    StopRuleException,
)


# === QUALITY THRESHOLDS ===

QUALITY_THRESHOLDS = {
    "high_confidence": 0.85,      # Strong evidence, proceed
    "acceptable": 0.70,           # Adequate evidence, proceed with note
    "marginal": 0.50,             # Weak evidence, consider abstaining
    "insufficient": 0.30,         # Too weak, should abstain
}

COUNTER_EVIDENCE_THRESHOLD = 0.15  # Max counter-evidence before abstaining
STALENESS_DAYS = 90                # Evidence older than this needs refresh


# === ABSTENTION CODES ===

ABSTENTION_REASONS = {
    "leak": {
        "code": "LEAK",
        "description": "Evidence integrity may be compromised",
        "user_message": (
            "The evidence chain shows signs of tampering or unauthorized access. "
            "We cannot make a determination until the data integrity is verified."
        ),
    },
    "no_counter_evidence": {
        "code": "NO_CE",
        "description": "Cannot rule out alternative explanations",
        "user_message": (
            "While the evidence points one way, we haven't been able to rule out "
            "other explanations. More investigation is needed before concluding."
        ),
    },
    "insufficient_data": {
        "code": "INSUF",
        "description": "Not enough information to decide",
        "user_message": (
            "There isn't enough data to make a reliable determination. "
            "We need more records or documentation before proceeding."
        ),
    },
    "conflicting_signals": {
        "code": "CONFLICT",
        "description": "Evidence points in multiple directions",
        "user_message": (
            "The evidence is contradictory - some signs point to fraud, "
            "others suggest legitimate activity. Manual review is required."
        ),
    },
    "stale_evidence": {
        "code": "STALE",
        "description": "Data is too old to trust",
        "user_message": (
            "The most recent evidence is over 90 days old. The situation may "
            "have changed significantly. Fresh data is needed."
        ),
    },
    "source_reliability": {
        "code": "SRC_REL",
        "description": "Evidence source has known reliability issues",
        "user_message": (
            "Some of this evidence comes from sources with known accuracy problems. "
            "We should verify with more reliable sources before deciding."
        ),
    },
}


# === CORE CLASSES ===

class EvidenceItem:
    """
    Represents a single piece of evidence with quality metadata.
    """

    def __init__(
        self,
        evidence_id: str,
        evidence_type: str,
        content: dict,
        source: str,
        timestamp: Optional[datetime] = None
    ):
        self.evidence_id = evidence_id
        self.evidence_type = evidence_type
        self.content = content
        self.source = source
        self.timestamp = timestamp or datetime.utcnow()
        self.quality_score = 1.0
        self.issues = []

    def add_issue(self, issue: str, severity: float):
        """Add a quality issue that reduces confidence."""
        self.issues.append({"issue": issue, "severity": severity})
        self.quality_score = max(0.0, self.quality_score - severity)

    @property
    def is_stale(self) -> bool:
        """Check if evidence is older than threshold."""
        age = datetime.utcnow() - self.timestamp
        return age.days > STALENESS_DAYS

    @property
    def days_old(self) -> int:
        """Get age of evidence in days."""
        return (datetime.utcnow() - self.timestamp).days

    def to_dict(self) -> dict:
        """Convert to dictionary for receipts."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "quality_score": round(self.quality_score, 2),
            "days_old": self.days_old,
            "is_stale": self.is_stale,
            "issues": self.issues,
        }


class EvidenceSet:
    """
    A collection of evidence items for a single determination.
    """

    def __init__(self, determination_id: str):
        self.determination_id = determination_id
        self.supporting_evidence: list[EvidenceItem] = []
        self.counter_evidence: list[EvidenceItem] = []
        self.created = datetime.utcnow()

    def add_supporting(self, evidence: EvidenceItem):
        """Add evidence that supports the finding."""
        self.supporting_evidence.append(evidence)

    def add_counter(self, evidence: EvidenceItem):
        """Add evidence that contradicts the finding."""
        self.counter_evidence.append(evidence)

    @property
    def total_supporting_weight(self) -> float:
        """Total quality-weighted supporting evidence."""
        return sum(e.quality_score for e in self.supporting_evidence)

    @property
    def total_counter_weight(self) -> float:
        """Total quality-weighted counter evidence."""
        return sum(e.quality_score for e in self.counter_evidence)

    @property
    def net_confidence(self) -> float:
        """Net confidence after considering counter-evidence."""
        if not self.supporting_evidence:
            return 0.0
        total = self.total_supporting_weight + self.total_counter_weight
        if total == 0:
            return 0.0
        return (self.total_supporting_weight - self.total_counter_weight) / total

    @property
    def counter_evidence_ratio(self) -> float:
        """Ratio of counter to total evidence."""
        total = self.total_supporting_weight + self.total_counter_weight
        if total == 0:
            return 0.0
        return self.total_counter_weight / total

    def has_stale_evidence(self) -> bool:
        """Check if any critical evidence is stale."""
        for e in self.supporting_evidence:
            if e.is_stale and e.quality_score > 0.5:
                return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for receipts."""
        return {
            "determination_id": self.determination_id,
            "supporting_count": len(self.supporting_evidence),
            "counter_count": len(self.counter_evidence),
            "supporting_weight": round(self.total_supporting_weight, 2),
            "counter_weight": round(self.total_counter_weight, 2),
            "net_confidence": round(self.net_confidence, 2),
            "counter_ratio": round(self.counter_evidence_ratio, 2),
            "has_stale": self.has_stale_evidence(),
        }


# === CORE FUNCTIONS ===

def evaluate_evidence_quality(evidence_set: EvidenceSet) -> dict:
    """
    Evaluate the quality of an evidence set and determine
    if a decision should be made or abstained.

    Args:
        evidence_set: Collection of evidence for a determination

    Returns:
        quality_receipt with recommendation
    """
    confidence = evidence_set.net_confidence
    counter_ratio = evidence_set.counter_evidence_ratio
    has_stale = evidence_set.has_stale_evidence()

    # Determine if we should abstain
    should_abstain = False
    abstention_reason = None

    if confidence < QUALITY_THRESHOLDS["insufficient"]:
        should_abstain = True
        abstention_reason = "insufficient_data"
    elif counter_ratio > COUNTER_EVIDENCE_THRESHOLD:
        should_abstain = True
        abstention_reason = "conflicting_signals" if counter_ratio > 0.3 else "no_counter_evidence"
    elif has_stale:
        should_abstain = True
        abstention_reason = "stale_evidence"

    # Determine quality level
    if confidence >= QUALITY_THRESHOLDS["high_confidence"]:
        quality_level = "high"
    elif confidence >= QUALITY_THRESHOLDS["acceptable"]:
        quality_level = "acceptable"
    elif confidence >= QUALITY_THRESHOLDS["marginal"]:
        quality_level = "marginal"
    else:
        quality_level = "insufficient"

    return emit_receipt("quality", {
        "tenant_id": TENANT_ID,
        "determination_id": evidence_set.determination_id,
        "evidence_summary": evidence_set.to_dict(),
        "quality_level": quality_level,
        "confidence_score": round(confidence, 2),
        "should_abstain": should_abstain,
        "abstention_reason": abstention_reason,
        "recommendation": (
            _get_abstention_message(abstention_reason)
            if should_abstain else
            _get_proceed_message(quality_level)
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def abstain(
    determination_id: str,
    reason: str,
    evidence_summary: Optional[dict] = None
) -> dict:
    """
    Emit an abstention receipt when we cannot make a determination.

    This is GOOD - it means the system knows its limits and won't
    make unreliable decisions.

    Args:
        determination_id: ID of the determination being abstained
        reason: Abstention reason code
        evidence_summary: Optional summary of available evidence

    Returns:
        abstain_receipt documenting the abstention
    """
    reason_info = ABSTENTION_REASONS.get(reason, {
        "code": "UNK",
        "description": "Unknown reason",
        "user_message": "Unable to make a determination at this time.",
    })

    return emit_receipt("abstain", {
        "tenant_id": TENANT_ID,
        "determination_id": determination_id,
        "because": reason,
        "reason_code": reason_info["code"],
        "reason_description": reason_info["description"],
        "user_message": reason_info["user_message"],
        "evidence_summary": evidence_summary,
        "action_required": "Gather additional evidence or escalate for manual review.",
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def check_counter_evidence(
    finding: dict,
    available_records: list
) -> dict:
    """
    Look for evidence that contradicts a finding.

    This is crucial for integrity - we must actively try to
    disprove our findings, not just confirm them.

    Args:
        finding: The fraud finding to check
        available_records: Records to search for counter-evidence

    Returns:
        counter_evidence_receipt with findings
    """
    finding_type = finding.get("anomaly_type", "unknown")
    finding_targets = finding.get("affected_entities", [])

    counter_evidence = []

    for record in available_records:
        # Look for evidence that contradicts the finding
        if finding_type == "ghost_vendor_compression":
            # Check for legitimate vendor activity
            if record.get("vendor") in finding_targets:
                if record.get("verification_status") == "verified":
                    counter_evidence.append({
                        "type": "verified_vendor_activity",
                        "record_id": record.get("id", "unknown"),
                        "weight": 0.8,
                    })
                if record.get("physical_inspection") == "passed":
                    counter_evidence.append({
                        "type": "physical_inspection_passed",
                        "record_id": record.get("id", "unknown"),
                        "weight": 0.9,
                    })

        elif finding_type == "compression_failure":
            # Check for legitimate reasons for repetitive patterns
            if record.get("contract_type") == "framework_agreement":
                counter_evidence.append({
                    "type": "framework_agreement_explains_repetition",
                    "record_id": record.get("id", "unknown"),
                    "weight": 0.6,
                })

        elif finding_type == "cost_cascade":
            # Check for authorized cost increases
            if record.get("modification_type") == "scope_change":
                if record.get("approval_level") == "contracting_officer":
                    counter_evidence.append({
                        "type": "authorized_scope_change",
                        "record_id": record.get("id", "unknown"),
                        "weight": 0.7,
                    })

    total_counter_weight = sum(c["weight"] for c in counter_evidence)
    has_significant_counter = total_counter_weight > 0.5

    return emit_receipt("counter_evidence", {
        "tenant_id": TENANT_ID,
        "finding_type": finding_type,
        "counter_evidence_found": len(counter_evidence),
        "counter_evidence_items": counter_evidence[:10],  # Limit to 10
        "total_counter_weight": round(total_counter_weight, 2),
        "has_significant_counter": has_significant_counter,
        "recommendation": (
            "Counter-evidence found. Review before finalizing finding."
            if has_significant_counter else
            "No significant counter-evidence found. Finding appears reliable."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def check_evidence_integrity(receipts: list) -> dict:
    """
    Verify the integrity of evidence chain.

    Checks for signs that evidence may have been tampered with
    or that the chain of custody has been broken.

    Args:
        receipts: Receipts to verify

    Returns:
        integrity_receipt with findings
    """
    issues = []

    # Check for hash chain integrity
    prev_hash = None
    for i, receipt in enumerate(receipts):
        current_hash = receipt.get("payload_hash")

        # Check for duplicate hashes (copy-paste)
        if current_hash == prev_hash:
            issues.append({
                "issue": "duplicate_hash",
                "severity": "high",
                "index": i,
                "message": "Two consecutive receipts have identical hashes.",
            })

        # Check for missing timestamps
        if not receipt.get("ts"):
            issues.append({
                "issue": "missing_timestamp",
                "severity": "high",
                "index": i,
                "message": "Receipt is missing timestamp.",
            })

        # Check for out-of-order timestamps
        if prev_hash and receipt.get("ts"):
            # Would check timestamp ordering here
            pass

        prev_hash = current_hash

    # Check for suspicious patterns
    hashes = [r.get("payload_hash", "") for r in receipts]
    unique_hashes = len(set(hashes))
    if unique_hashes < len(hashes) * 0.9:  # More than 10% duplicates
        issues.append({
            "issue": "excessive_duplicates",
            "severity": "medium",
            "message": f"Found {len(hashes) - unique_hashes} duplicate hashes in chain.",
        })

    integrity_score = 1.0 - (len(issues) * 0.1)
    integrity_score = max(0.0, integrity_score)

    has_integrity = integrity_score >= 0.9 and not any(
        i["severity"] == "high" for i in issues
    )

    return emit_receipt("integrity", {
        "tenant_id": TENANT_ID,
        "receipts_checked": len(receipts),
        "issues_found": len(issues),
        "issues": issues[:10],  # Limit to 10
        "integrity_score": round(integrity_score, 2),
        "has_integrity": has_integrity,
        "recommendation": (
            "Evidence chain appears intact."
            if has_integrity else
            "Integrity issues detected. Verify evidence source before proceeding."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def gate_decision(
    determination_id: str,
    evidence_set: EvidenceSet,
    finding: dict
) -> dict:
    """
    Apply quality gates before allowing a decision to proceed.

    This is the final checkpoint - if this passes, the finding
    can be reported. If not, we abstain.

    Args:
        determination_id: ID of the determination
        evidence_set: Collected evidence
        finding: The proposed finding

    Returns:
        gate_receipt with pass/fail and explanation
    """
    # Run all quality checks
    quality = evaluate_evidence_quality(evidence_set)
    integrity = check_evidence_integrity(
        [e.content for e in evidence_set.supporting_evidence]
    )

    gates_passed = []
    gates_failed = []

    # Gate 1: Minimum evidence quality
    if quality["quality_level"] in ["high", "acceptable"]:
        gates_passed.append("evidence_quality")
    else:
        gates_failed.append({
            "gate": "evidence_quality",
            "reason": f"Quality level '{quality['quality_level']}' below threshold",
        })

    # Gate 2: Evidence integrity
    if integrity["has_integrity"]:
        gates_passed.append("evidence_integrity")
    else:
        gates_failed.append({
            "gate": "evidence_integrity",
            "reason": f"Found {integrity['issues_found']} integrity issues",
        })

    # Gate 3: Counter-evidence ratio
    if evidence_set.counter_evidence_ratio <= COUNTER_EVIDENCE_THRESHOLD:
        gates_passed.append("counter_evidence")
    else:
        gates_failed.append({
            "gate": "counter_evidence",
            "reason": f"Counter-evidence ratio {evidence_set.counter_evidence_ratio:.1%} exceeds threshold",
        })

    # Gate 4: Evidence freshness
    if not evidence_set.has_stale_evidence():
        gates_passed.append("evidence_freshness")
    else:
        gates_failed.append({
            "gate": "evidence_freshness",
            "reason": "Critical evidence is stale (>90 days old)",
        })

    # Determine outcome
    all_passed = len(gates_failed) == 0
    decision = "proceed" if all_passed else "abstain"

    if not all_passed:
        # Emit abstention
        abstain(
            determination_id,
            gates_failed[0]["gate"] if gates_failed else "insufficient_data",
            evidence_set.to_dict()
        )

    return emit_receipt("gate", {
        "tenant_id": TENANT_ID,
        "determination_id": determination_id,
        "gates_passed": gates_passed,
        "gates_failed": gates_failed,
        "decision": decision,
        "all_passed": all_passed,
        "user_message": (
            "All quality gates passed. Finding can be reported."
            if all_passed else
            f"Quality gate(s) failed: {', '.join(g['gate'] for g in gates_failed)}. "
            "Cannot proceed with finding."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === HELPER FUNCTIONS ===

def _get_abstention_message(reason: str) -> str:
    """Get user-friendly abstention message."""
    reason_info = ABSTENTION_REASONS.get(reason, {})
    return reason_info.get(
        "user_message",
        "Unable to make a reliable determination with available evidence."
    )


def _get_proceed_message(quality_level: str) -> str:
    """Get proceed message based on quality level."""
    messages = {
        "high": "Evidence is strong. Proceed with high confidence.",
        "acceptable": "Evidence is adequate. Proceed with normal confidence.",
        "marginal": "Evidence is weak. Proceed with caution and flag for review.",
    }
    return messages.get(quality_level, "Review evidence quality before proceeding.")


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Guardian Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Create test evidence set
    evidence_set = EvidenceSet("test_determination_001")

    # Add supporting evidence
    strong_evidence = EvidenceItem(
        "ev_001",
        "compression_analysis",
        {"compression_ratio": 0.35},
        "automated_scanner"
    )
    evidence_set.add_supporting(strong_evidence)

    weak_evidence = EvidenceItem(
        "ev_002",
        "pattern_match",
        {"pattern": "repetitive_billing"},
        "heuristic_check"
    )
    weak_evidence.add_issue("source_unreliable", 0.3)
    evidence_set.add_supporting(weak_evidence)

    # Add some counter evidence
    counter = EvidenceItem(
        "ev_003",
        "manual_verification",
        {"status": "verified_legitimate"},
        "auditor_review"
    )
    counter.quality_score = 0.5
    evidence_set.add_counter(counter)

    # Test quality evaluation
    quality = evaluate_evidence_quality(evidence_set)
    assert "quality_level" in quality
    print(f"# Quality level: {quality['quality_level']}", file=sys.stderr)
    print(f"# Should abstain: {quality['should_abstain']}", file=sys.stderr)

    # Test abstention
    abstention = abstain(
        "test_determination_002",
        "conflicting_signals",
        {"note": "test abstention"}
    )
    assert abstention["because"] == "conflicting_signals"
    print(f"# Abstention reason: {abstention['reason_code']}", file=sys.stderr)

    # Test counter-evidence check
    records = [
        {"vendor": "Test Vendor", "verification_status": "verified"},
        {"vendor": "Test Vendor", "physical_inspection": "passed"},
    ]
    counter_result = check_counter_evidence(
        {"anomaly_type": "ghost_vendor_compression", "affected_entities": ["Test Vendor"]},
        records
    )
    assert counter_result["counter_evidence_found"] > 0
    print(f"# Counter evidence found: {counter_result['counter_evidence_found']}", file=sys.stderr)

    # Test integrity check
    test_receipts = [
        {"payload_hash": "abc123", "ts": "2024-01-01T00:00:00Z"},
        {"payload_hash": "def456", "ts": "2024-01-02T00:00:00Z"},
        {"payload_hash": "ghi789", "ts": "2024-01-03T00:00:00Z"},
    ]
    integrity = check_evidence_integrity(test_receipts)
    assert integrity["has_integrity"]
    print(f"# Integrity score: {integrity['integrity_score']}", file=sys.stderr)

    print("# PASS: guardian module self-test", file=sys.stderr)
