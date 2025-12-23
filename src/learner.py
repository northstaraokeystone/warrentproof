"""
WarrantProof Learner Module - Cross-Domain Pattern Transfer

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module enables the system to learn from fraud patterns in
one domain and apply that learning to detect similar patterns
in other domains.

The Insight:
Fraud follows patterns. The scheme used in naval logistics might
look different on the surface than aerospace parts gouging, but
the underlying structure - the fingerprint - is often similar.

How It Works:
1. Extract the "signature" of known fraud patterns
2. Generalize those signatures to be domain-independent
3. Match new data against generalized signatures
4. Transfer high-confidence patterns across domains

For Non-Technical Users:
Think of it like recognizing a scam artist's tricks. Once you've
seen how they work in one situation, you can spot similar tricks
in completely different contexts. The specific details change,
but the underlying pattern is the same.

Example:
The "Fat Leonard" naval logistics scandal involved repetitive
billing patterns. We can extract what made those patterns
suspicious and then look for similar patterns in Army
maintenance contracts or Air Force parts procurement.

Key Concepts:
- Pattern Signature: The essential characteristics of a fraud type
- Domain Transfer: Applying patterns from one area to another
- Generalization: Making patterns applicable across contexts
- Specificity: How narrow or broad a pattern is
"""

from datetime import datetime
from typing import Optional
import math

from .core import (
    TENANT_ID,
    DISCLAIMER,
    dual_hash,
    emit_receipt,
    get_citation,
    StopRuleException,
    MUTUAL_INFO_TRANSFER_THRESHOLD,
    CROSS_BRANCH_ACCURACY_TARGET,
)


# === PATTERN REGISTRY ===

# Known fraud pattern signatures (generalized from historical cases)
KNOWN_PATTERNS = {
    "repetitive_billing": {
        "name": "Repetitive Billing Pattern",
        "source_case": "Fat Leonard (GDMA)",
        "description": "Invoice descriptions that repeat with minimal variation",
        "signature": {
            "compression_ratio": {"operator": "<", "value": 0.50},
            "description_entropy": {"operator": "<", "value": 2.0},
            "unique_descriptions_pct": {"operator": "<", "value": 0.30},
        },
        "applicable_domains": ["logistics", "maintenance", "services"],
        "transferability": 0.85,  # How well this transfers to new domains
    },
    "price_gouging": {
        "name": "Price Gouging Pattern",
        "source_case": "TransDigm",
        "description": "Prices that far exceed comparable market rates",
        "signature": {
            "price_to_estimate_ratio": {"operator": ">", "value": 2.0},
            "sole_source_indicator": {"operator": "==", "value": True},
            "market_comparison_variance": {"operator": ">", "value": 1.5},
        },
        "applicable_domains": ["spare_parts", "consumables", "equipment"],
        "transferability": 0.90,
    },
    "shell_company": {
        "name": "Shell Company Pattern",
        "source_case": "General Fraud Pattern",
        "description": "Characteristics of fictitious vendors",
        "signature": {
            "registration_age_days": {"operator": "<", "value": 180},
            "physical_presence_verified": {"operator": "==", "value": False},
            "employee_count": {"operator": "<", "value": 3},
            "contract_diversity": {"operator": "<", "value": 0.20},
        },
        "applicable_domains": ["all"],
        "transferability": 0.95,
    },
    "conflict_of_interest": {
        "name": "Conflict of Interest Pattern",
        "source_case": "Boeing/Druyun",
        "description": "Signs of insider dealing or revolving door corruption",
        "signature": {
            "personnel_overlap": {"operator": ">", "value": 0},
            "timing_correlation": {"operator": ">", "value": 0.70},
            "approval_chain_anomaly": {"operator": "==", "value": True},
        },
        "applicable_domains": ["major_contracts", "sole_source", "modifications"],
        "transferability": 0.75,
    },
    "cost_escalation": {
        "name": "Cost Escalation Pattern",
        "source_case": "General Defense Pattern",
        "description": "Strategic low-balling followed by cost growth",
        "signature": {
            "initial_bid_ratio": {"operator": "<", "value": 0.80},  # Bid < 80% of estimate
            "final_cost_ratio": {"operator": ">", "value": 1.30},   # Final > 130% of estimate
            "modification_count": {"operator": ">", "value": 10},
        },
        "applicable_domains": ["construction", "shipbuilding", "aircraft", "systems"],
        "transferability": 0.80,
    },
}


# === CORE CLASSES ===

class PatternSignature:
    """
    Represents a generalized fraud pattern that can be matched
    against new data regardless of domain.
    """

    def __init__(self, pattern_id: str, pattern_data: dict):
        self.pattern_id = pattern_id
        self.name = pattern_data.get("name", pattern_id)
        self.source_case = pattern_data.get("source_case", "Unknown")
        self.description = pattern_data.get("description", "")
        self.signature = pattern_data.get("signature", {})
        self.applicable_domains = pattern_data.get("applicable_domains", ["all"])
        self.transferability = pattern_data.get("transferability", 0.5)
        self.match_count = 0
        self.false_positive_count = 0

    def matches(self, data: dict) -> tuple[bool, float, list]:
        """
        Check if data matches this pattern signature.

        Args:
            data: Data to check (dict with values for signature fields)

        Returns:
            (matches: bool, confidence: float, matched_rules: list)
        """
        matched_rules = []
        total_rules = len(self.signature)

        if total_rules == 0:
            return False, 0.0, []

        for field, rule in self.signature.items():
            if field not in data:
                continue

            value = data[field]
            operator = rule.get("operator", "==")
            threshold = rule.get("value")

            match = False
            if operator == "<" and value < threshold:
                match = True
            elif operator == ">" and value > threshold:
                match = True
            elif operator == "==" and value == threshold:
                match = True
            elif operator == "<=" and value <= threshold:
                match = True
            elif operator == ">=" and value >= threshold:
                match = True

            if match:
                matched_rules.append({
                    "field": field,
                    "value": value,
                    "rule": f"{operator} {threshold}",
                })

        # Calculate match confidence
        if len(matched_rules) == 0:
            return False, 0.0, []

        match_ratio = len(matched_rules) / total_rules
        confidence = match_ratio * self.transferability

        # Require at least 50% of rules to match
        matches = match_ratio >= 0.5

        return matches, confidence, matched_rules

    @property
    def accuracy(self) -> float:
        """Pattern accuracy based on historical matches."""
        if self.match_count == 0:
            return self.transferability  # Use default transferability
        return 1.0 - (self.false_positive_count / self.match_count)


class PatternLibrary:
    """
    Central library of known fraud patterns.
    """

    def __init__(self):
        self.patterns: dict[str, PatternSignature] = {}
        self._load_known_patterns()

    def _load_known_patterns(self):
        """Load patterns from KNOWN_PATTERNS registry."""
        for pattern_id, pattern_data in KNOWN_PATTERNS.items():
            self.patterns[pattern_id] = PatternSignature(pattern_id, pattern_data)

    def add_pattern(
        self,
        pattern_id: str,
        name: str,
        description: str,
        signature: dict,
        source_case: str = "User-defined",
        domains: Optional[list] = None,
        transferability: float = 0.7
    ):
        """Add a new pattern to the library."""
        self.patterns[pattern_id] = PatternSignature(pattern_id, {
            "name": name,
            "source_case": source_case,
            "description": description,
            "signature": signature,
            "applicable_domains": domains or ["all"],
            "transferability": transferability,
        })

    def find_matches(
        self,
        data: dict,
        domain: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> list:
        """
        Find all patterns that match the given data.

        Args:
            data: Data to check against patterns
            domain: Optional domain filter
            min_confidence: Minimum confidence to report

        Returns:
            List of matching patterns with confidence scores
        """
        matches = []

        for pattern_id, pattern in self.patterns.items():
            # Filter by domain if specified
            if domain and "all" not in pattern.applicable_domains:
                if domain not in pattern.applicable_domains:
                    continue

            match, confidence, rules = pattern.matches(data)

            if match and confidence >= min_confidence:
                matches.append({
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.name,
                    "confidence": round(confidence, 3),
                    "matched_rules": rules,
                    "description": pattern.description,
                    "source_case": pattern.source_case,
                })

        # Sort by confidence
        matches.sort(key=lambda x: -x["confidence"])

        return matches


# === GLOBAL INSTANCE ===

_library = PatternLibrary()


# === CORE FUNCTIONS ===

def match_patterns(
    data: dict,
    domain: Optional[str] = None,
    min_confidence: float = 0.5
) -> dict:
    """
    Check data against known fraud patterns.

    This is the main entry point for pattern matching. It takes
    raw data and identifies which known fraud patterns it resembles.

    Args:
        data: Data to analyze (dict with field values)
        domain: Optional domain for filtering (e.g., "logistics")
        min_confidence: Minimum confidence to report (default 0.5)

    Returns:
        match_receipt with all matching patterns
    """
    matches = _library.find_matches(data, domain, min_confidence)

    if not matches:
        return emit_receipt("pattern_match", {
            "tenant_id": TENANT_ID,
            "domain": domain,
            "patterns_checked": len(_library.patterns),
            "matches_found": 0,
            "matches": [],
            "risk_level": "low",
            "summary": "No known fraud patterns detected.",
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    # Determine overall risk level
    max_confidence = max(m["confidence"] for m in matches)
    if max_confidence >= 0.8:
        risk_level = "critical"
    elif max_confidence >= 0.6:
        risk_level = "high"
    elif max_confidence >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Build summary
    top_patterns = [m["pattern_name"] for m in matches[:3]]
    summary = f"Matched {len(matches)} known fraud pattern(s): {', '.join(top_patterns)}"

    return emit_receipt("pattern_match", {
        "tenant_id": TENANT_ID,
        "domain": domain,
        "patterns_checked": len(_library.patterns),
        "matches_found": len(matches),
        "matches": matches[:10],  # Limit to top 10
        "risk_level": risk_level,
        "highest_confidence": round(max_confidence, 3),
        "summary": summary,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def transfer_pattern(
    source_domain: str,
    target_domain: str,
    pattern_id: str,
    adaptation_notes: Optional[str] = None
) -> dict:
    """
    Transfer a pattern from one domain to another.

    Documents the intentional application of learnings from one
    area to detect fraud in another.

    Args:
        source_domain: Where the pattern was learned
        target_domain: Where the pattern is being applied
        pattern_id: Which pattern is being transferred
        adaptation_notes: Any adjustments made for the new domain

    Returns:
        transfer_receipt documenting the transfer
    """
    if pattern_id not in _library.patterns:
        return emit_receipt("transfer", {
            "tenant_id": TENANT_ID,
            "success": False,
            "error": f"Pattern '{pattern_id}' not found in library.",
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    pattern = _library.patterns[pattern_id]

    # Check if transfer is appropriate
    if "all" not in pattern.applicable_domains:
        if target_domain not in pattern.applicable_domains:
            # Pattern may not transfer well
            transfer_quality = "uncertain"
            confidence_adjustment = 0.5
        else:
            transfer_quality = "supported"
            confidence_adjustment = 1.0
    else:
        transfer_quality = "universal"
        confidence_adjustment = 0.9

    effective_transferability = pattern.transferability * confidence_adjustment

    return emit_receipt("transfer", {
        "tenant_id": TENANT_ID,
        "pattern_id": pattern_id,
        "pattern_name": pattern.name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "base_transferability": pattern.transferability,
        "confidence_adjustment": confidence_adjustment,
        "effective_transferability": round(effective_transferability, 3),
        "transfer_quality": transfer_quality,
        "adaptation_notes": adaptation_notes,
        "recommendation": _get_transfer_recommendation(
            transfer_quality, effective_transferability
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def learn_pattern(
    name: str,
    description: str,
    signature: dict,
    source_case: str,
    domains: Optional[list] = None,
    confidence: float = 0.7
) -> dict:
    """
    Add a new pattern to the library from observed data.

    Use this when you've identified a new fraud pattern that
    should be tracked going forward.

    Args:
        name: Human-readable name for the pattern
        description: What this pattern indicates
        signature: Rules that define the pattern
        source_case: Where this pattern was learned from
        domains: Which domains this applies to
        confidence: Initial confidence in pattern accuracy

    Returns:
        learn_receipt documenting the new pattern
    """
    # Generate pattern ID
    pattern_id = f"learned_{dual_hash(name)[:16]}"

    # Validate signature
    if not signature:
        return emit_receipt("learn", {
            "tenant_id": TENANT_ID,
            "success": False,
            "error": "Pattern signature cannot be empty.",
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    # Add to library
    _library.add_pattern(
        pattern_id=pattern_id,
        name=name,
        description=description,
        signature=signature,
        source_case=source_case,
        domains=domains,
        transferability=confidence
    )

    return emit_receipt("learn", {
        "tenant_id": TENANT_ID,
        "success": True,
        "pattern_id": pattern_id,
        "pattern_name": name,
        "signature_rules": len(signature),
        "applicable_domains": domains or ["all"],
        "initial_confidence": confidence,
        "message": f"Pattern '{name}' added to library. Will be matched against future data.",
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_pattern_for_users(pattern_match: dict) -> dict:
    """
    Explain a pattern match in plain language.

    Args:
        pattern_match: A single match from match_patterns

    Returns:
        insight_receipt with user-friendly explanation
    """
    pattern_id = pattern_match.get("pattern_id", "unknown")
    confidence = pattern_match.get("confidence", 0)
    matched_rules = pattern_match.get("matched_rules", [])

    # Get pattern details
    if pattern_id in _library.patterns:
        pattern = _library.patterns[pattern_id]
        source = pattern.source_case
        description = pattern.description
    else:
        source = "Unknown"
        description = pattern_match.get("description", "")

    # Build plain-language explanation
    if confidence >= 0.8:
        confidence_text = "very likely"
    elif confidence >= 0.6:
        confidence_text = "likely"
    elif confidence >= 0.4:
        confidence_text = "possibly"
    else:
        confidence_text = "might"

    # Explain what triggered the match
    triggers = []
    for rule in matched_rules[:3]:
        field = rule["field"].replace("_", " ")
        triggers.append(f"the {field}")

    trigger_text = ", ".join(triggers) if triggers else "several indicators"

    explanation = (
        f"This case {confidence_text} follows the '{pattern_match.get('pattern_name')}' "
        f"pattern, which was first identified in the {source} case. "
        f"The match was triggered by {trigger_text}. "
        f"{description}"
    )

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "pattern_explanation",
        "pattern_name": pattern_match.get("pattern_name"),
        "confidence_level": f"{confidence:.0%}",
        "confidence_text": confidence_text,
        "explanation": explanation,
        "what_to_do": (
            "Review the specific indicators that triggered this match. "
            "Compare against the original case for similarities."
        ),
        "historical_context": f"Pattern first seen in: {source}",
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def get_library_summary() -> dict:
    """
    Get summary of all patterns in the library.

    Returns:
        summary_receipt with pattern inventory
    """
    patterns = []
    domains = set()

    for pattern_id, pattern in _library.patterns.items():
        patterns.append({
            "pattern_id": pattern_id,
            "name": pattern.name,
            "source": pattern.source_case,
            "domains": pattern.applicable_domains,
            "transferability": pattern.transferability,
            "accuracy": round(pattern.accuracy, 2),
        })
        domains.update(pattern.applicable_domains)

    return emit_receipt("library_summary", {
        "tenant_id": TENANT_ID,
        "total_patterns": len(patterns),
        "domains_covered": list(domains),
        "patterns": patterns,
        "average_transferability": round(
            sum(p["transferability"] for p in patterns) / len(patterns)
            if patterns else 0, 2
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === HELPER FUNCTIONS ===

def _get_transfer_recommendation(quality: str, transferability: float) -> str:
    """Generate recommendation for pattern transfer."""
    if quality == "universal" and transferability >= 0.8:
        return "Pattern transfers well. Apply with high confidence."
    elif quality == "supported" and transferability >= 0.6:
        return "Pattern is applicable. Apply with normal confidence."
    elif quality == "uncertain":
        return (
            "Pattern may not transfer well to this domain. "
            "Use with caution and validate results carefully."
        )
    else:
        return "Review pattern applicability before relying on results."


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Learner Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test pattern matching
    test_data = {
        "compression_ratio": 0.35,
        "description_entropy": 1.5,
        "unique_descriptions_pct": 0.20,
    }
    matches = match_patterns(test_data)
    assert matches["matches_found"] > 0, "Should match repetitive_billing"
    print(f"# Pattern matches: {matches['matches_found']}", file=sys.stderr)
    print(f"# Top match: {matches['matches'][0]['pattern_name']}", file=sys.stderr)

    # Test pattern transfer
    transfer = transfer_pattern(
        source_domain="logistics",
        target_domain="maintenance",
        pattern_id="repetitive_billing"
    )
    assert "effective_transferability" in transfer
    print(f"# Transfer quality: {transfer['transfer_quality']}", file=sys.stderr)

    # Test learning new pattern
    new_pattern = learn_pattern(
        name="Test Pattern",
        description="A test pattern for self-test",
        signature={"test_field": {"operator": ">", "value": 100}},
        source_case="Self-Test",
        confidence=0.6
    )
    assert new_pattern["success"]
    print(f"# Learned new pattern: {new_pattern['pattern_id']}", file=sys.stderr)

    # Test pattern explanation
    if matches["matches"]:
        explanation = explain_pattern_for_users(matches["matches"][0])
        assert "explanation" in explanation
        print(f"# Explanation generated: {explanation['confidence_text']}", file=sys.stderr)

    # Test library summary
    summary = get_library_summary()
    assert summary["total_patterns"] >= len(KNOWN_PATTERNS)
    print(f"# Library contains {summary['total_patterns']} patterns", file=sys.stderr)

    print("# PASS: learner module self-test", file=sys.stderr)
