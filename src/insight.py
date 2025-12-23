"""
WarrantProof Insight Module - Plain-Language Fraud Explanations

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module translates complex physics-based fraud signals into
clear, actionable explanations that anyone can understand.

The Goal: Make fraud detection results understandable without
requiring knowledge of compression ratios, entropy, or Kolmogorov
complexity. Every finding is explained in plain English.

How It Works:
1. Takes raw analysis results from other modules
2. Converts technical metrics into human-readable summaries
3. Provides clear "what this means" explanations
4. Suggests next steps for investigators

Example Output:
  "This contract looks suspicious because the billing descriptions
   are unusually repetitive - like they were copied from a template
   rather than describing actual work done."
"""

from typing import Optional
from .core import (
    TENANT_ID,
    DISCLAIMER,
    KOLMOGOROV_THRESHOLD,
    KOLMOGOROV_LEGITIMATE_MIN,
    dual_hash,
    emit_receipt,
    get_citation,
    StopRuleException,
)


# === CONFIDENCE LEVELS (User-Friendly) ===

CONFIDENCE_LABELS = {
    (0.0, 0.3): ("low", "There are some minor indicators worth noting"),
    (0.3, 0.6): ("moderate", "Several patterns suggest this needs review"),
    (0.6, 0.8): ("high", "Strong evidence suggests a problem"),
    (0.8, 1.0): ("very_high", "Clear indicators of a serious issue"),
}


# === PLAIN-LANGUAGE EXPLANATIONS ===

ANOMALY_EXPLANATIONS = {
    "compression_failure": {
        "title": "Unusual Pattern Detected",
        "what_it_means": (
            "The billing records for this contract show unusually repetitive "
            "patterns - like the same phrases or amounts appearing over and over. "
            "In legitimate contracts, we expect natural variation in how work is "
            "described. This repetition can indicate copied invoices or templated billing."
        ),
        "why_it_matters": (
            "Real work generates unique descriptions. When we see cookie-cutter "
            "invoices, it may mean someone is billing for work that wasn't actually done."
        ),
        "suggested_action": "Review the original invoices to check if descriptions match actual work performed.",
    },
    "ghost_vendor_compression": {
        "title": "Possible Shell Company Activity",
        "what_it_means": (
            "This vendor's records show characteristics often seen with fake companies: "
            "random-looking business names, inconsistent contact information, or "
            "minimal documentation compared to legitimate contractors."
        ),
        "why_it_matters": (
            "Ghost vendors are fictitious companies set up to receive payments "
            "for goods or services that were never delivered."
        ),
        "suggested_action": "Verify the vendor exists: check their physical address, business registration, and previous contract history.",
    },
    "cert_fraud_compression": {
        "title": "Quality Certification Concerns",
        "what_it_means": (
            "The quality inspection records show unusual patterns - different "
            "inspectors using identical language, or certifications that don't "
            "follow normal inspection workflows."
        ),
        "why_it_matters": (
            "Fraudulent quality certifications can mean defective parts or materials "
            "were approved for use in critical systems."
        ),
        "suggested_action": "Cross-reference inspector credentials and verify physical inspection records.",
    },
    "kolmogorov_anomaly": {
        "title": "Artificially Simple Records",
        "what_it_means": (
            "These records are far simpler than they should be for this type of contract. "
            "It's like finding a 'paint by numbers' where we expected an original painting. "
            "The data was likely generated from a simple formula rather than reflecting "
            "real business activity."
        ),
        "why_it_matters": (
            "Legitimate procurement involves many independent decisions that create "
            "natural complexity. When records are too 'neat', they were probably manufactured."
        ),
        "suggested_action": "Request original source documents and compare against the digital records.",
    },
    "raf_cycle_detected": {
        "title": "Circular Money Flow Detected",
        "what_it_means": (
            "Money appears to be moving in a circle between related parties. "
            "For example: Company A pays Company B, which pays Company C, which "
            "then pays Company A. This is a classic sign of kickbacks or collusion."
        ),
        "why_it_matters": (
            "Circular payments often disguise the fact that money is being siphoned "
            "out of legitimate contracts into the pockets of conspirators."
        ),
        "suggested_action": "Map all relationships between entities in the payment chain and investigate any shared ownership or personnel.",
    },
    "bekenstein_violation": {
        "title": "Documentation Gap",
        "what_it_means": (
            "The amount of documentation for this transaction is unusually low "
            "given the dollar amount involved. Large contracts should generate "
            "substantial paper trails - approvals, specifications, inspections, etc."
        ),
        "why_it_matters": (
            "Missing documentation makes it impossible to verify that the "
            "government received what it paid for."
        ),
        "suggested_action": "Request complete transaction documentation and compare against similar contracts.",
    },
    "data_unavailable": {
        "title": "Missing Records",
        "what_it_means": (
            "Key records that should exist for this contract cannot be located. "
            "This could indicate poor record-keeping, but it can also indicate "
            "deliberate destruction of evidence."
        ),
        "why_it_matters": (
            "Complete records are essential for accountability. Their absence "
            "prevents any form of meaningful audit."
        ),
        "suggested_action": "Document what's missing, when it was last seen, and who had access.",
    },
    "time_anomaly": {
        "title": "Suspicious Timing",
        "what_it_means": (
            "The dates on these records don't make sense - approvals happening "
            "before requests, deliveries before orders, or payments before work. "
            "This often indicates backdated or fabricated documents."
        ),
        "why_it_matters": (
            "Proper sequence of events is fundamental to legitimate transactions. "
            "Broken timelines suggest after-the-fact paperwork created to cover fraud."
        ),
        "suggested_action": "Create a timeline of all events and identify logical impossibilities.",
    },
    "cost_cascade": {
        "title": "Cost Overrun Pattern",
        "what_it_means": (
            "Costs have consistently grown beyond estimates in ways that suggest "
            "deliberate low-balling of initial bids. The pattern indicates the "
            "contractor may have intentionally underestimated to win the contract, "
            "knowing they could increase costs later."
        ),
        "why_it_matters": (
            "Bid rigging through strategic underestimation undermines fair competition "
            "and can cost taxpayers billions."
        ),
        "suggested_action": "Compare original bid estimates against similar contracts and final costs.",
    },
}


# === CORE FUNCTIONS ===

def explain_anomaly(anomaly: dict) -> dict:
    """
    Convert a technical anomaly into a plain-language explanation.

    Args:
        anomaly: Raw anomaly dict from detection modules

    Returns:
        insight_receipt with human-readable explanation
    """
    anomaly_type = anomaly.get("anomaly_type", "unknown")

    # Get explanation template
    template = ANOMALY_EXPLANATIONS.get(anomaly_type, {
        "title": "Unusual Activity Detected",
        "what_it_means": "Our analysis found patterns that warrant further review.",
        "why_it_matters": "This may indicate issues with the transaction records.",
        "suggested_action": "Review the flagged records with appropriate subject matter experts.",
    })

    # Determine confidence level
    confidence = anomaly.get("fraud_likelihood", anomaly.get("confidence", 0.5))
    confidence_label, confidence_message = _get_confidence_label(confidence)

    # Build summary sentence
    summary = _build_summary(anomaly_type, confidence, anomaly)

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "anomaly_type": anomaly_type,
        "title": template["title"],
        "summary": summary,
        "what_it_means": template["what_it_means"],
        "why_it_matters": template["why_it_matters"],
        "suggested_action": template["suggested_action"],
        "confidence_level": confidence_label,
        "confidence_message": confidence_message,
        "confidence_score": round(confidence, 2),
        "technical_details": _sanitize_for_display(anomaly),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_compression_result(result: dict) -> dict:
    """
    Explain what a compression analysis result means in plain terms.

    Args:
        result: Compression receipt from compress.py

    Returns:
        insight_receipt with explanation
    """
    ratio = result.get("compression_ratio", 0.5)
    classification = result.get("classification", "unknown")

    if classification == "legitimate":
        title = "Records Look Normal"
        summary = (
            "The transaction records show the natural variation we expect "
            "from legitimate business activity. No unusual patterns detected."
        )
        recommendation = "No immediate action required. Include in routine audit sampling."
    elif classification == "suspicious":
        title = "Records Need Review"
        summary = (
            f"Some patterns in these records are unusual. About {int((1-ratio)*100)}% "
            "of the data shows repetitive patterns that could indicate templated billing, "
            "but more investigation is needed to determine if there's a problem."
        )
        recommendation = "Flag for priority review. Compare against similar contracts."
    else:  # fraudulent
        title = "Significant Concerns Identified"
        summary = (
            f"These records show strong signs of artificial generation. "
            f"Approximately {int((1-ratio)*100)}% of the content appears templated "
            "or copied, which is far outside normal ranges."
        )
        recommendation = "Escalate immediately for detailed investigation."

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "compression",
        "title": title,
        "summary": summary,
        "recommendation": recommendation,
        "metrics_explained": {
            "compression_ratio": f"{ratio:.1%} unique content (higher is better, >80% is normal)",
            "entropy": f"Information variety: {result.get('entropy_score', 0):.1f} bits (3-4 is typical)",
            "coherence": f"Pattern consistency: {result.get('coherence_score', 0):.1%} (>70% is normal)",
        },
        "classification": classification,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_kolmogorov_result(result: dict) -> dict:
    """
    Explain Kolmogorov complexity in simple terms.

    The key insight: If something can be described with a short formula,
    it's probably artificial. Real-world data resists compression.

    Args:
        result: Kolmogorov receipt from kolmogorov.py

    Returns:
        insight_receipt with explanation
    """
    k_ratio = result.get("kolmogorov_ratio", 0.5)
    is_fraud = result.get("is_fraud", False)

    if is_fraud:
        title = "Records Appear Artificially Generated"
        summary = (
            "These records can be described with a very simple formula, which suggests "
            "they were computer-generated rather than reflecting real business activity. "
            "Think of it like finding a perfectly repeating wallpaper pattern where "
            "you expected a photograph of actual work."
        )
        analogy = (
            "Imagine if someone's expense report could be recreated by just saying "
            "'repeat this same $500 meal charge every Tuesday'. Real expenses have "
            "variety - different amounts, different vendors, different days."
        )
    elif k_ratio >= KOLMOGOROV_LEGITIMATE_MIN:
        title = "Records Show Natural Complexity"
        summary = (
            "These records have the rich, varied content we expect from legitimate "
            "business activity. They can't be summarized with a simple formula, "
            "which is exactly what real-world data looks like."
        )
        analogy = (
            "Like a genuine photograph vs. a computer graphic - real things have "
            "subtle irregularities that are hard to fake."
        )
    else:
        title = "Records Show Mixed Signals"
        summary = (
            "The complexity of these records falls in a gray zone. Some aspects "
            "look natural, while others appear more formulaic. This warrants "
            "a closer look to determine what's going on."
        )
        analogy = (
            "Like a painting that might be original or might be a very good copy - "
            "expert review is needed."
        )

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "kolmogorov",
        "title": title,
        "summary": summary,
        "analogy": analogy,
        "complexity_explained": f"{k_ratio:.1%} complexity (>75% is normal, <65% is concerning)",
        "is_fraud": is_fraud,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_raf_result(result: dict) -> dict:
    """
    Explain network cycle detection in simple terms.

    Args:
        result: RAF receipt from raf.py

    Returns:
        insight_receipt with explanation
    """
    cycles = result.get("cycles_detected", 0)
    keystones = result.get("keystone_species", [])

    if cycles == 0:
        title = "No Circular Money Flows Detected"
        summary = (
            "Payments flow in expected patterns without suspicious circular paths. "
            "Money goes from the government to contractors for goods and services "
            "without doubling back through chains of related entities."
        )
    else:
        title = f"Detected {cycles} Circular Payment Pattern{'s' if cycles > 1 else ''}"
        keystone_text = ""
        if keystones:
            keystone_text = (
                f" The following entities appear in multiple cycles, making them "
                f"central to the pattern: {', '.join(keystones[:3])}"
                f"{' and others' if len(keystones) > 3 else ''}."
            )
        summary = (
            f"Found {cycles} instance{'s' if cycles > 1 else ''} where money "
            f"flows in a circle between related parties.{keystone_text} "
            "This pattern is often associated with kickback schemes where "
            "conspirators pass money between shell companies."
        )

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "network_cycles",
        "title": title,
        "summary": summary,
        "what_to_look_for": (
            "Check if the entities in these cycles share: common ownership, "
            "shared addresses, overlapping personnel, or were created around "
            "the same time. Legitimate business relationships rarely form "
            "perfect circles."
        ),
        "cycles_found": cycles,
        "key_entities": keystones[:5] if keystones else [],
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def generate_executive_summary(analysis_results: list) -> dict:
    """
    Create a high-level summary for decision-makers.

    Args:
        analysis_results: List of analysis receipts from various modules

    Returns:
        Executive summary insight receipt
    """
    # Count issues by severity
    critical = 0
    high = 0
    moderate = 0
    low = 0
    clean = 0

    key_findings = []

    for result in analysis_results:
        classification = result.get("classification", "")
        likelihood = result.get("fraud_likelihood", 0)

        if classification == "fraudulent" or likelihood > 0.8:
            critical += 1
            if len(key_findings) < 3:
                key_findings.append(_summarize_finding(result))
        elif classification == "suspicious" or likelihood > 0.6:
            high += 1
        elif likelihood > 0.3:
            moderate += 1
        elif likelihood > 0.1:
            low += 1
        else:
            clean += 1

    total = len(analysis_results)

    # Determine overall status
    if critical > 0:
        status = "requires_immediate_attention"
        status_message = (
            f"Found {critical} critical issue{'s' if critical > 1 else ''} "
            "requiring immediate investigation."
        )
    elif high > 0:
        status = "needs_review"
        status_message = (
            f"Found {high} significant concern{'s' if high > 1 else ''} "
            "that should be reviewed promptly."
        )
    elif moderate > 0:
        status = "monitor"
        status_message = (
            f"Found {moderate} minor pattern{'s' if moderate > 1 else ''} "
            "worth monitoring."
        )
    else:
        status = "normal"
        status_message = "No significant concerns identified in this analysis."

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "executive_summary",
        "status": status,
        "status_message": status_message,
        "summary_counts": {
            "critical_issues": critical,
            "high_concerns": high,
            "moderate_flags": moderate,
            "low_indicators": low,
            "clean_records": clean,
            "total_analyzed": total,
        },
        "key_findings": key_findings,
        "recommendation": _generate_recommendation(status, critical, high),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === HELPER FUNCTIONS ===

def _get_confidence_label(confidence: float) -> tuple:
    """Get human-readable confidence label and message."""
    for (low, high), (label, message) in CONFIDENCE_LABELS.items():
        if low <= confidence < high:
            return label, message
    return "very_high", "Clear indicators of a serious issue"


def _build_summary(anomaly_type: str, confidence: float, anomaly: dict) -> str:
    """Build a one-sentence summary of the finding."""
    confidence_word = {
        (0.0, 0.3): "may",
        (0.3, 0.6): "likely",
        (0.6, 0.8): "appears to",
        (0.8, 1.0): "clearly",
    }

    action_word = "may"
    for (low, high), word in confidence_word.items():
        if low <= confidence < high:
            action_word = word
            break

    type_descriptions = {
        "compression_failure": "involve copied or templated billing",
        "ghost_vendor_compression": "involve a fictitious vendor",
        "cert_fraud_compression": "have fraudulent quality certifications",
        "kolmogorov_anomaly": "contain artificially generated records",
        "raf_cycle_detected": "involve circular money flows between related parties",
        "bekenstein_violation": "be missing required documentation",
        "time_anomaly": "have backdated or fabricated documents",
        "cost_cascade": "involve deliberate cost manipulation",
    }

    description = type_descriptions.get(anomaly_type, "require further investigation")

    return f"This contract {action_word} {description}."


def _sanitize_for_display(data: dict) -> dict:
    """Remove overly technical fields for display."""
    exclude = ["payload_hash", "tenant_id", "simulation_flag", "citation"]
    return {k: v for k, v in data.items() if k not in exclude}


def _summarize_finding(result: dict) -> str:
    """Create a one-line summary of a finding."""
    anomaly_type = result.get("anomaly_type", result.get("receipt_type", "unknown"))
    likelihood = result.get("fraud_likelihood", result.get("confidence", 0))

    template = ANOMALY_EXPLANATIONS.get(anomaly_type, {})
    title = template.get("title", "Issue Detected")

    return f"{title} (confidence: {likelihood:.0%})"


def _generate_recommendation(status: str, critical: int, high: int) -> str:
    """Generate action recommendation based on findings."""
    if status == "requires_immediate_attention":
        return (
            f"URGENT: {critical} finding{'s' if critical > 1 else ''} "
            "require immediate investigation. Recommend suspending related "
            "payments pending review and briefing appropriate oversight officials."
        )
    elif status == "needs_review":
        return (
            f"Schedule priority review of {high} flagged item{'s' if high > 1 else ''}. "
            "Consider enhanced monitoring of related contracts and vendors."
        )
    elif status == "monitor":
        return (
            "Include flagged items in next routine audit cycle. "
            "No immediate action required."
        )
    else:
        return "Continue standard monitoring. No action required."


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Insight Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test anomaly explanation
    test_anomaly = {
        "anomaly_type": "compression_failure",
        "fraud_likelihood": 0.75,
        "compression_ratio": 0.42,
    }
    insight = explain_anomaly(test_anomaly)
    assert "title" in insight, "Insight must have title"
    assert "summary" in insight, "Insight must have summary"
    assert "what_it_means" in insight, "Insight must explain meaning"
    assert "suggested_action" in insight, "Insight must suggest action"
    print(f"# Anomaly explanation: {insight['title']}", file=sys.stderr)

    # Test compression explanation
    test_compression = {
        "compression_ratio": 0.45,
        "entropy_score": 5.2,
        "coherence_score": 0.35,
        "classification": "fraudulent",
    }
    insight = explain_compression_result(test_compression)
    assert insight["classification"] == "fraudulent"
    print(f"# Compression explanation: {insight['title']}", file=sys.stderr)

    # Test executive summary
    test_results = [
        {"classification": "legitimate", "fraud_likelihood": 0.1},
        {"classification": "suspicious", "fraud_likelihood": 0.5},
        {"classification": "fraudulent", "fraud_likelihood": 0.9},
    ]
    summary = generate_executive_summary(test_results)
    assert summary["summary_counts"]["critical_issues"] == 1
    assert summary["summary_counts"]["total_analyzed"] == 3
    print(f"# Executive summary status: {summary['status']}", file=sys.stderr)

    print("# PASS: insight module self-test", file=sys.stderr)
