"""
Gov-OS Temporal Insight Module - Plain-English Fraud Explanations

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY

Purpose: Generate human-readable explanations for audit transparency.
Per spec: "Insights plain-English for audit transparency"

This module translates technical anomaly data into explanations that:
1. Non-technical stakeholders can understand
2. Auditors can use in reports
3. Investigators can act upon
"""

from typing import Any, Dict

from .constants import DISCLAIMER, ZOMBIE_DAYS
from .receipt import emit_insight_receipt


def explain_temporal_anomaly(receipt: Dict[str, Any]) -> str:
    """
    Generate plain-English explanation for temporal anomaly.

    Example output:
    "This vendor was flagged because it maintained a relationship with
    [supplier] for [365] days without any transactions, which is
    physically improbable for legitimate business relationships."

    Args:
        receipt: temporal_anomaly_receipt dict

    Returns:
        Plain-English explanation string
    """
    from_node = receipt.get("from_node", "Entity A")
    to_node = receipt.get("to_node", "Entity B")
    days = receipt.get("days_since_last", 0)
    resistance = receipt.get("resistance", 0.0)
    expected = receipt.get("expected_weight", 0.0)
    observed = receipt.get("observed_weight", 0.0)

    # Determine strength of anomaly
    if resistance > 1.0:
        strength = "extremely"
    elif resistance > 0.5:
        strength = "highly"
    elif resistance > 0.2:
        strength = "moderately"
    else:
        strength = "somewhat"

    explanation = (
        f"This entity ({from_node}) was flagged because it maintained a "
        f"relationship with {to_node} for {days} days without any transactions. "
        f"Based on natural decay patterns, the relationship strength should have "
        f"decreased to {expected:.2f}, but it remained at {observed:.2f}. "
        f"This {strength} improbable pattern (resistance: {resistance:.1%}) "
        f"suggests artificial preservation of the relationship, which is a "
        f"common indicator of shell company or dormant fraud infrastructure."
    )

    return explanation


def explain_contagion(receipt: Dict[str, Any]) -> str:
    """
    Generate plain-English explanation for cross-domain contagion.

    Example output:
    "This vendor was flagged because its sister company in the
    healthcare sector just collapsed."

    Args:
        receipt: contagion_receipt dict

    Returns:
        Plain-English explanation string
    """
    source_domain = receipt.get("source_domain", "another domain")
    target_domain = receipt.get("target_domain", "this domain")
    shell_entity = receipt.get("shell_entity", "a shared entity")
    pre_invoice = receipt.get("pre_invoice_flag", False)

    # Map domain names to human-readable sectors
    domain_names = {
        "defense": "defense contracting sector",
        "medicaid": "healthcare/Medicaid sector",
        "unknown": "another sector",
    }

    source_name = domain_names.get(source_domain, f"{source_domain} sector")
    target_name = domain_names.get(target_domain, f"{target_domain} sector")

    if pre_invoice:
        timing = (
            "This flag was raised BEFORE any local evidence of wrongdoing "
            "in the target sector, demonstrating early warning capability."
        )
    else:
        timing = ""

    explanation = (
        f"This entity was flagged because a related organization in the "
        f"{source_name} was just identified in a fraud investigation. "
        f"Both entities share connections through {shell_entity}, which "
        f"appears to be a shell company linking the two sectors. "
        f"When fraud collapses in one sector, the financial pressure "
        f"propagates to connected entities in other sectors. {timing}"
    )

    return explanation


def explain_zombie(receipt: Dict[str, Any]) -> str:
    """
    Generate plain-English explanation for zombie entity.

    Example output:
    "This entity appears dormant (no activity for 400 months) but
    maintains active relationships, suggesting artificial preservation."

    Args:
        receipt: zombie_receipt dict

    Returns:
        Plain-English explanation string
    """
    entity_id = receipt.get("entity_id", "This entity")
    days_dormant = receipt.get("days_dormant", ZOMBIE_DAYS)
    preserved_weight = receipt.get("preserved_weight", 0.0)
    linked_domains = receipt.get("linked_domains", [])

    # Convert days to months for readability
    months_dormant = days_dormant // 30

    # Describe linked domains
    if linked_domains and len(linked_domains) > 1:
        domains_text = (
            f"spans {len(linked_domains)} different sectors "
            f"({', '.join(linked_domains)})"
        )
    elif linked_domains:
        domains_text = f"is primarily in the {linked_domains[0]} sector"
    else:
        domains_text = "operates across unknown sectors"

    explanation = (
        f"Entity '{entity_id}' appears dormant—it has had no recorded "
        f"transactions for approximately {months_dormant} months "
        f"({days_dormant} days). However, it maintains active relationship "
        f"weights totaling {preserved_weight:.2f}, which should have decayed "
        f"naturally over this time. This 'zombie' pattern—preserved "
        f"connections without activity—is physically improbable for "
        f"legitimate business relationships. The entity {domains_text}, "
        f"suggesting it may serve as dormant fraud infrastructure awaiting "
        f"future activation."
    )

    return explanation


def format_insight(
    anomaly_type: str,
    technical_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine plain-English with technical summary for complete insight.

    Emits insight_receipt with both human-readable and technical data.

    Args:
        anomaly_type: Type of anomaly (temporal_anomaly, contagion, zombie)
        technical_data: Raw anomaly receipt data

    Returns:
        insight_receipt dict
    """
    # Generate plain-English based on type
    if anomaly_type == "temporal_anomaly":
        plain_english = explain_temporal_anomaly(technical_data)
        confidence = min(0.95, 0.7 + technical_data.get("resistance", 0) * 0.2)
    elif anomaly_type == "contagion":
        plain_english = explain_contagion(technical_data)
        confidence = 0.85 if technical_data.get("pre_invoice_flag") else 0.7
    elif anomaly_type == "zombie":
        plain_english = explain_zombie(technical_data)
        days = technical_data.get("days_dormant", 0)
        confidence = min(0.95, 0.6 + (days / ZOMBIE_DAYS) * 0.3)
    else:
        plain_english = (
            f"An anomaly of type '{anomaly_type}' was detected. "
            f"Please review the technical data for details."
        )
        confidence = 0.5

    # Create technical summary
    technical_summary = {
        "anomaly_type": anomaly_type,
        "raw_data": technical_data,
        "detection_method": "temporal_decay_physics",
        "physics_basis": "exponential_decay_resistance",
    }

    # Emit and return insight receipt
    return emit_insight_receipt(
        anomaly_type=anomaly_type,
        plain_english=plain_english,
        technical_summary=technical_summary,
        confidence=confidence,
    )


def generate_executive_summary(
    anomalies: list,
    domain: str = "cross-domain",
) -> str:
    """
    Generate executive summary for a collection of anomalies.

    Args:
        anomalies: List of anomaly receipts
        domain: Domain context

    Returns:
        Executive summary string
    """
    if not anomalies:
        return f"No temporal anomalies detected in {domain} analysis."

    # Count by type
    type_counts = {}
    for a in anomalies:
        atype = a.get("receipt_type", "unknown")
        type_counts[atype] = type_counts.get(atype, 0) + 1

    # Generate summary
    lines = [
        f"TEMPORAL PHYSICS ANALYSIS SUMMARY - {domain.upper()}",
        "=" * 50,
        f"Total anomalies detected: {len(anomalies)}",
        "",
        "Breakdown by type:",
    ]

    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        readable_type = atype.replace("_receipt", "").replace("_", " ").title()
        lines.append(f"  - {readable_type}: {count}")

    lines.extend([
        "",
        "Key findings:",
    ])

    # Highlight most severe
    max_resistance = 0
    worst_entity = None
    for a in anomalies:
        r = a.get("resistance", 0)
        if r > max_resistance:
            max_resistance = r
            worst_entity = a.get("from_node") or a.get("entity_id")

    if worst_entity:
        lines.append(
            f"  - Highest resistance: {worst_entity} ({max_resistance:.1%} above expected)"
        )

    # Check for contagion
    contagion_count = type_counts.get("contagion_receipt", 0)
    if contagion_count > 0:
        lines.append(
            f"  - Cross-domain contagion detected: {contagion_count} propagation paths"
        )

    # Check for zombies
    zombie_count = type_counts.get("zombie_receipt", 0)
    if zombie_count > 0:
        lines.append(
            f"  - Zombie entities identified: {zombie_count} dormant but preserved"
        )

    lines.extend([
        "",
        "RECOMMENDATION: Review flagged entities for potential fraud infrastructure.",
        "",
        f"[SIMULATION DISCLAIMER: {DISCLAIMER}]",
    ])

    return "\n".join(lines)


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("# Gov-OS Temporal Insight Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test temporal anomaly explanation
    temporal_receipt = {
        "from_node": "SHELL_HOLDINGS_LLC",
        "to_node": "DEFENSE_SUPPLIER_A",
        "days_since_last": 400,
        "resistance": 0.75,
        "expected_weight": 0.5,
        "observed_weight": 0.875,
    }
    explanation = explain_temporal_anomaly(temporal_receipt)
    assert "SHELL_HOLDINGS_LLC" in explanation
    assert "400" in explanation
    print("# Temporal explanation generated", file=sys.stderr)

    # Test contagion explanation
    contagion_receipt = {
        "source_domain": "medicaid",
        "target_domain": "defense",
        "shell_entity": "SHELL_HOLDINGS_LLC",
        "pre_invoice_flag": True,
    }
    explanation = explain_contagion(contagion_receipt)
    assert "healthcare" in explanation.lower()
    assert "sister" in explanation.lower() or "related" in explanation.lower()
    print("# Contagion explanation generated", file=sys.stderr)

    # Test zombie explanation
    zombie_receipt = {
        "entity_id": "DORMANT_CORP_123",
        "days_dormant": 500,
        "preserved_weight": 3.5,
        "linked_domains": ["defense", "medicaid"],
    }
    explanation = explain_zombie(zombie_receipt)
    assert "DORMANT_CORP_123" in explanation
    assert "zombie" in explanation.lower() or "dormant" in explanation.lower()
    print("# Zombie explanation generated", file=sys.stderr)

    # Test format_insight
    insight = format_insight("temporal_anomaly", temporal_receipt)
    assert insight["receipt_type"] == "insight_receipt"
    assert "plain_english" in insight
    print("# Insight receipt formatted", file=sys.stderr)

    print("# PASS: insight module self-test", file=sys.stderr)
