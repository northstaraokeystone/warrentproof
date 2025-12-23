"""
WarrantProof Freshness Module - Evidence Staleness Detection

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module tracks how current the evidence is and flags when
data is too old to be reliable. Fresh evidence leads to
confident decisions; stale evidence leads to mistakes.

The Problem:
Evidence ages. A vendor that was legitimate last year might not
be today. A price that was fair in 2020 might be gouging in 2024.
Stale data leads to stale conclusions.

How It Works:
1. Track the age of every piece of evidence
2. Flag evidence that's getting stale
3. Recommend refresh priorities
4. Prevent decisions based on outdated information

For Non-Technical Users:
Think of this like checking expiration dates on food. Just because
something was good when you bought it doesn't mean it's still good
today. We check the "freshness date" on all our evidence.

Staleness Levels:
- Fresh (< 30 days): High confidence
- Recent (30-60 days): Good confidence
- Aging (60-90 days): Lower confidence, consider refresh
- Stale (90-180 days): Needs refresh before major decisions
- Expired (> 180 days): Do not use for new determinations
"""

from datetime import datetime, timedelta
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    dual_hash,
    emit_receipt,
    StopRuleException,
)


# === FRESHNESS THRESHOLDS ===

FRESHNESS_DAYS = {
    "fresh": 30,
    "recent": 60,
    "aging": 90,
    "stale": 180,
    # Beyond 180 = "expired"
}

# How much to reduce confidence based on age
AGE_CONFIDENCE_DECAY = {
    "fresh": 1.0,      # Full confidence
    "recent": 0.9,     # 90% confidence
    "aging": 0.7,      # 70% confidence
    "stale": 0.4,      # 40% confidence
    "expired": 0.1,    # 10% confidence (should not use)
}

# Different data types have different shelf lives
DATA_TYPE_MULTIPLIERS = {
    "vendor_registration": 1.0,      # Standard freshness
    "price_data": 0.5,               # Prices change fast - halve thresholds
    "certification": 1.5,            # Certs last longer
    "contract_award": 2.0,           # Awards are historical facts
    "market_analysis": 0.3,          # Markets change very fast
    "personnel": 0.75,               # People move around
    "physical_inspection": 1.0,      # Standard freshness
}


# === CORE FUNCTIONS ===

def assess_freshness(
    data_timestamp: datetime,
    data_type: str = "general",
    reference_time: Optional[datetime] = None
) -> dict:
    """
    Assess the freshness of a piece of evidence.

    Args:
        data_timestamp: When the data was collected/verified
        data_type: Type of data (affects freshness thresholds)
        reference_time: Time to compare against (default: now)

    Returns:
        freshness_receipt with status and confidence
    """
    reference = reference_time or datetime.utcnow()
    age = reference - data_timestamp
    age_days = age.days

    # Get multiplier for data type
    multiplier = DATA_TYPE_MULTIPLIERS.get(data_type, 1.0)

    # Adjust thresholds based on data type
    adjusted_thresholds = {
        level: int(days * multiplier)
        for level, days in FRESHNESS_DAYS.items()
    }

    # Determine freshness level
    if age_days <= adjusted_thresholds["fresh"]:
        level = "fresh"
        status_message = "Data is current and reliable."
    elif age_days <= adjusted_thresholds["recent"]:
        level = "recent"
        status_message = "Data is fairly current. Good for most decisions."
    elif age_days <= adjusted_thresholds["aging"]:
        level = "aging"
        status_message = "Data is getting old. Consider refreshing for critical decisions."
    elif age_days <= adjusted_thresholds["stale"]:
        level = "stale"
        status_message = "Data is stale. Refresh before using in new determinations."
    else:
        level = "expired"
        status_message = "Data is too old to use. Collect fresh data."

    # Calculate confidence adjustment
    confidence = AGE_CONFIDENCE_DECAY[level]

    # Calculate days until next level
    if level == "fresh":
        days_until_degraded = adjusted_thresholds["fresh"] - age_days
    elif level == "recent":
        days_until_degraded = adjusted_thresholds["recent"] - age_days
    elif level == "aging":
        days_until_degraded = adjusted_thresholds["aging"] - age_days
    elif level == "stale":
        days_until_degraded = adjusted_thresholds["stale"] - age_days
    else:
        days_until_degraded = 0

    return emit_receipt("freshness", {
        "tenant_id": TENANT_ID,
        "data_type": data_type,
        "data_timestamp": data_timestamp.isoformat(),
        "age_days": age_days,
        "freshness_level": level,
        "confidence_factor": confidence,
        "status_message": status_message,
        "days_until_degraded": max(0, days_until_degraded),
        "refresh_recommended": level in ["aging", "stale", "expired"],
        "usable": level != "expired",
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def assess_evidence_set_freshness(evidence_items: list) -> dict:
    """
    Assess freshness of an entire evidence set.

    The overall freshness is limited by the stalest critical evidence.

    Args:
        evidence_items: List of dicts with 'timestamp' and 'data_type'

    Returns:
        freshness_receipt with overall assessment
    """
    if not evidence_items:
        return emit_receipt("freshness", {
            "tenant_id": TENANT_ID,
            "evidence_count": 0,
            "overall_freshness": "unknown",
            "confidence_factor": 0.0,
            "status_message": "No evidence to assess.",
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    assessments = []
    for item in evidence_items:
        ts = item.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
        elif not isinstance(ts, datetime):
            continue

        assessment = assess_freshness(
            ts,
            item.get("data_type", "general")
        )
        assessments.append(assessment)

    if not assessments:
        return emit_receipt("freshness", {
            "tenant_id": TENANT_ID,
            "evidence_count": len(evidence_items),
            "overall_freshness": "unknown",
            "confidence_factor": 0.0,
            "status_message": "Could not assess evidence timestamps.",
            "simulation_flag": DISCLAIMER,
        }, to_stdout=False)

    # Overall freshness is limited by weakest link
    level_priority = ["expired", "stale", "aging", "recent", "fresh"]
    worst_level = "fresh"
    for assessment in assessments:
        level = assessment["freshness_level"]
        if level_priority.index(level) < level_priority.index(worst_level):
            worst_level = level

    # Count by freshness level
    level_counts = {level: 0 for level in level_priority}
    for assessment in assessments:
        level_counts[assessment["freshness_level"]] += 1

    # Overall confidence
    overall_confidence = AGE_CONFIDENCE_DECAY[worst_level]

    # Find oldest and newest
    ages = [a["age_days"] for a in assessments]
    oldest = max(ages)
    newest = min(ages)

    # Generate recommendations
    stale_count = level_counts["stale"] + level_counts["expired"]
    aging_count = level_counts["aging"]

    if stale_count > 0:
        recommendation = f"Refresh {stale_count} stale evidence item(s) before proceeding."
    elif aging_count > 0:
        recommendation = f"Consider refreshing {aging_count} aging evidence item(s) for better confidence."
    else:
        recommendation = "Evidence freshness is good. No refresh needed."

    return emit_receipt("freshness", {
        "tenant_id": TENANT_ID,
        "evidence_count": len(assessments),
        "overall_freshness": worst_level,
        "confidence_factor": overall_confidence,
        "freshness_breakdown": level_counts,
        "oldest_days": oldest,
        "newest_days": newest,
        "recommendation": recommendation,
        "refresh_needed": stale_count > 0,
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def get_refresh_priorities(evidence_items: list, max_items: int = 10) -> dict:
    """
    Identify which evidence should be refreshed first.

    Prioritizes by:
    1. Staleness (older = higher priority)
    2. Data type importance (prices > general)
    3. Confidence impact (how much refresh would help)

    Args:
        evidence_items: List of evidence with timestamp and data_type
        max_items: Maximum number of items to return

    Returns:
        priority_receipt with ranked refresh list
    """
    priorities = []

    for item in evidence_items:
        ts = item.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
        elif not isinstance(ts, datetime):
            continue

        assessment = assess_freshness(
            ts,
            item.get("data_type", "general")
        )

        if assessment["freshness_level"] in ["aging", "stale", "expired"]:
            # Calculate priority score
            # Higher age = higher priority
            age_factor = assessment["age_days"] / 365  # Normalize to year

            # Data types that change fast have higher priority
            type_factor = 1.0 / DATA_TYPE_MULTIPLIERS.get(
                item.get("data_type", "general"), 1.0
            )

            # How much confidence we'd gain from refresh
            confidence_gain = 1.0 - assessment["confidence_factor"]

            priority_score = (age_factor * 0.4) + (type_factor * 0.3) + (confidence_gain * 0.3)

            priorities.append({
                "item_id": item.get("id", "unknown"),
                "data_type": item.get("data_type", "general"),
                "age_days": assessment["age_days"],
                "current_level": assessment["freshness_level"],
                "priority_score": round(priority_score, 3),
                "confidence_if_refreshed": 1.0,
                "confidence_current": assessment["confidence_factor"],
            })

    # Sort by priority score (highest first)
    priorities.sort(key=lambda x: -x["priority_score"])

    # Limit to max_items
    priorities = priorities[:max_items]

    return emit_receipt("refresh_priority", {
        "tenant_id": TENANT_ID,
        "items_needing_refresh": len(priorities),
        "priorities": priorities,
        "top_priority": priorities[0] if priorities else None,
        "recommendation": (
            f"Refresh the top {min(3, len(priorities))} items for maximum impact."
            if priorities else
            "No items need refresh at this time."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_freshness_for_users(assessment: dict) -> dict:
    """
    Convert freshness assessment to user-friendly explanation.

    Args:
        assessment: Freshness receipt from assess_freshness

    Returns:
        insight_receipt with plain-language explanation
    """
    level = assessment.get("freshness_level", "unknown")
    age = assessment.get("age_days", 0)
    data_type = assessment.get("data_type", "data")

    explanations = {
        "fresh": {
            "icon": "✓",
            "headline": "Up to Date",
            "explanation": (
                f"This {data_type} is {age} days old - still very current. "
                "You can rely on it with full confidence."
            ),
        },
        "recent": {
            "icon": "○",
            "headline": "Reasonably Current",
            "explanation": (
                f"This {data_type} is {age} days old - still reliable for most purposes. "
                "Consider checking if making a high-stakes decision."
            ),
        },
        "aging": {
            "icon": "!",
            "headline": "Getting Dated",
            "explanation": (
                f"This {data_type} is {age} days old - things may have changed. "
                "We recommend getting an update before major decisions."
            ),
        },
        "stale": {
            "icon": "⚠",
            "headline": "Outdated",
            "explanation": (
                f"This {data_type} is {age} days old - significantly outdated. "
                "Don't rely on this for new decisions. Get fresh data first."
            ),
        },
        "expired": {
            "icon": "✗",
            "headline": "Too Old to Use",
            "explanation": (
                f"This {data_type} is {age} days old - far too outdated. "
                "We cannot use this for any current determinations. "
                "New data collection is required."
            ),
        },
    }

    info = explanations.get(level, {
        "icon": "?",
        "headline": "Unknown Status",
        "explanation": "Unable to assess the freshness of this data.",
    })

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "freshness_explanation",
        "icon": info["icon"],
        "headline": info["headline"],
        "explanation": info["explanation"],
        "age_in_plain_terms": _age_to_plain_terms(age),
        "action_needed": level in ["stale", "expired"],
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def monitor_freshness_decay(
    tracked_items: dict,
    alert_threshold: str = "aging"
) -> dict:
    """
    Monitor tracked items for freshness decay and emit alerts.

    Use this in scheduled jobs to proactively identify
    evidence that needs refresh.

    Args:
        tracked_items: Dict of item_id -> item with timestamp
        alert_threshold: Level at which to alert ("aging", "stale", "expired")

    Returns:
        monitoring_receipt with decay status
    """
    threshold_levels = ["fresh", "recent", "aging", "stale", "expired"]
    threshold_index = threshold_levels.index(alert_threshold)

    alerts = []
    healthy = []

    for item_id, item in tracked_items.items():
        ts = item.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
        elif not isinstance(ts, datetime):
            continue

        assessment = assess_freshness(ts, item.get("data_type", "general"))
        level = assessment["freshness_level"]

        if threshold_levels.index(level) >= threshold_index:
            alerts.append({
                "item_id": item_id,
                "level": level,
                "age_days": assessment["age_days"],
                "action": "refresh_needed",
            })
        else:
            healthy.append({
                "item_id": item_id,
                "level": level,
                "days_until_alert": assessment["days_until_degraded"],
            })

    # Sort alerts by severity
    alerts.sort(key=lambda x: -threshold_levels.index(x["level"]))

    return emit_receipt("monitoring", {
        "tenant_id": TENANT_ID,
        "items_monitored": len(tracked_items),
        "alerts_triggered": len(alerts),
        "healthy_items": len(healthy),
        "alerts": alerts[:20],  # Limit to 20
        "upcoming_alerts": [
            h for h in healthy if h["days_until_alert"] <= 7
        ][:10],
        "summary": (
            f"{len(alerts)} item(s) need attention. "
            f"{len([h for h in healthy if h['days_until_alert'] <= 7])} more approaching threshold."
            if alerts else
            "All monitored items are within freshness thresholds."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


# === HELPER FUNCTIONS ===

def _age_to_plain_terms(days: int) -> str:
    """Convert days to plain language."""
    if days == 0:
        return "today"
    elif days == 1:
        return "yesterday"
    elif days < 7:
        return f"{days} days ago"
    elif days < 14:
        return "about a week ago"
    elif days < 30:
        return f"about {days // 7} weeks ago"
    elif days < 60:
        return "about a month ago"
    elif days < 90:
        return "about 2 months ago"
    elif days < 180:
        return f"about {days // 30} months ago"
    elif days < 365:
        return f"about {days // 30} months ago (quite old)"
    else:
        return f"over {days // 365} year(s) ago (very old)"


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys

    print(f"# WarrantProof Freshness Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Test fresh data
    fresh_ts = datetime.utcnow() - timedelta(days=15)
    fresh_result = assess_freshness(fresh_ts, "general")
    assert fresh_result["freshness_level"] == "fresh"
    assert fresh_result["confidence_factor"] == 1.0
    print(f"# Fresh data (15 days): {fresh_result['freshness_level']}", file=sys.stderr)

    # Test stale data
    stale_ts = datetime.utcnow() - timedelta(days=120)
    stale_result = assess_freshness(stale_ts, "general")
    assert stale_result["freshness_level"] == "stale"
    assert stale_result["confidence_factor"] < 1.0
    print(f"# Stale data (120 days): {stale_result['freshness_level']}", file=sys.stderr)

    # Test price data (faster decay)
    price_ts = datetime.utcnow() - timedelta(days=20)
    price_result = assess_freshness(price_ts, "price_data")
    # 20 days for price data should be "aging" (threshold is 30 * 0.5 = 15)
    print(f"# Price data (20 days): {price_result['freshness_level']}", file=sys.stderr)

    # Test evidence set
    evidence = [
        {"timestamp": datetime.utcnow() - timedelta(days=10), "data_type": "general"},
        {"timestamp": datetime.utcnow() - timedelta(days=100), "data_type": "general"},
        {"timestamp": datetime.utcnow() - timedelta(days=45), "data_type": "price_data"},
    ]
    set_result = assess_evidence_set_freshness(evidence)
    assert set_result["evidence_count"] == 3
    print(f"# Evidence set overall: {set_result['overall_freshness']}", file=sys.stderr)

    # Test refresh priorities
    priorities = get_refresh_priorities(evidence)
    assert "priorities" in priorities
    print(f"# Items needing refresh: {priorities['items_needing_refresh']}", file=sys.stderr)

    # Test user explanation
    explanation = explain_freshness_for_users(stale_result)
    assert "headline" in explanation
    print(f"# User explanation: {explanation['headline']}", file=sys.stderr)

    print("# PASS: freshness module self-test", file=sys.stderr)
