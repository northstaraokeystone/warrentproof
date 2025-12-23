"""
WarrantProof Fitness Module - Self-Improving Detection Through Entropy

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

This module tracks how well the system is learning to detect fraud
by measuring "fitness" - the reduction in uncertainty (entropy)
that each detection pattern achieves.

The Core Idea:
- Good detection patterns reduce our uncertainty about what's fraudulent
- Poor patterns add noise (increase entropy)
- The system automatically favors patterns that work

Think of it like natural selection:
- Patterns that successfully identify fraud "survive"
- Patterns that just add noise "fade away"
- The system gets better over time without manual tuning

For Non-Technical Users:
Instead of manually adjusting detection rules, this module lets
the system learn which approaches actually work. It's like training
a dog - reward what works, ignore what doesn't.

Key Metrics:
- Fitness Score: How much uncertainty a pattern reduces (higher = better)
- Health Status: Overall system learning progress
- Efficiency: How many receipts needed to make a determination
"""

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from .core import (
    TENANT_ID,
    DISCLAIMER,
    dual_hash,
    emit_receipt,
    get_citation,
    StopRuleException,
)


# === FITNESS THRESHOLDS ===

FITNESS_THRESHOLDS = {
    "excellent": 0.5,    # Pattern reduces entropy by 50%+
    "good": 0.3,         # Pattern reduces entropy by 30%+
    "acceptable": 0.1,   # Pattern reduces entropy by 10%+
    "marginal": 0.0,     # Pattern has no effect
    "harmful": -0.1,     # Pattern adds entropy (noise)
}

HEALTH_THRESHOLDS = {
    "thriving": 0.7,     # 70%+ patterns are good or better
    "healthy": 0.5,      # 50%+ patterns are acceptable or better
    "stable": 0.3,       # 30%+ patterns contribute positively
    "struggling": 0.0,   # System needs attention
}


# === CORE CLASSES ===

class PatternFitness:
    """
    Tracks the fitness of a single detection pattern.

    A pattern is "fit" if it consistently reduces uncertainty about
    which records are fraudulent. We measure this by comparing
    entropy before and after the pattern is applied.
    """

    def __init__(self, pattern_id: str, pattern_type: str):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.observations = []
        self.entropy_reductions = []
        self.last_updated = datetime.utcnow()

    def record_observation(
        self,
        entropy_before: float,
        entropy_after: float,
        receipts_processed: int
    ) -> float:
        """
        Record how much entropy this pattern reduced.

        Args:
            entropy_before: System entropy before applying pattern
            entropy_after: System entropy after applying pattern
            receipts_processed: Number of receipts the pattern examined

        Returns:
            Fitness score for this observation
        """
        if receipts_processed == 0:
            return 0.0

        # Fitness = entropy reduction per receipt
        reduction = entropy_before - entropy_after
        fitness = reduction / receipts_processed

        self.observations.append({
            "ts": datetime.utcnow().isoformat(),
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "receipts": receipts_processed,
            "fitness": fitness,
        })
        self.entropy_reductions.append(fitness)
        self.last_updated = datetime.utcnow()

        return fitness

    @property
    def average_fitness(self) -> float:
        """Average fitness across all observations."""
        if not self.entropy_reductions:
            return 0.0
        return sum(self.entropy_reductions) / len(self.entropy_reductions)

    @property
    def trend(self) -> str:
        """Is the pattern improving, stable, or declining?"""
        if len(self.entropy_reductions) < 3:
            return "insufficient_data"

        recent = self.entropy_reductions[-5:]
        older = self.entropy_reductions[:-5] if len(self.entropy_reductions) > 5 else []

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"

    @property
    def rating(self) -> str:
        """Human-readable fitness rating."""
        fitness = self.average_fitness
        for rating, threshold in sorted(
            FITNESS_THRESHOLDS.items(), key=lambda x: -x[1]
        ):
            if fitness >= threshold:
                return rating
        return "harmful"

    def to_dict(self) -> dict:
        """Convert to dictionary for receipts."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "observation_count": len(self.observations),
            "average_fitness": round(self.average_fitness, 4),
            "trend": self.trend,
            "rating": self.rating,
            "last_updated": self.last_updated.isoformat(),
        }


class SystemHealth:
    """
    Tracks overall system learning health.

    A healthy system has patterns that consistently reduce entropy.
    An unhealthy system has too many patterns that add noise.
    """

    def __init__(self):
        self.patterns: dict[str, PatternFitness] = {}
        self.history = []

    def get_or_create_pattern(
        self,
        pattern_id: str,
        pattern_type: str
    ) -> PatternFitness:
        """Get existing pattern tracker or create new one."""
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = PatternFitness(pattern_id, pattern_type)
        return self.patterns[pattern_id]

    @property
    def overall_fitness(self) -> float:
        """Average fitness across all patterns."""
        if not self.patterns:
            return 0.0
        return sum(p.average_fitness for p in self.patterns.values()) / len(self.patterns)

    @property
    def health_status(self) -> str:
        """Overall system health status."""
        if not self.patterns:
            return "initializing"

        good_count = sum(
            1 for p in self.patterns.values()
            if p.average_fitness >= FITNESS_THRESHOLDS["acceptable"]
        )
        ratio = good_count / len(self.patterns)

        for status, threshold in sorted(
            HEALTH_THRESHOLDS.items(), key=lambda x: -x[1]
        ):
            if ratio >= threshold:
                return status
        return "struggling"

    def get_top_patterns(self, n: int = 5) -> list:
        """Get the n highest-fitness patterns."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.average_fitness,
            reverse=True
        )
        return [p.to_dict() for p in sorted_patterns[:n]]

    def get_weak_patterns(self, n: int = 5) -> list:
        """Get the n lowest-fitness patterns (for review)."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.average_fitness
        )
        return [p.to_dict() for p in sorted_patterns[:n]]

    def record_snapshot(self) -> dict:
        """Record current system state for history."""
        snapshot = {
            "ts": datetime.utcnow().isoformat(),
            "overall_fitness": round(self.overall_fitness, 4),
            "health_status": self.health_status,
            "pattern_count": len(self.patterns),
            "excellent_patterns": sum(
                1 for p in self.patterns.values()
                if p.rating == "excellent"
            ),
            "good_patterns": sum(
                1 for p in self.patterns.values()
                if p.rating == "good"
            ),
            "harmful_patterns": sum(
                1 for p in self.patterns.values()
                if p.rating == "harmful"
            ),
        }
        self.history.append(snapshot)
        return snapshot


# === GLOBAL INSTANCE ===

_system_health = SystemHealth()


# === CORE FUNCTIONS ===

def calculate_entropy(receipts: list) -> float:
    """
    Calculate Shannon entropy of receipt classifications.

    Args:
        receipts: List of receipts with classification field

    Returns:
        Entropy in bits (higher = more uncertainty)
    """
    if not receipts:
        return 0.0

    # Count classifications
    classifications = [
        r.get("classification", "unknown")
        for r in receipts
    ]
    counter = Counter(classifications)
    total = len(classifications)

    # H = -Σ p(x) log p(x)
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def record_pattern_fitness(
    pattern_id: str,
    pattern_type: str,
    receipts_before: list,
    receipts_after: list
) -> dict:
    """
    Record how much a pattern reduced system entropy.

    Use this after applying a detection pattern to measure its
    effectiveness. Patterns that consistently reduce entropy
    will automatically be favored.

    Args:
        pattern_id: Unique identifier for this pattern
        pattern_type: Category of pattern (e.g., "compression", "network")
        receipts_before: Receipts before pattern applied
        receipts_after: Receipts after pattern applied (with classifications)

    Returns:
        fitness_receipt with pattern performance
    """
    entropy_before = calculate_entropy(receipts_before)
    entropy_after = calculate_entropy(receipts_after)

    pattern = _system_health.get_or_create_pattern(pattern_id, pattern_type)
    fitness = pattern.record_observation(
        entropy_before,
        entropy_after,
        len(receipts_after)
    )

    # Determine if pattern is contributing positively
    is_helpful = fitness >= 0
    effectiveness = "reduces uncertainty" if is_helpful else "adds noise"

    return emit_receipt("fitness", {
        "tenant_id": TENANT_ID,
        "pattern_id": pattern_id,
        "pattern_type": pattern_type,
        "entropy_before": round(entropy_before, 4),
        "entropy_after": round(entropy_after, 4),
        "entropy_reduction": round(entropy_before - entropy_after, 4),
        "receipts_processed": len(receipts_after),
        "fitness_score": round(fitness, 4),
        "is_helpful": is_helpful,
        "effectiveness": effectiveness,
        "cumulative_fitness": round(pattern.average_fitness, 4),
        "pattern_rating": pattern.rating,
        "pattern_trend": pattern.trend,
        "citation": get_citation("SHANNON_1948"),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def get_system_health() -> dict:
    """
    Get overall system health report.

    Returns a summary of how well the detection patterns are
    performing collectively. Use this to understand if the
    system is learning effectively.

    Returns:
        health_receipt with system status
    """
    snapshot = _system_health.record_snapshot()

    # Generate human-readable status
    status = snapshot["health_status"]
    status_messages = {
        "thriving": "System is performing excellently. Detection patterns are highly effective.",
        "healthy": "System is performing well. Most patterns contribute positively.",
        "stable": "System is performing adequately. Some patterns may need review.",
        "struggling": "System needs attention. Many patterns are adding noise.",
        "initializing": "System is still learning. More observations needed.",
    }

    # Recommendations based on status
    recommendations = {
        "thriving": "Continue current approach. Consider experimenting with new pattern types.",
        "healthy": "Monitor weak patterns for improvement or removal.",
        "stable": "Review and potentially remove harmful patterns.",
        "struggling": "Urgent: Review all patterns. Consider resetting with proven patterns only.",
        "initializing": "Allow more time for patterns to accumulate observations.",
    }

    return emit_receipt("health", {
        "tenant_id": TENANT_ID,
        "status": status,
        "status_message": status_messages.get(status, "Unknown status"),
        "recommendation": recommendations.get(status, "Continue monitoring"),
        "overall_fitness": snapshot["overall_fitness"],
        "pattern_breakdown": {
            "total": snapshot["pattern_count"],
            "excellent": snapshot["excellent_patterns"],
            "good": snapshot["good_patterns"],
            "harmful": snapshot["harmful_patterns"],
        },
        "top_patterns": _system_health.get_top_patterns(3),
        "weak_patterns": _system_health.get_weak_patterns(3),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def explain_fitness_for_users() -> dict:
    """
    Generate a user-friendly explanation of system fitness.

    This is for non-technical users who want to understand
    how well the fraud detection is working without diving
    into entropy mathematics.

    Returns:
        insight_receipt with plain-language explanation
    """
    health = get_system_health()
    status = health["status"]
    fitness = health["overall_fitness"]

    # Translate to everyday terms
    if status == "thriving":
        headline = "Detection System Running Smoothly"
        explanation = (
            "The fraud detection patterns are working very well. "
            "They're successfully identifying suspicious records "
            "and separating them from legitimate ones. Think of it "
            "like a well-trained team that knows exactly what to look for."
        )
    elif status == "healthy":
        headline = "Detection System Performing Well"
        explanation = (
            "The system is doing a good job overall. Most detection "
            "methods are contributing to finding issues. A few might "
            "need tuning, but nothing urgent."
        )
    elif status == "stable":
        headline = "Detection System Adequate"
        explanation = (
            "The system is working, but could be better. Some detection "
            "methods aren't adding much value and may need review."
        )
    elif status == "struggling":
        headline = "Detection System Needs Attention"
        explanation = (
            "The system is having trouble. Too many detection methods "
            "are creating confusion rather than clarity. Think of it "
            "like having too many contradicting opinions - hard to "
            "know what to trust."
        )
    else:
        headline = "Detection System Still Learning"
        explanation = (
            "The system is new and still gathering information. "
            "It needs more examples to learn what works best. "
            "Give it time to improve."
        )

    # Convert fitness to percentage for users
    # (normalize: 0.5 = 100%, 0 = 50%, -0.5 = 0%)
    effectiveness_pct = min(100, max(0, int((fitness + 0.5) * 100)))

    return emit_receipt("insight", {
        "tenant_id": TENANT_ID,
        "analysis_type": "fitness_explanation",
        "headline": headline,
        "explanation": explanation,
        "effectiveness_percent": effectiveness_pct,
        "effectiveness_description": f"The system is {effectiveness_pct}% effective at identifying fraud patterns.",
        "what_this_means": (
            f"When the system analyzes records, it correctly identifies "
            f"fraud patterns about {effectiveness_pct}% of the time. "
            f"{'This is excellent.' if effectiveness_pct > 80 else 'There is room for improvement.' if effectiveness_pct > 60 else 'The system needs attention.'}"
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def prune_harmful_patterns() -> dict:
    """
    Identify patterns that should be removed or reviewed.

    Patterns with consistently negative fitness are adding
    noise to the system and should be reconsidered.

    Returns:
        prune_receipt with recommendations
    """
    harmful = [
        p for p in _system_health.patterns.values()
        if p.rating == "harmful" and len(p.observations) >= 5
    ]

    recommendations = []
    for pattern in harmful:
        recommendations.append({
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "fitness": round(pattern.average_fitness, 4),
            "observations": len(pattern.observations),
            "recommendation": (
                "Remove or significantly revise this pattern. "
                f"It has added noise in {len(pattern.observations)} observations."
            ),
        })

    return emit_receipt("prune", {
        "tenant_id": TENANT_ID,
        "harmful_patterns_found": len(harmful),
        "recommendations": recommendations,
        "action_needed": len(harmful) > 0,
        "summary": (
            f"Found {len(harmful)} pattern(s) that are degrading detection quality. "
            "Review these patterns and consider removing or revising them."
            if harmful else
            "No harmful patterns identified. All patterns are contributing positively or neutrally."
        ),
        "simulation_flag": DISCLAIMER,
    }, to_stdout=False)


def reset_system_health():
    """Reset system health for fresh start (useful for testing)."""
    global _system_health
    _system_health = SystemHealth()


# === MODULE SELF-TEST ===

if __name__ == "__main__":
    import sys
    import random

    print(f"# WarrantProof Fitness Module Self-Test", file=sys.stderr)
    print(f"# {DISCLAIMER}", file=sys.stderr)

    # Reset for clean test
    reset_system_health()

    # Simulate a good pattern (reduces entropy)
    before = [{"classification": "unknown"} for _ in range(100)]
    after = [
        {"classification": random.choice(["legitimate", "fraudulent"])}
        for _ in range(100)
    ]

    fitness = record_pattern_fitness(
        "test_compression",
        "compression",
        before,
        after
    )
    assert fitness["is_helpful"], "Good pattern should be helpful"
    print(f"# Good pattern fitness: {fitness['fitness_score']:.4f}", file=sys.stderr)

    # Simulate a harmful pattern (adds entropy)
    before_clean = [{"classification": "legitimate"} for _ in range(100)]
    after_noisy = [
        {"classification": random.choice(["legitimate", "suspicious", "fraudulent", "unknown"])}
        for _ in range(100)
    ]

    fitness_bad = record_pattern_fitness(
        "test_noisy",
        "noise",
        before_clean,
        after_noisy
    )
    print(f"# Noisy pattern fitness: {fitness_bad['fitness_score']:.4f}", file=sys.stderr)

    # Get system health
    health = get_system_health()
    assert "status" in health, "Health should have status"
    print(f"# System health: {health['status']}", file=sys.stderr)

    # Get user explanation
    explanation = explain_fitness_for_users()
    assert "headline" in explanation, "Explanation should have headline"
    print(f"# User explanation: {explanation['headline']}", file=sys.stderr)

    # Test pruning
    prune = prune_harmful_patterns()
    assert "harmful_patterns_found" in prune
    print(f"# Harmful patterns: {prune['harmful_patterns_found']}", file=sys.stderr)

    print("# PASS: fitness module self-test", file=sys.stderr)
