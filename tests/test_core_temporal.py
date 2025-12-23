"""
Tests for Gov-OS Temporal Physics Module (v5.1)

Tests:
- Exponential decay formula
- Resistance detection
- Zombie entity detection
- Decay integration with RAF

THIS IS A SIMULATION FOR ACADEMIC RESEARCH PURPOSES ONLY
"""

import math
import pytest
from datetime import datetime, timedelta

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# Import temporal functions
from src.core.temporal import (
    edge_weight_decay,
    resistance_to_decay,
    update_edge_with_decay,
    detect_zombies,
    identify_shell_entities,
    propagate_contagion,
    calculate_shared_entity_ratio,
    LAMBDA_NATURAL,
    RESISTANCE_THRESHOLD,
)
from src.core.constants import ZOMBIE_DAYS
from src.core.receipt import StopRuleException


class TestEdgeWeightDecay:
    """Tests for edge_weight_decay function."""

    def test_decay_formula_30_days(self):
        """Test 30-day decay: weight should decrease slightly."""
        result = edge_weight_decay(1.0, 30, LAMBDA_NATURAL)
        # With lambda=0.005/month, 30 days ≈ 1 month
        # W = 1.0 * e^(-0.005*1) ≈ 0.995
        assert 0.99 < result < 1.0
        assert result < 1.0  # Must decay

    def test_decay_zero_days(self):
        """Test zero days: no decay."""
        result = edge_weight_decay(1.0, 0, LAMBDA_NATURAL)
        assert result == 1.0

    def test_decay_large_days(self):
        """Test 3000 days (~8 years): significant decay."""
        result = edge_weight_decay(1.0, 3000, LAMBDA_NATURAL)
        # With lambda=0.005/month, 100 months
        # W = 1.0 * e^(-0.005*100) ≈ 0.606
        # But we divide lambda by 30 for daily, so:
        # W = 1.0 * e^(-0.005/30 * 3000) ≈ 0.606
        assert result < 0.7

    def test_decay_preserves_ratio(self):
        """Test that decay ratio is consistent across initial weights."""
        w1_initial = 1.0
        w2_initial = 2.0
        days = 100

        w1_decayed = edge_weight_decay(w1_initial, days, LAMBDA_NATURAL)
        w2_decayed = edge_weight_decay(w2_initial, days, LAMBDA_NATURAL)

        ratio1 = w1_decayed / w1_initial
        ratio2 = w2_decayed / w2_initial

        assert abs(ratio1 - ratio2) < 0.001

    def test_decay_half_life(self):
        """Test half-life calculation: ln(2)/lambda days."""
        # lambda = 0.005/month = 0.005/30 per day
        # half_life = ln(2) / (0.005/30) ≈ 4158 days ≈ 138.6 months
        half_life_days = int(math.log(2) / (LAMBDA_NATURAL / 30))
        result = edge_weight_decay(1.0, half_life_days, LAMBDA_NATURAL)
        assert abs(result - 0.5) < 0.01


class TestResistanceToDecay:
    """Tests for resistance_to_decay function."""

    def test_resistance_normal(self):
        """Equal weights = zero resistance."""
        result = resistance_to_decay(0.5, 0.5)
        assert result == 0.0

    def test_resistance_anomaly(self):
        """Double weight = 1.0 resistance."""
        result = resistance_to_decay(0.5, 1.0)
        assert result == 1.0

    def test_resistance_floor(self):
        """Lower than expected weight = zero (not negative)."""
        result = resistance_to_decay(0.5, 0.3)
        assert result == 0.0

    def test_resistance_high_anomaly(self):
        """Very high observed weight = high resistance."""
        result = resistance_to_decay(0.1, 0.5)
        assert result == 4.0  # (0.5/0.1) - 1 = 4

    def test_resistance_zero_expected(self):
        """Zero expected weight with observed = infinity."""
        result = resistance_to_decay(0.0, 0.5)
        assert result == float('inf')


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestUpdateEdgeWithDecay:
    """Tests for update_edge_with_decay function."""

    def test_update_emits_receipt_on_resistance(self):
        """High resistance should emit temporal_anomaly_receipt."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0, initial_weight=0.5, last_seen_date=datetime(2023, 1, 1))

        current_date = datetime(2024, 6, 1)  # 500+ days later
        last_seen = datetime(2023, 1, 1)

        resistance = update_edge_with_decay(
            G, "A", "B", current_date, last_seen, domain="test"
        )

        # Weight is 1.0 but expected is much lower after 500 days
        assert resistance > RESISTANCE_THRESHOLD

    def test_update_no_resistance_recent(self):
        """Recent edge should have zero resistance."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0, initial_weight=1.0)

        current_date = datetime(2024, 1, 2)
        last_seen = datetime(2024, 1, 1)  # Just yesterday

        resistance = update_edge_with_decay(
            G, "A", "B", current_date, last_seen, domain="test"
        )

        assert resistance < RESISTANCE_THRESHOLD


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestZombieDetection:
    """Tests for detect_zombies function."""

    def test_zombie_detection(self):
        """Dormant entity with preserved weight = zombie."""
        G = nx.DiGraph()
        # Add edge with old date but high weight
        old_date = datetime(2023, 1, 1)
        G.add_edge("ZOMBIE", "TARGET", weight=1.0, last_seen_date=old_date)

        current_date = datetime(2024, 6, 1)  # 500+ days later
        zombies = detect_zombies(G, current_date, zombie_days=365)

        assert len(zombies) >= 1
        assert any(z["entity_id"] == "ZOMBIE" for z in zombies)

    def test_no_zombie_active_entity(self):
        """Recently active entity = not zombie."""
        G = nx.DiGraph()
        recent_date = datetime(2024, 5, 1)
        G.add_edge("ACTIVE", "TARGET", weight=1.0, last_seen_date=recent_date)

        current_date = datetime(2024, 6, 1)  # Just 30 days later
        zombies = detect_zombies(G, current_date, zombie_days=365)

        assert not any(z["entity_id"] == "ACTIVE" for z in zombies)


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestSharedEntityRatio:
    """Tests for calculate_shared_entity_ratio function."""

    def test_shared_ratio_no_overlap(self):
        """No overlap = 0.0 ratio."""
        G1 = nx.DiGraph()
        G1.add_nodes_from(["A", "B", "C"])

        G2 = nx.DiGraph()
        G2.add_nodes_from(["D", "E", "F"])

        ratio = calculate_shared_entity_ratio(G1, G2)
        assert ratio == 0.0

    def test_shared_ratio_full_overlap(self):
        """Same nodes = 1.0 ratio."""
        G1 = nx.DiGraph()
        G1.add_nodes_from(["A", "B", "C"])

        G2 = nx.DiGraph()
        G2.add_nodes_from(["A", "B", "C"])

        ratio = calculate_shared_entity_ratio(G1, G2)
        assert ratio == 1.0

    def test_shared_ratio_partial_overlap(self):
        """Partial overlap = ratio between 0 and 1."""
        G1 = nx.DiGraph()
        G1.add_nodes_from(["A", "B", "C"])

        G2 = nx.DiGraph()
        G2.add_nodes_from(["B", "C", "D"])

        # Shared: B, C (2)
        # Total: A, B, C, D (4)
        # Ratio = 2/4 = 0.5
        ratio = calculate_shared_entity_ratio(G1, G2)
        assert ratio == 0.5


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX required")
class TestDecayIntegration:
    """Tests for RAF decay integration."""

    def test_decay_integration_with_raf(self):
        """Test that decay works with RAF graph building."""
        from src.raf import build_transaction_graph, add_transaction

        # Create transactions
        transactions = [
            {"source_duns": "A", "target_duns": "B", "amount_usd": 1000000, "date": datetime(2024, 1, 1)},
            {"source_duns": "B", "target_duns": "C", "amount_usd": 500000, "date": datetime(2024, 1, 1)},
        ]

        G = build_transaction_graph(transactions)

        # Add transaction with decay check
        resistance = add_transaction(
            G, "A", "B", 100000,
            tx_date=datetime(2024, 6, 1),  # 5 months later
            domain="test",
            apply_decay=True,
        )

        # Should have some resistance since weight was maintained
        assert G.has_edge("A", "B")
        assert G["A"]["B"]["weight"] > 0


class TestDecayConstants:
    """Tests for temporal constants."""

    def test_lambda_natural_value(self):
        """LAMBDA_NATURAL should be 0.005."""
        assert LAMBDA_NATURAL == 0.005

    def test_resistance_threshold_value(self):
        """RESISTANCE_THRESHOLD should be 0.1."""
        assert RESISTANCE_THRESHOLD == 0.1

    def test_zombie_days_value(self):
        """ZOMBIE_DAYS should be 365."""
        assert ZOMBIE_DAYS == 365
