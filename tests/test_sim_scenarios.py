"""
WarrantProof Simulation Scenario Tests

⚠️ SIMULATION ONLY - NOT REAL DoD DATA - FOR RESEARCH ONLY ⚠️

These tests validate the 6 mandatory scenarios:
1. BASELINE
2. SHIPYARD_STRESS
3. CROSS_BRANCH_INTEGRATION
4. FRAUD_DISCOVERY
5. REAL_TIME_OVERSIGHT
6. GODEL
"""

import pytest
import time

from src.sim import (
    run_simulation,
    SimConfig,
    SimState,
    validate_scenario,
    export_results,
    SCENARIOS,
)
from src.core import DISCLAIMER


class TestBaseline:
    """Tests for BASELINE scenario."""

    def test_baseline_runs(self):
        """Test BASELINE scenario completes."""
        config = SimConfig(n_cycles=5, n_transactions_per_cycle=50, scenario="BASELINE")
        result = run_simulation(config)

        assert result.cycle == 5
        assert len(result.receipts) > 0

    def test_baseline_generates_receipts(self):
        """Test BASELINE generates receipts."""
        config = SimConfig(n_cycles=5, n_transactions_per_cycle=50, scenario="BASELINE")
        result = run_simulation(config)

        assert len(result.receipts) >= 50  # At least one cycle worth

    def test_baseline_compression(self):
        """Test BASELINE performs compression analysis."""
        config = SimConfig(n_cycles=5, n_transactions_per_cycle=50, scenario="BASELINE")
        result = run_simulation(config)

        assert len(result.compressions) > 0

    def test_baseline_detection(self):
        """Test BASELINE runs detection."""
        config = SimConfig(n_cycles=5, n_transactions_per_cycle=50, scenario="BASELINE")
        result = run_simulation(config)

        # Detection runs even if no anomalies found
        assert isinstance(result.detections, list)


class TestShipyardStress:
    """Tests for SHIPYARD_STRESS scenario."""

    def test_shipyard_stress_runs(self):
        """Test SHIPYARD_STRESS scenario completes."""
        config = SimConfig(scenario="SHIPYARD_STRESS")
        result = run_simulation(config)

        assert len(result.receipts) > 0

    def test_shipyard_stress_generates_ships(self):
        """Test SHIPYARD_STRESS generates ship-related receipts."""
        config = SimConfig(scenario="SHIPYARD_STRESS")
        result = run_simulation(config)

        # Should have receipts for ships
        ship_receipts = [r for r in result.receipts
                        if "Trump-class" in str(r.get("program", ""))]
        assert len(ship_receipts) > 0

    def test_shipyard_stress_detects_cascade(self):
        """Test SHIPYARD_STRESS detects cost cascade."""
        config = SimConfig(scenario="SHIPYARD_STRESS")
        result = run_simulation(config)

        cascade_detections = [d for d in result.detections
                             if d.get("anomaly_type") == "cost_cascade"]
        assert len(cascade_detections) > 0


class TestCrossBranchIntegration:
    """Tests for CROSS_BRANCH_INTEGRATION scenario."""

    def test_cross_branch_runs(self):
        """Test CROSS_BRANCH_INTEGRATION scenario completes."""
        config = SimConfig(scenario="CROSS_BRANCH_INTEGRATION")
        result = run_simulation(config)

        assert len(result.receipts) > 0

    def test_cross_branch_covers_all_branches(self):
        """Test CROSS_BRANCH_INTEGRATION covers all branches."""
        config = SimConfig(scenario="CROSS_BRANCH_INTEGRATION")
        result = run_simulation(config)

        branches_covered = set()
        for receipt in result.receipts:
            if "branch" in receipt:
                branches_covered.add(receipt["branch"])

        # Should cover multiple branches
        assert len(branches_covered) >= 3

    def test_cross_branch_chain_verified(self):
        """Test CROSS_BRANCH_INTEGRATION verifies chain."""
        config = SimConfig(scenario="CROSS_BRANCH_INTEGRATION")
        result = run_simulation(config)

        assert "chain_result" in result.scenario_results


class TestFraudDiscovery:
    """Tests for FRAUD_DISCOVERY scenario."""

    def test_fraud_discovery_runs(self):
        """Test FRAUD_DISCOVERY scenario completes."""
        config = SimConfig(scenario="FRAUD_DISCOVERY")
        result = run_simulation(config)

        assert len(result.receipts) > 0

    def test_fraud_discovery_injects_fraud(self):
        """Test FRAUD_DISCOVERY injects fraud patterns."""
        config = SimConfig(scenario="FRAUD_DISCOVERY")
        result = run_simulation(config)

        assert result.fraud_injected_count > 0

    def test_fraud_discovery_compression_difference(self):
        """Test FRAUD_DISCOVERY shows compression difference."""
        config = SimConfig(scenario="FRAUD_DISCOVERY")
        result = run_simulation(config)

        # Should have multiple compression results
        assert len(result.compressions) > 1


class TestRealTimeOversight:
    """Tests for REAL_TIME_OVERSIGHT scenario."""

    def test_real_time_runs(self):
        """Test REAL_TIME_OVERSIGHT scenario completes."""
        config = SimConfig(scenario="REAL_TIME_OVERSIGHT")
        result = run_simulation(config)

        assert len(result.receipts) > 0

    def test_real_time_latency_recorded(self):
        """Test REAL_TIME_OVERSIGHT records latencies."""
        config = SimConfig(scenario="REAL_TIME_OVERSIGHT")
        result = run_simulation(config)

        assert "latencies" in result.scenario_results


class TestGodel:
    """Tests for GODEL (edge case) scenario."""

    def test_godel_runs(self):
        """Test GODEL scenario completes without crash."""
        config = SimConfig(scenario="GODEL")
        result = run_simulation(config)

        # Should complete without crashing
        assert isinstance(result, SimState)

    def test_godel_handles_edge_cases(self):
        """Test GODEL handles edge cases."""
        config = SimConfig(scenario="GODEL")
        result = run_simulation(config)

        assert "edge_cases" in result.scenario_results
        assert result.scenario_results.get("all_handled", False) == True

    def test_godel_stoprule_triggered(self):
        """Test GODEL triggers stoprule on hash mismatch."""
        config = SimConfig(scenario="GODEL")
        result = run_simulation(config)

        # Check that hash mismatch case was handled
        edge_cases = result.scenario_results.get("edge_cases", [])
        hash_case = next((c for c in edge_cases if c["case"] == "hash_mismatch"), None)

        assert hash_case is not None
        assert hash_case.get("stoprule_triggered", False) == True


class TestValidateScenario:
    """Tests for validate_scenario function."""

    def test_validate_returns_receipt(self):
        """Test validate_scenario returns simulation_receipt."""
        config = SimConfig(n_cycles=5, scenario="BASELINE")
        state = run_simulation(config)

        validation = validate_scenario(state, "BASELINE")

        assert validation["receipt_type"] == "simulation"
        assert "passed" in validation
        assert "results" in validation


class TestExportResults:
    """Tests for export_results function."""

    def test_export_includes_disclaimer(self):
        """Test export includes simulation disclaimer."""
        config = SimConfig(n_cycles=5, scenario="BASELINE")
        state = run_simulation(config)

        export = export_results(state)

        assert "simulation_disclaimer" in export
        assert export["simulation_disclaimer"] == DISCLAIMER

    def test_export_includes_citations(self):
        """Test export includes citations."""
        config = SimConfig(n_cycles=5, scenario="BASELINE")
        state = run_simulation(config)

        export = export_results(state)

        assert "all_citations" in export
        assert len(export["all_citations"]) > 0

    def test_export_includes_metrics(self):
        """Test export includes metrics."""
        config = SimConfig(n_cycles=5, scenario="BASELINE")
        state = run_simulation(config)

        export = export_results(state)

        assert "metrics" in export
        assert "summary" in export


class TestScenarioConfiguration:
    """Tests for scenario configurations."""

    def test_all_scenarios_defined(self):
        """Test all 6 mandatory scenarios are defined."""
        expected_scenarios = [
            "BASELINE",
            "SHIPYARD_STRESS",
            "CROSS_BRANCH_INTEGRATION",
            "FRAUD_DISCOVERY",
            "REAL_TIME_OVERSIGHT",
            "GODEL",
        ]

        for scenario in expected_scenarios:
            assert scenario in SCENARIOS, f"Missing scenario: {scenario}"

    def test_scenarios_have_pass_criteria(self):
        """Test all scenarios have pass criteria."""
        for scenario, config in SCENARIOS.items():
            assert "pass_criteria" in config, f"{scenario} missing pass_criteria"
            assert len(config["pass_criteria"]) > 0
