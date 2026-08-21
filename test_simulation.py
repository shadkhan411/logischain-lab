import pytest

from src.data.synthetic_generator import SupplyChainDataGenerator
from src.simulation.scenarios import SCENARIO_CATALOGUE, trigger_scenario, financial_impact_types
from src.simulation.engine import LogisChainLabEngine
from src.simulation.scoring import (
    ScoreBreakdown, score_financial_performance, score_sc_intelligence_use,
    score_learning_progression, CERTIFICATION_TIERS,
)
from src.simulation.game_modes import TradeFinancePortfolioMode, SupplyChainFinancePricingMode
import numpy as np


@pytest.fixture(scope="module")
def sim_graph():
    gen = SupplyChainDataGenerator(seed=5)
    return gen.generate_graph()


class TestScenarioCatalogue:
    def test_minimum_scenario_count(self):
        assert len(SCENARIO_CATALOGUE) >= 5  # Section D2.4.3 minimum

    def test_all_required_categories_present(self):
        required = {"port_congestion", "carrier_bankruptcy", "geopolitical_route_closure",
                    "supplier_quality_failure", "demand_whiplash"}
        assert required.issubset(set(SCENARIO_CATALOGUE.keys()))

    def test_every_scenario_has_three_plus_financial_impacts(self):
        for key, effect in SCENARIO_CATALOGUE.items():
            impacts = financial_impact_types(effect)
            assert len(impacts) >= 3, f"{key} only has {len(impacts)} impact types"

    def test_trigger_scenario_produces_valid_effect(self):
        rng = np.random.default_rng(0)
        ports = ["P1", "P2", "P3", "P4"]
        sc = trigger_scenario("port_congestion", current_turn=5, all_ports=ports, rng=rng)
        assert sc.duration_turns >= 1
        assert sc.start_turn == 5
        assert set(sc.affected_ports).issubset(set(ports))


class TestEngine:
    def test_engine_enforces_minimum_scale(self, sim_graph):
        nodes, edges = sim_graph
        engine = LogisChainLabEngine(nodes, edges, seed=1)
        assert len(engine.nodes) >= 100

    def test_engine_steps_advance_turn(self, sim_graph):
        nodes, edges = sim_graph
        engine = LogisChainLabEngine(nodes, edges, seed=1)
        for _ in range(10):
            engine.step()
        assert engine.turn == 10

    def test_intelligence_signals_well_formed(self, sim_graph):
        nodes, edges = sim_graph
        engine = LogisChainLabEngine(nodes, edges, seed=1)
        engine.step()
        sample = nodes[nodes.node_type == "supplier"].node_id.iloc[0]
        sig = engine.intelligence_signals(sample)
        assert 0 <= sig["sc_pd"] <= 1
        assert 0 <= sig["otif_now"] <= 1
        assert sig["port_congestion"] >= 0

    def test_reproducible_with_same_seed(self, sim_graph):
        nodes, edges = sim_graph
        e1 = LogisChainLabEngine(nodes, edges, seed=42)
        e2 = LogisChainLabEngine(nodes, edges, seed=42)
        for _ in range(15):
            e1.step()
            e2.step()
        assert e1.scenario_log == e2.scenario_log


class TestScoring:
    def test_score_breakdown_total_capped(self):
        s = ScoreBreakdown(300, 250, 200, 100, 150)
        assert s.total == 1000

    def test_certification_tiers_cover_full_range(self):
        assert CERTIFICATION_TIERS[0][0] == 0
        assert CERTIFICATION_TIERS[-1][1] == 1000

    def test_certification_lookup(self):
        s = ScoreBreakdown(240, 200, 150, 80, 121)  # totals to 791, matches worked example
        level, badge, equiv = s.certification()
        assert level == "Expert"
        assert badge == "Platinum"

    def test_financial_performance_score_bounded(self):
        score = score_financial_performance(0.02, 0.015, 0.01, 0.03, 0.02)
        assert 0 <= score <= 300

    def test_sc_intelligence_use_scales_linearly(self):
        assert score_sc_intelligence_use(1.0) == 200
        assert score_sc_intelligence_use(0.0) == 0
        assert score_sc_intelligence_use(0.5) == 100

    def test_learning_progression_rewards_upward_trend(self):
        improving = score_learning_progression([580, 620, 680, 740])
        flat = score_learning_progression([600, 600, 600, 600])
        assert improving > flat


class TestGameModes:
    def test_trade_finance_mode_runs_both_policies(self, sim_graph):
        nodes, edges = sim_graph
        mode = TradeFinancePortfolioMode(nodes, edges, n_clients=15, seed=11)
        res_sc = mode.run("sc_aware", n_turns=16)
        res_passive = mode.run("passive", n_turns=16)
        assert res_sc["score"]["total"] >= 0
        assert res_passive["score"]["total"] >= 0
        assert "certification_level" in res_sc["score"]

    def test_sc_aware_uses_signals_for_every_decision(self, sim_graph):
        nodes, edges = sim_graph
        mode = TradeFinancePortfolioMode(nodes, edges, n_clients=10, seed=11)
        res = mode.run("sc_aware", n_turns=8)
        assert res["score"]["sc_intelligence_use"] == 200  # 100% of decisions used signals

    def test_passive_uses_no_signals(self, sim_graph):
        nodes, edges = sim_graph
        mode = TradeFinancePortfolioMode(nodes, edges, n_clients=10, seed=11)
        res = mode.run("passive", n_turns=8)
        assert res["score"]["sc_intelligence_use"] == 0

    def test_scf_pricing_mode_runs(self, sim_graph):
        nodes, edges = sim_graph
        mode = SupplyChainFinancePricingMode(nodes, edges, n_suppliers=20, seed=11)
        res = mode.run("sc_aware", n_turns=16)
        assert res["financed_volume_usd"] >= 0
        assert 0 <= res["participation_rate"] <= 1
