import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import SupplyChainDataGenerator
from src.features.graph_features import build_entity_features, normalise_features, ENTITY_FEATURE_COLUMNS
from src.features.temporal_features import build_temporal_features
from src.features.financial_features import (
    build_lc_transaction_features, supply_chain_adjusted_pd, working_capital_velocity_index,
    trade_route_financial_stress_index, cash_conversion_cycle, LC_TRANSACTION_FEATURE_COLUMNS,
)


@pytest.fixture(scope="module")
def gen():
    return SupplyChainDataGenerator(seed=1)


@pytest.fixture(scope="module")
def graph(gen):
    return gen.generate_graph()


class TestSyntheticGenerator:
    def test_graph_meets_minimum_scale(self, graph):
        nodes, edges = graph
        assert len(nodes) >= 100
        assert (edges.edge_type == "transportation").sum() >= 500
        assert len(edges) >= 1000

    def test_graph_has_required_node_types(self, graph):
        nodes, _ = graph
        required = {"supplier", "manufacturer", "logistics_provider", "port",
                    "financial_institution", "customer"}
        assert required.issubset(set(nodes.node_type.unique()))

    def test_graph_has_required_edge_types(self, graph):
        _, edges = graph
        required = {"material_flow", "transportation", "financial", "ownership"}
        assert required.issubset(set(edges.edge_type.unique()))

    def test_default_rate_is_plausible(self, graph):
        nodes, _ = graph
        rate = nodes.default_12m.mean()
        assert 0.005 < rate < 0.30, f"default rate {rate} outside plausible range"

    def test_survival_fields_consistent(self, graph):
        nodes, _ = graph
        assert (nodes.survival_duration_days > 0).all()
        assert (nodes.survival_duration_days <= 730).all()
        assert set(nodes.survival_event.unique()).issubset({0, 1})

    def test_time_series_shape(self, gen):
        ts = gen.generate_time_series(n_days=100)
        assert ts.date.nunique() == 100
        assert (ts.throughput_teu > 0).all()
        assert (ts.freight_rate_usd_feu > 0).all()

    def test_shipments_labels_binary(self, gen):
        shp = gen.generate_shipments(n=200)
        assert set(shp.delayed.unique()).issubset({0, 1})
        assert set(shp.damaged.unique()).issubset({0, 1})
        assert (shp.delay_days >= 0).all()

    def test_lc_transactions_reproducible(self, graph):
        nodes, _ = graph
        gen1 = SupplyChainDataGenerator(seed=99)
        gen2 = SupplyChainDataGenerator(seed=99)
        lc1 = gen1.generate_trade_finance_transactions(n=100, nodes=nodes)
        lc2 = gen2.generate_trade_finance_transactions(n=100, nodes=nodes)
        pd.testing.assert_frame_equal(lc1, lc2)


class TestGraphFeatures:
    def test_entity_features_shape(self, graph):
        nodes, _ = graph
        feats = build_entity_features(nodes)
        assert len(feats) == len(nodes)
        assert set(ENTITY_FEATURE_COLUMNS).issubset(set(feats.columns))
        assert len(ENTITY_FEATURE_COLUMNS) == 21  # Section A5.1 worked example dimensionality

    def test_normalise_features_zero_mean(self, graph):
        nodes, _ = graph
        feats = build_entity_features(nodes)
        normed, stats = normalise_features(feats)
        for c in ENTITY_FEATURE_COLUMNS:
            assert abs(normed[c].mean()) < 1e-6
            assert abs(normed[c].std() - 1.0) < 1e-3


class TestTemporalFeatures:
    def test_temporal_features_no_lookahead_leakage(self, gen):
        ts = gen.generate_time_series(n_days=200)
        port0 = ts[ts.port == ts.port.iloc[0]]
        feats = build_temporal_features(port0, value_col="throughput_teu")
        # rolling mean at time t must only use data up to and including t
        manual_roll7 = port0.throughput_teu.rolling(7, min_periods=1).mean().values
        assert np.allclose(feats["roll_mean_7"].values, manual_roll7)

    def test_fourier_terms_bounded(self, gen):
        ts = gen.generate_time_series(n_days=100)
        port0 = ts[ts.port == ts.port.iloc[0]]
        feats = build_temporal_features(port0, value_col="throughput_teu")
        for col in [c for c in feats.columns if c.startswith("fourier_")]:
            assert feats[col].between(-1.0001, 1.0001).all()


class TestFinancialFeatures:
    def test_sc_pd_matches_worked_example(self):
        # Section A5.4: PD 2.5% -> SC-PD 3.33% for OTIF=85%, InvTurnover=4.8, 1 alt. supplier
        sc_pd = supply_chain_adjusted_pd(0.025, otif_actual=0.85, inv_turnover_actual=4.8,
                                          alternative_supplier_count=1)
        assert abs(float(sc_pd) - 0.0333) < 0.001

    def test_sc_pd_monotonic_in_otif(self):
        base = dict(traditional_pd=0.02, inv_turnover_actual=6.0, alternative_supplier_count=3)
        pd_high_otif = supply_chain_adjusted_pd(otif_actual=0.95, **base)
        pd_low_otif = supply_chain_adjusted_pd(otif_actual=0.70, **base)
        assert pd_low_otif > pd_high_otif

    def test_cash_conversion_cycle_worked_example(self):
        # Section A2.2 worked example: DIO=60.8, DSO=50.0, DPO=50.0 -> CCC=60.8
        ccc = cash_conversion_cycle(dio=60.8, dso=50.0, dpo=50.0)
        assert abs(ccc - 60.8) < 1e-6

    def test_wcvi_symmetry(self):
        wcvi_pos = working_capital_velocity_index(1.0, 1.0, -1.0)
        wcvi_neg = working_capital_velocity_index(-1.0, -1.0, 1.0)
        assert wcvi_pos == -wcvi_neg

    def test_trfsi_bounded_reasonable_range(self):
        trfsi = trade_route_financial_stress_index(0.5, 0.3, 0.2, 0.1)
        assert 0 <= trfsi <= 1

    def test_lc_transaction_features(self, graph):
        nodes, _ = graph
        gen2 = SupplyChainDataGenerator(seed=2)
        lc = gen2.generate_trade_finance_transactions(n=300, nodes=nodes)
        feats = build_lc_transaction_features(lc)
        assert set(LC_TRANSACTION_FEATURE_COLUMNS).issubset(set(feats.columns))
        assert feats["commodity_risk_num"].between(0, 2).all()
