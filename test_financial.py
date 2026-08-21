import numpy as np
import pytest

from src.data.synthetic_generator import SupplyChainDataGenerator
from src.financial.ccc_predictor import (
    simulate_working_capital_shocks, train_ccc_predictor, predict_ccc_change,
    covenant_breach_alert, FEATURE_COLS,
)
from src.financial.credit_risk_scorer import train_credit_risk_scorer, explain_prediction, N_SC_FEATURES
from src.financial.trade_finance_default import build_stacked_feature_matrix, train_trade_finance_default_model


@pytest.fixture(scope="module")
def small_graph():
    gen = SupplyChainDataGenerator(seed=6, n_suppliers=30, n_manufacturers=18, n_logistics=8,
                                    n_ports=6, n_financial_institutions=3, n_customers=14)
    return gen.generate_graph()


@pytest.fixture(scope="module")
def default_size_graph():
    # credit risk scorer needs enough default events in the test split to avoid a
    # degenerate single-class AUC; the full default-sized graph is still fast (XGBoost).
    gen = SupplyChainDataGenerator(seed=6)
    return gen.generate_graph()


class TestCCCPredictor:
    def test_simulated_shocks_shape(self):
        df = simulate_working_capital_shocks(n=500, seed=1)
        assert len(df) == 500
        assert set(FEATURE_COLS).issubset(set(df.columns))

    def test_ccc_change_equals_component_sum(self):
        df = simulate_working_capital_shocks(n=200, seed=1)
        recomputed = df.dio_change + df.dso_change - df.dpo_change
        assert np.allclose(df.ccc_change, recomputed)

    def test_train_ccc_predictor_runs(self):
        models, metrics = train_ccc_predictor(n=1500, verbose=False)
        assert "dio_change" in models and "dso_change" in models and "dpo_change" in models
        assert metrics["mae_days"] < 15  # sanity bound, not the (unstable) MAPE metric

    def test_meddevice_worked_example_direction(self):
        models, _ = train_ccc_predictor(n=1500, verbose=False)
        signals = dict(delta_otif=-0.12, delta_lead_time_std=4.2, delta_freight_cost_ratio=0.0,
                        delta_port_congestion=2.1, delta_inventory_turnover=0.0)
        result = predict_ccc_change(models, signals, horizon_days=90)
        # OTIF decline + lead-time variability increase should predict a *positive* CCC change
        assert result["ccc_change"] > 0
        assert result["dio_change"] > 0

    def test_covenant_breach_alert_logic(self):
        alert = covenant_breach_alert(current_ccc_days=72, predicted_ccc_change=26, covenant_threshold_days=90)
        assert alert["breach_predicted"] is True
        assert alert["predicted_ccc_days"] == 98.0
        no_breach = covenant_breach_alert(current_ccc_days=50, predicted_ccc_change=5, covenant_threshold_days=90)
        assert no_breach["breach_predicted"] is False


class TestCreditRiskScorer:
    def test_sc_feature_count_meets_minimum(self):
        assert N_SC_FEATURES >= 10  # Section D2.3.3 requirement

    def test_sc_enhanced_model_improves_over_financial_only(self, default_size_graph):
        nodes, _ = default_size_graph
        result = train_credit_risk_scorer(nodes, verbose=False)
        m = result["metrics"]
        assert m["auc_sc_enhanced_xgb"] >= m["auc_financial_only"]

    def test_shap_explanation_structure(self, default_size_graph):
        nodes, _ = default_size_graph
        result = train_credit_risk_scorer(nodes, verbose=False)
        sample_id = nodes[nodes.node_type == "supplier"].node_id.iloc[0]
        explanation = explain_prediction(result["model_sc_enhanced"], result["feats"], sample_id)
        assert 0 <= explanation["predicted_pd"] <= 1
        assert len(explanation["top_contributions"]) > 0


class TestTradeFinanceDefaultEnsemble:
    def test_stacked_feature_matrix_and_training_end_to_end(self, small_graph):
        from src.models.gnn import train_gnn
        from src.models.xgboost_model import train_xgboost_default_model
        from src.models.survival import fit_cox_ph
        from src.models.transformer import train_shipment_transformer

        nodes, edges = small_graph
        gen = SupplyChainDataGenerator(seed=6)
        lc_df = gen.generate_trade_finance_transactions(n=1500, nodes=nodes)
        shipments = gen.generate_shipments(n=400)

        _, _, gnn_embeddings, gnn_node_ids, _ = train_gnn(nodes, edges, epochs=15, verbose=False)
        xgb_model, _, _, _ = train_xgboost_default_model(nodes, n_trials=0, verbose=False)
        cox_model, _, _ = fit_cox_ph(nodes, penalizer=0.1)
        transformer_model, _ = train_shipment_transformer(shipments, epochs=5, verbose=False)

        X = build_stacked_feature_matrix(lc_df, nodes, xgb_model, cox_model, gnn_embeddings,
                                          gnn_node_ids, transformer_model)
        assert len(X) == len(lc_df)
        assert X.isna().sum().sum() == 0

        model, X2, split, metrics = train_trade_finance_default_model(
            lc_df, nodes, xgb_model, cox_model, gnn_embeddings, gnn_node_ids, transformer_model)
        assert 0 <= metrics["test_auc"] <= 1
        assert metrics["ece"] >= 0
