"""
End-to-end pipeline runner for LogisChain AI.

Generates data, trains every Level-0 base learner, the financial integration
models, and runs the LogisChain Lab simulation -- then writes a JSON results
summary to results/pipeline_results.json (used to generate the README's
"Results" section with real, reproduced numbers rather than hand-typed ones).

Usage: python -m demo.run_pipeline [--quick]
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main(quick: bool = False):
    t_start = time.time()
    results = {}

    from src.data.synthetic_generator import SupplyChainDataGenerator
    from src.models.gnn import train_gnn
    from src.models.tcn import train_tcn
    from src.models.transformer import train_shipment_transformer
    from src.models.xgboost_model import train_xgboost_default_model
    from src.models.survival import fit_cox_ph, train_deepsurv
    from src.financial.trade_finance_default import train_trade_finance_default_model
    from src.financial.ccc_predictor import train_ccc_predictor, predict_ccc_change, covenant_breach_alert
    from src.financial.credit_risk_scorer import train_credit_risk_scorer, explain_prediction
    from src.simulation.game_modes import TradeFinancePortfolioMode, SupplyChainFinancePricingMode

    print("=" * 70)
    print("LogisChain AI -- Full Pipeline Run")
    print("=" * 70)

    # ---- Phase 1: Data ----------------------------------------------------
    print("\n[Phase 1] Generating synthetic supply chain + trade finance data...")
    gen = SupplyChainDataGenerator(seed=42)
    nodes, edges = gen.generate_graph()
    ts = gen.generate_time_series(n_days=730)
    shipments = gen.generate_shipments(n=3000 if quick else 6000)
    lc_df = gen.generate_trade_finance_transactions(n=3000 if quick else 9000, nodes=nodes)
    results["data"] = dict(n_nodes=len(nodes), n_edges=len(edges),
                            n_transportation_links=int((edges.edge_type == "transportation").sum()),
                            n_shipments=len(shipments), n_lc_transactions=len(lc_df),
                            default_rate_nodes=float(nodes.default_12m.mean()),
                            default_rate_lc=float(lc_df.default.mean()))
    print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges "
          f"({results['data']['n_transportation_links']} transportation links)")

    # ---- Phase 2: Core ML models -------------------------------------------
    print("\n[Phase 2] Training core ML models...")
    print("  -> GNN (HetGAT)...")
    _, _, gnn_embeddings, gnn_node_ids, gnn_metrics = train_gnn(nodes, edges, verbose=False)
    results["gnn"] = gnn_metrics

    print("  -> TCN (port throughput + freight rate)...")
    _, _, tcn_throughput_metrics = train_tcn(ts, value_col="throughput_teu",
                                              exog_col="port_congestion_index",
                                              epochs=10 if quick else 15, max_entities=4 if quick else 6)
    _, _, tcn_freight_metrics = train_tcn(ts, value_col="freight_rate_usd_feu",
                                           exog_col="port_congestion_index",
                                           epochs=10 if quick else 15, max_entities=4 if quick else 6)
    results["tcn"] = dict(throughput=tcn_throughput_metrics, freight_rate=tcn_freight_metrics)

    print("  -> Transformer (shipment risk)...")
    transformer_model, transformer_metrics = train_shipment_transformer(
        shipments, epochs=15 if quick else 35, verbose=False)
    results["transformer"] = transformer_metrics

    print("  -> XGBoost (supplier default) + SHAP...")
    xgb_model, _, xgb_metrics, importances = train_xgboost_default_model(
        nodes, n_trials=0 if quick else 15, verbose=False)
    results["xgboost"] = {k: v for k, v in xgb_metrics.items() if k != "best_params"}
    results["xgboost"]["top_features"] = importances.head(10).round(4).to_dict()

    print("  -> Survival analysis (Cox PH + DeepSurv)...")
    cox_model, cox_fin_model, cox_metrics = fit_cox_ph(nodes)
    results["survival"] = dict(cox_ph=cox_metrics)
    if not quick:
        _, deepsurv_metrics = train_deepsurv(nodes, epochs=100, verbose=False, n_folds=5)
        results["survival"]["deepsurv"] = deepsurv_metrics

    # ---- Phase 3: Financial integration ------------------------------------
    print("\n[Phase 3] Financial model integration...")
    print("  -> Trade finance default (stacking ensemble)...")
    _, _, _, tf_metrics = train_trade_finance_default_model(
        lc_df, nodes, xgb_model, cox_model, gnn_embeddings, gnn_node_ids, transformer_model)
    results["trade_finance_default"] = tf_metrics

    print("  -> Working capital / CCC predictor...")
    ccc_models, ccc_metrics = train_ccc_predictor(n=4000 if quick else 6000, verbose=False)
    results["ccc_predictor"] = ccc_metrics
    meddevice_signals = dict(delta_otif=-0.12, delta_lead_time_std=4.2, delta_freight_cost_ratio=0.0,
                              delta_port_congestion=2.1, delta_inventory_turnover=0.0)
    meddevice_result = predict_ccc_change(ccc_models, meddevice_signals, horizon_days=90)
    meddevice_alert = covenant_breach_alert(72, meddevice_result["ccc_change"], 90)
    results["ccc_predictor"]["worked_example_meddevice"] = dict(**meddevice_result, **meddevice_alert)

    print("  -> Supply-chain-enhanced credit risk scorer (SC-PD)...")
    credit_result = train_credit_risk_scorer(nodes, verbose=False)
    results["credit_risk_scorer"] = credit_result["metrics"]

    # ---- Phase 4: LogisChain Lab simulation --------------------------------
    print("\n[Phase 4] Running LogisChain Lab simulation (both game modes, both policies)...")
    n_turns = 26 if quick else 52
    mode1 = TradeFinancePortfolioMode(nodes, edges, n_clients=45, seed=7)
    m1_sc = mode1.run("sc_aware", n_turns=n_turns)
    m1_passive = mode1.run("passive", n_turns=n_turns)
    mode2 = SupplyChainFinancePricingMode(nodes, edges, n_suppliers=60, seed=7)
    m2_sc = mode2.run("sc_aware", n_turns=n_turns)
    m2_passive = mode2.run("passive", n_turns=n_turns)

    results["simulation"] = dict(
        n_turns=n_turns,
        trade_finance_mode=dict(
            sc_aware=dict(realized_yield=m1_sc["realized_yield"], npl_ratio=m1_sc["npl_ratio"],
                          defaults=m1_sc["defaults"], score=m1_sc["score"]),
            passive=dict(realized_yield=m1_passive["realized_yield"], npl_ratio=m1_passive["npl_ratio"],
                        defaults=m1_passive["defaults"], score=m1_passive["score"]),
            n_scenarios_triggered=len(m1_sc["scenario_log"]),
        ),
        scf_pricing_mode=dict(
            sc_aware=dict(revenue_usd=m2_sc["total_discount_revenue_usd"],
                         default_rate=m2_sc["default_rate"], participation_rate=m2_sc["participation_rate"],
                         score=m2_sc["score"]),
            passive=dict(revenue_usd=m2_passive["total_discount_revenue_usd"],
                        default_rate=m2_passive["default_rate"], participation_rate=m2_passive["participation_rate"],
                        score=m2_passive["score"]),
        ),
    )

    elapsed = time.time() - t_start
    results["runtime_seconds"] = round(elapsed, 1)

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"Pipeline complete in {elapsed:.1f}s. Results written to {out_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Reduced scale for a fast smoke-test run")
    args = parser.parse_args()
    main(quick=args.quick)
