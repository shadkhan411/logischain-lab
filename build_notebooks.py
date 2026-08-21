"""Generates notebooks/*.ipynb. Cells mirror the already-tested code in src/
and demo/run_pipeline.py exactly (not re-derived), so opening and running
them end-to-end reproduces the same numbers as `demo/run_pipeline.py`."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------- 01_eda ----
nb1 = make_nb([
    md("# 01 — Exploratory Data Analysis\nLogisChain AI: supply chain network, "
       "temporal, and correlation analysis (Deliverable D2.1.2)."),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\n"
         "import pandas as pd, numpy as np\n"
         "from src.data.synthetic_generator import SupplyChainDataGenerator\n"
         "gen = SupplyChainDataGenerator(seed=42)\n"
         "nodes, edges = gen.generate_graph()\n"
         "print(nodes.shape, edges.shape)\nnodes.head()"),
    md("## (a) Supply chain network visualisation\nNode type counts, edge type "
       "counts, and network topology summary (degree/centrality distributions)."),
    code("print(nodes.node_type.value_counts())\nprint()\nprint(edges.edge_type.value_counts())"),
    code("import networkx as nx\n"
         "G = nx.DiGraph()\nG.add_nodes_from(nodes.node_id)\n"
         "G.add_edges_from(edges[['src','dst']].itertuples(index=False, name=None))\n"
         "print('Nodes:', G.number_of_nodes(), 'Edges:', G.number_of_edges())\n"
         "print('Density:', nx.density(G))\n"
         "degrees = pd.Series(dict(G.degree()))\n"
         "print(degrees.describe())"),
    md("## (b) Temporal pattern analysis\nSeasonality decomposition of port "
       "throughput and freight rates."),
    code("ts = gen.generate_time_series(n_days=730)\nts.groupby('port')[['throughput_teu','freight_rate_usd_feu']].describe().T.head(20)"),
    code("from src.features.temporal_features import build_temporal_features\n"
         "port0 = ts[ts.port == ts.port.iloc[0]]\n"
         "feats = build_temporal_features(port0, value_col='throughput_teu')\n"
         "feats[['date','throughput_teu','roll_mean_30','ewma_30','month_sin','month_cos']].tail(10)"),
    md("## (c) Correlation analysis: supply chain metrics vs. financial outcomes\n"
       "Reproduces the logic behind Section A1.3's metric-financial-impact table."),
    code("from src.features.graph_features import build_entity_features, ENTITY_FEATURE_COLUMNS\n"
         "feats_entity = build_entity_features(nodes)\n"
         "corr_target = nodes.set_index('node_id')['pd_annual_true']\n"
         "corrs = feats_entity.set_index('node_id')[ENTITY_FEATURE_COLUMNS].corrwith(corr_target)\n"
         "corrs.sort_values(key=abs, ascending=False)"),
    md("## (d) Missing data assessment\nThe synthetic generator produces complete data by "
       "construction (no missingness); in a production deployment against real AIS/customs/"
       "ERP feeds, `src/data` would need an imputation strategy for the completeness gaps "
       "documented in Section A4.3 (e.g. AIS destination field only ~60-70% populated)."),
    code("print('Missing values per column (nodes):')\nprint(nodes.isna().sum()[nodes.isna().sum() > 0])"),
])

# ------------------------------------------------------- 02_feature_engineering ----
nb2 = make_nb([
    md("# 02 — Feature Engineering\nBuilds the full feature catalog described in "
       "`docs/feature_catalog.md` (Deliverable D2.1.3, Section A5)."),
    code("import sys, os\nsys.path.insert(0, os.path.abspath('..'))\n"
         "from src.data.synthetic_generator import SupplyChainDataGenerator\n"
         "gen = SupplyChainDataGenerator(seed=42)\n"
         "nodes, edges = gen.generate_graph()\n"
         "ts = gen.generate_time_series(n_days=365)\n"
         "shipments = gen.generate_shipments(n=2000)\n"
         "lc = gen.generate_trade_finance_transactions(n=2000, nodes=nodes)"),
    md("## Entity (graph) features — Section A5.1"),
    code("from src.features.graph_features import build_entity_features, normalise_features, ENTITY_FEATURE_COLUMNS\n"
         "entity_feats = build_entity_features(nodes)\n"
         "print(f'{len(ENTITY_FEATURE_COLUMNS)} entity features')\n"
         "entity_feats[ENTITY_FEATURE_COLUMNS].describe().T"),
    md("## Temporal features — Section A5.2"),
    code("from src.features.temporal_features import build_temporal_features\n"
         "port0 = ts[ts.port == ts.port.iloc[0]]\n"
         "temporal_feats = build_temporal_features(port0, value_col='throughput_teu')\n"
         "new_cols = [c for c in temporal_feats.columns if c not in port0.columns]\n"
         "print(f'{len(new_cols)} engineered temporal features')\n"
         "temporal_feats[new_cols].head()"),
    md("## Trade finance transaction + cross-domain fusion features — Sections A5.3, A5.4"),
    code("from src.features.financial_features import (\n"
         "    build_lc_transaction_features, LC_TRANSACTION_FEATURE_COLUMNS,\n"
         "    supply_chain_adjusted_pd, working_capital_velocity_index, trade_route_financial_stress_index)\n"
         "lc_feats = build_lc_transaction_features(lc)\n"
         "print(f'{len(LC_TRANSACTION_FEATURE_COLUMNS)} LC transaction features')\n"
         "lc_feats[LC_TRANSACTION_FEATURE_COLUMNS].describe().T"),
    code("# Reproduce the brief's SC-PD worked example exactly (Section A5.4)\n"
         "sc_pd = supply_chain_adjusted_pd(0.025, otif_actual=0.85, inv_turnover_actual=4.8, alternative_supplier_count=1)\n"
         "print(f'SC-PD = {sc_pd:.4f} (brief worked example: 0.0333)')"),
])

# ------------------------------------------------------- 03_model_training ----
nb3 = make_nb([
    md("# 03 — Model Training\nTrains every Level-0 base learner: GNN, TCN, "
       "Transformer, XGBoost, Cox PH / DeepSurv (Section A3, Deliverables D4-D7).\n\n"
       "This mirrors `demo/run_pipeline.py` Phase 2 exactly. For a faster run reduce "
       "the epoch counts below; for the numbers reported in the README, use the "
       "defaults (or just run `python -m demo.run_pipeline`)."),
    code("import sys, os, warnings\nsys.path.insert(0, os.path.abspath('..'))\nwarnings.filterwarnings('ignore')\n"
         "from src.data.synthetic_generator import SupplyChainDataGenerator\n"
         "gen = SupplyChainDataGenerator(seed=42)\n"
         "nodes, edges = gen.generate_graph()\n"
         "ts = gen.generate_time_series(n_days=730)\n"
         "shipments = gen.generate_shipments(n=3000)"),
    md("## GNN (HetGAT) — Section A3.1"),
    code("from src.models.gnn import train_gnn\n"
         "gnn_model, gnn_heads, gnn_embeddings, gnn_node_ids, gnn_metrics = train_gnn(nodes, edges)\n"
         "gnn_metrics"),
    md("## TCN — Section A3.2"),
    code("from src.models.tcn import train_tcn\n"
         "tcn_model, tcn_scalers, tcn_metrics = train_tcn(ts, value_col='throughput_teu',\n"
         "    exog_col='port_congestion_index', epochs=15, max_entities=6)\n"
         "tcn_metrics"),
    md("## Transformer — Section A3.3"),
    code("from src.models.transformer import train_shipment_transformer\n"
         "transformer_model, transformer_metrics = train_shipment_transformer(shipments, epochs=35)\n"
         "transformer_metrics"),
    md("## XGBoost + SHAP — Section A7.1"),
    code("from src.models.xgboost_model import train_xgboost_default_model, compute_shap_values\n"
         "xgb_model, xgb_split, xgb_metrics, importances = train_xgboost_default_model(nodes, n_trials=15)\n"
         "print({k: v for k, v in xgb_metrics.items() if k != 'best_params'})\n"
         "importances.head(10)"),
    md("## Survival analysis: Cox PH + DeepSurv — Section A3.4"),
    code("from src.models.survival import fit_cox_ph, train_deepsurv\n"
         "cox_model, cox_fin_model, cox_metrics = fit_cox_ph(nodes)\n"
         "cox_metrics"),
    code("deepsurv_model, deepsurv_metrics = train_deepsurv(nodes, epochs=100, n_folds=5)\n"
         "deepsurv_metrics"),
])

# ------------------------------------------------- 04_financial_integration ----
nb4 = make_nb([
    md("# 04 — Financial Model Integration\nTrade finance default stacking ensemble, "
       "CCC/working-capital predictor, and the SC-PD credit risk scorer "
       "(Section D2.3, Deliverables D8-D10). Assumes the models from "
       "`03_model_training.ipynb` are already trained in-session, or re-trains "
       "them fresh below."),
    code("import sys, os, warnings\nsys.path.insert(0, os.path.abspath('..'))\nwarnings.filterwarnings('ignore')\n"
         "from src.data.synthetic_generator import SupplyChainDataGenerator\n"
         "from src.models.gnn import train_gnn\n"
         "from src.models.xgboost_model import train_xgboost_default_model\n"
         "from src.models.survival import fit_cox_ph\n"
         "from src.models.transformer import train_shipment_transformer\n\n"
         "gen = SupplyChainDataGenerator(seed=42)\n"
         "nodes, edges = gen.generate_graph()\n"
         "lc_df = gen.generate_trade_finance_transactions(n=6000, nodes=nodes)\n"
         "shipments = gen.generate_shipments(n=3000)\n\n"
         "_, _, gnn_embeddings, gnn_node_ids, _ = train_gnn(nodes, edges, verbose=False)\n"
         "xgb_model, _, _, _ = train_xgboost_default_model(nodes, n_trials=10, verbose=False)\n"
         "cox_model, _, _ = fit_cox_ph(nodes)\n"
         "transformer_model, _ = train_shipment_transformer(shipments, epochs=25, verbose=False)\n"
         "print('Level-0 base learners ready')"),
    md("## Trade finance default prediction — stacking ensemble (Section A7.2, Deliverable D2.3.1)"),
    code("from src.financial.trade_finance_default import train_trade_finance_default_model\n"
         "meta_model, X, split, tf_metrics = train_trade_finance_default_model(\n"
         "    lc_df, nodes, xgb_model, cox_model, gnn_embeddings, gnn_node_ids, transformer_model)\n"
         "tf_metrics"),
    md("## Working capital / CCC predictor (Deliverable D2.3.2, Section A6.4 worked example)"),
    code("from src.financial.ccc_predictor import train_ccc_predictor, predict_ccc_change, covenant_breach_alert\n"
         "ccc_models, ccc_metrics = train_ccc_predictor()\n"
         "print(ccc_metrics)\n\n"
         "# MedDevice Corp scenario\n"
         "signals = dict(delta_otif=-0.12, delta_lead_time_std=4.2, delta_freight_cost_ratio=0.0,\n"
         "               delta_port_congestion=2.1, delta_inventory_turnover=0.0)\n"
         "result = predict_ccc_change(ccc_models, signals, horizon_days=90)\n"
         "alert = covenant_breach_alert(72, result['ccc_change'], 90)\n"
         "print(result); print(alert)"),
    md("## Supply-chain-enhanced credit risk scorer / SC-PD (Deliverable D2.3.3, Section A8.2)"),
    code("from src.financial.credit_risk_scorer import train_credit_risk_scorer, explain_prediction\n"
         "credit_result = train_credit_risk_scorer(nodes)\n"
         "print(credit_result['metrics'])\n\n"
         "sample_id = nodes[nodes.node_type == 'supplier'].node_id.iloc[0]\n"
         "explanation = explain_prediction(credit_result['model_sc_enhanced'], credit_result['feats'], sample_id)\n"
         "explanation"),
])

# --------------------------------------------------------------- 05_evaluation ----
nb5 = make_nb([
    md("# 05 — Evaluation & LogisChain Lab Simulation\nModel evaluation summary "
       "(Section A6) and the LogisChain Lab gamified simulation (Part B), comparing "
       "an SC-aware policy against a passive, financial-only baseline."),
    code("import sys, os, warnings\nsys.path.insert(0, os.path.abspath('..'))\nwarnings.filterwarnings('ignore')\n"
         "import json\n"
         "from pathlib import Path\n"
         "results_path = Path('..') / 'results' / 'pipeline_results.json'\n"
         "if results_path.exists():\n"
         "    with open(results_path) as f:\n"
         "        results = json.load(f)\n"
         "    print(json.dumps(results, indent=2)[:2000])\n"
         "else:\n"
         "    print('Run `python -m demo.run_pipeline` first to generate results/pipeline_results.json')"),
    md("## LogisChain Lab: Trade Finance Portfolio Management + SCF Pricing\n"
       "Both mandatory game modes (Section B1.3), run under both policies over an "
       "identical scenario sequence."),
    code("from src.data.synthetic_generator import SupplyChainDataGenerator\n"
         "from src.simulation.game_modes import TradeFinancePortfolioMode, SupplyChainFinancePricingMode\n\n"
         "gen = SupplyChainDataGenerator(seed=42)\n"
         "nodes, edges = gen.generate_graph()\n\n"
         "mode1 = TradeFinancePortfolioMode(nodes, edges, n_clients=45, seed=7)\n"
         "res_sc = mode1.run('sc_aware', n_turns=52)\n"
         "res_passive = mode1.run('passive', n_turns=52)\n"
         "print('SC-aware score:', res_sc['score']['total'], res_sc['score']['certification_level'])\n"
         "print('Passive score:', res_passive['score']['total'], res_passive['score']['certification_level'])"),
    code("mode2 = SupplyChainFinancePricingMode(nodes, edges, n_suppliers=60, seed=7)\n"
         "res2_sc = mode2.run('sc_aware', n_turns=52)\n"
         "res2_passive = mode2.run('passive', n_turns=52)\n"
         "print('SC-aware revenue: $%.0f, score: %.1f' % (res2_sc['total_discount_revenue_usd'], res2_sc['score']['total']))\n"
         "print('Passive revenue:  $%.0f, score: %.1f' % (res2_passive['total_discount_revenue_usd'], res2_passive['score']['total']))"),
    md("## Scenario log — which disruptions were triggered this run"),
    code("import pandas as pd\npd.DataFrame(res_sc['scenario_log'])"),
])

for name, nb in [
    ("01_eda.ipynb", nb1), ("02_feature_engineering.ipynb", nb2),
    ("03_model_training.ipynb", nb3), ("04_financial_integration.ipynb", nb4),
    ("05_evaluation.ipynb", nb5),
]:
    with open(OUT / name, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {name}")
