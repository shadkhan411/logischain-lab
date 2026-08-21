"""
Supply-Chain-Enhanced Credit Risk Scorer - Deliverable D2.3.3.

Compares a financial-ratios-only baseline against the full supply-chain-
enhanced model (10+ SC features), quantifies the discrimination improvement,
applies the closed-form SC-PD fusion formula (Section A5.4) as a transparent
white-box cross-check, and produces SHAP-based local explanations plus a
regulatory "model card" (Section A8.4, SR 11-7 style).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.features.graph_features import ENTITY_FEATURE_COLUMNS, build_entity_features
from src.features.financial_features import supply_chain_adjusted_pd

FINANCIAL_ONLY_COLUMNS = ["current_ratio", "debt_to_equity", "ebitda_margin",
                          "interest_coverage", "working_capital_ratio"]
SC_ENHANCED_COLUMNS = ENTITY_FEATURE_COLUMNS  # all 21, financial + 16 SC/network/geo features
N_SC_FEATURES = len(SC_ENHANCED_COLUMNS) - len(FINANCIAL_ONLY_COLUMNS)  # >= 10 required


def train_credit_risk_scorer(nodes: pd.DataFrame, seed: int = 42, verbose: bool = True):
    assert N_SC_FEATURES >= 10, "Deliverable requires >=10 supply chain covariates"

    feats = build_entity_features(nodes)
    y = nodes["default_12m"].values
    Xtr_idx, Xte_idx = train_test_split(np.arange(len(nodes)), test_size=0.2,
                                         random_state=seed, stratify=y)

    def fit_eval(cols):
        X = feats[cols]
        pos_rate = y[Xtr_idx].mean()
        spw = (1 - pos_rate) / max(pos_rate, 1e-6)
        m = xgb.XGBClassifier(max_depth=3, n_estimators=250, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw,
                               eval_metric="auc", random_state=seed)
        m.fit(X.iloc[Xtr_idx], y[Xtr_idx])
        p = m.predict_proba(X.iloc[Xte_idx])[:, 1]
        auc = roc_auc_score(y[Xte_idx], p)
        return m, p, auc

    model_fin, p_fin, auc_fin = fit_eval(FINANCIAL_ONLY_COLUMNS)
    model_full, p_full, auc_full = fit_eval(SC_ENHANCED_COLUMNS)

    # White-box SC-PD cross-check (Section A5.4 closed-form formula), using the
    # financial-only model's PD as the "traditional PD" input.
    alt_supplier_count_proxy = (1.0 / feats["supplier_concentration_hhi"]).clip(upper=8)
    sc_pd_all = supply_chain_adjusted_pd(
        traditional_pd=model_fin.predict_proba(feats[FINANCIAL_ONLY_COLUMNS])[:, 1],
        otif_actual=feats["otif_rate"].values,
        inv_turnover_actual=feats["inventory_turnover"].values,
        alternative_supplier_count=alt_supplier_count_proxy.values,
    )
    sc_pd_test = sc_pd_all[Xte_idx]
    auc_sc_pd_formula = roc_auc_score(y[Xte_idx], sc_pd_test)

    metrics = dict(
        auc_financial_only=float(auc_fin),
        auc_sc_enhanced_xgb=float(auc_full),
        auc_improvement=float(auc_full - auc_fin),
        auc_sc_pd_formula=float(auc_sc_pd_formula),
        n_sc_features=int(N_SC_FEATURES),
        statistically_meaningful_improvement=bool((auc_full - auc_fin) > 0.03),
    )
    if verbose:
        print(f"  [Credit Risk Scorer] AUC financial-only={auc_fin:.4f}  "
              f"AUC SC-enhanced={auc_full:.4f}  (+{auc_full - auc_fin:.4f})")

    return dict(model_financial_only=model_fin, model_sc_enhanced=model_full,
                sc_pd_scores=sc_pd_all, metrics=metrics, test_idx=Xte_idx, feats=feats)


def explain_prediction(model_sc_enhanced, feats: pd.DataFrame, node_id: str,
                        portfolio_avg_pd: float | None = None) -> dict:
    """SHAP-based local explanation for a single entity (Section A8.2 worked example)."""
    import shap
    row = feats[feats.node_id == node_id]
    if row.empty:
        raise ValueError(f"Unknown node_id: {node_id}")
    X_row = row[SC_ENHANCED_COLUMNS]
    explainer = shap.TreeExplainer(model_sc_enhanced)
    shap_vals = explainer.shap_values(X_row)
    base_value = explainer.expected_value
    pd_pred = float(model_sc_enhanced.predict_proba(X_row)[:, 1][0])

    contributions = pd.Series(np.array(shap_vals).ravel(), index=SC_ENHANCED_COLUMNS)
    contributions = contributions.sort_values(key=np.abs, ascending=False)

    return dict(
        node_id=node_id, predicted_pd=pd_pred,
        base_value=float(base_value) if np.isscalar(base_value) else float(np.ravel(base_value)[0]),
        top_contributions=contributions.head(8).to_dict(),
    )


MODEL_CARD = {
    "model_purpose_and_scope": (
        "Supports trade-finance risk assessment, SCF pricing, working-capital "
        "monitoring, and cargo insurance pricing for logistics-dependent borrowers. "
        "Does NOT support unsecured consumer lending or market-risk VaR (Section A8.4)."
    ),
    "conceptual_soundness": (
        "Supply-chain covariates are included on the documented causal chain: "
        "operational degradation -> financial stress -> default (Section A8.4). "
        "Each covariate's sign is checked against economic intuition (e.g. OTIF decline "
        "and CCC extension should raise PD)."
    ),
    "outcome_analysis": "Quarterly backtest of predicted vs. actual default rate by decile; "
                          "review triggered if any decile deviates >2 std. dev. from expected.",
    "ongoing_monitoring": "Population Stability Index (PSI) computed monthly per input feature; "
                            "PSI > 0.10 triggers investigation, PSI > 0.25 triggers recalibration.",
    "explainability": "SHAP TreeExplainer provides global feature importance and per-prediction "
                        "local attributions for every scored entity (Section A8.2).",
}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    nodes, _ = gen.generate_graph()
    result = train_credit_risk_scorer(nodes)
    print("Metrics:", result["metrics"])

    sample_id = nodes[nodes.node_type == "supplier"].node_id.iloc[0]
    explanation = explain_prediction(result["model_sc_enhanced"], result["feats"], sample_id)
    print(f"\nSHAP explanation for {sample_id}: PD={explanation['predicted_pd']:.4f}")
    for feat, val in explanation["top_contributions"].items():
        print(f"  {feat}: {val:+.4f}")
