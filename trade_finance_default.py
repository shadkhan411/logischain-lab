"""
Trade Finance Default Prediction Model - Deliverable D2.3.1.

Integrates every Level-0 base learner into one LC-transaction-level feature
matrix, then fits the Level-1 stacking meta-learner (Section A7.2):

  - XGBoost tabular default probability for the applicant entity
  - Cox PH partial hazard for the applicant entity
  - GNN entity-embedding summary statistics for applicant & beneficiary
  - Transformer-derived shipment risk score for the LC's underlying shipment
    (synthesized from the LC's own route/congestion/discrepancy fields --
    the same signals a real trade-finance desk would pull from a shipment
    tracking feed for the goods behind the LC)
  - The raw LC transaction features (Section A5.3)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from src.features.financial_features import build_lc_transaction_features, LC_TRANSACTION_FEATURE_COLUMNS
from src.models.ensemble import fit_stacking_meta_learner
from src.models.transformer import build_event_sequences


def _embedding_lookup(node_ids_by_type: dict, embeddings: dict):
    """node_id -> embedding vector, across all node types."""
    lut = {}
    for ntype, ids in node_ids_by_type.items():
        emb = embeddings[ntype]
        for i, nid in enumerate(ids):
            lut[nid] = emb[i]
    return lut


def _synthesize_shipment_like_row(lc_row) -> dict:
    """Map an LC transaction's own fields onto the shipment-feature schema the
    Transformer expects, so the trained shipment-risk model can score the
    goods movement backing this specific LC."""
    return dict(
        origin_congestion_index=lc_row["port_congestion_origin"],
        destination_congestion_index=lc_row["port_congestion_destination"],
        vessel_speed_ratio=1.0 - 0.15 * lc_row["freight_rate_percentile"],
        carrier_reliability_score=1.0 - 0.5 * lc_row["historical_discrepancy_rate_beneficiary"],
        weather_risk_score=lc_row["trade_route_risk_score"] * 0.6,
        cargo_value_usd=lc_row["lc_amount_usd"],
    )


def build_stacked_feature_matrix(
    lc_df: pd.DataFrame, nodes: pd.DataFrame, xgb_model, cox_model,
    gnn_embeddings: dict, gnn_node_ids: dict, transformer_model,
) -> pd.DataFrame:
    from src.features.graph_features import ENTITY_FEATURE_COLUMNS, build_entity_features
    from src.models.survival import COX_COVARIATES

    lc = build_lc_transaction_features(lc_df).reset_index(drop=True)

    # ---- XGBoost entity-level PD, precomputed once for all nodes ----
    ent_feats = build_entity_features(nodes)
    xgb_pd_all = xgb_model.predict_proba(ent_feats[ENTITY_FEATURE_COLUMNS])[:, 1]
    node_to_xgb_pd = dict(zip(nodes.node_id, xgb_pd_all))

    # ---- Cox PH partial hazard, precomputed once for all nodes ----
    cox_input = ent_feats[COX_COVARIATES].copy()
    hazard_all = cox_model.predict_partial_hazard(cox_input).values.ravel()
    node_to_hazard = dict(zip(nodes.node_id, hazard_all))

    # ---- GNN embedding lookup ----
    emb_lut = _embedding_lookup(gnn_node_ids, gnn_embeddings)

    def emb_summary(node_id, prefix):
        vec = emb_lut.get(node_id)
        if vec is None:
            return {f"{prefix}_emb_mean": 0.0, f"{prefix}_emb_norm": 0.0}
        return {f"{prefix}_emb_mean": float(np.mean(vec)), f"{prefix}_emb_norm": float(np.linalg.norm(vec))}

    applicant_xgb_pd = lc["applicant_id"].map(node_to_xgb_pd).fillna(node_to_xgb_pd and np.mean(xgb_pd_all))
    applicant_hazard = lc["applicant_id"].map(node_to_hazard).fillna(np.mean(hazard_all))
    beneficiary_xgb_pd = lc["beneficiary_id"].map(node_to_xgb_pd).fillna(np.mean(xgb_pd_all))

    emb_rows = []
    for _, row in lc.iterrows():
        d = {}
        d.update(emb_summary(row["applicant_id"], "applicant"))
        d.update(emb_summary(row["beneficiary_id"], "beneficiary"))
        emb_rows.append(d)
    emb_df = pd.DataFrame(emb_rows)

    # ---- Transformer shipment-risk proxy ----
    shipment_like = pd.DataFrame([_synthesize_shipment_like_row(r) for _, r in lc.iterrows()])
    seqs, _ = build_event_sequences(shipment_like)
    transformer_model.eval()
    with torch.no_grad():
        out = transformer_model(torch.tensor(seqs))
        transformer_delay_prob = torch.sigmoid(out["delay_logit"]).numpy()
        transformer_risk_score = out["risk_score"].numpy()

    feature_df = lc[LC_TRANSACTION_FEATURE_COLUMNS].copy()
    feature_df["applicant_xgb_pd"] = applicant_xgb_pd.values
    feature_df["applicant_cox_hazard"] = applicant_hazard.values
    feature_df["beneficiary_xgb_pd"] = beneficiary_xgb_pd.values
    feature_df = pd.concat([feature_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    feature_df["transformer_delay_prob"] = transformer_delay_prob
    feature_df["transformer_risk_score"] = transformer_risk_score

    return feature_df


def train_trade_finance_default_model(lc_df, nodes, xgb_model, cox_model, gnn_embeddings,
                                       gnn_node_ids, transformer_model, seed: int = 42):
    X = build_stacked_feature_matrix(lc_df, nodes, xgb_model, cox_model, gnn_embeddings,
                                      gnn_node_ids, transformer_model)
    y = lc_df["default"].values
    model, split, metrics = fit_stacking_meta_learner(X, y, seed=seed)

    # Expected Calibration Error (Section A6.2) on the test split
    _, Xte, ytr, yte, p_test = split
    ece = _expected_calibration_error(yte, p_test)
    metrics["ece"] = ece
    metrics["n_features"] = X.shape[1]
    return model, X, split, metrics


def _expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1] if i < n_bins - 1 else p_pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(p_pred)) * abs(y_true[mask].mean() - p_pred[mask].mean())
    return float(ece)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator
    from src.models.gnn import train_gnn
    from src.models.xgboost_model import train_xgboost_default_model
    from src.models.survival import fit_cox_ph
    from src.models.transformer import train_shipment_transformer

    print("Generating data...")
    gen = SupplyChainDataGenerator()
    nodes, edges = gen.generate_graph()
    lc_df = gen.generate_trade_finance_transactions(n=4000, nodes=nodes)
    shipments = gen.generate_shipments(n=3000)

    print("Training Level-0 base learners...")
    _, _, gnn_embeddings, gnn_node_ids, gnn_metrics = train_gnn(nodes, edges, verbose=False)
    xgb_model, _, xgb_metrics, _ = train_xgboost_default_model(nodes, n_trials=10, verbose=False)
    cox_model, _, cox_metrics = fit_cox_ph(nodes)
    transformer_model, transformer_metrics = train_shipment_transformer(shipments, epochs=25, verbose=False)

    print("Training Level-1 stacking meta-learner (trade finance default)...")
    meta_model, X, split, metrics = train_trade_finance_default_model(
        lc_df, nodes, xgb_model, cox_model, gnn_embeddings, gnn_node_ids, transformer_model)
    print("Trade Finance Default Model metrics:", metrics)
