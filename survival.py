"""
Survival analysis for supplier default timing - Section A3.4.

Cox Proportional Hazards (lifelines) with supply-chain covariates, plus a
DeepSurv neural extension that captures non-linear interaction effects
(e.g. OTIF decline x leverage) the linear Cox model misses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

from src.features.graph_features import ENTITY_FEATURE_COLUMNS, build_entity_features, normalise_features

# A reduced covariate set keeps the Cox model well-conditioned (lifelines'
# penalizer handles the rest); DeepSurv uses the full feature set.
COX_COVARIATES = [
    "otif_rate", "inventory_turnover", "ccc_days", "supplier_concentration_hhi",
    "freight_cost_ratio", "betweenness_centrality", "debt_to_equity", "current_ratio",
    "ebitda_margin", "lead_time_std",
]


def fit_cox_ph(nodes: pd.DataFrame, penalizer: float = 0.05, seed: int = 42):
    feats = build_entity_features(nodes)
    df = feats[["node_id"] + COX_COVARIATES].merge(
        nodes[["node_id", "survival_duration_days", "survival_event"]], on="node_id"
    ).drop(columns=["node_id"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col="survival_duration_days", event_col="survival_event")

    c_index_train = concordance_index(train_df["survival_duration_days"],
                                       -cph.predict_partial_hazard(train_df), train_df["survival_event"])
    c_index_test = concordance_index(test_df["survival_duration_days"],
                                      -cph.predict_partial_hazard(test_df), test_df["survival_event"])

    # baseline (financial ratios only, no supply-chain covariates) for comparison
    financial_only_cols = ["debt_to_equity", "current_ratio", "ebitda_margin"]
    df_fin = feats[["node_id"] + financial_only_cols].merge(
        nodes[["node_id", "survival_duration_days", "survival_event"]], on="node_id"
    ).drop(columns=["node_id"])
    train_fin, test_fin = train_test_split(df_fin, test_size=0.2, random_state=seed)
    cph_fin = CoxPHFitter(penalizer=penalizer)
    cph_fin.fit(train_fin, duration_col="survival_duration_days", event_col="survival_event")
    c_index_fin_only = concordance_index(test_fin["survival_duration_days"],
                                          -cph_fin.predict_partial_hazard(test_fin), test_fin["survival_event"])

    metrics = dict(
        c_index_train=float(c_index_train),
        c_index_test_full=float(c_index_test),
        c_index_test_financial_only=float(c_index_fin_only),
        c_index_improvement=float(c_index_test - c_index_fin_only),
    )
    return cph, cph_fin, metrics


class DeepSurv(nn.Module):
    """h_i(t) = h_0(t) * exp(f_theta(x_i)) -- non-linear log-risk network (Section A3.4)."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # log-risk score f_theta(x)


def cox_partial_likelihood_loss(log_risk: torch.Tensor, duration: torch.Tensor, event: torch.Tensor):
    """Negative Cox partial log-likelihood (Breslow approximation), sorted by duration desc."""
    order = torch.argsort(duration, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    log_cumsum_risk = torch.logcumsumexp(log_risk, dim=0)
    diff = log_risk - log_cumsum_risk
    return -(diff * event).sum() / event.sum().clamp(min=1)


def train_deepsurv(nodes: pd.DataFrame, epochs: int = 150, lr: float = 5e-3, seed: int = 42,
                    verbose: bool = True, n_folds: int = 5):
    """Trains DeepSurv with k-fold cross-validated concordance evaluation.

    A single 80/20 holdout is too unstable to evaluate here: with a ~7-8%
    default rate over ~200 nodes, a 20% test slice contains only a handful of
    actual events, and the concordance index is only computed over
    "comparable pairs" involving an event -- so a single split can swing
    wildly (observed: anywhere from ~0.2 to ~0.8) purely from which few
    defaulters happen to land in the test fold. K-fold CV averages this out.
    """
    torch.manual_seed(seed)
    feats, _ = normalise_features(build_entity_features(nodes))
    X = feats[ENTITY_FEATURE_COLUMNS].values.astype(np.float32)
    duration = nodes["survival_duration_days"].values.astype(np.float32)
    event = nodes["survival_event"].values.astype(np.float32)

    rng = np.random.default_rng(seed)
    n = len(nodes)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    Xt = torch.tensor(X)
    dur_t = torch.tensor(duration)
    ev_t = torch.tensor(event)

    fold_c_indices = []
    final_model = None
    for fold_i in range(n_folds):
        test_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])

        model = DeepSurv(n_features=X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            log_risk = model(Xt[train_idx])
            loss = cox_partial_likelihood_loss(log_risk, dur_t[train_idx], ev_t[train_idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            risk_test = model(Xt[test_idx]).numpy()
        if event[test_idx].sum() > 0:
            fold_c = concordance_index(duration[test_idx], -risk_test, event[test_idx])
            fold_c_indices.append(fold_c)
        final_model = model
        if verbose:
            print(f"  [DeepSurv] fold {fold_i + 1}/{n_folds}  "
                  f"n_events_in_fold={int(event[test_idx].sum())}  c_index={fold_c_indices[-1]:.4f}")

    return final_model, dict(
        c_index_cv_mean=float(np.mean(fold_c_indices)),
        c_index_cv_std=float(np.std(fold_c_indices)),
        c_index_per_fold=[float(c) for c in fold_c_indices],
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    nodes, _ = gen.generate_graph()
    cph, cph_fin, metrics = fit_cox_ph(nodes)
    print("Cox PH metrics:", metrics)
    print("\nCox PH summary (top rows):")
    print(cph.summary[["coef", "exp(coef)", "p"]].head(10))

    model, ds_metrics = train_deepsurv(nodes, epochs=150)
    print("\nDeepSurv metrics:", ds_metrics)
