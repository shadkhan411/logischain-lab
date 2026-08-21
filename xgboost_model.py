"""
XGBoost tabular baseline for supplier default prediction - Section A7.1.

Trains on the 21 entity features (Section A5.1) to predict 12-month default.
Includes Optuna hyperparameter search and SHAP-based global/local
explainability (Section A8.2).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from src.features.graph_features import ENTITY_FEATURE_COLUMNS, build_entity_features


def train_xgboost_default_model(nodes: pd.DataFrame, n_trials: int = 25, seed: int = 42,
                                 verbose: bool = True):
    feats = build_entity_features(nodes)
    X = feats[ENTITY_FEATURE_COLUMNS]
    y = nodes["default_12m"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    pos_rate = ytr.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

    best_params = dict(max_depth=4, learning_rate=0.05, n_estimators=300,
                        min_child_weight=3, subsample=0.85, colsample_bytree=0.8,
                        gamma=0.05, reg_alpha=0.05, reg_lambda=1.0)

    if n_trials > 0:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = dict(
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    gamma=trial.suggest_float("gamma", 0.0, 0.4),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.5),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.0),
                )
                model = xgb.XGBClassifier(
                    **params, scale_pos_weight=scale_pos_weight, eval_metric="auc",
                    random_state=seed, n_jobs=2,
                )
                Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2,
                                                           random_state=seed, stratify=ytr)
                model.fit(Xtr2, ytr2)
                p = model.predict_proba(Xval)[:, 1]
                return roc_auc_score(yval, p)

            study = optuna.create_study(direction="maximize",
                                         sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            if verbose:
                print(f"  [XGBoost] Optuna best AUC (val): {study.best_value:.4f}")
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"  [XGBoost] Optuna tuning skipped ({e}); using default params.")

    model = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight,
                               eval_metric="auc", random_state=seed, n_jobs=2)
    model.fit(Xtr, ytr)

    p_test = model.predict_proba(Xte)[:, 1]
    metrics = dict(
        auc=float(roc_auc_score(yte, p_test)),
        gini=float(2 * roc_auc_score(yte, p_test) - 1),
        brier=float(brier_score_loss(yte, p_test)),
        average_precision=float(average_precision_score(yte, p_test)),
        best_params=best_params,
    )

    importances = pd.Series(model.feature_importances_, index=ENTITY_FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    return model, (Xtr, Xte, ytr, yte), metrics, importances


def compute_shap_values(model, X_sample: pd.DataFrame):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    nodes, _ = gen.generate_graph()
    model, (Xtr, Xte, ytr, yte), metrics, importances = train_xgboost_default_model(nodes, n_trials=15)
    print("Metrics:", {k: v for k, v in metrics.items() if k != "best_params"})
    print("\nTop 10 features by importance:")
    print(importances.head(10))

    explainer, shap_values = compute_shap_values(model, Xte.iloc[:20])
    print("\nSHAP values shape:", np.array(shap_values).shape)
