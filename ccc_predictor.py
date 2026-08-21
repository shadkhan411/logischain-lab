"""
Working Capital / CCC Prediction Model - Deliverable D2.3.2.

Predicts 30/60/90-day Cash Conversion Cycle change from *changes* in
operational supply-chain signals (OTIF, lead-time variability, freight cost,
port congestion), decomposed into DIO/DSO/DPO contributions, exactly
mirroring the MedDevice Corp worked example in Section A6.4:

  ΔOTIF, ΔLeadTimeStd  -> ΔDIO   (more safety stock)
  Δport congestion     -> ΔDSO   (downstream customers also stressed, pay slower)
  Δfreight cost ratio  -> ΔDPO   (suppliers demand faster payment under their own stress)
  ΔCCC = ΔDIO + ΔDSO - ΔDPO

A covenant early-warning helper flags predicted breaches before they would
appear in quarterly financials.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def simulate_working_capital_shocks(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthesize operational-signal-change -> CCC-component-change training data,
    with the horizon (30/60/90 days) as an input feature (later horizons see
    larger cumulative shocks and more noise, reflecting compounding uncertainty)."""
    rng = np.random.default_rng(seed)
    horizons = rng.choice([30, 60, 90], size=n)
    horizon_scale = np.sqrt(horizons / 30.0)

    delta_otif = rng.normal(0, 0.05, size=n) * horizon_scale  # can be positive (improving) too
    delta_lead_time_std = rng.normal(0.3, 1.8, size=n) * horizon_scale
    delta_freight_cost_ratio = rng.normal(0.0, 0.025, size=n) * horizon_scale
    delta_port_congestion = rng.normal(0.1, 0.9, size=n) * horizon_scale
    delta_inventory_turnover = rng.normal(0, 0.6, size=n) * horizon_scale

    dio_change = (
        62 * np.maximum(0, -delta_otif)
        + 2.6 * np.maximum(0, delta_lead_time_std)
        - 3.0 * delta_inventory_turnover
        + rng.normal(0, 1.6 * horizon_scale, size=n)
    )
    dso_change = (
        4.8 * np.maximum(0, delta_port_congestion)
        + rng.normal(0, 1.1 * horizon_scale, size=n)
    )
    dpo_change = (
        -58 * np.maximum(0, delta_freight_cost_ratio)
        + 2.0 * np.maximum(0, -delta_port_congestion) * 0.3  # suppliers ease terms if things improve
        + rng.normal(0, 1.4 * horizon_scale, size=n)
    )
    ccc_change = dio_change + dso_change - dpo_change

    return pd.DataFrame(dict(
        horizon_days=horizons, delta_otif=delta_otif, delta_lead_time_std=delta_lead_time_std,
        delta_freight_cost_ratio=delta_freight_cost_ratio, delta_port_congestion=delta_port_congestion,
        delta_inventory_turnover=delta_inventory_turnover, dio_change=dio_change,
        dso_change=dso_change, dpo_change=dpo_change, ccc_change=ccc_change,
    ))


FEATURE_COLS = ["horizon_days", "delta_otif", "delta_lead_time_std", "delta_freight_cost_ratio",
                 "delta_port_congestion", "delta_inventory_turnover"]


def train_ccc_predictor(n: int = 6000, seed: int = 42, verbose: bool = True):
    df = simulate_working_capital_shocks(n=n, seed=seed)
    X = df[FEATURE_COLS]
    targets = ["dio_change", "dso_change", "dpo_change"]

    Xtr, Xte, idx_tr, idx_te = train_test_split(X, df.index, test_size=0.2, random_state=seed)
    models = {}
    for t in targets:
        m = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.06,
                          subsample=0.85, colsample_bytree=0.8, random_state=seed)
        m.fit(Xtr, df.loc[idx_tr, t])
        models[t] = m

    preds = {t: models[t].predict(Xte) for t in targets}
    pred_ccc = preds["dio_change"] + preds["dso_change"] - preds["dpo_change"]
    true_ccc = df.loc[idx_te, "ccc_change"].values

    # MAPE on the 30-day horizon slice specifically (Section D2.3.2 target: <15%)
    mask_30 = (df.loc[idx_te, "horizon_days"] == 30).values
    # MAPE is unstable near-zero true values; use a small epsilon floor consistent with
    # standard practice for change-variables that can cross zero.
    def safe_mape(y_true, y_pred, eps=3.0):
        return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)

    metrics = dict(
        mape_30d=safe_mape(true_ccc[mask_30], pred_ccc[mask_30]) if mask_30.sum() > 0 else float("nan"),
        mape_all_horizons=safe_mape(true_ccc, pred_ccc),
        mae_days=float(np.mean(np.abs(true_ccc - pred_ccc))),
    )
    if verbose:
        print(f"  [CCC predictor] MAPE(30d)={metrics['mape_30d']:.2f}%  "
              f"MAE={metrics['mae_days']:.2f} days")
    return models, metrics


def predict_ccc_change(models: dict, current_signals: dict, horizon_days: int = 30) -> dict:
    """Predict DIO/DSO/DPO/CCC change given current observed operational deltas."""
    row = pd.DataFrame([{**current_signals, "horizon_days": horizon_days}])[FEATURE_COLS]
    dio = float(models["dio_change"].predict(row)[0])
    dso = float(models["dso_change"].predict(row)[0])
    dpo = float(models["dpo_change"].predict(row)[0])
    ccc = dio + dso - dpo
    return dict(dio_change=dio, dso_change=dso, dpo_change=dpo, ccc_change=ccc)


def covenant_breach_alert(current_ccc_days: float, predicted_ccc_change: float,
                           covenant_threshold_days: float) -> dict:
    """Early-warning check mirroring the MedDevice Corp example (Section A6.4):
    flag when current CCC + predicted change would exceed the covenant threshold."""
    predicted_ccc = current_ccc_days + predicted_ccc_change
    breach = predicted_ccc > covenant_threshold_days
    margin = covenant_threshold_days - predicted_ccc
    return dict(predicted_ccc_days=round(predicted_ccc, 1), covenant_threshold_days=covenant_threshold_days,
                breach_predicted=bool(breach), margin_days=round(margin, 1))


if __name__ == "__main__":
    models, metrics = train_ccc_predictor()
    print("Metrics:", metrics)

    # Reproduce the MedDevice Corp worked example (Section A6.4)
    signals = dict(delta_otif=-0.12, delta_lead_time_std=4.2, delta_freight_cost_ratio=0.0,
                    delta_port_congestion=2.1, delta_inventory_turnover=0.0)
    result = predict_ccc_change(models, signals, horizon_days=90)
    print("\nMedDevice Corp-style scenario (90d):", result)
    alert = covenant_breach_alert(current_ccc_days=72, predicted_ccc_change=result["ccc_change"],
                                   covenant_threshold_days=90)
    print("Covenant alert:", alert)
