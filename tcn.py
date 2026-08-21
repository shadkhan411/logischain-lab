"""
Temporal Convolutional Network (TCN) - Section A3.2.

Dilated causal convolutions, dilation factors [1,2,4,8,16,32,64] (7 layers ->
128-day receptive field, exactly as specified), residual blocks, and a
distributional (quantile regression) output head for 30/60/90-day horizons.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from src.features.temporal_features import build_temporal_features

DILATIONS = [1, 2, 4, 8, 16, 32, 64]
QUANTILES = [0.1, 0.5, 0.9]


class Chomp1d(nn.Module):
    """Removes the extra right-padding added for causal convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                  self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, n_features: int, n_horizons: int = 3, n_quantiles: int = 3,
                 num_filters: int = 32, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = n_features
        for d in DILATIONS:
            layers.append(TemporalBlock(in_ch, num_filters, kernel_size, dilation=d, dropout=dropout))
            in_ch = num_filters
        self.tcn = nn.Sequential(*layers)
        # Output heads: one linear layer producing (n_horizons * n_quantiles) values
        # from the representation at the last (most recent) time step.
        self.head = nn.Linear(num_filters, n_horizons * n_quantiles)
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

    def forward(self, x):
        # x: (batch, n_features, seq_len)
        h = self.tcn(x)              # (batch, num_filters, seq_len)
        last = h[:, :, -1]           # last time step representation
        out = self.head(last)        # (batch, n_horizons * n_quantiles)
        return out.view(-1, self.n_horizons, self.n_quantiles)


def pinball_loss(preds, target, quantiles=QUANTILES):
    """preds: (batch, n_horizons, n_quantiles); target: (batch, n_horizons)."""
    losses = []
    for qi, q in enumerate(quantiles):
        err = target - preds[:, :, qi]
        losses.append(torch.max((q - 1) * err, q * err))
    return torch.stack(losses, dim=-1).mean()


FEATURE_CHANNELS = ["value_scaled", "roll_mean_7", "roll_mean_30", "roll_std_7",
                     "ewma_7", "month_sin", "month_cos", "exog"]


def build_windows(feat_df: pd.DataFrame, value_col: str, exog_col: str, lookback: int = 128,
                   horizons=(30, 60, 90)):
    df = feat_df.copy()
    mu, sd = df[value_col].mean(), df[value_col].std() + 1e-8
    df["value_scaled"] = (df[value_col] - mu) / sd
    df["exog"] = (df[exog_col] - df[exog_col].mean()) / (df[exog_col].std() + 1e-8)

    channels = ["value_scaled", "roll_mean_7", "roll_mean_30", "roll_std_7", "ewma_7",
                "month_sin", "month_cos", "exog"]
    # normalise the rolling/ewma channels onto the same scale as value_scaled
    for c in ["roll_mean_7", "roll_mean_30", "ewma_7"]:
        df[c] = (df[c] - mu) / sd
    df["roll_std_7"] = df["roll_std_7"] / (sd + 1e-8)

    X, Y, idx = [], [], []
    max_h = max(horizons)
    n = len(df)
    for t in range(lookback, n - max_h):
        window = df[channels].iloc[t - lookback:t].values.T  # (n_features, lookback)
        targets = [df["value_scaled"].iloc[t + h] for h in horizons]
        X.append(window)
        Y.append(targets)
        idx.append(t)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), (mu, sd), idx


def train_tcn(ts_df: pd.DataFrame, value_col: str, exog_col: str = "port_congestion_index",
              entity_col: str = "port", lookback: int = 128, horizons=(30, 60, 90),
              epochs: int = 15, lr: float = 5e-3, seed: int = 42, verbose: bool = True,
              num_filters: int = 24, max_entities: int | None = None):
    torch.manual_seed(seed)
    all_X, all_Y = [], []
    scalers = {}
    test_sets = {}

    entities = ts_df[entity_col].unique().tolist()
    if max_entities is not None:
        entities = entities[:max_entities]
    ts_df = ts_df[ts_df[entity_col].isin(entities)]

    for entity, g in ts_df.groupby(entity_col):
        feats = build_temporal_features(g, value_col=value_col)
        X, Y, (mu, sd), idx = build_windows(feats, value_col, exog_col, lookback, horizons)
        scalers[entity] = (mu, sd)
        n = len(X)
        split = int(0.8 * n)
        all_X.append(X[:split]); all_Y.append(Y[:split])
        test_sets[entity] = (X[split:], Y[split:])

    X_train = np.concatenate(all_X, axis=0)
    Y_train = np.concatenate(all_Y, axis=0)
    Xt = torch.tensor(X_train)
    Yt = torch.tensor(Y_train)

    model = TCNModel(n_features=X_train.shape[1], n_horizons=len(horizons),
                      n_quantiles=len(QUANTILES), num_filters=num_filters)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    batch_size = 64
    n_train = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            preds = model(Xt[b])
            loss = pinball_loss(preds, Yt[b])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  [TCN] epoch {epoch:3d}  pinball_loss={epoch_loss / n_train:.4f}")

    # ---- Evaluation: MAPE on the median (50th pct) forecast, 30-day horizon ----
    model.eval()
    horizon_idx = horizons.index(30) if 30 in horizons else 0
    q50_idx = QUANTILES.index(0.5)
    mape_per_entity = {}
    with torch.no_grad():
        for entity, (Xte, Yte) in test_sets.items():
            if len(Xte) == 0:
                continue
            mu, sd = scalers[entity]
            preds = model(torch.tensor(Xte))[:, horizon_idx, q50_idx].numpy()
            true = Yte[:, horizon_idx]
            preds_orig = preds * sd + mu
            true_orig = true * sd + mu
            mape = np.mean(np.abs((true_orig - preds_orig) / np.clip(np.abs(true_orig), 1e-6, None))) * 100
            mape_per_entity[entity] = float(mape)

    metrics = dict(
        mape_30d_by_entity=mape_per_entity,
        mape_30d_mean=float(np.mean(list(mape_per_entity.values()))) if mape_per_entity else float("nan"),
    )
    return model, scalers, metrics


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    ts = gen.generate_time_series(n_days=730)

    print("=== Port throughput forecasting ===")
    model1, scalers1, metrics1 = train_tcn(ts, value_col="throughput_teu",
                                            exog_col="port_congestion_index", epochs=15,
                                            max_entities=6)
    print("MAPE (30d, mean across ports):", round(metrics1["mape_30d_mean"], 2), "%")

    print("=== Freight rate forecasting ===")
    model2, scalers2, metrics2 = train_tcn(ts, value_col="freight_rate_usd_feu",
                                            exog_col="port_congestion_index", epochs=15,
                                            max_entities=6)
    print("MAPE (30d, mean across ports):", round(metrics2["mape_30d_mean"], 2), "%")
