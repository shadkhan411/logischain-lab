"""
Transformer encoder for shipment-level risk prediction - Section A3.3.

Each shipment is represented as a sequence of 8 lifecycle event tokens
(booking_confirmed -> ... -> final_delivery), each embedded with temporal,
spatial, operational and contextual features, exactly the input
representation described in the brief. Multi-head self-attention (>=4 heads)
learns which events are most predictive of the final outcome. Multi-task
output heads: delay probability, damage probability, documentation
discrepancy probability, and a continuous 0-100 shipment risk score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss

EVENTS = ["booking_confirmed", "container_loaded", "vessel_departed", "transhipment_arrival",
          "transhipment_departure", "destination_port_arrival", "customs_cleared", "final_delivery"]
N_EVENTS = len(EVENTS)


def build_event_sequences(shipments: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Synthesize an 8-event token sequence per shipment from its summary features.

    Each event token = [event_type one-hot (8), position (1), origin_congestion (1),
    dest_congestion (1), vessel_speed_ratio (1), carrier_reliability (1),
    weather_risk (1), cargo_value_log (1)] = 15 dims. Congestion effects are made
    event-relevant (origin congestion matters near loading/departure; destination
    congestion matters near arrival/customs), mirroring the brief's discussion
    of *which* events are most predictive.
    """
    rng = np.random.default_rng(seed)
    n = len(shipments)
    n_feat = N_EVENTS + 7
    seqs = np.zeros((n, N_EVENTS, n_feat), dtype=np.float32)

    cargo_log = np.log(shipments["cargo_value_usd"].values.clip(min=1))
    cargo_log = (cargo_log - cargo_log.mean()) / (cargo_log.std() + 1e-8)

    for e in range(N_EVENTS):
        seqs[:, e, e] = 1.0  # event type one-hot
        seqs[:, e, N_EVENTS] = e / (N_EVENTS - 1)  # normalised position
        # origin congestion matters most for early events (booking..departure)
        origin_weight = max(0.25, 1.0 - e / 3.0)
        dest_weight = max(0.25, (e - 3) / 4.0) if e >= 1 else 0.25
        seqs[:, e, N_EVENTS + 1] = shipments["origin_congestion_index"].values * origin_weight
        seqs[:, e, N_EVENTS + 2] = shipments["destination_congestion_index"].values * dest_weight
        noise = rng.normal(0, 0.02, size=n)
        seqs[:, e, N_EVENTS + 3] = shipments["vessel_speed_ratio"].values + noise
        seqs[:, e, N_EVENTS + 4] = shipments["carrier_reliability_score"].values
        seqs[:, e, N_EVENTS + 5] = shipments["weather_risk_score"].values * (1.0 if e in (2, 3, 4) else 0.3)
        seqs[:, e, N_EVENTS + 6] = cargo_log

    labels = None
    if "delayed" in shipments.columns:
        labels = dict(
            delayed=shipments["delayed"].values.astype(np.float32),
            damaged=shipments["damaged"].values.astype(np.float32),
            doc_discrepancy=shipments["doc_discrepancy"].values.astype(np.float32),
            risk_score=(shipments["shipment_risk_score"].values.astype(np.float32) / 100.0),
        )
    return seqs, labels


class ShipmentRiskTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, seq_len: int = N_EVENTS, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)
        self.delay_head = nn.Linear(d_model, 1)
        self.damage_head = nn.Linear(d_model, 1)
        self.discrepancy_head = nn.Linear(d_model, 1)
        self.risk_head = nn.Linear(d_model, 1)

    def forward(self, x, return_attention=False):
        h = self.input_proj(x) + self.pos_embed
        if return_attention:
            attn_weights = []
            for layer in self.encoder.layers:
                h2, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
                attn_weights.append(w)
                h = layer.norm1(h + layer.dropout1(h2))
                h2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
                h = layer.norm2(h + layer.dropout2(h2))
        else:
            h = self.encoder(h)
        pooled = self.pool_norm(h.mean(dim=1))  # mean-pool across the event sequence
        out = dict(
            delay_logit=self.delay_head(pooled).squeeze(-1),
            damage_logit=self.damage_head(pooled).squeeze(-1),
            discrepancy_logit=self.discrepancy_head(pooled).squeeze(-1),
            risk_score=torch.sigmoid(self.risk_head(pooled).squeeze(-1)),
        )
        if return_attention:
            out["attention"] = attn_weights
        return out


def train_shipment_transformer(shipments: pd.DataFrame, epochs: int = 20, lr: float = 2e-3,
                                seed: int = 42, verbose: bool = True):
    torch.manual_seed(seed)
    seqs, labels = build_event_sequences(shipments, seed=seed)
    n = len(shipments)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X = torch.tensor(seqs)
    y_delay = torch.tensor(labels["delayed"])
    y_damage = torch.tensor(labels["damaged"])
    y_disc = torch.tensor(labels["doc_discrepancy"])
    y_risk = torch.tensor(labels["risk_score"])

    model = ShipmentRiskTransformer(n_features=seqs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce_delay = nn.BCEWithLogitsLoss()
    bce_damage = nn.BCEWithLogitsLoss()
    bce_disc = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Xtr, Xte = X[train_idx], X[test_idx]
    batch_size = 256
    n_train = len(train_idx)

    for epoch in range(epochs):
        model.train()
        bperm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            b = train_idx[bperm[i:i + batch_size].numpy()]
            opt.zero_grad()
            out = model(X[b])
            loss = (1.5 * bce_delay(out["delay_logit"], y_delay[b]) + bce_damage(out["damage_logit"], y_damage[b])
                    + bce_disc(out["discrepancy_logit"], y_disc[b]) + mse(out["risk_score"], y_risk[b]))
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        if verbose and (epoch % 4 == 0 or epoch == epochs - 1):
            print(f"  [Transformer] epoch {epoch:3d}  loss={epoch_loss / n_train:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(X[test_idx])
        delay_prob = torch.sigmoid(out["delay_logit"]).numpy()
        damage_prob = torch.sigmoid(out["damage_logit"]).numpy()
        disc_prob = torch.sigmoid(out["discrepancy_logit"]).numpy()
        y_delay_test = y_delay[test_idx].numpy()
        y_damage_test = y_damage[test_idx].numpy()
        y_disc_test = y_disc[test_idx].numpy()

    metrics = dict(
        delay_auc=float(roc_auc_score(y_delay_test, delay_prob)) if y_delay_test.sum() > 0 else float("nan"),
        delay_brier=float(brier_score_loss(y_delay_test, delay_prob)),
        damage_auc=float(roc_auc_score(y_damage_test, damage_prob)) if y_damage_test.sum() > 0 else float("nan"),
        doc_discrepancy_auc=float(roc_auc_score(y_disc_test, disc_prob)) if y_disc_test.sum() > 0 else float("nan"),
    )
    return model, metrics


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    shipments = gen.generate_shipments(n=6000)
    model, metrics = train_shipment_transformer(shipments, epochs=35, lr=1e-3)
    print("Metrics:", metrics)
