# LogisChain AI — Architecture

## 1. System overview

```
                    ┌─────────────────────────────────────────────┐
                    │            PHYSICAL DATA LAYER                │
                    │  AIS · customs · IoT · ERP · port · freight    │
                    │  (synthetic generator — see §2 below)          │
                    └───────────────────┬─────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                                │
┌───────▼────────┐          ┌──────────▼──────────┐          ┌─────────▼─────────┐
│   GNN (HetGAT)   │          │   TCN (dilated causal) │          │  Transformer (risk) │
│ entity risk       │          │ demand / throughput    │          │ shipment delay /     │
│ embeddings 128-d   │          │ forecasts, 30/60/90d   │          │ damage / discrepancy │
└───────┬────────┘          └──────────┬──────────┘          └─────────┬─────────┘
        │                               │                                │
        │                    ┌──────────▼──────────┐                    │
        │                    │  XGBoost (tabular)     │                    │
        │                    │ supplier default PD     │                    │
        │                    └──────────┬──────────┘                    │
        │                               │                                │
        │                    ┌──────────▼──────────┐                    │
        │                    │  Cox PH / DeepSurv      │                    │
        │                    │ default-timing hazard   │                    │
        │                    └──────────┬──────────┘                    │
        │                               │                                │
        └───────────────┬───────────────┴───────────────┬────────────────┘
                        │                               │
              ┌──────────▼──────────────────────────────▼──────────┐
              │      LEVEL-1 STACKING META-LEARNER (LightGBM)         │
              │  -> Trade Finance Default Prediction (D2.3.1)          │
              └──────────────────────┬──────────────────────────────┘
                                     │
      ┌──────────────────────────────┼──────────────────────────────┐
      │                              │                              │
┌──────▼──────┐          ┌───────────▼───────────┐          ┌─────────▼─────────┐
│ CCC / Working │          │  SC-PD Credit Risk       │          │  LogisChain Lab      │
│ Capital        │          │  Scorer (SHAP-explained) │          │  Simulation (2 modes) │
│ Predictor      │          │                          │          │                       │
└───────────────┘          └───────────────────────┘          └───────────────────────┘
```

## 2. Data strategy — why synthetic, and how it's kept honest

The brief's Section A4 data sources (UN Global Platform AIS, MarineTraffic,
Kpler, UN Comtrade, ICC Trade Register, SWIFT, Bloomberg/Refinitiv, ...) all
require paid subscriptions or registered API keys that aren't reachable from
this build environment's network allowlist. Rather than fabricate results
against data we don't have, `src/data/synthetic_generator.py` implements an
explicit **data-generating process (DGP)** that encodes the brief's own
causal chain (Section A8.4): *operational degradation → financial stress →
default*. Concretely:

- Node-level default probability is a logistic function of OTIF, leverage,
  inventory turnover, CCC, supplier concentration, lead-time variance,
  network centrality, and geopolitical/disaster risk — the same variables
  the brief lists as SHAP drivers in Section A8.2's worked example.
- Shipment delay/damage/discrepancy probabilities are logistic functions of
  vessel-speed deviation, port congestion, carrier reliability, and weather.
- LC default probability is a logistic function of applicant leverage,
  beneficiary OTIF, historical discrepancy rates, route risk, and currency
  volatility.

This means every model in the pipeline is learning a **real, recoverable
signal** rather than fitting noise — which is what lets the "does adding
supply-chain features improve on financial-only models" comparisons (the
project's central thesis) produce an honest answer instead of a foregone one.

Real API integration is a straightforward swap later: replace
`SupplyChainDataGenerator` with connectors to UN Global Platform AIS / UN
Comtrade / ICC Trade Register once credentials are available; every
downstream feature/model module consumes the same `nodes` / `edges` /
`time_series` / `shipments` / `lc_transactions` DataFrame schemas either way.

## 3. Model-by-model notes, including honest limitations

| Model | Target | Result | Brief's target | Met? |
|---|---|---|---|---|
| GNN node classification | Risk tier (3-way) | 77-83% accuracy | >70% | Yes |
| GNN link prediction | Edge existence | ~0.64 AUC | >0.75 | Partial (see below) |
| TCN (port throughput) | 30d value | 6.5% MAPE | <12% | Yes |
| TCN (freight rate) | 30d value | 15.4% MAPE | <12% | Partial (see below) |
| Transformer (delay) | Binary | ~0.68 AUC, 0.078 Brier | AUC>0.80, Brier<0.18 | Partial / Yes |
| XGBoost (supplier default) | Binary | 0.86-0.90 AUC, Gini 0.72-0.80 | Gini>0.55 | Yes |
| Cox PH (survival) | Time-to-default | 0.81 c-index, +0.13 vs financial-only | c-index>0.80, delta>0.05 | Yes |
| DeepSurv (survival) | Time-to-default | unstable (5-fold CV, high variance) | -- | Partial (see below) |
| Trade finance stacking ensemble | Binary | Gini approx 0.46-0.55 | Gini>0.55 | Borderline |
| SC-PD credit scorer | Binary | +0.33 AUC vs financial-only | delta>0.05 | Yes (large margin) |

**Why link prediction underperforms**: edges in the synthetic graph are
assigned independently of node features (a manufacturer's suppliers are
drawn uniformly at random), mirroring the fact that *which* counterparties
trade with each other in the real world depends on relationship history the
feature vector doesn't encode. A learnable per-node identity embedding
(`id_embed` in `HetGAT`) was added specifically to help the model memorise
structural roles, and it materially improved the metric (from ~0.50 to
~0.64), but full recovery would need either (a) real edges with genuine
feature-driven homophily, or (b) additional structural features like common-
neighbour counts as explicit link-prediction inputs.

**Why freight-rate and delay-probability forecasts underperform their
targets**: both are driven by discrete shock events (congestion spikes,
weather) layered on top of smoother baseline dynamics — inherently noisier
than throughput, which is dominated by smooth seasonal/trend structure. A
plain logistic-regression baseline on the *undiluted* shipment features
(bypassing the Transformer's sequence encoding entirely) tops out at ~0.70
AUC on this synthetic DGP, confirming the ceiling is a property of the
injected noise level, not an architecture deficiency.

**Why DeepSurv is unstable**: the synthetic default rate (~7%) over ~217
nodes means a 20%-held-out test fold contains only a handful of actual
default *events* — and the concordance index is only computed over
event-involving comparable pairs, so a single split can swing from ~0.2 to
~0.8 purely from which few defaulters land in the fold. We use 5-fold CV and
report the mean +/- std honestly rather than cherry-picking a favourable
split. Cox PH (a lower-variance linear model) is the more reliable estimator
at this data scale — a legitimate, general lesson about deep models needing
more data than simpler ones to reliably outperform them, not a bug.

## 4. LogisChain Lab simulation architecture

Three layers (Section B1.2), implemented in `src/simulation/`:

- **Physical Layer** (`engine.py`): port congestion + freight-rate state,
  updated by scenario triggers (`scenarios.py`, 10 scenario types).
- **Intelligence Layer** (`engine.intelligence_signals()`): computes SC-PD
  and TRFSI live from current physical state, using the exact fusion-feature
  formulas from `src/features/financial_features.py`.
- **Financial Layer** (`game_modes.py`): two mandatory modes (Trade Finance
  Portfolio Management, Supply Chain Finance Pricing), each run under both
  an `"sc_aware"` and a `"passive"` decision policy over an *identical*
  scenario sequence — isolating the value of supply-chain-intelligence
  integration exactly as the brief's B1.1 narrative claims.

Scoring (`scoring.py`) implements the 1000-point, 5-dimension framework from
Section B2.1 exactly, including the certification-tier lookup table.

**Known simplification**: this is an automated agent-vs-agent comparison
(both policies are algorithmic), not a human-in-the-loop UI — "Decision
Speed" and "Learning Progression" scores are computed from the *policy's*
reaction pattern rather than a real user's clicks. A human-playable frontend
(e.g. a Streamlit app driving the same `TradeFinancePortfolioMode.run()`
turn-by-turn) is a natural next step.

## 5. Reproducibility

Every stochastic component takes an explicit `seed`. `demo/run_pipeline.py`
runs the entire system end-to-end and writes real, reproduced metrics to
`results/pipeline_results.json` — the numbers in `README.md` are generated
from that file, not hand-typed.
