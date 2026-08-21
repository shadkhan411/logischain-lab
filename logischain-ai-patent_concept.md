# Patent Concept Document — LogisChain AI

**Deliverable D2.5.3.** All IP arising from this project is the exclusive
property of Zetheta Algorithms Private Limited, per the project brief's IP
Protection Protocol.

---

## 1. Problem Statement

Financial institutions that finance trade and supply chains — trade finance
desks, working-capital lenders, supply chain finance (SCF) platforms, cargo
insurers, and credit analysts — price and monitor risk almost entirely from
*financial* data (financial statements, historical default rates, credit
ratings). The *operational* data that determines whether the underlying
goods movement will actually succeed — vessel position, port congestion,
on-time-in-full (OTIF) performance, lead-time variance, carrier reliability —
is generated continuously by the supply chain itself but is not
systematically fed into financial risk models. This leaves financial
institutions blind to operational deterioration until it has already become
financial deterioration (a missed covenant, a defaulted LC, a claims spike),
by which point the loss has typically already been incurred.

## 2. Prior Art Analysis

- **Traditional credit risk models** (logistic regression / gradient-boosted
  trees on financial ratios) are well established (e.g. Altman Z-score
  successors, Moody's RiskCalc-style models) but use financial statement
  data with a 45-90 day publication lag and no operational inputs.
- **Supply chain visibility platforms** (e.g. project44, FourKites,
  MarineTraffic) provide operational tracking (ETAs, congestion) but do not
  translate that data into financial risk scores, pricing, or regulatory
  capital inputs.
- **Alternative-data credit scoring** (e.g. satellite imagery for commodity
  traders, utility-payment data for consumer credit) exists in adjacent
  domains but no publicly documented system combines (a) a heterogeneous
  graph-neural-network representation of the *multi-tier* supply chain
  network, (b) temporal convolutional forecasting of the *specific*
  operational metrics that drive working-capital requirements, and (c) a
  closed-form, auditable fusion layer translating both into standard credit
  risk quantities (PD, LGD, EAD, CCC) usable in an SR 11-7-style regulatory
  model governance framework.

## 3. Novel Contribution

LogisChain AI's contribution is the **fusion layer** connecting independently
well-understood techniques into a single, auditable pipeline:

1. **Heterogeneous graph attention network (HetGAT) risk propagation across
   a multi-tier trade network.** Node types (supplier, manufacturer,
   logistics provider, port, financial institution, customer) and typed
   edges (material flow, transportation, financial, ownership) with
   bidirectional message passing let a counterparty's risk score reflect
   both its own financials *and* its network position (Tier-2 supplier
   exposure, port concentration, customer concentration) — quantities no
   financial-statement-only model can see.

2. **Closed-form, white-box Supply-Chain-Adjusted PD (SC-PD).** Rather than
   treating the graph/temporal model outputs as an opaque uplift, LogisChain
   AI expresses the adjustment as an auditable formula:

   ```
   SC-PD = PD_traditional x (1 + 0.3 x OTIF_adj + 0.2 x InvTurnover_adj + 0.15 x NetworkResilience_adj)
   ```

   This preserves regulatory explainability (a requirement under Basel
   III/IV and SR 11-7 model-risk-management guidance) while still capturing
   the incremental signal from operational data — a middle ground between
   uninterpretable black-box uplift and no supply-chain integration at all.

3. **Leading-indicator Cash Conversion Cycle (CCC) forecasting with
   component decomposition.** Predicting DIO/DSO/DPO changes *separately*
   from observable operational deltas (ΔOTIF, Δlead-time variance, Δfreight
   cost, Δport congestion), then summing to a CCC forecast, gives lenders an
   early-warning covenant-breach signal 30-90 days before it would appear in
   quarterly financials — with a decomposition that shows *which* operational
   driver is responsible, supporting a targeted remediation conversation
   (e.g. "your DIO is rising because of Supplier X's lead-time variance")
   rather than a generic downgrade notice.

4. **Physical-financial cross-reference fraud detection.** Cross-referencing
   trade finance documentation (bills of lading, warehouse receipts) against
   independent physical evidence (AIS vessel tracking, IoT sensor feeds)
   detects document-collateral mismatches — phantom shipments, double-pledged
   warehouse receipts — that periodic (quarterly/semi-annual) physical
   inspection cannot catch between inspection dates. (Reference case:
   Section C5's Qingdao Port warehouse-receipt fraud, where the same physical
   copper stockpile was pledged to 5+ banks between inspections.)

## 4. Technical Claims

1. A method for computing a supply-chain-adjusted probability of default by
   applying a parameterised multiplicative adjustment to a baseline
   financial-statement-derived PD, where the adjustment factors are each a
   function of the deviation of a live operational metric (on-time-in-full
   rate, inventory turnover, count of qualified alternative suppliers) from
   a covenant or industry-benchmark threshold.
2. A method for forecasting a borrower's cash conversion cycle by
   separately predicting the components (days inventory outstanding, days
   sales outstanding, days payable outstanding) as functions of forecasted
   *changes* in upstream operational signals (lead-time variance, port
   congestion, freight cost), and aggregating the component forecasts into a
   covenant-breach probability with a specified lead time.
3. A heterogeneous graph neural network architecture for financial entity
   risk scoring in which node types correspond to supply-chain-participant
   roles (supplier, manufacturer, logistics provider, port/terminal,
   financial institution, end customer), edge types correspond to distinct
   flow relationships (material, transportation, financial, ownership), and
   a per-node learnable identity embedding is concatenated with financial
   and operational node features prior to message passing to preserve
   entity-specific structural signal through multiple aggregation layers.
4. A method for detecting trade-finance collateral fraud by computing an
   anomaly score from the discrepancy between (a) a claimed physical
   quantity/location asserted in a financing document (warehouse receipt,
   bill of lading) and (b) an independently observed physical quantity or
   location derived from vessel-tracking or IoT sensor data for the same
   asset and time window.

## 5. Commercial Application

- **Trade finance desks**: LC risk-adjusted pricing and document-discrepancy
  prediction at underwriting time.
- **Supply chain finance platforms**: per-supplier discount-rate setting
  and concentration-limit monitoring (directly responsive to the Greensill
  Capital failure mode analysed in Section C3 — concentration and
  phantom-receivable risk that a physical-financial cross-reference engine
  would have flagged).
- **Working capital lenders**: covenant early-warning 30-90 days ahead of
  breach, enabling proactive facility restructuring instead of reactive
  technical default (Section A6.4's MedDevice Corp scenario).
- **Cargo / trade disruption insurers**: dynamic, shipment-specific premium
  adjustment from live weather/congestion/carrier-reliability signals
  (Section A6.5).
