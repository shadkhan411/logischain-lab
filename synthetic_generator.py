"""
Synthetic Data Generator for LogisChain AI
============================================
Generates realistic, internally-consistent supply chain + trade finance data:

  1. A multi-relational supply chain knowledge graph (Section A3.1 node/edge schema)
  2. Daily time series for port throughput / freight rates (Section A3.2 TCN inputs)
  3. Shipment-level event sequences (Section A3.3 Transformer inputs)
  4. Letter-of-Credit trade finance transactions with default outcomes (Section A5.3)
  5. Supplier survival data (duration/event) for Cox PH / DeepSurv (Section A3.4)

Why synthetic data: the project brief's real sources (UN Global Platform AIS,
MarineTraffic, UN Comtrade, ICC Trade Register, SWIFT, Bloomberg, etc.) require
paid/registered API access that is not reachable from this build environment.
The generator implements a *data generating process* that mirrors the causal
chain documented in the brief (operational degradation -> financial stress ->
default, Section A8.4) so that supply-chain features carry genuine predictive
signal -- letting every downstream model honestly demonstrate the "SC features
improve financial risk prediction" hypothesis rather than faking metrics.

Everything is seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


COUNTRIES = [
    ("India", 62, 0.35, 0.22), ("China", 72, 0.20, 0.30), ("Vietnam", 58, 0.30, 0.18),
    ("Mexico", 58, 0.15, 0.20), ("Germany", 88, 0.05, 0.08), ("USA", 84, 0.18, 0.10),
    ("Brazil", 55, 0.25, 0.24), ("Bangladesh", 48, 0.40, 0.20), ("Turkey", 52, 0.12, 0.35),
    ("South Korea", 82, 0.10, 0.28), ("Indonesia", 56, 0.35, 0.15), ("Poland", 74, 0.08, 0.15),
]

PORTS = [
    "Shanghai", "Singapore", "Rotterdam", "Los Angeles", "Mumbai (JNPT)", "Manzanillo",
    "Hamburg", "Manila", "Santos", "Busan", "Felixstowe", "Jebel Ali",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SupplyChainDataGenerator:
    n_suppliers: int = 90
    n_manufacturers: int = 50
    n_logistics: int = 20
    n_ports: int = 12
    n_financial_institutions: int = 5
    n_customers: int = 40
    seed: int = 42
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # 1. SUPPLY CHAIN KNOWLEDGE GRAPH  (Section A3.1)
    # ------------------------------------------------------------------
    def generate_graph(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = self.rng
        rows = []
        node_id = 0

        def country_draw():
            return COUNTRIES[rng.integers(0, len(COUNTRIES))]

        specs = [
            ("supplier", self.n_suppliers),
            ("manufacturer", self.n_manufacturers),
            ("logistics_provider", self.n_logistics),
            ("port", self.n_ports),
            ("financial_institution", self.n_financial_institutions),
            ("customer", self.n_customers),
        ]

        for node_type, n in specs:
            for i in range(n):
                country, c_risk, disaster, geo_risk = country_draw()
                current_ratio = np.clip(rng.normal(1.5, 0.5), 0.4, 4.0)
                debt_to_equity = np.clip(rng.gamma(2.0, 0.4), 0.05, 5.0)
                ebitda_margin = np.clip(rng.normal(0.13, 0.06), -0.1, 0.4)
                interest_coverage = np.clip(rng.gamma(3.0, 1.5), 0.2, 25)
                revenue = float(np.round(rng.lognormal(mean=4.2, sigma=1.1) * 1e6, -3))
                otif = np.clip(rng.normal(0.90, 0.07), 0.45, 0.995)
                lead_time_mean = np.clip(rng.normal(14, 5), 2, 60)
                lead_time_std = np.clip(rng.gamma(2.0, 1.6), 0.3, 20)
                inv_turnover = np.clip(rng.normal(6.5, 2.2), 1.0, 20)
                fill_rate = np.clip(rng.normal(0.93, 0.05), 0.5, 0.999)
                dio = np.clip(365 / max(inv_turnover, 0.1), 5, 200)
                dso = np.clip(rng.normal(48, 15), 5, 150)
                dpo = np.clip(rng.normal(45, 14), 5, 150)
                ccc = dio + dso - dpo
                supplier_hhi = np.clip(rng.beta(2, 5), 0.05, 1.0)
                customer_hhi = np.clip(rng.beta(2, 5), 0.05, 1.0)
                capacity_util = np.clip(rng.normal(0.75, 0.15), 0.2, 0.99)
                reliability = np.clip(rng.normal(0.88, 0.08), 0.4, 0.999)
                freight_cost_ratio = np.clip(rng.normal(0.11, 0.04), 0.02, 0.35)

                rows.append(dict(
                    node_id=f"{node_type[:4].upper()}_{i:04d}", node_type=node_type,
                    country=country, country_risk_score=c_risk,
                    natural_disaster_exposure=disaster, geopolitical_risk_score=geo_risk,
                    current_ratio=current_ratio, debt_to_equity=debt_to_equity,
                    ebitda_margin=ebitda_margin, interest_coverage=interest_coverage,
                    revenue=revenue, otif_rate=otif, lead_time_mean=lead_time_mean,
                    lead_time_std=lead_time_std, inventory_turnover=inv_turnover,
                    fill_rate=fill_rate, dio=dio, dso=dso, dpo=dpo, ccc_days=ccc,
                    supplier_concentration_hhi=supplier_hhi,
                    customer_concentration_hhi=customer_hhi,
                    capacity_utilisation=capacity_util, reliability_score=reliability,
                    freight_cost_ratio=freight_cost_ratio,
                ))
                node_id += 1

        nodes = pd.DataFrame(rows)

        # ---- Edges -------------------------------------------------
        edges = []
        suppliers = nodes[nodes.node_type == "supplier"].node_id.tolist()
        manufacturers = nodes[nodes.node_type == "manufacturer"].node_id.tolist()
        logistics = nodes[nodes.node_type == "logistics_provider"].node_id.tolist()
        ports = nodes[nodes.node_type == "port"].node_id.tolist()
        fis = nodes[nodes.node_type == "financial_institution"].node_id.tolist()
        customers = nodes[nodes.node_type == "customer"].node_id.tolist()
        modes = ["ocean", "air", "rail", "road"]

        # material flow: supplier -> manufacturer (each manufacturer sources from 3-8 suppliers)
        for m in manufacturers:
            k = rng.integers(3, 9)
            chosen = rng.choice(suppliers, size=k, replace=False)
            shares = rng.dirichlet(np.ones(k))
            for s, share in zip(chosen, shares):
                edges.append(dict(src=s, dst=m, edge_type="material_flow",
                                   volume=float(rng.integers(500, 50000)),
                                   value=float(np.round(share * rng.lognormal(3.5, 1.0) * 1e5, 0)),
                                   mode=None, transit_time_days=None, cost=None,
                                   payment_terms_days=None, outstanding_balance=None))

        # material flow: manufacturer -> customer
        for c in customers:
            k = rng.integers(2, 5)
            chosen = rng.choice(manufacturers, size=k, replace=False)
            for m in chosen:
                edges.append(dict(src=m, dst=c, edge_type="material_flow",
                                   volume=float(rng.integers(200, 20000)),
                                   value=float(np.round(rng.lognormal(3.2, 1.0) * 1e5, 0)),
                                   mode=None, transit_time_days=None, cost=None,
                                   payment_terms_days=None, outstanding_balance=None))

        # transportation edges: supplier/manufacturer -> port -> port -> customer region (simplified: node -> port, port -> node)
        transport_capable = suppliers + manufacturers
        for n in transport_capable:
            k = rng.integers(2, 4)
            chosen = rng.choice(ports, size=k, replace=False)
            for p in chosen:
                mode = modes[rng.integers(0, len(modes))]
                edges.append(dict(src=n, dst=p, edge_type="transportation", volume=None, value=None,
                                   mode=mode, transit_time_days=float(np.clip(rng.normal(9, 4), 1, 40)),
                                   cost=float(np.round(rng.lognormal(6, 0.6), 0)),
                                   payment_terms_days=None, outstanding_balance=None))
        for c in customers:
            k = rng.integers(2, 4)
            chosen = rng.choice(ports, size=k, replace=False)
            for p in chosen:
                mode = modes[rng.integers(0, len(modes))]
                edges.append(dict(src=p, dst=c, edge_type="transportation", volume=None, value=None,
                                   mode=mode, transit_time_days=float(np.clip(rng.normal(7, 3), 1, 35)),
                                   cost=float(np.round(rng.lognormal(5.8, 0.6), 0)),
                                   payment_terms_days=None, outstanding_balance=None))
        # logistics providers serve manufacturers <-> ports
        for lp in logistics:
            k = rng.integers(5, 14)
            chosen = rng.choice(manufacturers, size=k, replace=False)
            for m in chosen:
                edges.append(dict(src=lp, dst=m, edge_type="transportation", volume=None, value=None,
                                   mode=modes[rng.integers(0, len(modes))],
                                   transit_time_days=float(np.clip(rng.normal(6, 2.5), 1, 25)),
                                   cost=float(np.round(rng.lognormal(5.5, 0.5), 0)),
                                   payment_terms_days=None, outstanding_balance=None))

        # financial edges: financial_institution <-> manufacturer/supplier (trade finance exposure)
        for fi in fis:
            k = rng.integers(20, 45)
            chosen = rng.choice(manufacturers + suppliers, size=k, replace=False)
            for cnode in chosen:
                edges.append(dict(src=fi, dst=cnode, edge_type="financial", volume=None,
                                   value=float(np.round(rng.lognormal(12, 1.1), 0)),
                                   mode=None, transit_time_days=None, cost=None,
                                   payment_terms_days=float(rng.choice([30, 45, 60, 90, 120])),
                                   outstanding_balance=float(np.round(rng.lognormal(11.5, 1.0), 0))))

        # ownership edges: sparse, manufacturer -> supplier (parent-subsidiary), ~8% of suppliers
        n_own = max(1, int(0.08 * len(suppliers)))
        owned_suppliers = rng.choice(suppliers, size=n_own, replace=False)
        for s in owned_suppliers:
            parent = manufacturers[rng.integers(0, len(manufacturers))]
            edges.append(dict(src=parent, dst=s, edge_type="ownership", volume=None, value=None,
                               mode=None, transit_time_days=None, cost=None,
                               payment_terms_days=None, outstanding_balance=None))

        edges = pd.DataFrame(edges)

        # ---- Network position features (via networkx) --------------
        G = nx.DiGraph()
        G.add_nodes_from(nodes.node_id.tolist())
        G.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))
        deg_in = dict(G.in_degree())
        deg_out = dict(G.out_degree())
        # betweenness on a (fast, approximate) subsample for large graphs
        betweenness = nx.betweenness_centrality(G, k=min(80, G.number_of_nodes()), seed=self.seed)
        clustering = nx.clustering(G.to_undirected())
        pagerank = nx.pagerank(G, alpha=0.85)

        nodes["in_degree"] = nodes.node_id.map(deg_in).fillna(0)
        nodes["out_degree"] = nodes.node_id.map(deg_out).fillna(0)
        nodes["betweenness_centrality"] = nodes.node_id.map(betweenness).fillna(0)
        nodes["clustering_coefficient"] = nodes.node_id.map(clustering).fillna(0)
        nodes["pagerank"] = nodes.node_id.map(pagerank).fillna(0)

        # ---- Ground-truth default generating process (Section A8.4 causal chain) ----
        z = (
            -4.35
            + 1.35 * nodes.debt_to_equity
            - 1.55 * (nodes.current_ratio - 1.0)
            - 3.1 * (nodes.otif_rate - 0.90)
            - 0.22 * (nodes.inventory_turnover - 6.5)
            + 0.018 * (nodes.ccc_days - 55)
            + 1.9 * (nodes.supplier_concentration_hhi - 0.3)
            + 0.06 * (nodes.lead_time_std - 4.0)
            - 2.6 * nodes.ebitda_margin
            + 2.2 * nodes.betweenness_centrality
            + 0.9 * nodes.geopolitical_risk_score
            + 0.55 * nodes.natural_disaster_exposure
        )
        z += self.rng.normal(0, 0.30, size=len(nodes))  # idiosyncratic noise
        pd_annual = _sigmoid(z)
        nodes["pd_annual_true"] = pd_annual
        nodes["default_12m"] = (self.rng.uniform(size=len(nodes)) < pd_annual).astype(int)

        # survival duration: exponential-ish time-to-default within a 2-year (730 day) horizon
        hazard_scale = np.clip(pd_annual, 1e-4, 0.99)
        raw_time = self.rng.exponential(scale=(365.0 / np.maximum(hazard_scale, 1e-3)))
        duration = np.minimum(raw_time, 730)
        event = (raw_time <= 730).astype(int)
        nodes["survival_duration_days"] = duration.round(1)
        nodes["survival_event"] = event

        return nodes, edges

    # ------------------------------------------------------------------
    # 2. TIME SERIES (Section A3.2 TCN inputs)  -- port throughput & freight rate
    # ------------------------------------------------------------------
    def generate_time_series(self, n_days: int = 730, start: str = "2023-01-01") -> pd.DataFrame:
        rng = self.rng
        dates = pd.date_range(start=start, periods=n_days, freq="D")
        frames = []
        for p_idx, port in enumerate(PORTS[: self.n_ports]):
            base = rng.uniform(80_000, 150_000)
            t = np.arange(n_days)
            annual = 0.10 * base * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0, 2 * np.pi))
            weekly = 0.04 * base * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 2 * np.pi))
            trend = base * 0.00008 * t
            noise = rng.normal(0, base * 0.02, size=n_days)
            # random disruption shocks (congestion events)
            throughput = base + annual + weekly + trend + noise
            n_shocks = rng.integers(2, 6)
            congestion_index = np.clip(rng.normal(1.8, 0.5, size=n_days), 0.2, 5.0)
            for _ in range(n_shocks):
                shock_start = rng.integers(0, n_days - 20)
                shock_len = rng.integers(5, 20)
                depth = rng.uniform(0.15, 0.45)
                idx = slice(shock_start, min(shock_start + shock_len, n_days))
                throughput[idx] *= (1 - depth)
                congestion_index[idx] = np.clip(congestion_index[idx] + depth * 6, 0.2, 5.0)
            freight_rate = np.clip(
                1500 + 300 * congestion_index ** 1.4 + rng.normal(0, 150, size=n_days)
                + 400 * np.sin(2 * np.pi * t / 365.25 + 1.0), 400, 15000
            )
            frames.append(pd.DataFrame({
                "date": dates, "port": port,
                "throughput_teu": np.round(throughput, 0),
                "port_congestion_index": np.round(congestion_index, 2),
                "freight_rate_usd_feu": np.round(freight_rate, 0),
            }))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. SHIPMENT EVENT SEQUENCES (Section A3.3 Transformer inputs)
    # ------------------------------------------------------------------
    EVENT_SEQUENCE = ["booking_confirmed", "container_loaded", "vessel_departed",
                       "transhipment_arrival", "transhipment_departure",
                       "destination_port_arrival", "customs_cleared", "final_delivery"]

    def generate_shipments(self, n: int = 6000, start: str = "2023-01-01", n_days: int = 730) -> pd.DataFrame:
        rng = self.rng
        rows = []
        for i in range(n):
            origin = PORTS[rng.integers(0, len(PORTS))]
            dest = PORTS[rng.integers(0, len(PORTS))]
            while dest == origin:
                dest = PORTS[rng.integers(0, len(PORTS))]
            depart_day = rng.integers(0, n_days - 60)
            vessel_speed_pct_of_plan = np.clip(rng.normal(1.0, 0.08), 0.6, 1.25)
            origin_congestion = np.clip(rng.normal(2.0, 0.9), 0.1, 5.0)
            dest_congestion = np.clip(rng.normal(2.0, 0.9), 0.1, 5.0)
            carrier_reliability = np.clip(rng.normal(0.87, 0.09), 0.3, 0.999)
            cargo_value = float(np.round(rng.lognormal(11.5, 1.1), 0))
            weather_risk = np.clip(rng.beta(1.5, 6), 0, 1)
            base_transit = rng.uniform(12, 38)

            delay_z = (
                -2.9
                + 1.6 * (1 - vessel_speed_pct_of_plan)
                + 0.55 * (dest_congestion - 2.0)
                + 0.35 * (origin_congestion - 2.0)
                + 1.3 * (1 - carrier_reliability)
                + 1.8 * weather_risk
                + rng.normal(0, 0.32)
            )
            delay_prob = _sigmoid(delay_z)
            delayed = int(rng.uniform() < delay_prob)
            delay_days = float(np.round(max(0, rng.gamma(2.0, 3.0)) * delayed, 1))

            damage_z = -3.4 + 1.6 * weather_risk + 0.9 * (1 - carrier_reliability) + rng.normal(0, 0.4)
            damaged = int(rng.uniform() < _sigmoid(damage_z))

            discrepancy_z = -2.2 + 0.8 * delayed + 0.6 * (dest_congestion - 2.0) / 2 + rng.normal(0, 0.5)
            doc_discrepancy = int(rng.uniform() < _sigmoid(discrepancy_z))

            risk_score = float(np.round(100 * _sigmoid(0.9 * delay_z + 0.4 * damage_z + 0.3 * discrepancy_z - 1.0), 1))

            rows.append(dict(
                shipment_id=f"SHP_{i:06d}", origin_port=origin, destination_port=dest,
                depart_date=pd.Timestamp(start) + pd.Timedelta(days=int(depart_day)),
                planned_transit_days=round(base_transit, 1),
                vessel_speed_ratio=round(vessel_speed_pct_of_plan, 3),
                origin_congestion_index=round(origin_congestion, 2),
                destination_congestion_index=round(dest_congestion, 2),
                carrier_reliability_score=round(carrier_reliability, 3),
                cargo_value_usd=cargo_value, weather_risk_score=round(weather_risk, 3),
                delay_probability_true=round(delay_prob, 4), delayed=delayed, delay_days=delay_days,
                damaged=damaged, doc_discrepancy=doc_discrepancy, shipment_risk_score=risk_score,
            ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. TRADE FINANCE (LC) TRANSACTIONS (Section A5.3)
    # ------------------------------------------------------------------
    def generate_trade_finance_transactions(self, n: int = 9000, nodes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        rng = self.rng
        if nodes is None:
            nodes, _ = self.generate_graph()
        applicants = nodes[nodes.node_type.isin(["manufacturer", "customer"])].reset_index(drop=True)
        beneficiaries = nodes[nodes.node_type.isin(["supplier", "manufacturer"])].reset_index(drop=True)

        rows = []
        for i in range(n):
            a = applicants.iloc[rng.integers(0, len(applicants))]
            b = beneficiaries.iloc[rng.integers(0, len(beneficiaries))]
            lc_amount = float(np.round(rng.lognormal(13.2, 1.15), 0))
            tenor = int(rng.choice([30, 60, 90, 120, 180]))
            commodity_risk = rng.choice(["Low", "Medium", "High"], p=[0.35, 0.45, 0.20])
            route_risk = np.clip(rng.beta(2, 4), 0, 1)
            hist_discrepancy_applicant = np.clip(rng.beta(2, 5), 0, 1)
            hist_discrepancy_beneficiary = np.clip(rng.beta(1.5, 6), 0, 1)
            port_cong_origin = np.clip(rng.normal(2.0, 0.9), 0.1, 5.0)
            port_cong_dest = np.clip(rng.normal(2.0, 0.9), 0.1, 5.0)
            freight_pctile = rng.uniform(0, 1)
            seasonal_factor = np.clip(rng.normal(1.0, 0.12), 0.7, 1.5)
            country_risk_diff = a.country_risk_score - b.country_risk_score
            currency_vol = np.clip(rng.gamma(2.0, 0.04), 0.01, 0.4)
            commodity_risk_num = {"Low": 0, "Medium": 1, "High": 2}[commodity_risk]

            z = (
                -5.15
                + 0.85 * (a.debt_to_equity - 0.8)
                - 3.0 * (b.otif_rate - 0.90)
                + 1.9 * hist_discrepancy_applicant
                + 0.7 * commodity_risk_num
                + 1.1 * route_risk
                + 0.35 * (port_cong_dest - 2.0)
                - 0.02 * country_risk_diff
                + 2.6 * currency_vol
                + 0.015 * (tenor - 90)
                + rng.normal(0, 0.29)
            )
            default_prob = _sigmoid(z)
            default = int(rng.uniform() < default_prob)

            rows.append(dict(
                lc_id=f"LC_{i:06d}", applicant_id=a.node_id, beneficiary_id=b.node_id,
                applicant_country=a.country, beneficiary_country=b.country,
                lc_amount_usd=lc_amount, tenor_days=tenor, commodity_risk_category=commodity_risk,
                trade_route_risk_score=round(route_risk, 3),
                applicant_leverage=round(a.debt_to_equity, 3),
                applicant_current_ratio=round(a.current_ratio, 3),
                beneficiary_otif_rate=round(b.otif_rate, 4),
                historical_discrepancy_rate_applicant=round(hist_discrepancy_applicant, 4),
                historical_discrepancy_rate_beneficiary=round(hist_discrepancy_beneficiary, 4),
                port_congestion_origin=round(port_cong_origin, 2),
                port_congestion_destination=round(port_cong_dest, 2),
                freight_rate_percentile=round(freight_pctile, 3),
                seasonal_factor=round(seasonal_factor, 3),
                country_risk_differential=round(country_risk_diff, 1),
                currency_volatility_30d=round(currency_vol, 4),
                default_probability_true=round(default_prob, 5),
                default=default,
            ))
        df = pd.DataFrame(rows)
        return df


if __name__ == "__main__":
    gen = SupplyChainDataGenerator()
    nodes, edges = gen.generate_graph()
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(nodes.node_type.value_counts())
    print(f"Default rate (12m): {nodes.default_12m.mean():.3%}")
    ts = gen.generate_time_series()
    print(f"Time series rows: {len(ts)}")
    shp = gen.generate_shipments(n=2000)
    print(f"Shipments: {len(shp)}, delay rate: {shp.delayed.mean():.2%}")
    lc = gen.generate_trade_finance_transactions(n=3000, nodes=nodes)
    print(f"LC transactions: {len(lc)}, default rate: {lc.default.mean():.3%}")
