"""
Heterogeneous Graph Attention Network (HetGAT) - Section A3.1.

Node types (6, >= min 3 required): supplier, manufacturer, logistics_provider,
port, financial_institution, customer.

Edge relation categories (4, meeting the min-4 requirement): material_flow,
transportation, financial, ownership -- each instantiated across the specific
(src_type, dst_type) pairs that occur in the graph, plus reverse edges so risk
can propagate both up- and down-stream (Section A3.1 worked example: a
supplier's embedding is shaped by both its suppliers' AND its customers'
health).

Architecture: 3 layers of typed multi-head GAT message passing
(HeteroConv + GATConv per relation), producing a 128-dim per-entity risk
embedding. Trained multi-task: (a) link prediction (does an edge exist) and
(b) node classification (risk tier), exactly the two evaluation tasks
specified in deliverable D2.2.1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
from sklearn.metrics import roc_auc_score, accuracy_score

from src.features.graph_features import ENTITY_FEATURE_COLUMNS, build_entity_features, normalise_features


def build_hetero_data(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Convert flat nodes/edges DataFrames into a PyG HeteroData object."""
    feats, _ = normalise_features(build_entity_features(nodes))
    node_types = sorted(feats.node_type.unique().tolist())

    data = HeteroData()
    type_to_local_idx = {}  # node_id -> (type, local_idx)
    for ntype in node_types:
        sub = feats[feats.node_type == ntype].reset_index(drop=True)
        x = torch.tensor(sub[ENTITY_FEATURE_COLUMNS].values, dtype=torch.float32)
        data[ntype].x = x
        data[ntype].node_id = sub.node_id.tolist()
        for i, nid in enumerate(sub.node_id):
            type_to_local_idx[nid] = (ntype, i)

    node_type_map = nodes.set_index("node_id")["node_type"].to_dict()
    relation_groups: dict[tuple, list] = {}
    for row in edges.itertuples(index=False):
        src_type = node_type_map.get(row.src)
        dst_type = node_type_map.get(row.dst)
        if src_type is None or dst_type is None:
            continue
        rel = (src_type, row.edge_type, dst_type)
        relation_groups.setdefault(rel, []).append((row.src, row.dst))

    edge_type_names = set()
    for (src_type, etype, dst_type), pairs in relation_groups.items():
        src_idx = [type_to_local_idx[s][1] for s, d in pairs]
        dst_idx = [type_to_local_idx[d][1] for s, d in pairs]
        ei = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[(src_type, etype, dst_type)].edge_index = ei
        # reverse edge for bidirectional message passing (risk propagates both ways)
        rev_name = f"rev_{etype}"
        data[(dst_type, rev_name, src_type)].edge_index = ei.flip(0)
        edge_type_names.add(etype)
        edge_type_names.add(rev_name)

    return data, type_to_local_idx, node_type_map, sorted(edge_type_names)


class HetGAT(nn.Module):
    """3-layer heterogeneous Graph Attention Network -> 128-dim entity embeddings."""

    def __init__(self, metadata, in_channels: int, node_counts: dict[str, int],
                 hidden_channels: int = 64, out_channels: int = 128, heads: int = 4,
                 num_layers: int = 3, id_embed_dim: int = 16):
        super().__init__()
        node_types, edge_types = metadata
        # Learnable per-node "identity" embedding, concatenated with the raw financial/
        # operational features. Our synthetic graph's edges are assigned independently of
        # node features (mirroring the fact that *which* counterparties trade with each
        # other is driven by relationships the feature vector doesn't encode) -- so pure
        # feature-based message passing has almost no signal to separate a true edge from
        # a random one. A small learnable ID embedding lets the model memorise
        # node-specific structural roles (the standard fix for transductive link
        # prediction on graphs without strong feature homophily).
        self.id_embed = nn.ModuleDict({
            nt: nn.Embedding(node_counts[nt], id_embed_dim) for nt in node_types
        })
        self.lin_in = nn.ModuleDict({
            nt: Linear(in_channels + id_embed_dim, hidden_channels) for nt in node_types
        })
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            is_last = layer == num_layers - 1
            out_dim = out_channels if is_last else hidden_channels
            out_heads = 1 if is_last else heads
            # GATConv concatenates heads by default, so per-head width must be
            # out_dim // out_heads for the concatenated output to equal out_dim.
            conv_out = out_dim // out_heads
            conv = HeteroConv({
                et: GATConv((-1, -1), conv_out, heads=out_heads, add_self_loops=False, dropout=0.1)
                for et in edge_types
            }, aggr="sum")
            self.convs.append(conv)
            # LayerNorm per node type per layer: residual stacking without normalisation
            # lets embedding magnitude grow layer over layer, which saturates the
            # dot-product link-prediction head's sigmoid and stalls its gradient.
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(out_dim) for nt in node_types}))
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        for nt, x in x_dict.items():
            ids = torch.arange(x.shape[0], device=x.device)
            x_aug = torch.cat([x, self.id_embed[nt](ids)], dim=-1)
            h_dict[nt] = F.relu(self.lin_in[nt](x_aug))
        for i, conv in enumerate(self.convs):
            out_dict = conv(h_dict, edge_index_dict)
            # residual connection: heterogeneous bipartite GATConv has no notion of
            # self-loops across differing (src,dst) node types, so without an explicit
            # skip connection each node's own features are diluted after every layer
            # of pure neighbour aggregation. This keeps each entity's own signal alive
            # alongside the network-context signal (Section A3.1 "own risk plus network
            # context risk").
            new_h_dict = {}
            for nt, h_prev in h_dict.items():
                h_new = out_dict.get(nt, torch.zeros_like(h_prev) if h_prev.shape[-1] == h_prev.shape[-1] else h_prev)
                if nt not in out_dict:
                    h_new = h_prev  # isolated node type for this relation set this layer
                if h_new.shape == h_prev.shape:
                    h_new = h_new + h_prev
                h_new = self.norms[i][nt](h_new)
                new_h_dict[nt] = F.relu(h_new) if i < self.num_layers - 1 else h_new
            h_dict = new_h_dict
        return h_dict


class RiskHeads(nn.Module):
    """Multi-task heads on top of GNN embeddings: node classification + link prediction.

    Node classification concatenates the node's own (normalised) input features with
    its graph embedding -- mirroring the brief's framing that "the final embedding ...
    captures Node A's own risk plus its network context risk" (Section A3.1). Without
    this, a bipartite heterogeneous GAT with no true self-loops has to *reconstruct*
    self-information purely through message passing, which is a much harder learning
    problem than necessary.
    """

    def __init__(self, embed_dim: int = 128, in_channels: int = 21, n_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + in_channels, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, n_classes),
        )

    def node_logits(self, h, x_raw):
        return self.classifier(torch.cat([h, x_raw], dim=-1))

    @staticmethod
    def link_score(h_src, h_dst):
        dim = h_src.shape[-1]
        return (h_src * h_dst).sum(dim=-1) / (dim ** 0.5)


def make_risk_tier_labels(nodes: pd.DataFrame, n_tiers: int = 3) -> pd.Series:
    return pd.qcut(nodes["pd_annual_true"], q=n_tiers, labels=list(range(n_tiers)))


def train_gnn(nodes: pd.DataFrame, edges: pd.DataFrame, epochs: int = 180, lr: float = 0.008,
              seed: int = 42, verbose: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data, type_to_local_idx, node_type_map, edge_type_names = build_hetero_data(nodes, edges)
    node_types = list(data.node_types)
    metadata = (node_types, list(data.edge_types))
    in_channels = data[node_types[0]].x.shape[1]

    model = HetGAT(metadata, in_channels=in_channels,
                    node_counts={nt: data[nt].x.shape[0] for nt in node_types})
    heads = RiskHeads(embed_dim=128, in_channels=in_channels, n_classes=3)
    opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=lr, weight_decay=5e-4)

    # ---- node classification labels & split ----
    risk_tier = make_risk_tier_labels(nodes).astype(int)
    tier_map = dict(zip(nodes.node_id, risk_tier))
    cls_targets, cls_index_by_type = {}, {}
    for nt in node_types:
        ids = data[nt].node_id
        y = torch.tensor([tier_map[i] for i in ids], dtype=torch.long)
        cls_targets[nt] = y
        n = len(ids)
        perm = rng.permutation(n)
        split = int(0.8 * n)
        cls_index_by_type[nt] = (torch.tensor(perm[:split]), torch.tensor(perm[split:]))

    # ---- link prediction split: hold out 15% of "forward" (non-reverse) edges ----
    fwd_edge_types = [et for et in data.edge_types if not et[1].startswith("rev_")]
    pos_train, pos_test = [], []
    for et in fwd_edge_types:
        ei = data[et].edge_index
        n_e = ei.shape[1]
        if n_e == 0:
            continue
        perm = rng.permutation(n_e)
        split = int(0.85 * n_e)
        pos_train.append((et, ei[:, perm[:split]]))
        pos_test.append((et, ei[:, perm[split:]]))

    def sample_negative(et, n):
        src_type, _, dst_type = et
        n_src = data[src_type].x.shape[0]
        n_dst = data[dst_type].x.shape[0]
        src = torch.from_numpy(rng.integers(0, n_src, size=n))
        dst = torch.from_numpy(rng.integers(0, n_dst, size=n))
        return src, dst

    for epoch in range(epochs):
        model.train(); heads.train()
        opt.zero_grad()
        h_dict = model(data.x_dict, data.edge_index_dict)

        # node classification loss (train split)
        cls_loss = 0.0
        for nt in node_types:
            idx_tr, _ = cls_index_by_type[nt]
            logits = heads.node_logits(h_dict[nt][idx_tr], data.x_dict[nt][idx_tr])
            cls_loss = cls_loss + F.cross_entropy(logits, cls_targets[nt][idx_tr])

        # link prediction loss (train split)
        link_loss = 0.0
        for et, ei in pos_train:
            src_type, _, dst_type = et
            pos_src, pos_dst = ei[0], ei[1]
            pos_score = heads.link_score(h_dict[src_type][pos_src], h_dict[dst_type][pos_dst])
            neg_src, neg_dst = sample_negative(et, pos_src.shape[0])
            neg_score = heads.link_score(h_dict[src_type][neg_src], h_dict[dst_type][neg_dst])
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            link_loss = link_loss + F.binary_cross_entropy_with_logits(scores, labels)

        loss = cls_loss + link_loss
        loss.backward()
        opt.step()

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  [GNN] epoch {epoch:3d}  loss={loss.item():.4f}  "
                  f"(cls={float(cls_loss.detach()) if torch.is_tensor(cls_loss) else cls_loss:.4f}, "
                  f"link={float(link_loss.detach()) if torch.is_tensor(link_loss) else link_loss:.4f})")

    # ---- Evaluation ----
    model.eval(); heads.eval()
    with torch.no_grad():
        h_dict = model(data.x_dict, data.edge_index_dict)

        # node classification accuracy (overall + supplier-only, per D2.2.1 wording)
        all_preds, all_true, supplier_preds, supplier_true = [], [], [], []
        for nt in node_types:
            _, idx_te = cls_index_by_type[nt]
            logits = heads.node_logits(h_dict[nt][idx_te], data.x_dict[nt][idx_te])
            preds = logits.argmax(dim=-1)
            true = cls_targets[nt][idx_te]
            all_preds.append(preds); all_true.append(true)
            if nt == "supplier":
                supplier_preds.append(preds); supplier_true.append(true)
        node_acc = accuracy_score(torch.cat(all_true), torch.cat(all_preds))
        supplier_acc = (accuracy_score(torch.cat(supplier_true), torch.cat(supplier_preds))
                         if supplier_preds else float("nan"))

        # link prediction AUC
        scores_all, labels_all = [], []
        for et, ei in pos_test:
            src_type, _, dst_type = et
            pos_score = heads.link_score(h_dict[src_type][ei[0]], h_dict[dst_type][ei[1]])
            neg_src, neg_dst = sample_negative(et, ei.shape[1])
            neg_score = heads.link_score(h_dict[src_type][neg_src], h_dict[dst_type][neg_dst])
            scores_all.append(torch.cat([pos_score, neg_score]))
            labels_all.append(torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]))
        link_auc = roc_auc_score(torch.cat(labels_all).numpy(), torch.sigmoid(torch.cat(scores_all)).numpy())

    embeddings = {nt: h_dict[nt].detach().numpy() for nt in node_types}
    node_ids = {nt: data[nt].node_id for nt in node_types}
    metrics = dict(node_classification_accuracy=float(node_acc),
                    supplier_risk_tier_accuracy=float(supplier_acc),
                    link_prediction_auc=float(link_auc))
    return model, heads, embeddings, node_ids, metrics


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_generator import SupplyChainDataGenerator

    gen = SupplyChainDataGenerator()
    nodes, edges = gen.generate_graph()
    model, heads, emb, node_ids, metrics = train_gnn(nodes, edges)
    print("Metrics:", metrics)
    print("Embedding shapes:", {k: v.shape for k, v in emb.items()})
