# brep_encoder.py
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs

from encoders_graphtransformer import (
    UVNetCurveEncoderMasked,   # Edge: [Ne, L, 4] -> [Ne, 16]
    UVNetSurfaceEncoder,       # Face: [Nf, H, W, 4] -> [Nf, 32]
    PointTransformerV3Encoder, # Face (full): [Nf, H, W, 11] -> [Nf, 32]
    PointNetPlusPlus1DEncoder, # Edge (full): [Ne, L, 9] -> [Ne, 16]
    UVNetGraphEncoder          # Fusion: (32,16,32)->128, with global pooling
)

class BrepEncoder(nn.Module):
    """
    UV-Net only: B-Rep graph -> (node features [per face 128], graph-level features [per graph 128])

    - Face lightweight features: g.ndata['x']         : [Nf, H, W, 4]  (xyz + pad_mask)
    - Face full features:        g.ndata['x_full']    : [Nf, H, W, 11] (see conversion script)
    - Edge lightweight features: g.edata['x']         : [Ne, L, 4]     (xyz + pad_mask)
    - Edge full features:        g.edata['x_full']    : [Ne, L, 9]     (see conversion script)
    """
    def __init__(self,
                 srf_in_channels=4,
                 crv_in_channels=4,
                 face_emb_dim=32,   # Surface encoder output
                 edge_emb_dim=16,   # Curve encoder output
                 point_emb_dim=32,  # PTv3 output
                 graph_emb_dim=128  # Final node feature dim (Gf+Ef+Ff)
                 ):
        super().__init__()

        # 1) Face (lightweight) 2D CNN: [Nf,H,W,4] -> [Nf,32]
        self.surf_encoder = UVNetSurfaceEncoder(in_channels=srf_in_channels, output_dims=face_emb_dim)

        # 2) Edge (lightweight) 1D CNN: [Ne,L,4] -> [Ne,16] (for EGAT Gf layer)
        self.curv_encoder = UVNetCurveEncoderMasked(in_channels=crv_in_channels, output_dims=edge_emb_dim)

        # 2b) Edge (full) PointNet++: [Ne,L,9] -> [Ne,16] (for NNConv Ef layer)
        self.pointnet_edge_encoder = PointNetPlusPlus1DEncoder(input_dim=9, output_dim=edge_emb_dim)

        # 3) Face (full) PointTransformerV3: [Nf,H,W,11] -> [Nf,32] (for Ff layer)
        self.point_encoder = PointTransformerV3Encoder(input_dim=11, output_dim=point_emb_dim)

        # 4) Graph fusion: output node features [Nf,128] and graph-level features [B,128]
        self.graph_encoder = UVNetGraphEncoder(
            face_feature_dim=face_emb_dim,
            edge_feature_dim=edge_emb_dim,
            output_dim=graph_emb_dim
        )

    @torch.no_grad()
    def _check_keys(self, g: dgl.DGLGraph):
        miss = []
        for k in ["x", "x_full"]:
            if k not in g.ndata:
                miss.append(f"g.ndata['{k}']")
        for k in ["x", "x_full"]:
            if g.num_edges() > 0 and k not in g.edata:
                miss.append(f"g.edata['{k}']")
        if miss:
            raise KeyError("Missing required features in graph: " + ", ".join(miss))

    def _encode_one_batch(self, batched_graph: dgl.DGLGraph):
        # Read features
        face_feat      = batched_graph.ndata["x"]       # [Nf, H, W, 4]
        face_feat_full = batched_graph.ndata["x_full"]  # [Nf, H, W, 11]
        edge_feat      = batched_graph.edata["x"]       # [Ne, L, 4]
        edge_feat_full = batched_graph.edata["x_full"]  # [Ne, L, 9]

        # 1) Face (lightweight) 2D CNN
        face_features = self.surf_encoder(face_feat)           # [Nf, 32]

        # 2) Edge (lightweight) 1D CNN for Gf
        edge_features = self.curv_encoder(edge_feat)           # [Ne, 16]

        # 2b) Edge (full) PointNet++ for Ef
        edge_features_pointnet = self.pointnet_edge_encoder(edge_feat_full)  # [Ne, 16]

        # 3) Face (full) PointTransformerV3 for Ff
        face_point_features = self.point_encoder(face_feat_full)  # [Nf, 32]

        # 4) Graph fusion, returns node features and graph-level features
        node_emb, graph_emb = self.graph_encoder(
            batched_graph,
            face_features,
            edge_features,          # -> Gf
            face_point_features,    # -> Ff
            edge_features_pointnet  # -> Ef
        )  # node_emb: [Nf,128], graph_emb: [B,128]

        return node_emb, graph_emb

    def forward(self, batched_graph: dgl.DGLGraph):
        self._check_keys(batched_graph)
        return self._encode_one_batch(batched_graph)


def load_bin_graph(bin_path: Path) -> dgl.DGLGraph:
    graphs, _ = load_graphs(str(bin_path))
    if not graphs:
        raise RuntimeError(f"No graphs found in {bin_path}")
    # save_graphs(path, [graph]) saves a single graph; take the first one
    g = graphs[0]
    return g


def build_dummy_graph(device="cpu") -> dgl.DGLGraph:
    # Used for self-check when no .bin file is available
    g = dgl.graph(([0, 1, 2, 2], [1, 2, 0, 3])).to(device)
    Nf = 4
    Ne = g.num_edges()

    H = W = 16
    L = 32

    g.ndata["x"]       = torch.randn(Nf, H, W, 4,  device=device)
    g.ndata["x_full"]  = torch.randn(Nf, H, W, 11, device=device)
    g.edata["x"]       = torch.randn(Ne, L, 4, device=device)
    g.edata["x_full"]  = torch.randn(Ne, L, 9, device=device)

    # Construct padding mask (last dimension)
    g.ndata["x"][:, :, :,  -1]  = (torch.rand_like(g.ndata["x"][:, :, :,  -1])  > 0.2).float()
    g.ndata["x_full"][:, :, :, -1] = (torch.rand_like(g.ndata["x_full"][:, :, :, -1]) > 0.2).float()
    g.edata["x"][:, :,    -1]   = (torch.rand_like(g.edata["x"][:, :,    -1])   > 0.2).float()
    g.edata["x_full"][:, :, -1] = (torch.rand_like(g.edata["x_full"][:, :, -1]) > 0.2).float()
    return g


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="B-Rep UV-Net Encoder (no Q-Former)")
    parser.add_argument("--bin", type=str, default="", help="Path to a .bin graph file saved by DGL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    # Build model
    model = BrepEncoder(
        srf_in_channels=4,   # Face lightweight features: xyz+pad
        crv_in_channels=4,   # Edge lightweight features: xyz+pad
        face_emb_dim=32,
        edge_emb_dim=16,
        point_emb_dim=32,
        graph_emb_dim=128
    ).to(device)

    print("========== Model ==========")
    print(model)
    total, trainable = count_params(model)
    print(f"Total params: {total/1e6:.3f}M | Trainable: {trainable/1e6:.3f}M")

    # Load graph
    if args.bin:
        bin_path = Path(args.bin)
        print(f"\nLoading bin graph from: {bin_path}")
        g = load_bin_graph(bin_path).to(device)
    else:
        print("\nNo --bin provided, using a dummy graph for a quick sanity check.")
        g = build_dummy_graph(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        node_emb, graph_emb = model(g)

    # Output shapes
    print("\n========== Outputs ==========")
    print(f"Num nodes (faces): {g.num_nodes()}, Num edges: {g.num_edges()}")
    print(f"node_emb shape : {tuple(node_emb.shape)}   # [total_faces_in_batch, 128]")
    print(f"graph_emb shape: {tuple(graph_emb.shape)}  # [batch_size, 128] (batched graphs OK)")

    print("\nDone.")


if __name__ == "__main__":
    main()
