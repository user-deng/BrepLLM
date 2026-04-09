#encoders_graphtransformer.py
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import NNConv

from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling

def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation
    """
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=8, output_dims=64):
        """
        1D convolutional network for B-rep edge geometry
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        if x.dim() == 3 and x.shape[2] == self.in_channels:
             x = x.permute(0, 2, 1)

        assert x.size(1) == self.in_channels, f"Input channel mismatch in CurveEncoder. Expected {self.in_channels}, got {x.size(1)}"
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=11,  # 10-dim features + 1-dim mask
        output_dims=32,  # Output 32-dim face features
    ):
        """
        Masked 2D CNN encoder for face feature encoding.
        Input: [N, 16, 16, 11] (padded face features + mask)
        Output: [N, 32] (face features)
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        # If mask is present, actual feature dim is in_channels-1
        actual_feature_dim = in_channels - 1 if in_channels == 4 else in_channels
        self.conv1 = _conv2d(actual_feature_dim, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        x: [N, 16, 16, 11] or [N, 11, 16, 16]
        """
        if x.dim() == 4 and x.shape[3] == self.in_channels:
            x = x.permute(0, 3, 1, 2)  # [N, 11, 16, 16]

        assert x.size(1) == self.in_channels, f"Input channel mismatch in SurfaceEncoder. Expected {self.in_channels}, got {x.size(1)}"

        # Extract mask and apply to features
        if self.in_channels == 4:  # Has mask
            features = x[:, :3, :, :]  # [N, 3, 16, 16] 3D point cloud
            mask = x[:, 3:4, :, :]     # [N, 1, 16, 16] mask
            x = features * mask  # Apply mask, zero out padded regions
        else:
            mask = None

        batch_size = x.size(0)
        x = self.conv1(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.conv2(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.conv3(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetCurveEncoderMasked(nn.Module):
    def __init__(self, in_channels=9, output_dims=16):  # 8-dim features + 1-dim mask
        """
        Masked 1D CNN encoder for edge feature encoding.
        Input: [N, 16, 9] (padded edge features + mask)
        Output: [N, 16] (edge features)
        """
        super(UVNetCurveEncoderMasked, self).__init__()
        self.in_channels = in_channels
        # If mask is present, actual feature dim is in_channels-1
        actual_feature_dim = in_channels - 1 if in_channels == 4 else in_channels
        self.conv1 = _conv1d(actual_feature_dim, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        x: [N, 16, 9] or [N, 9, 16]
        """
        if x.dim() == 3 and x.shape[2] == self.in_channels:
             x = x.permute(0, 2, 1)  # [N, 9, 16]

        assert x.size(1) == self.in_channels, f"Input channel mismatch in CurveEncoderMasked. Expected {self.in_channels}, got {x.size(1)}"

        # Extract mask and apply to features
        if self.in_channels == 4:  # Has mask
            features = x[:, :3, :]  # [N, 3, 16] 3D point cloud
            mask = x[:, 3:4, :]     # [N, 1, 16] mask
            x = features * mask  # Apply mask
        else:
            mask = None

        batch_size = x.size(0)
        x = self.conv1(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.conv2(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.conv3(x)
        if mask is not None:
            x = x * mask  # Re-apply mask after each conv layer

        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# encoders_graphtransformer.py
class PointNetPlusPlus1DEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=16):
        super(PointNetPlusPlus1DEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        actual_feat_dim = input_dim - 1  # Exclude mask
        self.point_encoder = nn.Sequential(
            nn.Linear(actual_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        self.final_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        x: [B, T, D] ; D==9(8+mask) or D==4(3+mask) or other (last dim is mask)
        """
        assert x.dim() == 3, f"Expected 3D tensor, got {x.shape}"
        B, T, D = x.shape

        if self.input_dim == 9:
            features = x[:, :, :8]   # [B,T,8]
            mask     = x[:, :, 8]    # [B,T]
        elif self.input_dim == 4:
            features = x[:, :, :3]   # [B,T,3]
            mask     = x[:, :, 3]    # [B,T]
        else:
            features = x[:, :, :-1]
            mask     = x[:, :, -1]   # [B,T]

        # Only encode valid points
        flat_feat = features.reshape(B * T, features.size(-1))
        flat_mask = mask.reshape(B * T)
        valid = flat_mask > 0.5
        if valid.sum() == 0:
            return torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)

        encoded_valid = self.point_encoder(flat_feat[valid])  # [N_valid,128]

        encoded_all = torch.zeros(B * T, 128, device=x.device, dtype=encoded_valid.dtype)
        encoded_all[valid] = encoded_valid
        encoded_seq = encoded_all.view(B, T, 128)  # [B,T,128]

        # Note: key_padding_mask=True means "ignore"
        key_padding_mask = (mask == 0)
        attn_out, _ = self.self_attention(
            encoded_seq, encoded_seq, encoded_seq, key_padding_mask=key_padding_mask
        )

        valid_mask = mask.unsqueeze(-1)  # [B,T,1]
        summed = (attn_out * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp(min=1e-8)
        pooled = summed / denom  # [B,128]

        return self.final_proj(pooled)  # [B,output_dim]


class PointTransformerV3Encoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=32, grid_size=0.02, dropout=0.1):
        """
        PointTransformerV3 encoder for intra-face self-attention.
        Input: [N, 16, 16, 11] (face features + mask)
        Output: [N, 32] (face point cloud features)
        """
        super(PointTransformerV3Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.dropout = dropout

        try:
            # Try to import the real PointTransformerV3
            from PointTransformerV3_main.model import PointTransformerV3
            self.use_real_ptv3 = True

            # Compute actual feature dim (excluding mask)
            if input_dim == 11:
                actual_feat_dim = 10  # 10-dim features + 1-dim mask
            elif input_dim == 4:
                actual_feat_dim = 3   # 3-dim point cloud + 1-dim mask
            else:
                actual_feat_dim = input_dim - 1 if input_dim > 3 else 3

            # Use PointTransformerV3
            self.ptv3 = PointTransformerV3(
                in_channels=actual_feat_dim,
                enc_depths=(2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256),
                enc_num_head=(2, 4, 8, 16),
                enc_patch_size=(1024, 1024, 1024, 1024),
                stride=(2, 2, 2),  # len(stride) + 1 = len(enc_depths)
                cls_mode=True,  # Classification mode, encoder only
            )

            # Lock BatchNorm in PointTransformerV3 to eval mode
            self._lock_bn_eval(self.ptv3)
            # Override train() to re-lock BN each time
            _old_train = self.ptv3.train
            def _bn_safe_train(mode: bool = True):
                ret = _old_train(mode)
                self._lock_bn_eval(self.ptv3)
                return ret
            self.ptv3.train = _bn_safe_train

        except ImportError:
            # If import fails, use simplified version
            print("Warning: PointTransformerV3 not found, using simplified version")
            self.use_real_ptv3 = False

            # Simplified self-attention implementation
            self.point_encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU()
            )

            self.self_attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=8,
                batch_first=True
            )

        # Final projection layer
        self.final_proj = nn.Sequential(
            nn.Linear(256 if self.use_real_ptv3 else 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def _lock_bn_eval(self, module: nn.Module):
        """Lock all BatchNorm layers in the module to eval mode."""
        for m in module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                # Use running stats (won't be updated); affine (weight/bias) remain trainable
                m.track_running_stats = True

    def _prepare_single_face_ptv3_input(self, face_points, device, mask=None):
        # Select coordinates and features
        if self.input_dim == 11:
            coord = face_points[:, :3]     # (H*W, 3)
            feat  = face_points[:, :10]    # (H*W, 10)
            mask_col = face_points[:, 10]
        elif self.input_dim == 4:
            coord = face_points[:, :3]     # (H*W, 3)
            feat  = face_points[:, :3]     # (H*W, 3)
            mask_col = face_points[:, 3]
        else:
            coord = face_points[:, :3]
            feat  = face_points

        # Valid point filtering
        if mask is not None:
            valid_mask = mask
        else:
            valid_mask = (mask_col > 0) if face_points.shape[1] >= 4 else torch.ones(len(face_points), dtype=torch.bool, device=device)

        if valid_mask.sum() < 2:
            # Fewer than 2 points, return None so upper layer falls back to zero vector
            return None

        coord = coord[valid_mask]   # (N_valid, 3)
        feat  = feat[valid_mask]    # (N_valid, C)

        # Key: construct non-negative grid_coord with at least two distinct levels
        # Scale coordinates to [0, S-1]
        S = 64  # Target discrete resolution
        cmin = coord.min(dim=0).values
        cmax = coord.max(dim=0).values
        span = (cmax - cmin).clamp_min(1e-6)  # Prevent division by zero

        # Linear scaling to integer grid
        grid_coord = torch.floor((coord - cmin) / span * (S - 1)).to(torch.int32)
        grid_coord = grid_coord.clamp(min=0, max=S - 1)

        # Assemble input for PTv3 (no grid_size needed)
        data_dict = {
            "coord": coord,
            "feat":  feat,
            "grid_coord": grid_coord,
            "offset": torch.tensor([0, len(coord)], device=device, dtype=torch.long),
            "batch":  torch.zeros(len(coord), device=device, dtype=torch.long),
        }
        return data_dict


    def forward(self, x):
        """
        x: [N, 16, 16, 11]
        """
        batch_size = x.size(0)
        device = x.device

        if self.use_real_ptv3:
            # Use the real PointTransformerV3
            face_features = []

            for i in range(batch_size):
                face_pts = x[i].view(-1, self.input_dim)  # [256, 4]

                # Filter invalid points (mask == 0)
                if self.input_dim == 11:
                    valid_pts_mask = (face_pts[:, 10] > 0)
                elif self.input_dim == 4:
                    valid_pts_mask = (face_pts[:, 3] > 0)
                else:
                    valid_pts_mask = torch.ones(len(face_pts), dtype=torch.bool, device=device)
                if valid_pts_mask.sum() == 0:
                    # No valid points
                    face_feat = torch.zeros(self.output_dim, device=device)
                else:
                    # Encode with PTv3, passing mask info
                    data_dict = self._prepare_single_face_ptv3_input(face_pts, device, valid_pts_mask)

                    if data_dict is None:
                        face_feat = torch.zeros(self.output_dim, device=device)
                    else:
                        if len(data_dict['coord']) == 0:
                            face_feat = torch.zeros(self.output_dim, device=device)
                        else:
                            point_output = self.ptv3(data_dict)

                            if hasattr(point_output, 'feat'):
                                # Global pooling
                                global_feat = point_output.feat.mean(dim=0)  # (256,)
                                face_feat = self.final_proj(global_feat)  # (output_dim,)
                            else:
                                face_feat = torch.zeros(self.output_dim, device=device)

                face_features.append(face_feat)

            return torch.stack(face_features)  # [N, output_dim]

        else:
            # Use simplified version
            # Reshape to point cloud format [N, 256, 4]
            points = x.view(batch_size, -1, self.input_dim)

            # Extract mask
            if self.input_dim == 4:
                features = points[:, :, :3]   # [N, 256, 3] 3D point cloud
                mask = points[:, :, 3]        # [N, 256] mask

                # Encode point features
                point_feats = self.point_encoder(features)  # [N, 256, 128]

                # Apply self-attention with mask
                key_padding_mask = (mask == 0)  # True means positions to mask
                attn_out, _ = self.self_attention(
                    point_feats, point_feats, point_feats,
                    key_padding_mask=key_padding_mask
                )

                # Pool: average over valid points
                valid_mask = mask.unsqueeze(-1)  # [N, 256, 1]
                masked_feats = attn_out * valid_mask
                pooled_feats = masked_feats.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
            else:
                # No mask case
                point_feats = self.point_encoder(points)
                attn_out, _ = self.self_attention(point_feats, point_feats, point_feats)
                pooled_feats = attn_out.mean(dim=1)

            # Output projection
            output = self.final_proj(pooled_feats)  # [N, 32]
            return output


class _EdgeConv(nn.Module):
    def __init__(self, edge_feats, out_feats, node_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))
    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h))
        return h

class _NodeConv(nn.Module):
    def __init__(self, node_feats, out_feats, edge_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))
    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        face_feature_dim=32,  # Face feature dimension
        edge_feature_dim=16,  # Edge feature dimension
        output_dim=128,       # Final node feature dimension (Gf+Ef+Ff = 64+32+32)
        **kwargs,
    ):
        """
        Three-level feature architecture graph encoder:
        1. Global features Gf (64-dim): GAT fusing face-edge info
        2. Mid-level features Ef (32-dim): NNConv processing neighboring edges
        3. Low-level features Ff (32-dim): Point Transformer processing face features
        Final node features: Gf + Ef + Ff = 128-dim
        """
        super(UVNetGraphEncoder, self).__init__()

        # Three-level feature dimensions
        self.gf_dim = 64  # Global features
        self.ef_dim = 32  # Mid-level features
        self.ff_dim = 32  # Low-level features
        self.total_dim = self.gf_dim + self.ef_dim + self.ff_dim  # 128

        assert output_dim == self.total_dim, f"output_dim should be {self.total_dim}, but got {output_dim}"

        # ===== 1. Global features Gf: EGAT fusing face-edge info =====
        from dgl.nn.pytorch import EGATConv
        self.face_proj = nn.Linear(face_feature_dim, self.gf_dim)
        self.edge_proj = nn.Linear(edge_feature_dim, self.gf_dim)

        # Use single-head attention to avoid dimension issues
        self.egat_layers = nn.ModuleList([
            EGATConv(
                in_node_feats=self.gf_dim,
                in_edge_feats=self.gf_dim,
                out_node_feats=self.gf_dim,
                out_edge_feats=self.gf_dim,
                num_heads=1
            ),
            EGATConv(
                in_node_feats=self.gf_dim,
                in_edge_feats=self.gf_dim,
                out_node_feats=self.gf_dim,
                out_edge_feats=self.gf_dim,
                num_heads=1
            )
        ])

        # ===== 2. Mid-level features Ef: NNConv processing neighboring edge info =====
        self.edge_pointnet_for_ef = PointNetPlusPlus1DEncoder(input_dim=4, output_dim=edge_feature_dim)
        self.neighbor_edge_conv = NNConv(
            in_feats=face_feature_dim,
            out_feats=self.ef_dim,
            edge_func=nn.Linear(edge_feature_dim, face_feature_dim * self.ef_dim),
            aggregator_type="mean"
        )

        # ===== 3. Low-level features Ff: Point Transformer processing face features =====
        # This part directly uses the output of PointTransformerV3Encoder in forward()

        # ===== 4. Global graph pooling (optional, for graph-level prediction) =====
        gate_nn = nn.Linear(self.total_dim, 1)
        self.graph_pool = GlobalAttentionPooling(gate_nn)

        # ===== 5. Final output layer =====
        self.output_linear = nn.Linear(self.total_dim, output_dim)

    def forward(self, g, face_features, edge_features, face_point_features=None, edge_features_pointnet=None):
        """
        Args:
            g: DGL graph
            face_features: [N_faces, 32] face features from 2D CNN
            edge_features: [N_edges, 16] edge features from 1D CNN (for Gf layer)
            face_point_features: [N_faces, 32] face features from Point Transformer (for Ff layer)
            edge_features_pointnet: [N_edges, 16] edge features from PointNet++ (for Ef layer)

        Returns:
            node_features: [N_faces, 128] 128-dim features for each node
            graph_feature: [128] graph-level feature (optional)
        """
        device = face_features.device

        # ===== 1. Compute global features Gf (64-dim) =====
        # Project face and edge features to the same dimension
        h_face = self.face_proj(face_features)  # [N_faces, 64]
        h_edge = self.edge_proj(edge_features)  # [N_edges, 64]

        # Fuse adjacency info through EGAT layers
        h_node, h_edge_out = h_face, h_edge
        for egat_layer in self.egat_layers:
            h_node, h_edge_out = egat_layer(g, h_node, h_edge_out)  # [N_faces, 64], [N_edges, 64]

        Gf = h_node  # [N_faces, 64] Global features

        # ===== 2. Compute mid-level features Ef (32-dim) =====
        # Use NNConv to process neighboring edge info with PointNet++ edge features
        if edge_features_pointnet is not None:
            Ef = self.neighbor_edge_conv(g, face_features, edge_features_pointnet)  # [N_faces, 32]
        else:
            # Backward compatible: if PointNet++ edge features not provided, use 1D CNN edge features
            Ef = self.neighbor_edge_conv(g, face_features, edge_features)  # [N_faces, 32]

        # ===== 3. Get low-level features Ff (32-dim) =====
        if face_point_features is not None:
            Ff = face_point_features  # [N_faces, 32] from Point Transformer
        else:
            # If not provided, fill with zeros (backward compatible)
            Ff = torch.zeros(face_features.size(0), self.ff_dim, device=device)

        # ===== 4. Concatenate three-level features =====
        # Ensure all tensors are 2D
        if Gf.dim() > 2:
            Gf = Gf.squeeze()
        if Ef.dim() > 2:
            Ef = Ef.squeeze()
        if Ff.dim() > 2:
            Ff = Ff.squeeze()

        node_features = torch.cat([Gf, Ef, Ff], dim=-1)  # [N_faces, 128]

        # ===== 5. Compute global graph feature (optional) =====
        graph_feat = self.graph_pool(g, node_features)  # [128]

        return node_features, graph_feat
