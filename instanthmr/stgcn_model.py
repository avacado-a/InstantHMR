"""
Spatiotemporal Graph Convolutional Network (ST-GCN)
===================================================
PyTorch Neural Network for Parkinson's Disease Movement and Gait Assessment.

Architecture:
- Input: (B, C=6, T, V=17) [3D Canonical Position + 3D Camera-Relative Velocity]
- 6–9 ST-GCN Residual Blocks:
    - Spatial Graph Conv (Partitioned Adjacency A0, A1, A2)
    - Temporal 1D Conv (Kernel=9 over Time dimension T)
    - Batch Normalization + Dropout + Residual Connections
- Global Average Pooling over (Time T, Joints V=17) -> 256-D feature vector
- Linear Classification Head + Sigmoid -> Parkinson's Risk Probability [0.0 to 1.0]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gait_features import COCOGraph


class ConvTemporalGraphical(nn.Module):
    """
    Spatial Graph Convolution layer with learnable edge weighting.
    Applies partitioned adjacency multiplication across K=3 partitions (Self, Centripetal, Centrifugal).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, V)
        # A: (K, V, V)
        assert A.size(0) == self.kernel_size

        x_conv = self.conv(x)  # (B, out_channels * K, T, V)
        n, kc, t, v = x_conv.size()
        x_conv = x_conv.view(n, self.kernel_size, kc // self.kernel_size, t, v)

        # Graph aggregation across partitions: einsum 'nkctv,kvw->nctw'
        out = torch.einsum("nkctv,kvw->nctw", (x_conv, A)).contiguous()
        return out


class STGCNBlock(nn.Module):
    """
    Standard Spatiotemporal GCN Residual Block:
    Spatial Graph Conv -> BatchNorm -> ReLU -> Temporal 1D Conv (k=9) -> BatchNorm -> Dropout -> Residual Add
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size=3)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Temporal 1D Convolution spanning 9 consecutive frames
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        # Residual shortcut
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.bn_gcn(self.gcn(x, A)))
        x = self.tcn(x)
        return self.relu(x + res)


class ParkinsonSTGCN(nn.Module):
    """
    Full End-to-End Spatiotemporal Graph Convolutional Network for Parkinson's Disease.
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        dropout: float = 0.25,
        edge_importance_weighting: bool = True
    ):
        super().__init__()

        # Graph Adjacency Setup (3, 17, 17)
        graph = COCOGraph(num_nodes=17)
        self.register_buffer("A", torch.tensor(graph.A, dtype=torch.float32))

        # Learnable Edge Importance Mask
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in range(6)
            ])
        else:
            self.edge_importance = [1.0] * 6

        # Input Data BatchNorm
        self.data_bn = nn.BatchNorm1d(in_channels * 17)

        # 6-Block ST-GCN Backbone (64 -> 64 -> 128 -> 128 -> 256 -> 256 channels)
        self.st_gcn1 = STGCNBlock(in_channels, 64, stride=1, dropout=dropout)
        self.st_gcn2 = STGCNBlock(64, 64, stride=1, dropout=dropout)
        self.st_gcn3 = STGCNBlock(64, 128, stride=2, dropout=dropout)
        self.st_gcn4 = STGCNBlock(128, 128, stride=1, dropout=dropout)
        self.st_gcn5 = STGCNBlock(128, 256, stride=2, dropout=dropout)
        self.st_gcn6 = STGCNBlock(256, 256, stride=1, dropout=dropout)

        # Classification Head
        self.fc = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass up to the 256-D global representation vector.
        x shape: (B, C=6, T, V=17)
        """
        N, C, T, V = x.size()

        # Batch Normalization over channels & vertices
        x_flat = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x_norm = self.data_bn(x_flat)
        x = x_norm.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # ST-GCN Convolutional Blocks
        x = self.st_gcn1(x, self.A * self.edge_importance[0])
        x = self.st_gcn2(x, self.A * self.edge_importance[1])
        x = self.st_gcn3(x, self.A * self.edge_importance[2])
        x = self.st_gcn4(x, self.A * self.edge_importance[3])
        x = self.st_gcn5(x, self.A * self.edge_importance[4])
        x = self.st_gcn6(x, self.A * self.edge_importance[5])

        # Global Average Pooling over Time (T) and Joints (V)
        # (B, 256, T, V) -> (B, 256, 1, 1) -> (B, 256)
        x_pool = F.avg_pool2d(x, x.size()[2:])
        feature_vec = x_pool.view(N, -1)
        return feature_vec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prob: Parkinson's risk probability [0.0 to 1.0] (B, 1)
            features: 256-dimensional clinical feature vector (B, 256)
        """
        features = self.extract_features(x)
        logits = self.fc(features)
        prob = torch.sigmoid(logits)
        return prob, features
