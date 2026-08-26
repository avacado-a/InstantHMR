"""
MHR-70 Spatiotemporal Graph Convolutional Network Architecture
=============================================================
Processes all 70 keypoints (Body, Face Proxies, Olecranon/Acromion, Feet, Full 5-Finger Hands).
Extracts fine-grained resting tremors, finger tapping bradykinesia, postural instability,
and gait asymmetries simultaneously across high-density graph topologies.

Input: (B, C=6, T, V=70) -> [3D Canonical Positions + 3D Camera Velocities]
Output: Binary Screening Probability & Multiclass Severity Logits + 256-D Deep Kinematic Vector.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.skeleton import SKELETON_EDGES, JOINT_NAMES

# Complete 70-joint anatomical edges including all 5-finger kinematic chains
FULL_MHR70_EDGES = list(SKELETON_EDGES) + [
    # Right Hand Digits
    (21, 22), (22, 23), (23, 24), # Thumb
    (25, 26), (26, 27), (27, 28), # Index
    (29, 30), (30, 31), (31, 32), # Middle
    (33, 34), (34, 35), (35, 36), # Ring
    (37, 38), (38, 39), (39, 40), # Pinky
    # Left Hand Digits
    (42, 43), (43, 44), (44, 45), # Thumb
    (46, 47), (47, 48), (48, 49), # Index
    (50, 51), (51, 52), (52, 53), # Middle
    (54, 55), (55, 56), (56, 57), # Ring
    (58, 59), (59, 60), (60, 61), # Pinky
    # Face & Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Elbow / Shoulder Details
    (7, 63), (7, 65), (8, 64), (8, 66),
    (5, 67), (6, 68),
    # Feet
    (13, 15), (13, 16), (14, 18), (14, 19)
]

# Graph depth from root (pelvis 9, 10)
GRAPH_DEPTH_MHR70 = {}
for i in range(70):
    if i in (9, 10): # Hips
        GRAPH_DEPTH_MHR70[i] = 0
    elif i in (5, 6, 11, 12, 69): # Shoulders, Knees, Neck
        GRAPH_DEPTH_MHR70[i] = 1
    elif i in (0, 7, 8, 13, 14, 67, 68): # Nose, Elbows, Ankles, Acromion
        GRAPH_DEPTH_MHR70[i] = 2
    elif i in (1, 2, 41, 62, 15, 16, 17, 18, 19, 20, 63, 64, 65, 66): # Wrists, Eyes, Feet
        GRAPH_DEPTH_MHR70[i] = 3
    else: # Fingers and Ears
        GRAPH_DEPTH_MHR70[i] = 4


class MHR70Graph:
    """Constructs 3-partition adjacency matrix for the 70-joint MHR skeleton."""
    def __init__(self):
        self.num_nodes = 70
        self.edges = FULL_MHR70_EDGES
        self.depths = GRAPH_DEPTH_MHR70
        self.A = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        V = 70
        A0 = np.eye(V, dtype=np.float32)
        A1 = np.zeros((V, V), dtype=np.float32)
        A2 = np.zeros((V, V), dtype=np.float32)

        for i, j in self.edges:
            d_i = self.depths.get(i, 2)
            d_j = self.depths.get(j, 2)
            if d_i > d_j:
                A1[j, i] = 1.0 # Inward / centripetal
                A2[i, j] = 1.0 # Outward / centrifugal
            elif d_i < d_j:
                A1[i, j] = 1.0
                A2[j, i] = 1.0
            else:
                A1[i, j] = A1[j, i] = 0.5
                A2[i, j] = A2[j, i] = 0.5

        def norm_adj(mat):
            r = np.sum(mat, axis=1)
            r[r == 0] = 1.0
            return np.dot(np.diag(1.0 / r), mat)

        return np.stack([norm_adj(A0), norm_adj(A1), norm_adj(A2)], axis=0).astype(np.float32)


class SpatialGraphConv70(nn.Module):
    def __init__(self, in_c: int, out_c: int, partitions: int = 3):
        super().__init__()
        self.partitions = partitions
        self.conv = nn.Conv2d(in_c, out_c * partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V=70), A: (3, 70, 70)
        x_c = self.conv(x)
        b, pc, t, v = x_c.size()
        x_c = x_c.view(b, self.partitions, pc // self.partitions, t, v)
        return torch.einsum("nkctv,kvw->nctw", (x_c, A)).contiguous()


class MHR70STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, dropout: float = 0.25):
        super().__init__()
        self.gcn = SpatialGraphConv70(in_c, out_c)
        self.bn_gcn = nn.BatchNorm2d(out_c)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout)
        )
        if in_c != out_c or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.res = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        x = self.relu(self.bn_gcn(self.gcn(x, A)))
        x = self.tcn(x)
        return self.relu(x + res)


class HighDensityMHR70STGCN(nn.Module):
    """
    High-Capacity 70-Joint Spatiotemporal Graph Convolutional Network.
    Optimized for multi-site generalization and fine-grained motor tremor detection.
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 1, dropout: float = 0.25):
        super().__init__()
        graph = MHR70Graph()
        self.register_buffer("A", torch.tensor(graph.A, dtype=torch.float32))

        # Learnable Edge Importance
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size())) for _ in range(6)
        ])

        self.data_bn = nn.BatchNorm1d(in_channels * 70)

        # 6-Block Residual Backbone
        self.b1 = MHR70STGCNBlock(in_channels, 64, stride=1, dropout=dropout)
        self.b2 = MHR70STGCNBlock(64, 64, stride=1, dropout=dropout)
        self.b3 = MHR70STGCNBlock(64, 128, stride=2, dropout=dropout)
        self.b4 = MHR70STGCNBlock(128, 128, stride=1, dropout=dropout)
        self.b5 = MHR70STGCNBlock(128, 256, stride=2, dropout=dropout)
        self.b6 = MHR70STGCNBlock(256, 256, stride=1, dropout=dropout)

        self.fc = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, T, 70)
        B, C, T, V = x.size()
        x_flat = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x_flat).view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        x = self.b1(x, self.A * self.edge_importance[0])
        x = self.b2(x, self.A * self.edge_importance[1])
        x = self.b3(x, self.A * self.edge_importance[2])
        x = self.b4(x, self.A * self.edge_importance[3])
        x = self.b5(x, self.A * self.edge_importance[4])
        x = self.b6(x, self.A * self.edge_importance[5])

        x_pool = F.avg_pool2d(x, x.size()[2:])
        return x_pool.view(B, -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        prob = torch.sigmoid(logits)
        return prob, feat
