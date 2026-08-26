"""
Parkinson's Disease 3D Gait & Kinematics Preprocessor
=====================================================
Transforms raw 3D joint predictions into the normalized (B, C=6, T, V=17) tensor
for Spatiotemporal Graph Convolutional Networks (ST-GCN).

Pipeline Stages:
1. Extraction of 17 COCO joints from MHR70 predictions.
2. Spatial Canonicalization & Invariance:
   - Root-Centering: Subtract mid-hip J_root(t) from all joints.
   - Heading-Vector Alignment: Coronal rotation around Y-axis to face +Z.
   - Scale Normalization: Normalize by torso segment length.
3. Feature Channel Stacking:
   - C[0..3]: Normalized canonical spatial coordinates [X, Y, Z].
   - C[3..6]: Camera-relative temporal velocity vectors:
              v_cam(t) = J_cam(t) - J_cam(t - 1)
4. Graph Topology Linking & Spatial Partitioning (A0 self, A1 centripetal, A2 centrifugal).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

# COCO-17 Standard Order & Mapping
COCO17_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_hip", "right_hip",
  "left_knee", "right_knee",
  "left_ankle", "right_ankle"
]

# 18 Anatomical Spatial Bone Edges
COCO_ANATOMICAL_EDGES = [
    # Head Group
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper Torso
    (0, 5), (0, 6), (5, 6),
    # Upper Limbs
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Trunk & Pelvis
    (5, 11), (6, 12), (11, 12),
    # Lower Limbs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Distance from body root (for centripetal / centrifugal graph partitioning)
# 0: Root (Pelvis) -> higher numbers are extremities
JOINT_GRAPH_DEPTH = {
    11: 0, 12: 0,        # Hips / Pelvis
    5: 1, 6: 1,          # Shoulders
    0: 2, 13: 1, 14: 1,  # Nose, Knees
    1: 3, 2: 3, 7: 2, 8: 2, 15: 2, 16: 2, # Eyes, Elbows, Ankles
    3: 4, 4: 4, 9: 3, 10: 3                # Ears, Wrists
}


def extract_coco17_from_mhr70(mhr_joints: np.ndarray, mapping_file: str = None) -> np.ndarray:
    """
    Extract (T, 17, 3) COCO-17 keypoints from (T, 70, 3) MHR keypoints.
    """
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            cfg = json.load(f)
            indices = cfg.get("mhr_indices", [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14])
    else:
        # Default MHR70 -> COCO17
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
    
    indices = np.array(indices, dtype=np.int32)
    return mhr_joints[:, indices, :]


def canonicalize_and_stack_features(
    joints_local_3d: np.ndarray,
    joints_cam_3d: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Stage 1 & 2: Spatial Canonicalization + Camera-Relative Velocity Feature Stacking.

    Args:
        joints_local_3d: (T, 17, 3) body-centered or local 3D coordinates.
        joints_cam_3d: (T, 17, 3) 3D coordinates in metric camera space.
        fps: Frame rate for velocity scaling.

    Returns:
        feature_tensor: (6, T, 17) float32 numpy array ready for ST-GCN.
    """
    T, V, C = joints_local_3d.shape
    assert V == 17 and C == 3, f"Expected (T, 17, 3), got ({T}, {V}, {C})"

    # -------------------------------------------------------------
    # 1. Spatial Canonicalization & Invariance (Per-Frame Normalization)
    # -------------------------------------------------------------
    P = np.zeros((T, 17, 3), dtype=np.float32)

    for t in range(T):
        frame = joints_local_3d[t].copy()

        # A. Root-Centering (Translation Invariance)
        # Mid-hip root: (Left Hip [11] + Right Hip [12]) / 2
        root = (frame[11] + frame[12]) / 2.0
        frame_centered = frame - root

        # B. Heading-Vector Alignment (Rotational Invariance)
        # Pelvis coronal vector: Right Hip [12] -> Left Hip [11]
        hip_vec = frame_centered[11] - frame_centered[12]
        # Angle in the horizontal X-Z plane
        theta = np.arctan2(hip_vec[2], hip_vec[0])
        # Rotation matrix around Y (vertical axis) to align coronal vector with X and face +Z
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R_y = np.array([
            [cos_t, 0.0, sin_t],
            [0.0,   1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float32)

        frame_aligned = np.dot(frame_centered, R_y.T)

        # C. Scale Normalization (Torso Length Invariance)
        # Mid-shoulder
        mid_shoulder = (frame_aligned[5] + frame_aligned[6]) / 2.0
        # Mid-hip
        mid_hip = (frame_aligned[11] + frame_aligned[12]) / 2.0
        torso_len = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_len < 1e-4:
            torso_len = 1.0

        P[t] = frame_aligned / torso_len

    # -------------------------------------------------------------
    # 2. Kinematic Feature Channel Stacking with Camera-Relative Velocity
    # -------------------------------------------------------------
    # Camera-relative velocity preserves true physical movement, forward speed,
    # and natural gait velocity relative to the real world / camera frame.
    V_cam = np.zeros((T, 17, 3), dtype=np.float32)
    dt = 1.0 / max(1e-4, fps)

    # First-order backward difference
    for t in range(1, T):
        V_cam[t] = (joints_cam_3d[t] - joints_cam_3d[t - 1]) / dt
    # First frame holds frame-1 velocity
    if T > 1:
        V_cam[0] = V_cam[1]

    # Combine Canonical Position C[0..3] and Camera Velocity C[3..6]
    # Shape: (T, 17, 6) -> transpose to (6, T, 17)
    features = np.concatenate([P, V_cam], axis=-1)  # (T, 17, 6)
    features = np.transpose(features, (2, 0, 1))    # (6, T, 17)

    return features.astype(np.float32)


# -----------------------------------------------------------------
# 3. Graph Edge Link Construction & Spatial Partitioning (A0, A1, A2)
# -----------------------------------------------------------------
class COCOGraph:
    """
    Constructs the 3-partition spatial adjacency matrices for 17-joint COCO ST-GCN:
    - A0: Self-loops (Identity)
    - A1: Centripetal (inward toward pelvis/spine root)
    - A2: Centrifugal (outward toward extremities)
    """
    def __init__(self, num_nodes: int = 17):
        self.num_nodes = num_nodes
        self.edges = COCO_ANATOMICAL_EDGES
        self.depths = JOINT_GRAPH_DEPTH
        self.A = self._build_spatial_partition_matrices()

    def _build_spatial_partition_matrices(self) -> np.ndarray:
        V = self.num_nodes
        A0 = np.eye(V, dtype=np.float32)
        A1 = np.zeros((V, V), dtype=np.float32)
        A2 = np.zeros((V, V), dtype=np.float32)

        for i, j in self.edges:
            d_i = self.depths.get(i, 0)
            d_j = self.depths.get(j, 0)

            if d_i > d_j:
                # Joint i is further from root than j (Centripetal from i -> j)
                A1[j, i] = 1.0
                # Centrifugal from j -> i
                A2[i, j] = 1.0
            elif d_i < d_j:
                A1[i, j] = 1.0
                A2[j, i] = 1.0
            else:
                # Same hierarchy level (e.g. shoulder crossbar 5 <-> 6, hip crossbar 11 <-> 12)
                A1[i, j] = 0.5
                A1[j, i] = 0.5
                A2[i, j] = 0.5
                A2[j, i] = 0.5

        # Degree normalization: D^{-1} A
        def normalize_adj(A_mat):
            row_sum = np.sum(A_mat, axis=1)
            row_sum[row_sum == 0] = 1.0
            d_inv = np.diag(1.0 / row_sum)
            return np.dot(d_inv, A_mat)

        A0_norm = normalize_adj(A0)
        A1_norm = normalize_adj(A1)
        A2_norm = normalize_adj(A2)

        # Output shape: (3, 17, 17)
        return np.stack([A0_norm, A1_norm, A2_norm], axis=0).astype(np.float32)
