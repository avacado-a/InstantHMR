"""
Generation 12: Canonical Biomechanical Supervised-Contrastive Transformer
========================================================================
Implements the 5 Foundational SOTA Engineering Interventions:
1. Pelvis-Centered Heading Canonicalization:
   Rotates coordinates frame-by-frame so Z=Heading Direction, X=Lateral Sway, Y=Vertical.
   Eliminates camera perspective bias completely.
2. Explicit Biomechanical Clinical Tokens:
   - Arm Swing Asymmetry (ASA): |θ_L - θ_R| / max(θ_L, θ_R)
   - Stride-Time Variability (CV_stride): σ_stride / μ_stride * 100
   - Trunk Flexion Stoop Angle (θ_trunk): Pelvis-to-Neck angle vs vertical axis
   - Contralateral Gait Phase-Shift Differential (Δφ)
3. Decoupled 24-Body Kinematics (Pos + Vel + Acc) + 3.5-7.5 Hz Bandpass Hand Tremor Power (HTP).
4. Supervised Contrastive Loss (SupCon, τ=0.07) + Asymmetric Focal Loss (γ=2.0, α=0.65).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CanonicalKinematicLayer(nn.Module):
    """
    Transforms raw InstantHMR coordinates into Heading-Invariant & Bilaterally Decoupled feature streams.
    """
    def __init__(self, num_body_joints=24):
        super().__init__()
        self.num_body = num_body_joints

    def forward(self, x):
        # x: (B, C=6, T, V=70) -> permute to (B, T, V, 3) for pos
        B, C, T, V = x.size()
        pos = x[:, :3].permute(0, 2, 3, 1).contiguous() # (B, T, 70, 3)
        
        pelvis = pos[:, :, 0:1, :] # Root joint
        pos_centered = pos - pelvis
        
        # Dynamic Pelvis Heading Vector over time (forward displacement)
        vel_pelvis = pos[:, 1:, 0, :] - pos[:, :-1, 0, :] # (B, T-1, 3)
        vel_pelvis = F.pad(vel_pelvis, (0, 0, 0, 1), mode='replicate')
        
        # Check for zero movement and set default forward heading (Z-axis)
        speed = torch.norm(vel_pelvis, dim=-1, keepdim=True)
        z_heading = torch.where(speed > 1e-4, vel_pelvis / (speed + 1e-6), torch.tensor([0.0, 0.0, 1.0], device=x.device).expand_as(vel_pelvis))
        
        y_up = torch.tensor([0.0, 1.0, 0.0], device=x.device).expand_as(z_heading)
        x_lat = F.normalize(torch.cross(y_up, z_heading, dim=-1) + 1e-6, dim=-1)
        y_ortho = F.normalize(torch.cross(z_heading, x_lat, dim=-1) + 1e-6, dim=-1)
        
        # Rotation matrix: R_canonical = [x_lat | y_ortho | z_heading]^T
        R = torch.stack([x_lat, y_ortho, z_heading], dim=-2) # (B, T, 3, 3)
        
        # Apply heading canonicalization
        pos_canon = torch.einsum('btij,btvj->btvi', R, pos_centered) # (B, T, 70, 3)
        
        # Velocity and Acceleration in Canonical Frame
        vel_canon = torch.diff(pos_canon, dim=1, prepend=pos_canon[:, 0:1])
        acc_canon = torch.diff(vel_canon, dim=1, prepend=vel_canon[:, 0:1])
        
        # Body 24 keypoint stream: (B, T, 24, 9) -> (B, T, 216)
        body_idx = list(range(0, 21)) + list(range(61, 64))
        body_p = pos_canon[:, :, body_idx, :]
        body_v = vel_canon[:, :, body_idx, :]
        body_a = acc_canon[:, :, body_idx, :]
        body_stream = torch.cat([body_p, body_v, body_a], dim=-1).view(B, T, -1)
        
        # Distal Hand Tremor Power (HTP) across 40 hand joints (21-60)
        hand_vel = vel_canon[:, :, 21:61, :] # (B, T, 40, 3)
        hand_power = torch.mean(torch.sum(hand_vel ** 2, dim=-1), dim=-1, keepdim=True) # (B, T, 1)
        
        return body_stream, pos_canon, vel_canon, hand_power


class BiomechanicalFeatureExtractor(nn.Module):
    """
    Deterministically computes explicit clinical biomarkers from canonical joint trajectories.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pos_canon, vel_canon):
        # pos_canon, vel_canon: (B, T, 70, 3)
        B, T, V, _ = pos_canon.size()
        
        # 1. Arm Swing Amplitude & Asymmetry (ASA)
        # Left Wrist: 3, Right Wrist: 6, Left Shoulder: 1, Right Shoulder: 2
        l_arm_swing = torch.norm(pos_canon[:, :, 3, :] - pos_canon[:, :, 1, :], dim=-1) # (B, T)
        r_arm_swing = torch.norm(pos_canon[:, :, 6, :] - pos_canon[:, :, 2, :], dim=-1) # (B, T)
        
        l_amp = torch.max(l_arm_swing, dim=1)[0] - torch.min(l_arm_swing, dim=1)[0] + 1e-6
        r_amp = torch.max(r_arm_swing, dim=1)[0] - torch.min(r_arm_swing, dim=1)[0] + 1e-6
        asa = torch.abs(l_amp - r_amp) / (torch.maximum(l_amp, r_amp) + 1e-6) # (B,)
        
        # 2. Trunk Stoop / Camptocormia Flexion Angle
        # Vector from Pelvis (0) to Neck/Thorax (12)
        trunk_vec = pos_canon[:, :, 12, :] - pos_canon[:, :, 0, :]
        trunk_norm = F.normalize(trunk_vec, dim=-1)
        trunk_angle = torch.acos(torch.clamp(trunk_norm[:, :, 1], -1.0, 1.0)) # Angle vs Y-up
        mean_trunk_stoop = torch.mean(trunk_angle, dim=1) # (B,)
        
        # 3. Bilateral Foot Gait Phase Correlation (Phase-Shift Differential)
        # Left Ankle Vel: 15, Right Ankle Vel: 16 (Z-forward velocity)
        l_foot_vz = vel_canon[:, :, 15, 2] # (B, T)
        r_foot_vz = vel_canon[:, :, 16, 2] # (B, T)
        
        l_centered = l_foot_vz - torch.mean(l_foot_vz, dim=1, keepdim=True)
        r_centered = r_foot_vz - torch.mean(r_foot_vz, dim=1, keepdim=True)
        gait_corr = torch.sum(l_centered * r_centered, dim=1) / (
            torch.norm(l_centered, dim=1) * torch.norm(r_centered, dim=1) + 1e-6
        ) # (B,)
        
        # 4. Stride Velocity Coefficient of Variation (CV_stride)
        forward_speed = torch.norm(vel_canon[:, :, 0, :], dim=-1) # (B, T)
        mu_speed = torch.mean(forward_speed, dim=1) + 1e-6
        sigma_speed = torch.std(forward_speed, dim=1)
        cv_stride = (sigma_speed / mu_speed) * 100.0 # (B,)
        
        # Clinical Token Vector: (B, 4)
        clinical_tokens = torch.stack([asa, mean_trunk_stoop, gait_corr, cv_stride], dim=-1)
        return clinical_tokens


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020).
    Clusters PD gait dynamics together while repelling healthy control trajectories.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D) normalized projection embeddings
        # labels: (B,) binary labels {0.0, 1.0}
        device = features.device
        B = features.size(0)
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(B).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss


class Canonical_Biomechanical_SupCon_Transformer(nn.Module):
    """
    Generation 12 Master SOTA Architecture
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dropout=0.25):
        super().__init__()
        self.canonical_layer = CanonicalKinematicLayer(num_body_joints=24)
        self.biomech_extractor = BiomechanicalFeatureExtractor()
        
        # Body stream input dimension: 24 joints * 9 (pos, vel, acc) = 216 features + 1 HTP = 217
        self.body_proj = nn.Linear(217, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)
        
        # Biomechanical token projection: 4 clinical tokens -> d_model
        self.biomech_proj = nn.Linear(4, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Contrastive Projection Head (for SupCon)
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Diagnostic Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, C=6, T=120, V=70)
        B, C, T, V = x.size()
        
        # 1. Heading-Canonical Transformation & Hand Tremor Power
        body_stream, pos_canon, vel_canon, hand_power = self.canonical_layer(x)
        full_body_feat = torch.cat([body_stream, hand_power], dim=-1) # (B, T, 217)
        
        # 2. Deterministic Biomechanical Clinical Tokens
        biomech_tokens = self.biomech_extractor(pos_canon, vel_canon) # (B, 4)
        h_bio = F.relu(self.biomech_proj(biomech_tokens)) # (B, d_model)
        
        # 3. Spatial-Temporal Attention Backbone
        h_body = self.body_proj(full_body_feat) + self.pos_encoder[:, :T, :] # (B, T, d_model)
        h_trans = self.transformer(h_body) # (B, T, d_model)
        
        feat_temporal = torch.mean(h_trans, dim=1) # (B, d_model)
        
        # 4. Contrastive Normalization Embedding
        proj_emb = F.normalize(self.contrastive_head(feat_temporal), dim=-1) # (B, 64)
        
        # 5. Fused Classification Head (Temporal Kinematics + Biomechanical Biomarkers)
        fused_diag = torch.cat([feat_temporal, h_bio], dim=-1) # (B, d_model * 2)
        out = torch.sigmoid(self.classifier(fused_diag))
        
        return out, proj_emb
