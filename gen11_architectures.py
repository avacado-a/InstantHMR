"""
Generation 11: Asymmetric Multi-Crop Hierarchical Kinematic Transformer (Asym_MCT_GCN)
=====================================================================================
Direct synthesis of the 3 champion archetypes to break through the 78% accuracy barrier:
  1. High-Precision Macroscopic Temporal Transformer Backbone (UltraLong)
  2. Decoupled Cartesian Position + Velocity/Jerk Kinematics (DualStream)
  3. Bilateral Asymmetric Difference Stream (Δ_bilateral = J_right - J_left)
  4. Hierarchical Sub-Graph GCN (40 Hand Keypoints vs 24 Body Keypoints)
  5. Multi-Crop Temporal Sliding Consensus with Asymmetric Focal Loss (γ=2.0, α=0.65)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hierarchical_SubGraph_GCN(nn.Module):
    """
    Decouples local distal hand tremor/bradykinesia (40 hand keypoints: joints 21-60)
    from macroscopic pelvic/torso/gait displacement (24 body keypoints: joints 0-20, 61-63).
    """
    def __init__(self, in_c=6, d_model=128):
        super().__init__()
        # Body Stream (24 joints)
        self.body_proj = nn.Linear(in_c * 24, d_model // 2)
        self.body_conv = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        
        # Hand Distal Stream (40 joints)
        self.hand_proj = nn.Linear(in_c * 40, d_model // 2)
        self.hand_conv = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU()
        )
        
        self.merge = nn.Linear(d_model, d_model)
        self.bn_merge = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: (B, C=6, T, V=70)
        B, C, T, V = x.size()
        
        # Body keypoints: 0-20 and 61-63 (total 24 joints)
        body_idx = list(range(0, 21)) + list(range(61, 64))
        # Hand keypoints: 21-60 (total 40 joints: 20 per hand)
        hand_idx = list(range(21, 61))
        
        x_body = x[:, :, :, body_idx].permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        x_hand = x[:, :, :, hand_idx].permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        
        h_body = self.body_conv(self.body_proj(x_body).permute(0, 2, 1)) # (B, d//2, T)
        h_hand = self.hand_conv(self.hand_proj(x_hand).permute(0, 2, 1)) # (B, d//2, T)
        
        h_hier = torch.cat([h_body, h_hand], dim=1).permute(0, 2, 1) # (B, T, d_model)
        h_hier = F.relu(self.bn_merge(self.merge(h_hier).permute(0, 2, 1)).permute(0, 2, 1))
        return h_hier


class Bilateral_Difference_Stream(nn.Module):
    """
    Computes explicit differential tokens: Δ_bilateral = J_right(t) - J_left(t)
    across paired anatomical landmarks:
      - Right Upper Limb & Hand: Joints 21-41
      - Left Upper Limb & Hand: Joints 42-62
    """
    def __init__(self, in_c=6, d_model=128):
        super().__init__()
        # Paired limb feature dimension: 6 channels * 21 joints = 126
        self.diff_proj = nn.Linear(in_c * 21, d_model)
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.asym_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C=6, T, V=70)
        B, C, T, V = x.size()
        r_limb = x[:, :, :, 21:42].permute(0, 2, 1, 3).contiguous().view(B, T, -1) # (B, T, 126)
        l_limb = x[:, :, :, 42:63].permute(0, 2, 1, 3).contiguous().view(B, T, -1) # (B, T, 126)
        
        diff = torch.abs(r_limb - l_limb)
        h_diff = self.tcn(self.diff_proj(diff).permute(0, 2, 1)).permute(0, 2, 1) # (B, T, d_model)
        
        h_r = self.diff_proj(r_limb)
        h_l = self.diff_proj(l_limb)
        gate = self.asym_gate(torch.cat([h_r, h_l], dim=-1))
        
        return h_diff * gate


class Asym_MCT_GCN(nn.Module):
    """
    Generation 11 Master Model: Asymmetric Multi-Crop Hierarchical Transformer
    """
    def __init__(self, in_c=6, num_c=1, d_model=128, nhead=8, num_layers=4, dropout=0.25):
        super().__init__()
        self.hier_gcn = Hierarchical_SubGraph_GCN(in_c=in_c, d_model=d_model)
        self.asym_stream = Bilateral_Difference_Stream(in_c=in_c, d_model=d_model)
        
        # Cross-Limb & Cross-Pathway Fusion
        self.cross_fusion = nn.Linear(d_model * 2, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)
        
        # UltraLong Temporal Transformer Bottleneck
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Asymmetric Decision Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_c)
        )

    def forward(self, x):
        # x: (B, C=6, T=120, V=70)
        B, C, T, V = x.size()
        
        # Branch 1: Hierarchical Hand/Body Decoupling
        h_hier = self.hier_gcn(x) # (B, T, d_model)
        
        # Branch 2: Explicit Bilateral Asymmetry Differential
        h_asym = self.asym_stream(x) # (B, T, d_model)
        
        # Cross-Limb Attention Fusion
        h_fused = self.cross_fusion(torch.cat([h_hier, h_asym], dim=-1))
        h_fused = h_fused + self.pos_encoder[:, :T, :]
        
        # Temporal Bottleneck
        h_trans = self.transformer(h_fused)
        
        # Multi-Scale Global Pooling
        feat_mean = torch.mean(h_trans, dim=1)
        feat_max, _ = torch.max(h_trans, dim=1)
        feat = (feat_mean + feat_max) * 0.5
        
        out = torch.sigmoid(self.classifier(feat))
        return out, feat


class MultiCrop_Consensus_Wrapper(nn.Module):
    """
    Evaluates multi-crop temporal sliding windows (5 overlapping sub-sequences per subject)
    and aggregates confidence to prevent single-window truncation errors.
    """
    def __init__(self, base_model, crop_len=90, num_crops=5):
        super().__init__()
        self.base_model = base_model
        self.crop_len = crop_len
        self.num_crops = num_crops

    def forward(self, x):
        # x: (B, 6, T=120, 70)
        B, C, T, V = x.size()
        if T <= self.crop_len:
            return self.base_model(x)
            
        crop_preds = []
        crop_feats = []
        step = (T - self.crop_len) // (self.num_crops - 1)
        
        for i in range(self.num_crops):
            start = i * step
            crop_x = x[:, :, start:start+self.crop_len, :]
            # Pad or interpolate crop to 120 frames for uniform transformer positional encoding
            crop_interp = F.interpolate(
                crop_x.permute(0, 1, 3, 2).contiguous().view(B, C*V, self.crop_len),
                size=120, mode='linear', align_corners=False
            ).view(B, C, V, 120).permute(0, 1, 3, 2)
            
            p, f = self.base_model(crop_interp)
            crop_preds.append(p)
            crop_feats.append(f)
            
        # Consensus voting
        avg_pred = torch.stack(crop_preds, dim=0).mean(dim=0)
        avg_feat = torch.stack(crop_feats, dim=0).mean(dim=0)
        return avg_pred, avg_feat
