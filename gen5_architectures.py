"""
Generation 5: Multi-Scale Wavelet Kinematic-Transformer (MST_KinematicTransformer)
=================================================================================
Synthesizes all breakthroughs discovered so far:
1. Dual-Pathway Kinematics (Positions + Velocities)
2. Multi-Scale Temporal Wavelet Dilations (d=1, 2, 4) to capture 1-8 Hz tremor frequencies
3. Multi-Head Cross-Attention Self-Attention (d_model=128, nhead=4)
4. Tuned Focal Loss (gamma=1.5, alpha=0.55)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScale_Wavelet_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c_quarter = out_channels // 4
        self.b1 = nn.Conv1d(in_channels, c_quarter, kernel_size=3, padding=1, dilation=1)
        self.b2 = nn.Conv1d(in_channels, c_quarter, kernel_size=5, padding=2, dilation=1)
        self.b3 = nn.Conv1d(in_channels, c_quarter, kernel_size=5, padding=4, dilation=2)
        self.b4 = nn.Conv1d(in_channels, c_quarter, kernel_size=5, padding=8, dilation=4)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return F.relu(self.bn(out))

class MST_KinematicTransformer(nn.Module):
    def __init__(self, in_c=6, num_c=1, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        # Pathway 1: Raw 3D Joint Geometry
        self.pos_emb = nn.Linear(3 * 70, d_model)
        self.pos_wavelet = MultiScale_Wavelet_Block(d_model, d_model)
        
        # Pathway 2: Velocity & Acceleration Kinematics
        self.mot_emb = nn.Linear(3 * 70, d_model)
        self.mot_wavelet = MultiScale_Wavelet_Block(d_model, d_model)
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_c)
        )

    def forward(self, x):
        # x: (B, 6, T, 70)
        B, C, T, V = x.size()
        pos = x[:, :3].permute(0, 2, 1, 3).contiguous().view(B, T, 3 * V)
        mot = x[:, 3:6].permute(0, 2, 1, 3).contiguous().view(B, T, 3 * V)
        
        # Linear projections
        hp = self.pos_emb(pos).permute(0, 2, 1) # (B, d_model, T)
        hp = self.pos_wavelet(hp).permute(0, 2, 1) # (B, T, d_model)
        hp = hp + self.pos_encoder[:, :T, :]
        
        hm = self.mot_emb(mot).permute(0, 2, 1)
        hm = self.mot_wavelet(hm).permute(0, 2, 1)
        
        # Fused Representation
        h_fused = self.fusion(torch.cat([hp, hm], dim=-1))
        h_trans = self.transformer(h_fused)
        
        feat = torch.mean(h_trans, dim=1)
        out = torch.sigmoid(self.classifier(feat))
        return out, feat
