"""
Generation 6: Bilateral Asymmetric Kinematic-Transformer (Asym_KinematicTransformer)
====================================================================================
Incorporates the key clinical hallmark of early Parkinson's Disease:
Unilateral Asymmetry (uneven arm swing amplitude + unilateral hand tremor divergence)
fused with Decoupled Velocity/Jerk Kinematics & Multi-Head Self-Attention.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Asym_KinematicTransformer(nn.Module):
    def __init__(self, in_c=6, num_c=1, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.pos_emb = nn.Linear(3 * 70, d_model)
        self.mot_emb = nn.Linear(3 * 70, d_model)
        
        # Bilateral Asymmetry Cross-Attention (Right vs Left Upper Limbs: Joints 21-41 vs 42-62)
        self.asym_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)
        self.fusion = nn.Linear(d_model * 3, d_model)
        
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
        
        # Global embeddings
        hp = self.pos_emb(pos) + self.pos_encoder[:, :T, :]
        hm = self.mot_emb(mot)
        
        # Bilateral Limb Asymmetry Extraction
        r_limb = x[:, :, :, 21:42].permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        l_limb = x[:, :, :, 42:63].permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        
        if r_limb.size(-1) < 128:
            r_limb = F.pad(r_limb, (0, 128 - r_limb.size(-1)))
            l_limb = F.pad(l_limb, (0, 128 - l_limb.size(-1)))
            
        asym_diff = torch.abs(r_limb - l_limb)
        gate = self.asym_gate(torch.cat([r_limb, l_limb], dim=-1))
        h_asym = asym_diff * gate
        
        # Tri-modal Fusion: Position + Motion Velocity + Bilateral Asymmetry Gate
        h_fused = self.fusion(torch.cat([hp, hm, h_asym], dim=-1))
        h_trans = self.transformer(h_fused)
        
        feat = torch.mean(h_trans, dim=1)
        out = torch.sigmoid(self.classifier(feat))
        return out, feat
