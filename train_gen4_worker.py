"""
Generation 4 Autonomous Mutation Trainer
========================================
Trains 4 distinct hyperparameter mutations of the new Kinematic_Spectral_Transformer champion:
  1. Gen4_KinematicTrans_Base     (lr=1e-3, drop=0.2, 4 heads)
  2. Gen4_KinematicTrans_Deep     (lr=5e-4, drop=0.25, 6 layers, 8 heads)
  3. Gen4_KinematicTrans_CosineWD (lr=8e-4, wd=1e-3, CosineAnnealingWarmRestarts)
  4. Gen4_KinematicTrans_Focal    (BinaryFocalLoss gamma=1.5, alpha=0.55)
"""

import os
import sys
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen4_architectures import Kinematic_Spectral_Transformer
from round3_architectures import BinaryFocalLoss

CONFIGS = {
    "1_Gen4_KinematicTrans_Base": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 1e-3, "wd": 1e-4, "loss": "bce"},
    "2_Gen4_KinematicTrans_Deep": {"d_model": 128, "nhead": 8, "layers": 4, "lr": 5e-4, "wd": 1e-4, "loss": "bce"},
    "3_Gen4_KinematicTrans_CosineWD": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 8e-4, "wd": 1e-3, "loss": "bce"},
    "4_Gen4_KinematicTrans_Focal": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 8e-4, "wd": 1e-4, "loss": "focal"},
}

os.makedirs("models/gen4", exist_ok=True)

def evaluate(model, val_x, val_y, batch_size=64):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(val_x), batch_size):
            bx = val_x[i:i+batch_size]
            by = val_y[i:i+batch_size]
            p, _ = model(bx)
            all_preds.extend(p.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(by.cpu().numpy().tolist())
            
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    binary = (preds_arr >= 0.50).astype(float)
    acc = np.mean(binary == labels_arr) * 100.0
    
    tp = np.sum((binary == 1.0) & (labels_arr == 1.0))
    fn = np.sum((binary == 0.0) & (labels_arr == 1.0))
    fp = np.sum((binary == 1.0) & (labels_arr == 0.0))
    sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1 = (2 * (prec/100) * (sens/100) / ((prec/100) + (sens/100))) if (prec + sens) > 0 else 0.0
    
    b35 = (preds_arr >= 0.35).astype(float)
    tp35 = np.sum((b35 == 1.0) & (labels_arr == 1.0))
    fn35 = np.sum((b35 == 0.0) & (labels_arr == 1.0))
    sens35 = (tp35 / (tp35 + fn35) * 100.0) if (tp35 + fn35) > 0 else 0.0
    return acc, sens, sens35, prec, f1

def train_worker(name, cfg, X_train, y_train, X_val, y_val, epochs=60):
    torch.set_num_threads(2)
    model = Kinematic_Spectral_Transformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["layers"]
    )
    
    criterion = BinaryFocalLoss(gamma=1.5) if cfg["loss"] == "focal" else nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_acc, best_sens, best_s35, best_prec, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    best_ep = 0
    save_path = f"models/gen4/{name}_best.pt"
    
    N = len(X_train)
    batch_size = 32
    print(f"🚀 [{name}] Launching Gen 4 Mutation Training ({epochs} Epochs)...", flush=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        train_loss = 0.0
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx = X_train[idx]
            by = y_train[idx]
            
            if np.random.rand() > 0.5:
                bx = bx.clone()
                bx[:, :3] += torch.randn_like(bx[:, :3]) * 0.0015
                
            optimizer.zero_grad()
            pred, _ = model(bx)
            loss = criterion(pred.squeeze(-1), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)
            
        scheduler.step()
        train_loss /= N
        acc, sens, s35, prec, f1 = evaluate(model, X_val, y_val)
        
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc, best_sens, best_s35, best_prec, best_f1 = acc, sens, s35, prec, f1
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
            
        print(f"[{name}] Ep +{epoch:02d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {acc:.2f}% (Peak: {best_acc:.2f}% @ +{best_ep}) | Sens: {sens:.2f}% (T=0.35: {s35:.2f}%) | Prec: {prec:.2f}% | F1: {f1:.4f}", flush=True)

if __name__ == "__main__":
    m_name = sys.argv[1] if len(sys.argv) > 1 else "1_Gen4_KinematicTrans_Base"
    eps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    cached = torch.load("data/cached_mhr70_tensors.pt", map_location="cpu", weights_only=False)
    X_train, y_train = cached["X_train"], cached["y_train"]
    X_val, y_val = cached["X_val"], cached["y_val"]
    
    train_worker(m_name, CONFIGS[m_name], X_train, y_train, X_val, y_val, eps)
