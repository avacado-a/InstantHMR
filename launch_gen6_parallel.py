"""
Generation 6 Master Parallel Launcher
=====================================
Spawns 4 distinct variations of Bilateral Asymmetric Kinematic-Transformer (Asym_KinematicTransformer):
  1. Gen6_Asym_CosineWD_Base  (lr=8e-4, wd=1e-3, CosineAnnealingLR)
  2. Gen6_Asym_Focal_Tuned    (lr=8e-4, wd=1e-4, FocalLoss gamma=1.5)
  3. Gen6_Asym_DeepAttn       (lr=5e-4, wd=5e-4, 4 layers, 8 heads, FocalLoss gamma=1.5)
  4. Gen6_Asym_HighReg        (lr=1e-3, wd=2e-3, drop=0.3, BCE Loss)
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

from gen6_architectures import Asym_KinematicTransformer
from round3_architectures import BinaryFocalLoss

CONFIGS = {
    "1_Gen6_Asym_CosineWD_Base": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 8e-4, "wd": 1e-3, "loss": "bce", "gamma": 1.0},
    "2_Gen6_Asym_Focal_Tuned": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 8e-4, "wd": 1e-4, "loss": "focal", "gamma": 1.5},
    "3_Gen6_Asym_DeepAttn": {"d_model": 128, "nhead": 8, "layers": 4, "lr": 5e-4, "wd": 5e-4, "loss": "focal", "gamma": 1.5},
    "4_Gen6_Asym_HighReg": {"d_model": 128, "nhead": 4, "layers": 3, "lr": 1e-3, "wd": 2e-3, "loss": "bce", "gamma": 1.0},
}

os.makedirs("models/gen6", exist_ok=True)
LOG_FILE = "data/gen6_convergence.log"

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
    model = Asym_KinematicTransformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["layers"]
    )
    
    criterion = BinaryFocalLoss(gamma=cfg["gamma"]) if cfg["loss"] == "focal" else nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_acc, best_sens, best_s35, best_prec, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    best_ep = 0
    save_path = f"models/gen6/{name}_best.pt"
    
    N = len(X_train)
    batch_size = 32
    print(f"🚀 [{name}] Starting Gen 6 Asymmetry Training ({epochs} Epochs)...", flush=True)
    
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
    if len(sys.argv) > 1 and sys.argv[1] in CONFIGS:
        m_name = sys.argv[1]
        eps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        cached = torch.load("data/cached_mhr70_tensors.pt", map_location="cpu", weights_only=False)
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_val, y_val = cached["X_val"], cached["y_val"]
        train_worker(m_name, CONFIGS[m_name], X_train, y_train, X_val, y_val, eps)
    else:
        print("=" * 85)
        print("⚡ LAUNCHING GENERATION 6 BILATERAL ASYMMETRIC TRANSFORMER (4 PARALLEL WORKERS)")
        print("=" * 85)
        
        procs = []
        for name in CONFIGS.keys():
            cmd = ["/opt/anaconda3/bin/python3", "-u", "launch_gen6_parallel.py", name, "60"]
            log_h = open(f"data/{name}.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_h, stderr=subprocess.STDOUT)
            procs.append((name, proc, log_h))
            print(f"🚀 [GEN 6 WORKER STARTED] PID {proc.pid}: {name}")
            
        print(f"\n✓ All 4 Gen-6 candidates running simultaneously in parallel!")
        running = True
        while running:
            time.sleep(2)
            running = any(p.poll() is None for _, p, _ in procs)
            with open(LOG_FILE, "w") as master:
                master.write("=" * 85 + "\n")
                master.write("⚡ GENERATION 6 ASYM-KINEMATIC-TRANSFORMER LIVE AGGREGATED LOGS\n")
                master.write("=" * 85 + "\n\n")
                for m_name, p, _ in procs:
                    status = "RUNNING" if p.poll() is None else f"DONE (code {p.returncode})"
                    master.write(f"--- [WORKER: {m_name} | {status}] ---\n")
                    if os.path.exists(f"data/{m_name}.log"):
                        with open(f"data/{m_name}.log", "r") as mf:
                            lines = mf.readlines()[-12:]
                            master.write("".join(lines) + "\n")
                    master.write("\n")
                    
        print("🌟 ALL 4 GENERATION 6 CANDIDATES COMPLETED TRAINING!")
