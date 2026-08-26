"""
MPS GPU-Accelerated Grand-Fusion Ultra-STGCN Trainer
=====================================================
Trains the Grand-Fusion ParkinsonUltraSTGCN on Apple Silicon Metal Performance Shaders (MPS GPU).
Features:
- Native Apple Silicon GPU execution (`device="mps"`).
- Dual-Stream (Coordinates + Camera Acceleration/Velocity).
- Multi-Scale Temporal Dilations (d=1, 2, 4).
- Channel-Wise Graph Topology Refinement (CTR-GCN).
- Bilateral Asymmetry Gating (Left vs. Right Arm/Leg comparison).
- 2-Layer Temporal Self-Attention Transformer.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.ultra_fusion_model import ParkinsonUltraSTGCN
from train_parallel_tournament import preload_and_cache_tensors


def train_gpu_ultrafusion(epochs=30, batch_size=32, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"🚀 TRAINING PARKINSON-ULTRA-STGCN ON APPLE SILICON GPU ({device})")
    print("=" * 80, flush=True)

    cached_data = preload_and_cache_tensors()
    train_ds = TensorDataset(cached_data["X_train"], cached_data["y_train"])
    val_ds   = TensorDataset(cached_data["X_val"], cached_data["y_val"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ParkinsonUltraSTGCN(in_c=6, num_c=1, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc, best_rec, best_f1 = 0.0, 0.0, 0.0
    save_path = "models/parkinson_ultrafusion_gpu_best.pt"

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_corr, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, _ = model(x)
            pred = pred.squeeze(-1)
            loss = criterion(pred, y * 0.9 + 0.05)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_corr += ((pred >= 0.5) == (y >= 0.5)).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = (train_corr / total) * 100.0

        # Val evaluation
        model.eval()
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, _ = model(x)
                pred = pred.squeeze(-1)
                p_bin = (pred >= 0.35).float()

                tp += int(((p_bin == 1) & (y == 1)).sum().item())
                tn += int(((p_bin == 0) & (y == 0)).sum().item())
                fp += int(((p_bin == 1) & (y == 0)).sum().item())
                fn += int(((p_bin == 0) & (y == 1)).sum().item())

        val_tot = tp + tn + fp + fn
        acc = (tp + tn) / max(1, val_tot) * 100.0
        rec = tp / max(1, tp + fn) * 100.0
        f1  = 2 * tp / max(1, 2 * tp + fp + fn)

        if acc > best_acc or (acc == best_acc and rec > best_rec):
            best_acc = acc
            best_rec = rec
            best_f1  = f1
            torch.save(model.state_dict(), save_path)

        if epoch % 2 == 0 or epoch == epochs:
            print(f"  [Epoch {epoch:02d}/{epochs:02d}] Train Acc: {train_acc:5.1f}% | Val Acc: {acc:5.1f}% | Sensitivity: {rec:5.1f}% | Best Acc: {best_acc:5.1f}%", flush=True)

    t_tot = time.perf_counter() - t0
    print(f"\n🎉 GPU Training Finished in {t_tot:.1f}s!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}% | Best Sensitivity: {best_rec:.2f}% | F1: {best_f1:.4f}")
    print(f"💾 Checkpoint Saved to: {save_path}\n", flush=True)


if __name__ == "__main__":
    train_gpu_ultrafusion(epochs=30)
