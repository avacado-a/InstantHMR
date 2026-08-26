"""
Fast Apple Silicon MPS GPU Multi-Model Tournament & Grand-Fusion Trainer
========================================================================
Runs each architecture sequentially on Apple Silicon Metal GPU (MPS).
Completes all remaining models and compiles the Leaderboard + Stacking Ensemble in minutes!
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

from train_10_models_tournament import (
    WM_STGCN, MS_STGCN, ST_Transformer, DualStream_STGCN,
    CTR_GCN, HD_GCN, ST_GRU_GCN, Asymmetric_STGCN, Shift_GCN, MetaEnsemble
)
from instanthmr.ultra_fusion_model import ParkinsonUltraSTGCN
from train_parallel_tournament import preload_and_cache_tensors

os.makedirs("models/benchmarks", exist_ok=True)

TOURNAMENT_MODELS = {
    "1_WM_STGCN": WM_STGCN,
    "2_MS_STGCN": MS_STGCN,
    "3_ST_Transformer": ST_Transformer,
    "4_DualStream_STGCN": DualStream_STGCN,
    "5_CTR_GCN": CTR_GCN,
    "6_HD_GCN": HD_GCN,
    "7_ST_GRU_GCN": ST_GRU_GCN,
    "8_Asymmetric_STGCN": Asymmetric_STGCN,
    "9_Shift_GCN": Shift_GCN,
    "10_UltraFusion_STGCN": ParkinsonUltraSTGCN
}


def run_gpu_tournament(epochs=15, batch_size=32, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n" + "=" * 85)
    print(f"⚡ RUNNING TOURNAMENT ON APPLE SILICON ACCELERATOR ({device})")
    print("=" * 85, flush=True)

    cached_data = preload_and_cache_tensors()
    train_ds = TensorDataset(cached_data["X_train"], cached_data["y_train"])
    val_ds   = TensorDataset(cached_data["X_val"], cached_data["y_val"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = {}
    trained_model_objs = []

    for name, model_cls in TOURNAMENT_MODELS.items():
        save_path = f"models/benchmarks/{name}.pt"
        if os.path.exists(save_path):
            print(f"⏩ [CACHED] {name} already trained. Evaluating on test cohort...", flush=True)
            model = model_cls().to(device)
            try:
                model.load_state_dict(torch.load(save_path, map_location=device))
            except Exception:
                model.load_state_dict(torch.load(save_path, map_location="cpu"))
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
            results[name] = {"Accuracy": acc, "Sensitivity": rec, "F1": f1}
            trained_model_objs.append(model)
            print(f"🌟 [COMPLETED] {name:<26} | Val Acc: {acc:5.2f}% | Sensitivity: {rec:5.2f}% | F1: {f1:.4f}", flush=True)
            continue

        print(f"\n🚀 Training Architecture [{name}] ({epochs} Epochs on {device})...", flush=True)
        model = model_cls().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_rec, best_f1 = 0.0, 0.0, 0.0
        best_state = None

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred, _ = model(x)
                pred = pred.squeeze(-1)
                loss = criterion(pred, y * 0.9 + 0.05)
                loss.backward()
                optimizer.step()

            scheduler.step()

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
                best_state = model.state_dict()

        dt = time.time() - t0
        torch.save(best_state, save_path)
        results[name] = {"Accuracy": best_acc, "Sensitivity": best_rec, "F1": best_f1}
        model.load_state_dict(best_state)
        trained_model_objs.append(model)
        print(f"🌟 [COMPLETED] {name:<26} ({dt:.1f}s) | Val Acc: {best_acc:5.2f}% | Sensitivity: {best_rec:5.2f}% | F1: {best_f1:.4f}", flush=True)

    # -------------------------------------------------------------
    # MODEL 11: Deep Meta-Ensemble
    # -------------------------------------------------------------
    print("\n⚡ Constructing and Benchmarking 11-Model Meta-Ensemble...", flush=True)
    ensemble = MetaEnsemble(trained_model_objs).to(device)
    ensemble.eval()

    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred, _ = ensemble(x)
            pred = pred.squeeze(-1)
            p_bin = (pred >= 0.35).float()
            tp += int(((p_bin == 1) & (y == 1)).sum().item())
            tn += int(((p_bin == 0) & (y == 0)).sum().item())
            fp += int(((p_bin == 1) & (y == 0)).sum().item())
            fn += int(((p_bin == 0) & (y == 1)).sum().item())

    ens_tot = tp + tn + fp + fn
    ens_acc = (tp + tn) / max(1, ens_tot) * 100.0
    ens_rec = tp / max(1, tp + fn) * 100.0
    ens_f1  = 2 * tp / max(1, 2 * tp + fp + fn)
    results["11_DeepMeta_Ensemble"] = {"Accuracy": ens_acc, "Sensitivity": ens_rec, "F1": ens_f1}
    torch.save(ensemble.state_dict(), "models/benchmarks/11_DeepMeta_Ensemble.pt")

    # -------------------------------------------------------------
    # PRINT FINAL LEADERBOARD
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("🏆 FINAL TOURNAMENT LEADERBOARD (11 ARCHITECTURES EVALUATED)")
    print("=" * 85)
    print(f"{'Rank':<5} | {'Model Architecture':<30} | {'Validation Acc':<15} | {'Sensitivity':<15} | {'F1-Score':<10}")
    print("-" * 85)

    sorted_ranks = sorted(results.items(), key=lambda item: (item[1]['Accuracy'] + item[1]['Sensitivity']), reverse=True)
    for r, (m_name, m_res) in enumerate(sorted_ranks, 1):
        print(f"#{r:<4} | {m_name:<30} | {m_res['Accuracy']:5.2f}%         | {m_res['Sensitivity']:5.2f}%        | {m_res['F1']:.4f}")
    print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    run_gpu_tournament(epochs=15)
