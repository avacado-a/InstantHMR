"""
Lightning Fast Direct-Tensor Tournament Runner (Zero DataLoader Overhead)
========================================================================
Runs directly on in-memory PyTorch tensors in chunks of 64 for lightning-fast training & evaluation.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_10_models_tournament import (
    WM_STGCN, MS_STGCN, ST_Transformer, DualStream_STGCN,
    CTR_GCN, HD_GCN, ST_GRU_GCN, Asymmetric_STGCN, Shift_GCN, MetaEnsemble
)
from instanthmr.ultra_fusion_model import ParkinsonUltraSTGCN

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


def eval_tensor_dataset(model, X, y, batch_size=64):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    N = X.size(0)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            pred, _ = model(xb)
            pred = pred.squeeze(-1)
            p_bin = (pred >= 0.35).float()
            tp += int(((p_bin == 1) & (yb == 1)).sum().item())
            tn += int(((p_bin == 0) & (yb == 0)).sum().item())
            fp += int(((p_bin == 1) & (yb == 0)).sum().item())
            fn += int(((p_bin == 0) & (yb == 1)).sum().item())
    val_tot = tp + tn + fp + fn
    acc = (tp + tn) / max(1, val_tot) * 100.0
    rec = tp / max(1, tp + fn) * 100.0
    f1  = 2 * tp / max(1, 2 * tp + fp + fn)
    return acc, rec, f1


def main():
    torch.set_num_threads(8)
    print("=" * 85)
    print("⚡ INSTANT HIGH-SPEED DIRECT-TENSOR TOURNAMENT")
    print("=" * 85, flush=True)

    cache_path = "data/cached_mhr70_tensors.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    X_train, y_train = cached["X_train"], cached["y_train"]
    X_val, y_val     = cached["X_val"], cached["y_val"]
    print(f"✓ Loaded {len(X_train)} Train and {len(X_val)} Validation tensors directly into RAM.\n", flush=True)

    results = {}
    trained_model_objs = []
    epochs = 15
    batch_size = 64

    for name, model_cls in TOURNAMENT_MODELS.items():
        save_path = f"models/benchmarks/{name}.pt"
        if os.path.exists(save_path):
            print(f"⏩ [CACHED] Evaluating {name}...", flush=True)
            model = model_cls()
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
            acc, rec, f1 = eval_tensor_dataset(model, X_val, y_val, batch_size=batch_size)
            results[name] = {"Accuracy": acc, "Sensitivity": rec, "F1": f1}
            trained_model_objs.append(model)
            print(f"🌟 [COMPLETED] {name:<26} | Val Acc: {acc:5.2f}% | Sensitivity: {rec:5.2f}% | F1: {f1:.4f}\n", flush=True)
            continue

        print(f"🚀 Training Architecture [{name}] ({epochs} Epochs)...", flush=True)
        model = model_cls()
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_rec, best_f1 = 0.0, 0.0, 0.0
        best_state = None
        t0 = time.time()
        N = X_train.size(0)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(N)
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]

            for i in range(0, N, batch_size):
                xb = X_shuff[i:i+batch_size]
                yb = y_shuff[i:i+batch_size]
                optimizer.zero_grad()
                pred, _ = model(xb)
                loss = criterion(pred.squeeze(-1), yb * 0.9 + 0.05)
                loss.backward()
                optimizer.step()

            scheduler.step()
            acc, rec, f1 = eval_tensor_dataset(model, X_val, y_val, batch_size=batch_size)
            print(f"   ↳ Epoch {epoch:02d}/{epochs:02d} | Val Acc: {acc:5.2f}% | Sens: {rec:5.2f}% | F1: {f1:.4f}", flush=True)

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
        print(f"🌟 [COMPLETED] {name:<26} ({dt:.1f}s) | Val Acc: {best_acc:5.2f}% | Sensitivity: {best_rec:5.2f}% | F1: {best_f1:.4f}\n", flush=True)

    # Meta Ensemble
    print("=" * 85)
    print("⚡ CONSTRUCTING 11-MODEL META-ENSEMBLE...")
    print("=" * 85, flush=True)
    ensemble = MetaEnsemble(trained_model_objs)
    ens_acc, ens_rec, ens_f1 = eval_tensor_dataset(ensemble, X_val, y_val, batch_size=batch_size)
    results["11_DeepMeta_Ensemble"] = {"Accuracy": ens_acc, "Sensitivity": ens_rec, "F1": ens_f1}
    torch.save(ensemble.state_dict(), "models/benchmarks/11_DeepMeta_Ensemble.pt")
    print(f"🌟 [COMPLETED] 11_DeepMeta_Ensemble        | Val Acc: {ens_acc:5.2f}% | Sensitivity: {ens_rec:5.2f}% | F1: {ens_f1:.4f}\n", flush=True)

    # Leaderboard
    print("\n" + "=" * 85)
    print("🏆 FINAL TOURNAMENT LEADERBOARD (ALL ARCHITECTURES EVALUATED)")
    print("=" * 85)
    print(f"{'Rank':<5} | {'Model Architecture':<30} | {'Validation Acc':<15} | {'Sensitivity':<15} | {'F1-Score':<10}")
    print("-" * 85)

    sorted_ranks = sorted(results.items(), key=lambda item: (item[1]['Accuracy'] + item[1]['Sensitivity']), reverse=True)
    for r, (m_name, m_res) in enumerate(sorted_ranks, 1):
        print(f"#{r:<4} | {m_name:<30} | {m_res['Accuracy']:5.2f}%         | {m_res['Sensitivity']:5.2f}%        | {m_res['F1']:.4f}")
    print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    main()
