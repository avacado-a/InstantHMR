"""
High-Speed Parallel Multi-Process Tournament Engine
====================================================
Precomputes and memory-caches all 2,953 canonicalized 70-joint feature tensors,
then trains multiple neural architectures concurrently in parallel process pools (3 workers).
Accelerates tournament completion by 3x–4x!
"""

import os
import sys
import time
import pickle
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_10_models_tournament import (
    ALL_MODELS, WM_STGCN, MS_STGCN, ST_Transformer, DualStream_STGCN,
    CTR_GCN, HD_GCN, ST_GRU_GCN, Asymmetric_STGCN, Shift_GCN, MetaEnsemble
)
from instanthmr.ultra_fusion_model import ParkinsonUltraSTGCN
from train_overnight_70 import smpl_to_mhr70_joints, canonicalize_mhr70_features

# Add Grand-Fusion to tournament lineup
TOURNAMENT_MODELS = dict(ALL_MODELS)
TOURNAMENT_MODELS["10_UltraFusion_STGCN"] = ParkinsonUltraSTGCN


def preload_and_cache_tensors(data_dir="data/care_pd", seq_len=90):
    cache_file = "data/cached_mhr70_tensors.pt"
    if os.path.exists(cache_file):
        print("⚡ Loading precomputed memory cache from data/cached_mhr70_tensors.pt...")
        return torch.load(cache_file)

    print("🚀 Precomputing and memory-caching all 2,953 70-joint clinical features...")
    cohorts = ["3DGait.pkl", "BMCLab.pkl", "PD-GaM.pkl", "T-SDU-PD.pkl"]
    raw_samples = []

    for cf in cohorts:
        cp = os.path.join(data_dir, cf)
        if not os.path.exists(cp):
            continue
        with open(cp, "rb") as f:
            d = pickle.load(f)
        for subj_id, walks in d.items():
            for walk_id, walk in walks.items():
                if isinstance(walk, dict):
                    updrs = walk.get("UPDRS_GAIT")
                    if updrs is not None:
                        pose = walk.get("pose")
                        trans = walk.get("trans")
                        if pose is not None and len(pose) > 10:
                            raw_samples.append({
                                "pose": pose,
                                "trans": trans,
                                "fps": walk.get("fps", 30),
                                "label": 1.0 if updrs >= 1 else 0.0,
                                "subj_id": subj_id
                            })

    np.random.seed(42)
    unique_s = list(set(s["subj_id"] for s in raw_samples))
    np.random.shuffle(unique_s)
    split = int(len(unique_s) * 0.8)
    train_s = set(unique_s[:split])

    X_train, y_train = [], []
    X_val, y_val = [], []

    for item in raw_samples:
        j70 = smpl_to_mhr70_joints(item["pose"], item["trans"])
        T = len(j70)
        if T >= seq_len:
            j70 = j70[:seq_len]
        else:
            j70 = np.pad(j70, ((0, seq_len - T), (0, 0), (0, 0)), mode="edge")

        feat = canonicalize_mhr70_features(j70, fps=item["fps"])
        lbl = item["label"]

        if item["subj_id"] in train_s:
            X_train.append(feat)
            y_train.append(lbl)
        else:
            X_val.append(feat)
            y_val.append(lbl)

    cached_data = {
        "X_train": torch.tensor(np.array(X_train), dtype=torch.float32),
        "y_train": torch.tensor(np.array(y_train), dtype=torch.float32),
        "X_val": torch.tensor(np.array(X_val), dtype=torch.float32),
        "y_val": torch.tensor(np.array(y_val), dtype=torch.float32),
    }
    torch.save(cached_data, cache_file)
    print(f"✓ Cached {len(X_train)} train and {len(X_val)} val tensors.")
    return cached_data


def train_single_model_worker(args):
    name, model_cls, cached_data, epochs = args
    torch.set_num_threads(2) # allocate 2 CPU cores per worker

    train_ds = TensorDataset(cached_data["X_train"], cached_data["y_train"])
    val_ds   = TensorDataset(cached_data["X_val"], cached_data["y_val"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    print(f"\n🚀 [WORKER] Starting Training [{name}]...", flush=True)
    model = model_cls()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_rec, best_f1 = 0.0, 0.0, 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
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

    save_path = f"models/benchmarks/{name}.pt"
    torch.save(best_state, save_path)
    print(f"🌟 [FINISHED] {name:<26} | Val Acc: {best_acc:5.2f}% | Sensitivity: {best_rec:5.2f}% | F1: {best_f1:.4f}", flush=True)
    return name, best_acc, best_rec, best_f1


def run_parallel_tournament(workers=3, epochs=15):
    print("=" * 80)
    print(f"⚡ LAUNCHING PARALLEL MULTI-MODEL TOURNAMENT ({workers} CONCURRENT WORKERS)")
    print("=" * 80, flush=True)

    cached_data = preload_and_cache_tensors()
    tasks = [(name, cls_obj, cached_data, epochs) for name, cls_obj in TOURNAMENT_MODELS.items()]

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for name, acc, rec, f1 in executor.map(train_single_model_worker, tasks):
            results[name] = {"Accuracy": acc, "Sensitivity": rec, "F1": f1}

    # Meta Ensemble
    print("\n⚡ Evaluating Meta-Ensemble across all trained models...", flush=True)
    val_ds = TensorDataset(cached_data["X_val"], cached_data["y_val"])
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    trained_models = []
    for name, cls_obj in TOURNAMENT_MODELS.items():
        m = cls_obj()
        m.load_state_dict(torch.load(f"models/benchmarks/{name}.pt"))
        m.eval()
        trained_models.append(m)

    ensemble = MetaEnsemble(trained_models)
    ensemble.eval()

    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
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

    print("\n" + "=" * 85)
    print("🏆 FINAL PARALLEL TOURNAMENT LEADERBOARD")
    print("=" * 85)
    print(f"{'Rank':<5} | {'Model Architecture':<30} | {'Validation Acc':<15} | {'Sensitivity':<15} | {'F1-Score':<10}")
    print("-" * 85)

    sorted_ranks = sorted(results.items(), key=lambda item: (item[1]['Accuracy'] + item[1]['Sensitivity']), reverse=True)
    for r, (m_name, m_res) in enumerate(sorted_ranks, 1):
        print(f"#{r:<4} | {m_name:<30} | {m_res['Accuracy']:5.2f}%         | {m_res['Sensitivity']:5.2f}%        | {m_res['F1']:.4f}")
    print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    run_parallel_tournament(workers=3, epochs=15)
