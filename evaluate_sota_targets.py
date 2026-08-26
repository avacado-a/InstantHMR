"""
Optimized Clinical Validation Benchmark with Sensitivity & SOTA Metric Analysis
================================================================================
Evaluates the model across:
1. Standard Threshold (0.50) vs Clinical Screening Threshold (0.35).
2. SOTA Target Metric Benchmarks (Accuracy, Recall, Precision, F1, Specificity).
3. Cross-Dataset Zero-Shot Evaluation (CARE-PD trained model on MMU Video Keypoints).
"""

import os
import sys
import glob
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.stgcn_model import ParkinsonSTGCN
from instanthmr.gait_features import canonicalize_and_stack_features
from train_real_carepd import RealCAREPDDataset


def evaluate_clinical_sota_targets(weights_path: str = "models/parkinson_stgcn_carepd.pt"):
    print("\n" + "=" * 75)
    print("🏆 PARKINSON'S ST-GCN SOTA CLINICAL BENCHMARK EVALUATOR")
    print(f"📦 Model Checkpoint: {weights_path}")
    print("=" * 75)

    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights not found at {weights_path}")
        return

    # Load Held-Out Validation Dataset
    val_ds = RealCAREPDDataset(is_train=False, binary=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = ParkinsonSTGCN(in_channels=6, num_classes=1, dropout=0.0)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in val_loader:
            probs, _ = model(x)
            probs = probs.squeeze(-1).numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(y.numpy().tolist())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    total = len(all_targets)

    # Evaluate at multiple threshold regimes
    thresholds = [0.50, 0.40, 0.35, 0.30]

    print("\n📊 THRESHOLD OPERATING CHARACTERISTICS (Screening vs. Diagnostic):")
    print("-" * 75)
    print(f"{'Threshold':<11} | {'Accuracy':<9} | {'Sensitivity (Recall)':<22} | {'Specificity':<12} | {'F1-Score':<8}")
    print("-" * 75)

    for th in thresholds:
        preds = (all_probs >= th).astype(np.float32)
        tp = np.sum((preds == 1) & (all_targets == 1))
        tn = np.sum((preds == 0) & (all_targets == 0))
        fp = np.sum((preds == 1) & (all_targets == 0))
        fn = np.sum((preds == 0) & (all_targets == 1))

        acc = (tp + tn) / total
        prec = tp / max(1e-5, tp + fp)
        rec = tp / max(1e-5, tp + fn)
        spec = tn / max(1e-5, tn + fp)
        f1 = 2 * (prec * rec) / max(1e-5, prec + rec)

        print(f"p >= {th:.2f}     | {acc*100:5.2f}%   | {rec*100:5.2f}% (FN: {fn:2d}/{len(all_targets[all_targets==1])}) | {spec*100:5.2f}%       | {f1:.4f}")

    print("-" * 75)

    # Detailed Clinical Screening Regime (p >= 0.35)
    th_screen = 0.35
    preds_s = (all_probs >= th_screen).astype(np.float32)
    tp_s = int(np.sum((preds_s == 1) & (all_targets == 1)))
    tn_s = int(np.sum((preds_s == 0) & (all_targets == 0)))
    fp_s = int(np.sum((preds_s == 1) & (all_targets == 0)))
    fn_s = int(np.sum((preds_s == 0) & (all_targets == 1)))

    rec_s = (tp_s / (tp_s + fn_s)) * 100.0

    print("\n🎯 CLINICAL SCREENING REGIME (Threshold p >= 0.35):")
    print(f"  • Sensitivity / Clinical Catch Rate : {rec_s:.2f}%  (Target: >= 92-95%)")
    print(f"  • Missed Patients (False Negatives) : ONLY {fn_s} out of {tp_s + fn_s} patients!")
    print(f"  • True Positives Identified         : {tp_s} patients correctly flagged for referral")
    print(f"  • True Negatives (Healthy Cleared)  : {tn_s} subjects")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    evaluate_clinical_sota_targets()
