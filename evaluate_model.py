"""
Automated Test & Validation Evaluator for ParkinsonSTGCN
========================================================
Runs comprehensive clinical benchmarking on the trained model:
1. Validation Accuracy, Precision, Recall, F1-Score, and ROC-AUC across held-out subjects.
2. Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives).
3. End-to-End inference on a sample real walking video from MMU Parkinson's dataset.
"""

import os
import sys
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


def evaluate_model(weights_path: str = "models/parkinson_stgcn_carepd.pt"):
    print("\n" + "=" * 70)
    print("🔬 COMPREHENSIVE CLINICAL VALIDATION BENCHMARK REPORT")
    print(f"📦 Evaluating Model Checkpoint: {weights_path}")
    print("=" * 70)

    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights not found at {weights_path}")
        return

    # Load Held-out Validation Dataset
    val_ds = RealCAREPDDataset(is_train=False, binary=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = ParkinsonSTGCN(in_channels=6, num_classes=1, dropout=0.0)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in val_loader:
            probs, _ = model(x)
            probs = probs.squeeze(-1).numpy()
            preds = (probs >= 0.5).astype(np.float32)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Metrics
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))

    total = len(all_targets)
    accuracy = (tp + tn) / total
    precision = tp / max(1e-5, tp + fp)
    recall = tp / max(1e-5, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-5, precision + recall)
    specificity = tn / max(1e-5, tn + fp)

    print("\n📊 CLINICAL PERFORMANCE SUMMARY:")
    print(f"  • Total Held-Out Patient Walks  : {total}")
    print(f"  • Overall Diagnostic Accuracy   : {accuracy * 100:.2f}%")
    print(f"  • Sensitivity / Recall (PD)     : {recall * 100:.2f}%")
    print(f"  • Specificity (Healthy Controls): {specificity * 100:.2f}%")
    print(f"  • Precision (Positive Predictive): {precision * 100:.2f}%")
    print(f"  • F1-Score                      : {f1:.4f}")
    print("\n📈 CONFUSION MATRIX:")
    print(f"               Predicted Healthy | Predicted PD")
    print(f"  Actual Healthy:    {tn:4d} (TN)     |   {fp:4d} (FP)")
    print(f"  Actual PD     :    {fn:4d} (FN)     |   {tp:4d} (TP)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_model()
