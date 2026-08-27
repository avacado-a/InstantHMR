import os
import torch
import numpy as np
from gen5_architectures import MST_KinematicTransformer

cached = torch.load("data/cached_mhr70_tensors.pt", map_location="cpu", weights_only=False)
X_val, y_val = cached["X_val"], cached["y_val"]
labels = y_val.numpy()

models = [
    ("1_Gen5_MST_Focal_Base", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("2_Gen5_MST_Focal_HeavyWD", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("3_Gen5_MST_Cosine_BCE", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("4_Gen5_MST_DeepCrossAttn", {"d_model": 128, "nhead": 8, "layers": 4}),
]

print("=" * 85)
print("🏆 GENERATION 5 MST-KINEMATIC-TRANSFORMER MUTATION BENCHMARKS")
print("=" * 85)
print(f"{'Model Name':<32} | {'Val Acc':<9} | {'Sens (0.50)':<12} | {'Recall (0.35)':<13} | {'Prec':<8} | {'F1':<6}")
print("-" * 85)

preds_list = []
for name, cfg in models:
    pth = f"models/gen5/{name}_best.pt"
    if os.path.exists(pth):
        m = MST_KinematicTransformer(d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["layers"])
        m.load_state_dict(torch.load(pth, map_location="cpu", weights_only=False))
        m.eval()
        with torch.no_grad():
            p, _ = m(X_val)
        p_arr = p.squeeze(-1).numpy()
        preds_list.append((name, p_arr))
        
        b50 = (p_arr >= 0.50).astype(float)
        b35 = (p_arr >= 0.35).astype(float)
        acc = np.mean(b50 == labels) * 100.0
        
        tp = np.sum((b50 == 1.0) & (labels == 1.0))
        fn = np.sum((b50 == 0.0) & (labels == 1.0))
        fp = np.sum((b50 == 1.0) & (labels == 0.0))
        sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2 * (prec/100) * (sens/100) / ((prec/100) + (sens/100))) if (prec+sens)>0 else 0.0
        
        tp35 = np.sum((b35 == 1.0) & (labels == 1.0))
        fn35 = np.sum((b35 == 0.0) & (labels == 1.0))
        sens35 = (tp35 / (tp35 + fn35) * 100.0) if (tp35 + fn35) > 0 else 0.0
        
        print(f"{name:<32} | {acc:6.2f}%  | {sens:6.2f}%     | {sens35:6.2f}%       | {prec:5.2f}%  | {f1:.4f}")

# Cross-Generation Meta-Ensemble: Gen4-Focal + Gen4-CosineWD + Gen5-CosineBCE + Gen3-DualStream
print("-" * 85)
from gen4_architectures import Kinematic_Spectral_Transformer
from round3_architectures import DualStream_MultiCrop_GCN

m_g4_foc = Kinematic_Spectral_Transformer()
m_g4_foc.load_state_dict(torch.load("models/gen4/4_Gen4_KinematicTrans_Focal_best.pt", map_location="cpu", weights_only=False))
m_g4_foc.eval()

m_g4_cos = Kinematic_Spectral_Transformer()
m_g4_cos.load_state_dict(torch.load("models/gen4/3_Gen4_KinematicTrans_CosineWD_best.pt", map_location="cpu", weights_only=False))
m_g4_cos.eval()

m_g3_dual = DualStream_MultiCrop_GCN()
m_g3_dual.load_state_dict(torch.load("models/round3/2_DualStream_MultiCrop_best.pt", map_location="cpu", weights_only=False))
m_g3_dual.eval()

with torch.no_grad():
    p_g4_foc, _ = m_g4_foc(X_val)
    p_g4_cos, _ = m_g4_cos(X_val)
    p_g3_dual, _ = m_g3_dual(X_val)

p1 = p_g4_foc.squeeze(-1).numpy()
p2 = p_g4_cos.squeeze(-1).numpy()
p3 = p_g3_dual.squeeze(-1).numpy()

# Weighted Meta-Ensemble
meta_p = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
b_meta = (meta_p >= 0.50).astype(float)
acc_m = np.mean(b_meta == labels) * 100.0
tp_m = np.sum((b_meta == 1.0) & (labels == 1.0))
fn_m = np.sum((b_meta == 0.0) & (labels == 1.0))
fp_m = np.sum((b_meta == 1.0) & (labels == 0.0))
sens_m = (tp_m / (tp_m + fn_m) * 100.0) if (tp_m + fn_m) > 0 else 0.0
prec_m = (tp_m / (tp_m + fp_m) * 100.0) if (tp_m + fp_m) > 0 else 0.0
f1_m = (2 * (prec_m/100) * (sens_m/100) / ((prec_m/100) + (sens_m/100))) if (prec_m+sens_m)>0 else 0.0

print(f"{'👑 Cross-Generation Meta-Ensemble':<32} | {acc_m:6.2f}%  | {sens_m:6.2f}%     | {sens_m:6.2f}%       | {prec_m:5.2f}%  | {f1_m:.4f}")
print("=" * 85)
