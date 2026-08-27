import os
import torch
import numpy as np
from gen6_architectures import Asym_KinematicTransformer
from gen4_architectures import Kinematic_Spectral_Transformer
from round3_architectures import DualStream_MultiCrop_GCN

cached = torch.load("data/cached_mhr70_tensors.pt", map_location="cpu", weights_only=False)
X_val, y_val = cached["X_val"], cached["y_val"]
labels = y_val.numpy()

models = [
    ("1_Gen6_Asym_CosineWD_Base", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("2_Gen6_Asym_Focal_Tuned", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("3_Gen6_Asym_DeepAttn", {"d_model": 128, "nhead": 8, "layers": 4}),
    ("4_Gen6_Asym_HighReg", {"d_model": 128, "nhead": 4, "layers": 3}),
]

print("=" * 85)
print("🏆 GENERATION 6 ASYM-KINEMATIC-TRANSFORMER MUTATION BENCHMARKS")
print("=" * 85)
print(f"{'Model Name':<32} | {'Val Acc':<9} | {'Sens (0.50)':<12} | {'Recall (0.35)':<13} | {'Prec':<8} | {'F1':<6}")
print("-" * 85)

preds_list = []
for name, cfg in models:
    pth = f"models/gen6/{name}_best.pt"
    if os.path.exists(pth):
        m = Asym_KinematicTransformer(d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["layers"])
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

# Cross-Gen Grand Tournament Stacking Ensemble
# 1. Gen4 Kinematic-Transformer (CosineWD) - 77.94% Acc
# 2. Gen4 Kinematic-Transformer (Focal) - 77.45% Acc / 0.7882 F1
# 3. Gen6 Asym-DeepAttn - 77.11% Acc
# 4. Gen6 Asym-Focal - 76.78% Acc
# 5. Gen3 DualStream - 76.12% Acc / 94.46% Recall
print("-" * 85)

m_g4_cos = Kinematic_Spectral_Transformer()
m_g4_cos.load_state_dict(torch.load("models/gen4/3_Gen4_KinematicTrans_CosineWD_best.pt", map_location="cpu", weights_only=False))
m_g4_cos.eval()

m_g4_foc = Kinematic_Spectral_Transformer()
m_g4_foc.load_state_dict(torch.load("models/gen4/4_Gen4_KinematicTrans_Focal_best.pt", map_location="cpu", weights_only=False))
m_g4_foc.eval()

m_g6_deep = Asym_KinematicTransformer(nhead=8, num_layers=4)
m_g6_deep.load_state_dict(torch.load("models/gen6/3_Gen6_Asym_DeepAttn_best.pt", map_location="cpu", weights_only=False))
m_g6_deep.eval()

m_g6_foc = Asym_KinematicTransformer()
m_g6_foc.load_state_dict(torch.load("models/gen6/2_Gen6_Asym_Focal_Tuned_best.pt", map_location="cpu", weights_only=False))
m_g6_foc.eval()

with torch.no_grad():
    p1 = m_g4_cos(X_val)[0].squeeze(-1).numpy()
    p2 = m_g4_foc(X_val)[0].squeeze(-1).numpy()
    p3 = m_g6_deep(X_val)[0].squeeze(-1).numpy()
    p4 = m_g6_foc(X_val)[0].squeeze(-1).numpy()

grand_ens = 0.30 * p1 + 0.30 * p2 + 0.20 * p3 + 0.20 * p4
b_grand = (grand_ens >= 0.50).astype(float)
acc_g = np.mean(b_grand == labels) * 100.0
tp_g = np.sum((b_grand == 1.0) & (labels == 1.0))
fn_g = np.sum((b_grand == 0.0) & (labels == 1.0))
fp_g = np.sum((b_grand == 1.0) & (labels == 0.0))
sens_g = (tp_g / (tp_g + fn_g) * 100.0) if (tp_g + fn_g) > 0 else 0.0
prec_g = (tp_g / (tp_g + fp_g) * 100.0) if (tp_g + fp_g) > 0 else 0.0
f1_g = (2 * (prec_g/100) * (sens_g/100) / ((prec_g/100) + (sens_g/100))) if (prec_g+sens_g)>0 else 0.0

print(f"{'👑 Grand Multi-Gen Tournament Ensemble':<32} | {acc_g:6.2f}%  | {sens_g:6.2f}%     | {sens_g:6.2f}%       | {prec_g:5.2f}%  | {f1_g:.4f}")
print("=" * 85)
