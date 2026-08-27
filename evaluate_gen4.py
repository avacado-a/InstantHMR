import os
import torch
import numpy as np
from gen4_architectures import Kinematic_Spectral_Transformer

cached = torch.load("data/cached_mhr70_tensors.pt", map_location="cpu", weights_only=False)
X_val, y_val = cached["X_val"], cached["y_val"]
labels = y_val.numpy()

models = [
    ("1_Gen4_KinematicTrans_Base", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("2_Gen4_KinematicTrans_Deep", {"d_model": 128, "nhead": 8, "layers": 4}),
    ("3_Gen4_KinematicTrans_CosineWD", {"d_model": 128, "nhead": 4, "layers": 3}),
    ("4_Gen4_KinematicTrans_Focal", {"d_model": 128, "nhead": 4, "layers": 3}),
]

print("=" * 85)
print("🏆 GENERATION 4 KINEMATIC-TRANSFORMER MUTATION BENCHMARKS")
print("=" * 85)
print(f"{'Model Name':<32} | {'Val Acc':<9} | {'Sens (0.50)':<12} | {'Recall (0.35)':<13} | {'Prec':<8} | {'F1':<6}")
print("-" * 85)

preds_list = []
for name, cfg in models:
    pth = f"models/gen4/{name}_best.pt"
    if os.path.exists(pth):
        m = Kinematic_Spectral_Transformer(d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["layers"])
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

# Generation 4 Ensemble (CosineWD + Focal)
if len(preds_list) >= 4:
    p_cos = preds_list[2][1]
    p_foc = preds_list[3][1]
    ens_p = 0.45 * p_cos + 0.55 * p_foc
    b_ens = (ens_p >= 0.50).astype(float)
    acc_e = np.mean(b_ens == labels) * 100.0
    tp_e = np.sum((b_ens == 1.0) & (labels == 1.0))
    fn_e = np.sum((b_ens == 0.0) & (labels == 1.0))
    fp_e = np.sum((b_ens == 1.0) & (labels == 0.0))
    sens_e = (tp_e / (tp_e + fn_e) * 100.0) if (tp_e + fn_e) > 0 else 0.0
    prec_e = (tp_e / (tp_e + fp_e) * 100.0) if (tp_e + fp_e) > 0 else 0.0
    f1_e = (2 * (prec_e/100) * (sens_e/100) / ((prec_e/100) + (sens_e/100))) if (prec_e+sens_e)>0 else 0.0
    print("-" * 85)
    print(f"{'★ Gen-4 Dual-Consensus Ensemble':<32} | {acc_e:6.2f}%  | {sens_e:6.2f}%     | {sens_e:6.2f}%       | {prec_e:5.2f}%  | {f1_e:.4f}")
print("=" * 85)
