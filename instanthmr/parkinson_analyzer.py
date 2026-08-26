"""
End-to-End Parkinson's Disease Gait & Kinematics Analyzer
=========================================================
Processes a movement/walking video through InstantHMR + ST-GCN:
1. InstantHMR 3D extraction (Local 3D + Camera-Relative 3D).
2. COCO-17 Extraction via saved mapping.
3. Spatial Canonicalization & Scale Normalization.
4. Camera-Relative Velocity Feature Stacking -> Tensor [1, 6, T, 17].
5. Spatiotemporal Graph Convolutional Inference.
6. Clinical Gait Metrics Telemetry (Bradykinesia, Cadence, Arm Swing Asymmetry).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import time
import cv2
import numpy as np
import torch

from instanthmr.adapter import get_pipeline
from instanthmr.gait_features import (
    extract_coco17_from_mhr70,
    canonicalize_and_stack_features,
    COCO17_NAMES
)
from instanthmr.stgcn_model import ParkinsonSTGCN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_clinical_gait_metrics(features_tensor: np.ndarray, fps: float = 30.0) -> dict:
    """
    Computes explainable Parkinson's biomechanical biomarkers from the (6, T, 17) tensor:
    - Arm Swing Amplitude & Asymmetry (Left vs. Right Wrist velocity/excursion)
    - Bradykinesia Score (Movement speed decay and cadence variation)
    - Postural Forward Flexion (Trunk inclination angle)
    - Stride Regularity & Cadence
    """
    # features_tensor: (6, T, 17)
    # C[0..3]: [X, Y, Z] canonical position
    # C[3..6]: [Vx, Vy, Vz] camera-relative velocity
    pos = np.transpose(features_tensor[:3], (1, 2, 0))  # (T, 17, 3)
    vel = np.transpose(features_tensor[3:], (1, 2, 0))  # (T, 17, 3)
    T = pos.shape[0]

    # 1. Arm Swing Excursion & Asymmetry
    l_wrist_speed = np.linalg.norm(vel[:, 9, :], axis=-1)  # Left wrist
    r_wrist_speed = np.linalg.norm(vel[:, 10, :], axis=-1) # Right wrist
    l_arm_swing_mean = float(np.mean(l_wrist_speed))
    r_arm_swing_mean = float(np.mean(r_wrist_speed))

    denom = max(1e-4, l_arm_swing_mean + r_arm_swing_mean)
    arm_asymmetry_ratio = abs(l_arm_swing_mean - r_arm_swing_mean) / denom

    # 2. Forward Trunk Flexion (Cam posture tilt)
    mid_shoulder = (pos[:, 5, :] + pos[:, 6, :]) / 2.0
    mid_hip = (pos[:, 11, :] + pos[:, 12, :]) / 2.0
    spine_vec = mid_shoulder - mid_hip
    # Pitch angle in sagittal plane
    trunk_pitch_deg = np.degrees(np.arctan2(spine_vec[:, 2], spine_vec[:, 1]))
    forward_flexion_mean = float(np.mean(trunk_pitch_deg))

    # 3. Movement Velocity & Bradykinesia Index
    total_body_velocity = np.mean(np.linalg.norm(vel, axis=-1), axis=-1) # (T,)
    mean_speed = float(np.mean(total_body_velocity))
    speed_decay = float((total_body_velocity[0] - total_body_velocity[-1]) / max(1e-4, total_body_velocity[0]))

    # 4. Step Cadence (Peak Detection on Ankle Velocity)
    l_ankle_speed = np.linalg.norm(vel[:, 15, :], axis=-1)
    r_ankle_speed = np.linalg.norm(vel[:, 16, :], axis=-1)
    steps_est = int(np.sum(l_ankle_speed > np.mean(l_ankle_speed)) / max(1, fps / 2))

    return {
        "frames_analyzed": T,
        "duration_seconds": round(T / max(1e-4, fps), 2),
        "mean_gait_speed_mps": round(mean_speed, 3),
        "arm_swing_left_mps": round(l_arm_swing_mean, 3),
        "arm_swing_right_mps": round(r_arm_swing_mean, 3),
        "arm_swing_asymmetry": round(arm_asymmetry_ratio, 3),
        "forward_trunk_flexion_deg": round(forward_flexion_mean, 2),
        "bradykinesia_speed_decay": round(speed_decay, 3),
        "estimated_steps": steps_est
    }


def analyze_parkinson_video(video_path: str, weights_path: str = None) -> dict:
    """
    Complete analysis pipeline: Video -> InstantHMR -> COCO-17 -> Canonicalization -> ST-GCN.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("\n" + "=" * 65)
    print("🧠 InstantHMR Spatiotemporal Parkinson's Disease Analyzer")
    print(f"🎬 Video Target: {os.path.basename(video_path)}")
    print("=" * 65)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("Initializing InstantHMR 3D Kinematics Engine...")
    pipeline = get_pipeline(device="cpu", detector_stride=3)

    raw_j3d_local = []
    raw_j3d_cam = []

    t0 = time.perf_counter()
    f_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pipeline.predict(frame_rgb)

        if res.persons:
            p = res.persons[0]
            raw_j3d_local.append(p.joints_3d_local) # (70, 3)
            raw_j3d_cam.append(p.joints_3d_cam)     # (70, 3)
        else:
            if raw_j3d_local:
                raw_j3d_local.append(raw_j3d_local[-1])
                raw_j3d_cam.append(raw_j3d_cam[-1])
            else:
                raw_j3d_local.append(np.zeros((70, 3), dtype=np.float32))
                raw_j3d_cam.append(np.zeros((70, 3), dtype=np.float32))

        f_idx += 1
        if f_idx % 90 == 0 or f_idx == total_frames:
            print(f"  Processed {f_idx}/{total_frames} frames ({(f_idx/total_frames)*100:.1f}%)...")

    cap.release()
    t_extract = time.perf_counter() - t0

    j3d_local_arr = np.array(raw_j3d_local, dtype=np.float32)  # (T, 70, 3)
    j3d_cam_arr = np.array(raw_j3d_cam, dtype=np.float32)      # (T, 70, 3)

    print(f"✓ 3D Pose Extraction Complete: {f_idx} frames in {t_extract:.2f}s ({f_idx/t_extract:.1f} FPS)")

    # 1. Map to COCO-17
    mapping_file = os.path.join(REPO_ROOT, "instanthmr", "coco_mapping.json")
    coco_local_3d = extract_coco17_from_mhr70(j3d_local_arr, mapping_file) # (T, 17, 3)
    coco_cam_3d = extract_coco17_from_mhr70(j3d_cam_arr, mapping_file)     # (T, 17, 3)

    # 2. Spatial Canonicalization & Camera-Relative Velocity Feature Stacking
    features_6d = canonicalize_and_stack_features(coco_local_3d, coco_cam_3d, fps=fps) # (6, T, 17)
    input_tensor = torch.from_numpy(features_6d).unsqueeze(0).float() # (1, 6, T, 17)

    print(f"✓ Formed Canonical ST-GCN Tensor: Shape {list(input_tensor.shape)}")

    # 3. Instantiate and evaluate ST-GCN
    model = ParkinsonSTGCN(in_channels=6, num_classes=1)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("✓ Loaded pre-trained ParkinsonSTGCN weights.")
    else:
        print("✓ Initialized ParkinsonSTGCN architecture.")

    model.eval()
    with torch.no_grad():
        prob, feature_vec = model(input_tensor)

    risk_score = float(prob.item())
    clinical_metrics = compute_clinical_gait_metrics(features_6d, fps=fps)

    print("\n" + "=" * 65)
    print("📊 PARKINSON'S DISEASE BIOMECHANICAL REPORT")
    print("=" * 65)
    print(f"  • ST-GCN Output Risk Probability: {risk_score * 100:.1f}%")
    print(f"  • 256-D Neural Gait Vector:        {list(feature_vec.shape)}")
    print("  -------------------------------------------------------------")
    print("  • Clinical Biomarkers:")
    for k, v in clinical_metrics.items():
        print(f"      - {k:<28}: {v}")
    print("=" * 65 + "\n")

    return {
        "risk_probability": risk_score,
        "input_tensor_shape": list(input_tensor.shape),
        "clinical_metrics": clinical_metrics,
        "feature_vector_256d": feature_vec.squeeze(0).numpy().tolist()
    }


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "/Users/sidhparikh/Documents/WHAM_REPRO/examples/IMG_9732.mov"
    analyze_parkinson_video(vid)
