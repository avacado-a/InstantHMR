#!/usr/bin/env python3
"""
InstantHMR v2 - RF-DETR Nano vs YOLO26s Detector Benchmark on IMG_0111.mp4
Head-to-head evaluation:
- Left: RF-DETR Nano (Transformer-based, Multi-Scale Deformable Attention)
- Right: YOLO26s (CNN Edge-First, NMS-free & DFL-free, STAL)
- Shows real-time comparative FPS speeds, detector latencies, HMR latencies, and 3D studios
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from instanthmr.models import resolve_model
from instanthmr.inference import InstantHMR
from instanthmr.detector import PersonDetector
from instanthmr.skeleton import SKELETON_EDGES, NUM_JOINTS
from instanthmr.smoothing import TemporalSmoother

# UI Colors (BGR)
COLOR_BG = (20, 22, 26)
COLOR_PANEL_BG = (14, 16, 20)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 145, 155)
COLOR_CYAN = (240, 200, 0)
COLOR_GREEN = (80, 225, 100)
COLOR_ORANGE = (40, 140, 255)
COLOR_PURPLE = (220, 100, 180)

VIDEO_PATH = Path("/Users/sidhparikh/Downloads/IMG_0111.mp4")
OUTPUT_PATH = Path("/Users/sidhparikh/Downloads/compare_rfdetr_nano_vs_yolo26s_IMG_0111.mp4")
TEMP_RAW = OUTPUT_PATH.with_suffix(".raw.mp4")


def get_bone_color(idx1: int, idx2: int):
    if idx1 in (0, 1, 2, 3, 4, 69) or idx2 in (0, 1, 2, 3, 4, 69):
        return (255, 220, 60)
    if idx1 in range(21, 41) or idx2 in range(21, 41):
        return (60, 240, 255)
    if idx1 in range(42, 62) or idx2 in range(42, 62):
        return (240, 80, 220)
    if idx1 in (5, 7, 41, 63, 65, 67) or idx2 in (5, 7, 41, 63, 65, 67) or \
       idx1 in (6, 8, 62, 64, 66, 68) or idx2 in (6, 8, 62, 64, 66, 68):
        return (40, 160, 255)
    if idx1 in range(9, 21) or idx2 in range(9, 21):
        return (255, 110, 50)
    return (80, 230, 120)


def draw_person_overlay(img: np.ndarray, person, model_name: str, box_color: tuple, tag_color: tuple):
    h, w = img.shape[:2]
    joints_2d = person.joints_2d

    # 1. Bones
    for a, b in SKELETON_EDGES:
        if a < len(joints_2d) and b < len(joints_2d):
            pt1 = (int(round(joints_2d[a, 0])), int(round(joints_2d[a, 1])))
            pt2 = (int(round(joints_2d[b, 0])), int(round(joints_2d[b, 1])))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                color = get_bone_color(a, b)
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # 2. Keypoints (70 joints)
    for i in range(min(len(joints_2d), NUM_JOINTS)):
        pt = (int(round(joints_2d[i, 0])), int(round(joints_2d[i, 1])))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            radius = 2 if (21 <= i <= 61) else 3
            color = (255, 255, 255) if (21 <= i <= 61) else ((0, 240, 255) if i in (0, 1, 2, 3, 4, 69) else (0, 255, 120))
            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    # 3. Box & Tag
    bbox = person.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3])
    if x2 > x1 and y2 > y1:
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
        corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), box_color, 2)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), box_color, 2)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), box_color, 2)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), box_color, 2)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), box_color, 2)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), box_color, 2)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, 2)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), box_color, 2)

        z_depth = person.cam_trans[2] if hasattr(person, "cam_trans") else 0.0
        tag = f"{model_name}: {person.confidence*100:.0f}% | Z:{z_depth:.1f}m"
        (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tag_y1 = max(0, y1 - th_text - 6)
        tag_y2 = max(th_text + 6, y1)
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 8, tag_y2), tag_color, -1)
        cv2.putText(img, tag, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def render_3d_skeleton_studio(joints_3d: np.ndarray | None, width: int, height: int,
                               azim: float = 24.0, elev: float = -8.0,
                               z_depth: float | None = None, model_tag: str = "DETECTOR") -> np.ndarray:
    canvas = np.full((height, width, 3), COLOR_PANEL_BG, dtype=np.uint8)

    cy_floor = int(height * 0.88)
    for gx in range(20, width, max(20, width // 12)):
        cv2.line(canvas, (gx, cy_floor), (gx + (gx - width // 2) // 2, height - 6), (28, 32, 40), 1)
    for gy in range(cy_floor, height - 6, max(12, (height - cy_floor) // 5)):
        cv2.line(canvas, (10, gy), (width - 10, gy), (28, 32, 40), 1)

    if joints_3d is None or len(joints_3d) == 0:
        cv2.putText(canvas, "Searching for subject...", (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
        return canvas

    rad_az = np.radians(azim)
    rad_el = np.radians(elev)
    R_az = np.array([[np.cos(rad_az), 0, np.sin(rad_az)],
                     [0, 1, 0],
                     [-np.sin(rad_az), 0, np.cos(rad_az)]])
    R_el = np.array([[1, 0, 0],
                     [0, np.cos(rad_el), -np.sin(rad_el)],
                     [0, np.sin(rad_el), np.cos(rad_el)]])
    R = R_el @ R_az

    rot_j = joints_3d @ R.T
    center = rot_j.mean(axis=0)
    j_c = rot_j - center

    span_y = j_c[:, 1].max() - j_c[:, 1].min()
    scale = (height * 0.68) / max(span_y, 0.1)

    cx, cy = width / 2.0, height * 0.46
    u_j = (j_c[:, 0] * scale + cx).astype(np.int32)
    v_j = (j_c[:, 1] * scale + cy).astype(np.int32)
    pts_joints = np.stack([u_j, v_j], axis=1)

    # 3D Bones
    for a, b in SKELETON_EDGES:
        if a < len(pts_joints) and b < len(pts_joints):
            pt1 = tuple(pts_joints[a])
            pt2 = tuple(pts_joints[b])
            if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                color = get_bone_color(a, b)
                cv2.line(canvas, pt1, pt2, (color[0] // 2, color[1] // 2, color[2] // 2), 4, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

    # 3D Joints
    for i in range(min(len(pts_joints), NUM_JOINTS)):
        pt = tuple(pts_joints[i])
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            radius = 2 if (21 <= i <= 61) else 3
            color = (255, 255, 255) if (21 <= i <= 61) else (0, 240, 255)
            cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Title & Depth
    cv2.putText(canvas, f"3D SKELETON STUDIO (MHR 70)", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 255), 1, cv2.LINE_AA)
    depth_str = f"Depth Z: {z_depth:.2f}m" if z_depth is not None else "70 Articulated Joints"
    cv2.putText(canvas, f"{model_tag} | {depth_str}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_GRAY, 1, cv2.LINE_AA)

    # 3D axes widget
    ax_x, ax_y = width - 36, height - 28
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x + 18, ax_y), (80, 80, 240), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x, ax_y - 18), (80, 240, 80), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x - 10, ax_y + 10), (240, 160, 60), 1, tipLength=0.3)
    cv2.putText(canvas, "3D", (ax_x - 8, ax_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.26, COLOR_WHITE, 1)

    return canvas


def make_header_hud(width: int, height: int, frame_idx: int, total_frames: int, t_frame_s: float,
                    fps_rf: float, det_rf_ms: float, fps_yolo: float, det_yolo_ms: float, hmr_ms: float) -> np.ndarray:
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.line(hud, (0, height - 1), (width, height - 1), (45, 50, 60), 1)

    # Row 1
    cv2.putText(hud, "DETECTOR HEAD-TO-HEAD: RF-DETR Nano vs YOLO26s (IMG_0111)", (16, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"Frame: {frame_idx:4d} / {total_frames:4d} | Time: {t_frame_s:5.2f}s", (960, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # Row 2: Telemetry Pills
    # 1. RF-DETR Nano Pill
    cv2.rectangle(hud, (16, 34), (390, 62), (32, 28, 20), -1)
    cv2.rectangle(hud, (16, 34), (390, 62), COLOR_ORANGE, 1)
    cv2.putText(hud, f"RF-DETR Nano: {fps_rf:4.1f} FPS | Det: {det_rf_ms:4.1f}ms", (26, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_ORANGE, 1, cv2.LINE_AA)

    # 2. YOLO26s Pill
    cv2.rectangle(hud, (402, 34), (776, 62), (20, 38, 28), -1)
    cv2.rectangle(hud, (402, 34), (776, 62), COLOR_GREEN, 1)
    cv2.putText(hud, f"YOLO26s: {fps_yolo:4.1f} FPS | Det: {det_yolo_ms:4.1f}ms", (412, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GREEN, 1, cv2.LINE_AA)

    # 3. Batched InstantHMR CoreML Regressor Pill
    cv2.rectangle(hud, (788, 34), (1050, 62), (32, 28, 42), -1)
    cv2.rectangle(hud, (788, 34), (1050, 62), COLOR_PURPLE, 1)
    cv2.putText(hud, f"HMR CoreML: {hmr_ms:4.1f}ms (Batch N=2)", (798, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 140, 215), 1, cv2.LINE_AA)

    # 4. Hardware Badge
    cv2.rectangle(hud, (1062, 34), (1264, 62), (24, 26, 32), -1)
    cv2.rectangle(hud, (1062, 34), (1264, 62), (90, 95, 105), 1)
    speed_diff = (det_rf_ms / max(det_yolo_ms, 1.0)) if det_yolo_ms > 0 else 1.0
    cv2.putText(hud, f"YOLO26s is {speed_diff:.1f}x Faster", (1072, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1, cv2.LINE_AA)

    return hud


def main():
    print("=" * 70)
    print("STARTING DETECTOR BENCHMARK: RF-DETR Nano vs YOLO26s on IMG_0111")
    print("=" * 70)

    # 1. Initialize Detectors
    print("Loading RF-DETR Nano (Transformer)...")
    det_rf = PersonDetector("nano", device="mps", confidence=0.35)

    print("Loading YOLO26s (19.5MB, STAL, NMS-free)...")
    det_yolo = PersonDetector("yolo26s", device="mps", confidence=0.25)

    # 2. Initialize InstantHMR Medium
    print("Loading InstantHMR Medium (CoreML/MPS, crop_expand=1.4, max_batch=2)...")
    ckpt = resolve_model("medium")
    hmr = InstantHMR(ckpt, device="mps", with_mesh=False, crop_expand=1.4, max_batch=2)
    hmr.warmup(max_batch=2)

    smoother_rf = TemporalSmoother(beta_scale=1.5)
    smoother_yolo = TemporalSmoother(beta_scale=1.5)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas_w = 1280
    canvas_h = 720
    hud_h = 70
    content_h = canvas_h - hud_h  # 650
    half_w = canvas_w // 2        # 640

    v2d_w = int(round(content_h * (w_src / h_src)))
    if v2d_w % 2 != 0:
        v2d_w += 1
    v3d_w = half_w - v2d_w  # 274

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(TEMP_RAW), fourcc, src_fps, (canvas_w, canvas_h))

    cached_box_rf, cached_score_rf = None, None
    cached_box_yolo, cached_score_yolo = None, None
    last_det_rf_ms, last_det_yolo_ms = 0.0, 0.0

    history_fps_rf = []
    history_fps_yolo = []

    t_start = time.perf_counter()

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_frame_s = idx / src_fps
        frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 1. Detector Step (Stride 4)
        run_det = (idx % 4 == 0) or (cached_box_rf is None) or (cached_box_yolo is None)
        if run_det:
            t0 = time.perf_counter()
            b_rf, s_rf = det_rf(rgb)
            last_det_rf_ms = (time.perf_counter() - t0) * 1000.0
            cached_box_rf = b_rf[0] if len(b_rf) > 0 else None
            cached_score_rf = s_rf[0] if len(s_rf) > 0 else 0.0

            t0 = time.perf_counter()
            b_yolo, s_yolo = det_yolo(rgb)
            last_det_yolo_ms = (time.perf_counter() - t0) * 1000.0
            cached_box_yolo = b_yolo[0] if len(b_yolo) > 0 else None
            cached_score_yolo = s_yolo[0] if len(s_yolo) > 0 else 0.0

        # 2. Batched InstantHMR CoreML Step (N=2)
        boxes_for_hmr = []
        conf_for_hmr = []
        mapping = {}

        if cached_box_rf is not None:
            mapping["rf"] = len(boxes_for_hmr)
            boxes_for_hmr.append(cached_box_rf)
            conf_for_hmr.append(cached_score_rf)
        if cached_box_yolo is not None:
            mapping["yolo"] = len(boxes_for_hmr)
            boxes_for_hmr.append(cached_box_yolo)
            conf_for_hmr.append(cached_score_yolo)

        t0 = time.perf_counter()
        if len(boxes_for_hmr) > 0:
            preds = hmr.predict(rgb, boxes_for_hmr, confidences=conf_for_hmr)
            t_hmr_ms = (time.perf_counter() - t0) * 1000.0
        else:
            preds = []
            t_hmr_ms = 0.0

        p_rf = preds[mapping["rf"]] if "rf" in mapping else None
        p_yolo = preds[mapping["yolo"]] if "yolo" in mapping else None

        # 3. Smoothing (1-Euro filter with tuned beta=1.0)
        if p_rf is not None:
            s_res = smoother_rf([p_rf], t_frame_s)
            p_rf = s_res[0] if s_res else p_rf

            if len(p_rf.joints_2d) > 0:
                k_min = p_rf.joints_2d.min(axis=0)
                k_max = p_rf.joints_2d.max(axis=0)
                margin_x = max(12.0, (k_max[0] - k_min[0]) * 0.12)
                margin_y = max(12.0, (k_max[1] - k_min[1]) * 0.12)
                cached_box_rf = np.array([
                    max(0.0, k_min[0] - margin_x),
                    max(0.0, k_min[1] - margin_y),
                    min(float(v2d_w), k_max[0] + margin_x),
                    min(float(content_h), k_max[1] + margin_y)
                ], dtype=np.float32)

        if p_yolo is not None:
            s_res = smoother_yolo([p_yolo], t_frame_s)
            p_yolo = s_res[0] if s_res else p_yolo

            if len(p_yolo.joints_2d) > 0:
                k_min = p_yolo.joints_2d.min(axis=0)
                k_max = p_yolo.joints_2d.max(axis=0)
                margin_x = max(12.0, (k_max[0] - k_min[0]) * 0.12)
                margin_y = max(12.0, (k_max[1] - k_min[1]) * 0.12)
                cached_box_yolo = np.array([
                    max(0.0, k_min[0] - margin_x),
                    max(0.0, k_min[1] - margin_y),
                    min(float(v2d_w), k_max[0] + margin_x),
                    min(float(content_h), k_max[1] + margin_y)
                ], dtype=np.float32)

        # 4. Latencies & Rolling FPS
        lat_rf = (last_det_rf_ms if run_det else 0.0) + (t_hmr_ms if p_rf is not None else 0.0)
        history_fps_rf.append(1000.0 / max(lat_rf, 1.0))
        if len(history_fps_rf) > 30:
            history_fps_rf.pop(0)
        rolling_fps_rf = float(np.mean(history_fps_rf))

        lat_yolo = (last_det_yolo_ms if run_det else 0.0) + (t_hmr_ms if p_yolo is not None else 0.0)
        history_fps_yolo.append(1000.0 / max(lat_yolo, 1.0))
        if len(history_fps_yolo) > 30:
            history_fps_yolo.pop(0)
        rolling_fps_yolo = float(np.mean(history_fps_yolo))

        # 5. Render Left Panel (RF-DETR Nano)
        rf_2d = frame_resized.copy()
        if p_rf is not None:
            draw_person_overlay(rf_2d, p_rf, "RF-DETR Nano", COLOR_ORANGE, (30, 100, 200))
        cv2.putText(rf_2d, "RF-DETR Nano (Transformer)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ORANGE, 1, cv2.LINE_AA)

        rot_azim = 24.0 + 8.0 * np.sin(idx * 0.05)
        z_rf = p_rf.cam_trans[2] if p_rf is not None else None
        j3d_rf = p_rf.joints_3d_local if p_rf is not None else None
        rf_3d = render_3d_skeleton_studio(j3d_rf, v3d_w, content_h, azim=rot_azim, z_depth=z_rf, model_tag="RF-DETR")
        panel_left = np.hstack([rf_2d, rf_3d])

        # 6. Render Right Panel (YOLO26s)
        yolo_2d = frame_resized.copy()
        if p_yolo is not None:
            draw_person_overlay(yolo_2d, p_yolo, "YOLO26s", COLOR_GREEN, (50, 180, 80))
        cv2.putText(yolo_2d, "YOLO26s (NMS-Free + STAL)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GREEN, 1, cv2.LINE_AA)

        z_yolo = p_yolo.cam_trans[2] if p_yolo is not None else None
        j3d_yolo = p_yolo.joints_3d_local if p_yolo is not None else None
        yolo_3d = render_3d_skeleton_studio(j3d_yolo, v3d_w, content_h, azim=rot_azim, z_depth=z_yolo, model_tag="YOLO26s")
        panel_right = np.hstack([yolo_2d, yolo_3d])

        # 7. Divider Line & HUD
        body_content = np.hstack([panel_left, panel_right])
        cv2.line(body_content, (half_w, 0), (half_w, content_h), (50, 58, 70), 2)

        hud = make_header_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                              rolling_fps_rf, last_det_rf_ms,
                              rolling_fps_yolo, last_det_yolo_ms,
                              t_hmr_ms)

        full_frame = np.vstack([hud, body_content])
        out_writer.write(full_frame)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
            print(f"  Frame {idx + 1:4d}/{total_frames} | "
                  f"RF-DETR: {last_det_rf_ms:4.1f}ms ({rolling_fps_rf:4.1f} FPS) | "
                  f"YOLO26s: {last_det_yolo_ms:4.1f}ms ({rolling_fps_yolo:4.1f} FPS) | "
                  f"HMR: {t_hmr_ms:4.1f}ms")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"\nBenchmark finished in {dur:.1f}s ({total_frames / max(dur, 0.1):.1f} FPS)")

    print(f"Transcoding to H.264: {OUTPUT_PATH.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(TEMP_RAW),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUTPUT_PATH)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if TEMP_RAW.exists():
        TEMP_RAW.unlink()

    print(f"\n>>> SAVED BENCHMARK VIDEO: {OUTPUT_PATH}")
    print(f">>> LAUNCHING ON SCREEN IN QUICKTIME PLAYER...")
    subprocess.run(["open", str(OUTPUT_PATH)])
    print("=" * 70)


if __name__ == "__main__":
    main()
