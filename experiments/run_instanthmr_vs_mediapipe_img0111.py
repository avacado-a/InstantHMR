#!/usr/bin/env python3
"""
InstantHMR v2 vs MediaPipe Pose Heavy (Large) - 3D Points & Pose Benchmark on IMG_0111
Head-to-head evaluation:
- Left: InstantHMR v2 (YOLO26s + 1-Euro Filter) -> 70 Articulated 3D Joints & Keypoints
- Right: MediaPipe Pose Heavy (Large Model, 30.6MB) -> 33 World Landmarks & Keypoints
- Both models feature synchronized 2D video overlays and 3D Studio orbital viewports
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
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
COLOR_PURPLE = (220, 100, 180)

VIDEO_PATH = Path("/Users/sidhparikh/Downloads/IMG_0111.mp4")
OUTPUT_PATH = Path("/Users/sidhparikh/Downloads/instanthmr_vs_mediapipe_IMG_0111.mp4")
TEMP_RAW = OUTPUT_PATH.with_suffix(".raw.mp4")

MP_MODEL_PATH = "/Users/sidhparikh/Documents/mediapipe_video_pose/pose_landmarker.task"

MEDIAPIPE_POSE_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


def get_bone_color_instanthmr(idx1: int, idx2: int):
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


def draw_instanthmr_2d(img: np.ndarray, person):
    h, w = img.shape[:2]
    joints_2d = person.joints_2d

    # 1. Bones
    for a, b in SKELETON_EDGES:
        if a < len(joints_2d) and b < len(joints_2d):
            pt1 = (int(round(joints_2d[a, 0])), int(round(joints_2d[a, 1])))
            pt2 = (int(round(joints_2d[b, 0])), int(round(joints_2d[b, 1])))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                color = get_bone_color_instanthmr(a, b)
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 210, 255), 1, cv2.LINE_AA)
        corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), (0, 210, 255), 2)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), (0, 210, 255), 2)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), (0, 210, 255), 2)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), (0, 210, 255), 2)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), (0, 210, 255), 2)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), (0, 210, 255), 2)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), (0, 210, 255), 2)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), (0, 210, 255), 2)

        z_depth = person.cam_trans[2] if hasattr(person, "cam_trans") else 0.0
        tag = f"InstantHMR (YOLO26s): {person.confidence*100:.0f}% | Z:{z_depth:.1f}m"
        (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        tag_y1 = max(0, y1 - th_text - 6)
        tag_y2 = max(th_text + 6, y1)
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 8, tag_y2), (0, 180, 240), -1)
        cv2.putText(img, tag, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)


def draw_mediapipe_2d(img: np.ndarray, landmarks):
    h, w = img.shape[:2]
    pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in landmarks]

    # 1. Bones
    for a, b in MEDIAPIPE_POSE_EDGES:
        if a < len(pts) and b < len(pts):
            pt1 = pts[a]
            pt2 = pts[b]
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, (80, 225, 100), 2, cv2.LINE_AA)

    # 2. Keypoints (33 points)
    for i, pt in enumerate(pts):
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 4, (40, 160, 60), 1, cv2.LINE_AA)

    # 3. Box & Tag
    xs = [p[0] for p in pts if 0 <= p[0] < w]
    ys = [p[1] for p in pts if 0 <= p[1] < h]
    if len(xs) > 0 and len(ys) > 0:
        pad = 12
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 225, 100), 1, cv2.LINE_AA)
        corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), (80, 225, 100), 2)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), (80, 225, 100), 2)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), (80, 225, 100), 2)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), (80, 225, 100), 2)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), (80, 225, 100), 2)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), (80, 225, 100), 2)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), (80, 225, 100), 2)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), (80, 225, 100), 2)

        tag = f"MediaPipe Heavy: 33 Landmarks"
        (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        tag_y1 = max(0, y1 - th_text - 6)
        tag_y2 = max(th_text + 6, y1)
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 8, tag_y2), (50, 180, 80), -1)
        cv2.putText(img, tag, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)


def render_3d_skeleton_studio(joints_3d: np.ndarray | None, width: int, height: int,
                               azim: float = 24.0, elev: float = -8.0,
                               z_depth: float | None = None, is_mediapipe: bool = False) -> np.ndarray:
    canvas = np.full((height, width, 3), COLOR_PANEL_BG, dtype=np.uint8)

    # Ground grid
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

    edges = MEDIAPIPE_POSE_EDGES if is_mediapipe else SKELETON_EDGES
    theme_color = (80, 225, 100) if is_mediapipe else (0, 210, 255)

    # 3D Bones
    for a, b in edges:
        if a < len(pts_joints) and b < len(pts_joints):
            pt1 = tuple(pts_joints[a])
            pt2 = tuple(pts_joints[b])
            if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                bone_c = (80, 220, 120) if is_mediapipe else get_bone_color_instanthmr(a, b)
                cv2.line(canvas, pt1, pt2, (bone_c[0] // 2, bone_c[1] // 2, bone_c[2] // 2), 4, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, bone_c, 2, cv2.LINE_AA)

    # 3D Joints
    num_pts = min(len(pts_joints), 33 if is_mediapipe else NUM_JOINTS)
    for i in range(num_pts):
        pt = tuple(pts_joints[i])
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            radius = 2 if (not is_mediapipe and 21 <= i <= 61) else 3
            color = (255, 255, 255) if (not is_mediapipe and 21 <= i <= 61) else theme_color
            cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Title & Stats
    if not is_mediapipe:
        cv2.putText(canvas, "3D SKELETON STUDIO (MHR 70)", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 255), 1, cv2.LINE_AA)
        depth_str = f"Depth Z: {z_depth:.2f}m" if z_depth is not None else "70 Articulated Joints"
        cv2.putText(canvas, f"Rigid Kinematic Bones | {depth_str}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_GRAY, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "3D WORLD LANDMARKS (MP 33)", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 225, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, "33 World Points | Free Lengths", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_GRAY, 1, cv2.LINE_AA)

    # 3D axes widget
    ax_x, ax_y = width - 36, height - 28
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x + 18, ax_y), (80, 80, 240), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x, ax_y - 18), (80, 240, 80), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x - 10, ax_y + 10), (240, 160, 60), 1, tipLength=0.3)
    cv2.putText(canvas, "3D", (ax_x - 8, ax_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.26, COLOR_WHITE, 1)

    return canvas


def make_header_hud(width: int, height: int, frame_idx: int, total_frames: int, t_frame_s: float,
                    fps_ihmr: float, ihmr_ms: float, fps_mp: float, mp_ms: float) -> np.ndarray:
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.line(hud, (0, height - 1), (width, height - 1), (45, 50, 60), 1)

    # Row 1
    cv2.putText(hud, "3D HEAD-TO-HEAD: InstantHMR v2 (70 Joints) vs MediaPipe Pose Heavy (33 Points)", (16, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"IMG_0111.mp4 | Frame: {frame_idx:4d} / {total_frames:4d} | Time: {t_frame_s:5.2f}s", (920, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # Row 2: Telemetry Pills
    # InstantHMR Pill
    cv2.rectangle(hud, (16, 34), (430, 62), (32, 38, 24), -1)
    cv2.rectangle(hud, (16, 34), (430, 62), COLOR_CYAN, 1)
    cv2.putText(hud, f"InstantHMR (YOLO26s + 1-Euro): {fps_ihmr:4.1f} FPS | Latency: {ihmr_ms:4.1f}ms", (26, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_CYAN, 1, cv2.LINE_AA)

    # MediaPipe Pill
    cv2.rectangle(hud, (442, 34), (860, 62), (20, 38, 28), -1)
    cv2.rectangle(hud, (442, 34), (860, 62), COLOR_GREEN, 1)
    cv2.putText(hud, f"MediaPipe Heavy (33 World LMs): {fps_mp:4.1f} FPS | Latency: {mp_ms:4.1f}ms", (452, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GREEN, 1, cv2.LINE_AA)

    # Hardware & Detail Badge
    cv2.rectangle(hud, (872, 34), (1264, 62), (24, 26, 32), -1)
    cv2.rectangle(hud, (872, 34), (1264, 62), (90, 95, 105), 1)
    cv2.putText(hud, "InstantHMR: 70 Kinematic Joints vs MediaPipe: 33 Points", (882, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_GRAY, 1, cv2.LINE_AA)

    return hud


def main():
    print("=" * 70)
    print("STARTING 3D HEAD-TO-HEAD BENCHMARK: InstantHMR v2 vs MediaPipe Large on IMG_0111")
    print("=" * 70)

    # 1. Initialize InstantHMR v2 (YOLO26s + 1-Euro filter)
    print("Loading YOLO26s (19.5MB, STAL, NMS-free)...")
    det_s = PersonDetector("yolo26s", device="mps", confidence=0.25, use_pose_envelope=False)

    print("Loading InstantHMR Medium (CoreML/MPS, crop_expand=1.4)...")
    ckpt = resolve_model("medium")
    hmr = InstantHMR(ckpt, device="mps", with_mesh=False, crop_expand=1.4, max_batch=1)
    hmr.warmup(max_batch=1)
    smoother = TemporalSmoother(beta_scale=1.5)

    # 2. Initialize MediaPipe Pose Heavy
    print("Loading MediaPipe Pose Heavy (30.6MB)...")
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    mp_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

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

    cached_box = None
    last_det_ms = 0.0
    history_fps_ihmr = []
    history_fps_mp = []

    t_start = time.perf_counter()

    with PoseLandmarker.create_from_options(mp_options) as mp_landmarker:
        for idx in range(total_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t_frame_s = idx / src_fps
            ts_ms = int(idx * 1000 / src_fps)
            frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # -----------------------------------------------------------------
            # 1. InstantHMR Pipeline (YOLO26s + 1-Euro Filter)
            # -----------------------------------------------------------------
            t0_ihmr = time.perf_counter()
            run_det = (idx % 4 == 0) or (cached_box is None)
            if run_det:
                t0_det = time.perf_counter()
                b, s = det_s(rgb)
                last_det_ms = (time.perf_counter() - t0_det) * 1000.0
                cached_box = b[0] if len(b) > 0 else None

            if cached_box is not None:
                preds = hmr.predict(rgb, [cached_box])
                t_hmr_ms = (time.perf_counter() - t0_ihmr) * 1000.0
                p_ihmr = preds[0] if len(preds) > 0 else None
            else:
                p_ihmr = None
                t_hmr_ms = 0.0

            # 1-Euro smoothing
            if p_ihmr is not None:
                s_res = smoother([p_ihmr], t_frame_s)
                p_ihmr = s_res[0] if s_res else p_ihmr

                # Dynamic box update
                if len(p_ihmr.joints_2d) > 0:
                    k_min = p_ihmr.joints_2d.min(axis=0)
                    k_max = p_ihmr.joints_2d.max(axis=0)
                    margin_x = max(12.0, (k_max[0] - k_min[0]) * 0.12)
                    margin_y = max(12.0, (k_max[1] - k_min[1]) * 0.12)
                    cached_box = np.array([
                        max(0.0, k_min[0] - margin_x),
                        max(0.0, k_min[1] - margin_y),
                        min(float(v2d_w), k_max[0] + margin_x),
                        min(float(content_h), k_max[1] + margin_y)
                    ], dtype=np.float32)

            t_ihmr_tot = (time.perf_counter() - t0_ihmr) * 1000.0
            history_fps_ihmr.append(1000.0 / max(t_ihmr_tot, 1.0))
            if len(history_fps_ihmr) > 30:
                history_fps_ihmr.pop(0)
            rolling_fps_ihmr = float(np.mean(history_fps_ihmr))

            # -----------------------------------------------------------------
            # 2. MediaPipe Heavy Pipeline (33 Landmarks & 3D World Points)
            # -----------------------------------------------------------------
            t0_mp = time.perf_counter()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res_mp = mp_landmarker.detect_for_video(mp_image, ts_ms)
            t_mp_tot = (time.perf_counter() - t0_mp) * 1000.0
            history_fps_mp.append(1000.0 / max(t_mp_tot, 1.0))
            if len(history_fps_mp) > 30:
                history_fps_mp.pop(0)
            rolling_fps_mp = float(np.mean(history_fps_mp))

            mp_lms = res_mp.pose_landmarks[0] if (res_mp.pose_landmarks and len(res_mp.pose_landmarks) > 0) else None
            mp_world_lms = res_mp.pose_world_landmarks[0] if (res_mp.pose_world_landmarks and len(res_mp.pose_world_landmarks) > 0) else None

            # -----------------------------------------------------------------
            # 3. Synchronized 3D Azimuth & Elevation
            # -----------------------------------------------------------------
            rot_azim = 24.0 + 8.0 * np.sin(idx * 0.05)
            elev = -8.0

            # -----------------------------------------------------------------
            # 4. Render Left Panel (InstantHMR v2: 2D Video + 3D Studio)
            # -----------------------------------------------------------------
            ihmr_2d = frame_resized.copy()
            if p_ihmr is not None:
                draw_instanthmr_2d(ihmr_2d, p_ihmr)
            cv2.putText(ihmr_2d, "InstantHMR v2 (70 Joints)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

            z_ihmr = p_ihmr.cam_trans[2] if p_ihmr is not None else None
            j3d_ihmr = p_ihmr.joints_3d_local if p_ihmr is not None else None
            ihmr_3d = render_3d_skeleton_studio(j3d_ihmr, v3d_w, content_h, azim=rot_azim, elev=elev,
                                                z_depth=z_ihmr, is_mediapipe=False)
            panel_left = np.hstack([ihmr_2d, ihmr_3d])

            # -----------------------------------------------------------------
            # 5. Render Right Panel (MediaPipe Heavy: 2D Video + 3D Studio)
            # -----------------------------------------------------------------
            mp_2d = frame_resized.copy()
            if mp_lms is not None:
                draw_mediapipe_2d(mp_2d, mp_lms)
            cv2.putText(mp_2d, "MediaPipe Heavy (33 Points)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 240, 120), 1, cv2.LINE_AA)

            if mp_world_lms is not None:
                pts_mp_3d = np.array([[lm.x, lm.y, lm.z] for lm in mp_world_lms], dtype=np.float32)
            else:
                pts_mp_3d = None
            mp_3d = render_3d_skeleton_studio(pts_mp_3d, v3d_w, content_h, azim=rot_azim, elev=elev,
                                              z_depth=None, is_mediapipe=True)
            panel_right = np.hstack([mp_2d, mp_3d])

            # -----------------------------------------------------------------
            # 6. Assemble Full Canvas & HUD
            # -----------------------------------------------------------------
            body_content = np.hstack([panel_left, panel_right])
            cv2.line(body_content, (half_w, 0), (half_w, content_h), (50, 58, 70), 2)

            hud = make_header_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                                  rolling_fps_ihmr, t_ihmr_tot,
                                  rolling_fps_mp, t_mp_tot)

            full_frame = np.vstack([hud, body_content])
            out_writer.write(full_frame)

            if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
                print(f"  Frame {idx + 1:4d}/{total_frames} | "
                      f"InstantHMR: {t_ihmr_tot:4.1f}ms ({rolling_fps_ihmr:4.1f} FPS) | "
                      f"MediaPipe: {t_mp_tot:4.1f}ms ({rolling_fps_mp:4.1f} FPS)")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"\nProcessing finished in {dur:.1f}s ({total_frames / max(dur, 0.1):.1f} FPS)")

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
