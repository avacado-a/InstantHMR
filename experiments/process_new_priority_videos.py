#!/usr/bin/env python3
"""
InstantHMR v2 - Top Priority Video Benchmark Processor (100% RAW, NO SMOOTHING)
Processes newly downloaded videos (IMG_0109, IMG_0093, IMG_0111).
Completely removes 1-Euro filter to deliver zero-lag, raw InstantHMR predictions.

Features:
- 100% RAW: Zero smoothing filter, zero phase delay / frame lag
- Authentic YOLO26n vs YOLO26s (2D Only, No 3D Skeleton Studio)
- 70-joint MHR keypoints + colored skeleton bones directly overlaid on current frame
- Dynamic box centering on intermediate frames from joints_2d (zero box lag)
- Batched InstantHMR CoreML inference (N=2)
- Stride 4 detector
- Crop expand 1.4 for extended limb reach & equipment
- Real-time telemetry HUD: rolling FPS, detector ms, HMR ms, distance Z
- Automatically launches each video on screen in QuickTime Player as soon as it completes
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

# UI Colors (BGR)
COLOR_BG = (20, 22, 26)
COLOR_PANEL_BG = (14, 16, 20)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 145, 155)
COLOR_CYAN = (240, 200, 0)
COLOR_GREEN = (80, 225, 100)
COLOR_PURPLE = (220, 100, 180)

OUTPUT_DIR = Path("/Users/sidhparikh/Downloads")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_VIDEOS = [
    {
        "id": "IMG_0109",
        "path": Path("/Users/sidhparikh/Downloads/IMG_0109.mp4"),
        "desc": "Priority 1: IMG_0109 (Raw Zero-Lag Detection)",
    },
    {
        "id": "IMG_0093",
        "path": Path("/Users/sidhparikh/Downloads/IMG_0093.mp4"),
        "desc": "Priority 2: IMG_0093 (High Frame-rate Agility)",
    },
    {
        "id": "IMG_0111",
        "path": Path("/Users/sidhparikh/Downloads/IMG_0111.mp4"),
        "desc": "Priority 3: IMG_0111 (Long Duration Exercise Flow)",
    },
]


def get_bone_color(idx1: int, idx2: int):
    # Head / neck: 0-4, 69
    if idx1 in (0, 1, 2, 3, 4, 69) or idx2 in (0, 1, 2, 3, 4, 69):
        return (255, 220, 60)
    # Hands / fingers: 21-40 (right), 42-61 (left)
    if idx1 in range(21, 41) or idx2 in range(21, 41):
        return (60, 240, 255)
    if idx1 in range(42, 62) or idx2 in range(42, 62):
        return (240, 80, 220)
    # Arms:
    if idx1 in (5, 7, 41, 63, 65, 67) or idx2 in (5, 7, 41, 63, 65, 67) or \
       idx1 in (6, 8, 62, 64, 66, 68) or idx2 in (6, 8, 62, 64, 66, 68):
        return (40, 160, 255)
    # Legs / feet: 9-20
    if idx1 in range(9, 21) or idx2 in range(9, 21):
        return (255, 110, 50)
    # Torso:
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

    # 2. Keypoints (70 articulated, raw position on current frame)
    for i in range(min(len(joints_2d), NUM_JOINTS)):
        pt = (int(round(joints_2d[i, 0])), int(round(joints_2d[i, 1])))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            radius = 2 if (21 <= i <= 61) else 3
            color = (255, 255, 255) if (21 <= i <= 61) else ((0, 240, 255) if i in (0, 1, 2, 3, 4, 69) else (0, 255, 120))
            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    # 3. Bounding box & tag (tightly tracks joints_2d with zero lag)
    if len(joints_2d) > 0:
        k_min = joints_2d.min(axis=0)
        k_max = joints_2d.max(axis=0)
        pad_x = max(10.0, (k_max[0] - k_min[0]) * 0.08)
        pad_y = max(10.0, (k_max[1] - k_min[1]) * 0.08)
        x1 = max(0, int(round(k_min[0] - pad_x)))
        y1 = max(0, int(round(k_min[1] - pad_y)))
        x2 = min(w - 1, int(round(k_max[0] + pad_x)))
        y2 = min(h - 1, int(round(k_max[1] + pad_y)))
    else:
        bbox = person.bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3])

    if x2 > x1 and y2 > y1:
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
        corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), box_color, 2)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), box_color, 2)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), box_color, 2)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), box_color, 2)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), box_color, 2)
        cv2.line(img, (x1, y2), (x1 - corner_len, y2), box_color, 2)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, 2)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, 2)

        z_depth = person.cam_trans[2] if hasattr(person, "cam_trans") else 0.0
        tag = f"{model_name}: {person.confidence*100:.0f}% | Z:{z_depth:.1f}m [RAW]"
        (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tag_y1 = max(0, y1 - th_text - 6)
        tag_y2 = max(th_text + 6, y1)
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 8, tag_y2), tag_color, -1)
        cv2.putText(img, tag, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def make_header_hud(width: int, height: int, vid_name: str, vid_idx: int, total_vids: int,
                    frame_idx: int, total_frames: int, t_frame_s: float,
                    fps_n: float, det_n_ms: float, fps_s: float, det_s_ms: float, hmr_ms: float) -> np.ndarray:
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.line(hud, (0, height - 1), (width, height - 1), (45, 50, 60), 1)

    # Row 1
    cv2.putText(hud, "INSTANTHMR v2 | RAW RESULTS (NO SMOOTHING / ZERO LAG)", (16, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"Video {vid_idx}/{total_vids}: {vid_name}", (530, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(hud, f"Frame: {frame_idx:4d} / {total_frames:4d} | Time: {t_frame_s:5.2f}s", (960, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # Row 2: Telemetry Pills
    fps_n_col = COLOR_CYAN
    cv2.rectangle(hud, (16, 34), (340, 62), (32, 38, 24), -1)
    cv2.rectangle(hud, (16, 34), (340, 62), fps_n_col, 1)
    cv2.putText(hud, f"YOLO26n: {fps_n:4.1f} FPS | Det: {det_n_ms:4.1f}ms", (26, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, fps_n_col, 1, cv2.LINE_AA)

    fps_s_col = COLOR_GREEN
    cv2.rectangle(hud, (352, 34), (680, 62), (20, 38, 28), -1)
    cv2.rectangle(hud, (352, 34), (680, 62), fps_s_col, 1)
    cv2.putText(hud, f"YOLO26s: {fps_s:4.1f} FPS | Det: {det_s_ms:4.1f}ms", (362, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, fps_s_col, 1, cv2.LINE_AA)

    cv2.rectangle(hud, (692, 34), (990, 62), (32, 28, 42), -1)
    cv2.rectangle(hud, (692, 34), (990, 62), COLOR_PURPLE, 1)
    cv2.putText(hud, f"HMR CoreML: {hmr_ms:4.1f}ms (Batch N=2)", (702, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 140, 215), 1, cv2.LINE_AA)

    cv2.rectangle(hud, (1002, 34), (1264, 62), (24, 26, 32), -1)
    cv2.rectangle(hud, (1002, 34), (1264, 62), (90, 95, 105), 1)
    cv2.putText(hud, "Apple Silicon M1 | RAW KPS", (1014, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1, cv2.LINE_AA)

    return hud


def process_video(vid_info: dict, vid_idx: int, total_vids: int,
                  det_n: PersonDetector, det_s: PersonDetector, hmr: InstantHMR) -> Path:
    vid_id = vid_info["id"]
    vid_path = vid_info["path"]
    vid_desc = vid_info["desc"]

    print(f"\n" + "=" * 70)
    print(f"[{vid_idx}/{total_vids}] PROCESSING PRIORITY VIDEO: {vid_id}.mp4 (100% RAW)")
    print(f"Description: {vid_desc}")
    print("=" * 70)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_individual = OUTPUT_DIR / f"showcase_raw_{vid_id}.mp4"
    temp_raw = OUTPUT_DIR / f"temp_{vid_id}.mp4"

    canvas_w = 1280
    canvas_h = 720
    hud_h = 70
    content_h = canvas_h - hud_h  # 650
    half_w = canvas_w // 2        # 640

    scale = min(half_w / w_src, content_h / h_src)
    fit_w = int(round(w_src * scale))
    fit_h = int(round(h_src * scale))
    if fit_w % 2 != 0:
        fit_w += 1
    if fit_h % 2 != 0:
        fit_h += 1
    pad_x = (half_w - fit_w) // 2
    pad_y = (content_h - fit_h) // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_raw), fourcc, src_fps, (canvas_w, canvas_h))

    cached_box_n, cached_score_n = None, None
    cached_box_s, cached_score_s = None, None
    last_det_n_ms, last_det_s_ms = 0.0, 0.0

    history_fps_n = []
    history_fps_s = []

    t_start = time.perf_counter()

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_frame_s = idx / src_fps
        display_frame = cv2.resize(frame_bgr, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # 1. Detector Step (Stride 4)
        run_det = (idx % 4 == 0) or (cached_box_n is None) or (cached_box_s is None)
        if run_det:
            t0 = time.perf_counter()
            bn, sn = det_n(rgb)
            last_det_n_ms = (time.perf_counter() - t0) * 1000.0
            cached_box_n = bn[0] if len(bn) > 0 else None
            cached_score_n = sn[0] if len(sn) > 0 else 0.0

            t0 = time.perf_counter()
            bs, ss = det_s(rgb)
            last_det_s_ms = (time.perf_counter() - t0) * 1000.0
            cached_box_s = bs[0] if len(bs) > 0 else None
            cached_score_s = ss[0] if len(ss) > 0 else 0.0

        # 2. Batched InstantHMR CoreML Step
        boxes_for_hmr = []
        conf_for_hmr = []
        mapping = {}

        if cached_box_n is not None:
            mapping["n"] = len(boxes_for_hmr)
            boxes_for_hmr.append(cached_box_n)
            conf_for_hmr.append(cached_score_n)
        if cached_box_s is not None:
            mapping["s"] = len(boxes_for_hmr)
            boxes_for_hmr.append(cached_box_s)
            conf_for_hmr.append(cached_score_s)

        t0 = time.perf_counter()
        if len(boxes_for_hmr) > 0:
            preds = hmr.predict(rgb, boxes_for_hmr, confidences=conf_for_hmr)
            t_hmr_ms = (time.perf_counter() - t0) * 1000.0
        else:
            preds = []
            t_hmr_ms = 0.0

        pred_n = preds[mapping["n"]] if "n" in mapping else None
        pred_s = preds[mapping["s"]] if "s" in mapping else None

        # 3. Dynamic Zero-Lag Centering for intermediate stride frames
        if pred_n is not None and len(pred_n.joints_2d) > 0:
            k_min = pred_n.joints_2d.min(axis=0)
            k_max = pred_n.joints_2d.max(axis=0)
            margin_x = max(12.0, (k_max[0] - k_min[0]) * 0.12)
            margin_y = max(12.0, (k_max[1] - k_min[1]) * 0.12)
            cached_box_n = np.array([
                max(0.0, k_min[0] - margin_x),
                max(0.0, k_min[1] - margin_y),
                min(float(fit_w), k_max[0] + margin_x),
                min(float(fit_h), k_max[1] + margin_y)
            ], dtype=np.float32)

        if pred_s is not None and len(pred_s.joints_2d) > 0:
            k_min = pred_s.joints_2d.min(axis=0)
            k_max = pred_s.joints_2d.max(axis=0)
            margin_x = max(12.0, (k_max[0] - k_min[0]) * 0.12)
            margin_y = max(12.0, (k_max[1] - k_min[1]) * 0.12)
            cached_box_s = np.array([
                max(0.0, k_min[0] - margin_x),
                max(0.0, k_min[1] - margin_y),
                min(float(fit_w), k_max[0] + margin_x),
                min(float(fit_h), k_max[1] + margin_y)
            ], dtype=np.float32)

        # 4. Latency & Rolling FPS
        lat_n = (last_det_n_ms if run_det else 0.0) + (t_hmr_ms if pred_n is not None else 0.0)
        fps_n_inst = 1000.0 / max(lat_n, 1.0)
        history_fps_n.append(fps_n_inst)
        if len(history_fps_n) > 30:
            history_fps_n.pop(0)
        rolling_fps_n = float(np.mean(history_fps_n))

        lat_s = (last_det_s_ms if run_det else 0.0) + (t_hmr_ms if pred_s is not None else 0.0)
        fps_s_inst = 1000.0 / max(lat_s, 1.0)
        history_fps_s.append(fps_s_inst)
        if len(history_fps_s) > 30:
            history_fps_s.pop(0)
        rolling_fps_s = float(np.mean(history_fps_s))

        # 5. Build YOLO26n Panel (Left) - 100% RAW
        frame_n = display_frame.copy()
        if pred_n is not None:
            draw_person_overlay(frame_n, pred_n, "YOLO26n", (0, 210, 255), (0, 180, 240))
        panel_n = np.full((content_h, half_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
        panel_n[pad_y:pad_y + fit_h, pad_x:pad_x + fit_w] = frame_n

        if pad_y >= 30:
            cv2.rectangle(panel_n, (pad_x + 12, pad_y - 34), (pad_x + 360, pad_y - 6), (28, 34, 44), -1)
            cv2.rectangle(panel_n, (pad_x + 12, pad_y - 34), (pad_x + 360, pad_y - 6), COLOR_CYAN, 1)
            cv2.putText(panel_n, "YOLO26n (Nano 5.3MB) | RAW (No Filter)", (pad_x + 20, pad_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)
            if content_h - (pad_y + fit_h) >= 28:
                z_n = pred_n.cam_trans[2] if pred_n is not None else 0.0
                conf_n = pred_n.confidence * 100 if pred_n is not None else 0.0
                cv2.putText(panel_n, f"Detection: {conf_n:.0f}% | Depth Z: {z_n:.2f}m | Zero Lag",
                            (pad_x + 16, pad_y + fit_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(panel_n, (pad_x + 10, pad_y + 12), (pad_x + 290, pad_y + 38), (20, 26, 36), -1)
            cv2.rectangle(panel_n, (pad_x + 10, pad_y + 12), (pad_x + 290, pad_y + 38), COLOR_CYAN, 1)
            cv2.putText(panel_n, "YOLO26n | RAW Zero-Lag", (pad_x + 18, pad_y + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_CYAN, 1, cv2.LINE_AA)

        # 6. Build YOLO26s Panel (Right) - 100% RAW
        frame_s = display_frame.copy()
        if pred_s is not None:
            draw_person_overlay(frame_s, pred_s, "YOLO26s", (80, 230, 120), (50, 180, 80))
        panel_s = np.full((content_h, half_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
        panel_s[pad_y:pad_y + fit_h, pad_x:pad_x + fit_w] = frame_s

        if pad_y >= 30:
            cv2.rectangle(panel_s, (pad_x + 12, pad_y - 34), (pad_x + 360, pad_y - 6), (20, 38, 28), -1)
            cv2.rectangle(panel_s, (pad_x + 12, pad_y - 34), (pad_x + 360, pad_y - 6), COLOR_GREEN, 1)
            cv2.putText(panel_s, "YOLO26s (Small 19.5MB) | RAW (No Filter)", (pad_x + 20, pad_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GREEN, 1, cv2.LINE_AA)
            if content_h - (pad_y + fit_h) >= 28:
                z_s = pred_s.cam_trans[2] if pred_s is not None else 0.0
                conf_s = pred_s.confidence * 100 if pred_s is not None else 0.0
                cv2.putText(panel_s, f"Detection: {conf_s:.0f}% | Depth Z: {z_s:.2f}m | Zero Lag",
                            (pad_x + 16, pad_y + fit_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(panel_s, (pad_x + 10, pad_y + 12), (pad_x + 290, pad_y + 38), (18, 36, 24), -1)
            cv2.rectangle(panel_s, (pad_x + 10, pad_y + 12), (pad_x + 290, pad_y + 38), COLOR_GREEN, 1)
            cv2.putText(panel_s, "YOLO26s | RAW Zero-Lag", (pad_x + 18, pad_y + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GREEN, 1, cv2.LINE_AA)

        # 7. Divider Line
        body_content = np.hstack([panel_n, panel_s])
        cv2.line(body_content, (half_w, 0), (half_w, content_h), (50, 58, 70), 2)

        # 8. HUD
        hud = make_header_hud(canvas_w, hud_h, f"{vid_id}.mp4", vid_idx, total_vids,
                              idx + 1, total_frames, t_frame_s,
                              rolling_fps_n, last_det_n_ms,
                              rolling_fps_s, last_det_s_ms,
                              t_hmr_ms)

        full_frame = np.vstack([hud, body_content])
        out_writer.write(full_frame)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
            print(f"  [{vid_id}] Frame {idx + 1:4d}/{total_frames} | "
                  f"YOLO26n: {last_det_n_ms:4.1f}ms ({rolling_fps_n:4.1f} FPS) | "
                  f"YOLO26s: {last_det_s_ms:4.1f}ms ({rolling_fps_s:4.1f} FPS) | "
                  f"HMR: {t_hmr_ms:4.1f}ms")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"\nFinished {vid_id} in {dur:.1f}s ({total_frames / max(dur, 0.1):.1f} FPS)")

    print(f"Transcoding to H.264: {out_individual.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_individual)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw.exists():
        temp_raw.unlink()

    print(f"  >>> SAVED RAW VIDEO: {out_individual}")
    print(f"  >>> OPENING ON SCREEN IN QUICKTIME PLAYER: {out_individual.name}")
    subprocess.run(["open", str(out_individual)])

    return out_individual


def main():
    print("=" * 70)
    print("STARTING INSTANTHMR v2 (100% RAW, NO SMOOTHING / ZERO LAG)")
    print("=" * 70)

    # 1. Initialize Authentic YOLO26 Detectors
    print("Loading authentic YOLO26n (Nano, 5.3MB)...")
    det_n = PersonDetector("yolo26n", device="mps", confidence=0.25, use_pose_envelope=False)

    print("Loading authentic YOLO26s (Small, 19.5MB)...")
    det_s = PersonDetector("yolo26s", device="mps", confidence=0.25, use_pose_envelope=False)

    # 2. Initialize InstantHMR Medium
    print("Loading InstantHMR Medium (CoreML/MPS, crop_expand=1.4, max_batch=4)...")
    ckpt = resolve_model("medium")
    hmr = InstantHMR(ckpt, device="mps", with_mesh=False, crop_expand=1.4, max_batch=4)
    hmr.warmup(max_batch=2)

    completed_videos = []
    t_global_start = time.perf_counter()

    for idx, vid_info in enumerate(NEW_VIDEOS, 1):
        if not vid_info["path"].exists():
            print(f"WARNING: Video file not found: {vid_info['path']}. Skipping.")
            continue
        try:
            out_vid = process_video(vid_info, idx, len(NEW_VIDEOS), det_n, det_s, hmr)
            completed_videos.append(out_vid)
        except Exception as e:
            print(f"ERROR processing {vid_info['id']}: {e}")
            import traceback
            traceback.print_exc()

    total_dur = time.perf_counter() - t_global_start
    print("\n" + "=" * 70)
    print(f"ALL {len(completed_videos)} PRIORITY RAW VIDEOS PROCESSED IN {total_dur:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
