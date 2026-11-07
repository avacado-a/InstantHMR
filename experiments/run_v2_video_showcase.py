#!/usr/bin/env python3
"""
InstantHMR v2 Showcase Video Generator (MPS / Apple Silicon + YOLO26n + Keypoints only)
Runs InstantHMR v2 with YOLO26n person detector (stride 4) on MPS or CPU,
omits the dense MHR 3D mesh (keypoints & skeleton only), tracks exact real-time FPS
and per-stage latency, and renders a synchronized video opened on screen.
"""

from __future__ import annotations

import argparse
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

from instanthmr import PosePipeline
from instanthmr.skeleton import SKELETON_EDGES, NUM_JOINTS, JOINT_NAMES
from instanthmr.smoothing import TemporalSmoother

# UI Colors (BGR)
COLOR_BG = (20, 22, 26)
COLOR_PANEL_BG = (14, 16, 20)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 145, 155)
COLOR_DARK_GRAY = (40, 44, 52)
COLOR_CYAN = (240, 210, 0)
COLOR_GREEN = (80, 225, 100)
COLOR_ORANGE = (30, 140, 255)
COLOR_BLUE = (255, 120, 40)
COLOR_GOLD = (50, 205, 255)
COLOR_PURPLE = (220, 80, 180)


def get_bone_color(idx1: int, idx2: int):
    # Head / neck: 0-4, 69
    if idx1 in (0, 1, 2, 3, 4, 69) or idx2 in (0, 1, 2, 3, 4, 69):
        return (255, 220, 60)
    # Hands / fingers: 21-40 (right), 42-61 (left)
    if idx1 in range(21, 41) or idx2 in range(21, 41):
        return (60, 240, 255)  # Cyan for right hand
    if idx1 in range(42, 62) or idx2 in range(42, 62):
        return (240, 80, 220)  # Magenta for left hand
    # Arms:
    if idx1 in (5, 7, 41, 63, 65, 67) or idx2 in (5, 7, 41, 63, 65, 67) or \
       idx1 in (6, 8, 62, 64, 66, 68) or idx2 in (6, 8, 62, 64, 66, 68):
        return (40, 160, 255)
    # Legs / feet: 9-20
    if idx1 in range(9, 21) or idx2 in range(9, 21):
        return (255, 110, 50)
    # Torso:
    return (80, 230, 120)


def draw_keypoints_and_skeleton(img: np.ndarray, person):
    """Draws bounding box, 70 keypoints, and skeleton bones."""
    h, w = img.shape[:2]
    joints_2d = person.joints_2d

    # 1. Draw Skeleton Bones
    for a, b in SKELETON_EDGES:
        if a < len(joints_2d) and b < len(joints_2d):
            pt1 = (int(round(joints_2d[a, 0])), int(round(joints_2d[a, 1])))
            pt2 = (int(round(joints_2d[b, 0])), int(round(joints_2d[b, 1])))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                color = get_bone_color(a, b)
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # 2. Draw 70 Keypoints
    for i in range(min(len(joints_2d), NUM_JOINTS)):
        pt = (int(round(joints_2d[i, 0])), int(round(joints_2d[i, 1])))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            # Finger joints smaller radius
            if i in range(21, 62):
                radius = 2
                color = (255, 255, 255)
            elif i in (0, 1, 2, 3, 4, 69):
                radius = 3
                color = (0, 240, 255)
            else:
                radius = 4
                color = (0, 255, 120)

            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    # 3. Draw Bounding Box with styled corner brackets
    bbox = person.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 240), 1, cv2.LINE_AA)
    corner_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
    thick = 2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), (0, 230, 255), thick)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), (0, 230, 255), thick)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), (0, 230, 255), thick)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), (0, 230, 255), thick)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), (0, 230, 255), thick)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), (0, 230, 255), thick)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), (0, 230, 255), thick)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), (0, 230, 255), thick)

    tag = f"YOLO26n: {person.confidence*100:.0f}%"
    cv2.rectangle(img, (x1, max(0, y1 - 18)), (x1 + 92, max(18, y1)), (0, 180, 240), -1)
    cv2.putText(img, tag, (x1 + 4, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)


def render_3d_skeleton_studio(joints_3d: np.ndarray, width: int = 360, height: int = 480,
                               azim: float = 24.0, elev: float = -8.0) -> np.ndarray:
    """Renders the 3D Motion-Capture Skeleton in an orbital studio viewport."""
    canvas = np.full((height, width, 3), COLOR_PANEL_BG, dtype=np.uint8)

    # Perspective ground grid
    cy_floor = int(height * 0.88)
    for gx in range(20, width, 25):
        cv2.line(canvas, (gx, cy_floor), (gx + (gx - width//2)//2, height - 8), (28, 32, 40), 1)
    for gy in range(cy_floor, height - 8, 14):
        cv2.line(canvas, (15, gy), (width - 15, gy), (28, 32, 40), 1)

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

    # Center on hips
    center = rot_j.mean(axis=0)
    j_c = rot_j - center

    span_y = j_c[:, 1].max() - j_c[:, 1].min()
    scale = (height * 0.70) / max(span_y, 0.1)

    cx, cy = width / 2.0, height * 0.46
    u_j = (j_c[:, 0] * scale + cx).astype(np.int32)
    v_j = (j_c[:, 1] * scale + cy).astype(np.int32)
    pts_joints = np.stack([u_j, v_j], axis=1)

    # Draw 3D bones with depth glow
    for a, b in SKELETON_EDGES:
        if a < len(pts_joints) and b < len(pts_joints):
            pt1 = tuple(pts_joints[a])
            pt2 = tuple(pts_joints[b])
            if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                color = get_bone_color(a, b)
                cv2.line(canvas, pt1, pt2, (color[0]//2, color[1]//2, color[2]//2), 4, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw 3D Joints
    for i in range(min(len(pts_joints), NUM_JOINTS)):
        pt = tuple(pts_joints[i])
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            radius = 2 if i in range(21, 62) else 4
            color = (255, 255, 255) if i in range(21, 62) else (0, 240, 255)
            cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Labels
    cv2.putText(canvas, "3D SKELETON STUDIO (MHR70)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "70 Articulated Joints (No Mesh)", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_GRAY, 1, cv2.LINE_AA)

    # 3D axes widget
    ax_x, ax_y = width - 38, height - 32
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x + 20, ax_y), (80, 80, 240), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x, ax_y - 20), (80, 240, 80), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x - 12, ax_y + 12), (240, 160, 60), 1, tipLength=0.3)
    cv2.putText(canvas, "3D", (ax_x - 10, ax_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_WHITE, 1)

    return canvas


def make_header_hud(width: int, height: int, frame_idx: int, total_frames: int,
                    time_s: float, det_ms: float, hmr_ms: float, total_ms: float,
                    rolling_fps: float, device_str: str) -> np.ndarray:
    """Draws the top telemetry bar."""
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.line(hud, (0, height - 1), (width, height - 1), (45, 50, 60), 1)

    # App Title & Device Badge
    cv2.putText(hud, "INSTANTHMR v2 (KEYPOINTS ONLY)", (16, 25), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 220, 255), 1, cv2.LINE_AA)
    dev_badge = f"DEV: {device_str.upper()} | DET: YOLO26n (stride 4)"
    cv2.putText(hud, dev_badge, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 210, 240), 1, cv2.LINE_AA)

    # Live FPS Pill
    fps_text = f"{rolling_fps:4.1f} FPS"
    fps_color = (60, 235, 120) if rolling_fps >= 15.0 else (40, 200, 255)
    pill_w = 110
    cv2.rectangle(hud, (244, 12), (244 + pill_w, 48), (32, 38, 48), -1)
    cv2.rectangle(hud, (244, 12), (244 + pill_w, 48), fps_color, 1)
    cv2.putText(hud, fps_text, (256, 36), cv2.FONT_HERSHEY_DUPLEX, 0.58, fps_color, 1, cv2.LINE_AA)

    # Latency Breakdown
    col_x = 384
    cv2.putText(hud, f"Inference Latency: {total_ms:5.1f} ms", (col_x, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(hud, f"YOLO26n: {det_ms:4.1f} ms  |  HMR Regressor: {hmr_ms:4.1f} ms", (col_x, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1, cv2.LINE_AA)

    # Frame & Time
    col_x2 = width - 175
    cv2.putText(hud, f"Frame: {frame_idx:3d} / {total_frames}", (col_x2, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(hud, f"Time:  {time_s:4.2f}s (30 fps)", (col_x2, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1, cv2.LINE_AA)

    return hud


def parse_args():
    p = argparse.ArgumentParser(description="Run InstantHMR v2 on MPS/CPU with YOLO26n and stride 4 (Keypoints only)")
    p.add_argument("--device", default="mps", choices=("mps", "cpu", "coreml", "cuda"),
                   help="computation device (default: mps)")
    p.add_argument("--detector", default="yolo26n", help="detector model (default: yolo26n)")
    p.add_argument("--detector-stride", type=int, default=4, help="detector stride (default: 4)")
    p.add_argument("--model", default="medium", choices=("nano", "tiny", "small", "medium"),
                   help="InstantHMR model variant (default: medium)")
    p.add_argument("--input", default="/Users/sidhparikh/Downloads/m2-res_480p.mp4", help="input video path")
    p.add_argument("--output", default="/Users/sidhparikh/Downloads/instanthmr_v2_yolo26n_mps.mp4", help="output video path")
    return p.parse_args()


def main():
    args = parse_args()
    video_input = Path(args.input)
    video_output = Path(args.output)
    frames_dir = Path("/Users/sidhparikh/Downloads/instanthmr_v2_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not video_input.exists():
        sys.exit(f"Input video not found: {video_input}")

    print(f"\n==================================================================")
    print(f"InstantHMR v2: Keypoints Only | Device: {args.device.upper()} | Detector: {args.detector} (stride {args.detector_stride})")
    print(f"Input: {video_input}")
    print(f"==================================================================")

    cap = cv2.VideoCapture(str(video_input))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video specs: {w_orig}x{h_orig} @ {orig_fps:.1f} FPS, {total_frames} frames ({total_frames/orig_fps:.2f}s)")

    print(f"Initializing pipeline with InstantHMR v2 ({args.model}) on {args.device} (with_mesh=False)...")
    pipeline = PosePipeline(args.model, device=args.device, detector=args.detector,
                            detector_stride=args.detector_stride, with_mesh=False)
    smoother = TemporalSmoother(beta_scale=1.2)
    print("Pipeline ready! Running without dense MHR mesh calculations.")

    hud_h = 60
    panel_w, panel_h = w_orig, h_orig
    canvas_w = panel_w * 2
    canvas_h = panel_h + hud_h

    temp_raw_output = video_output.with_suffix(".temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_raw_output), fourcc, orig_fps, (canvas_w, canvas_h))

    history_fps = []
    t_start_all = time.perf_counter()

    print("Processing video frames...")
    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_frame_s = idx / orig_fps
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = pipeline(rgb)
        persons = smoother(result.persons, t_frame_s) if smoother else result.persons
        t_tot_ms = (time.perf_counter() - t0) * 1000

        current_fps = 1000.0 / max(t_tot_ms, 1.0)
        history_fps.append(current_fps)
        if len(history_fps) > 30:
            history_fps.pop(0)
        rolling_fps = float(np.mean(history_fps))

        # Left Panel: 2D Video + 70 Keypoints & Skeleton (No MHR Mesh)
        left_img = frame_bgr.copy()
        if persons:
            draw_keypoints_and_skeleton(left_img, persons[0])

        cv2.putText(left_img, "INPUT + 70 KEYPOINTS", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)

        # Right Panel: 3D Skeleton Motion Capture (No Mesh)
        if persons:
            p = persons[0]
            rot_azim = 24.0 + 8.0 * np.sin(idx * 0.05)
            right_img = render_3d_skeleton_studio(p.joints_3d_local, width=panel_w, height=panel_h,
                                                  azim=rot_azim, elev=-8.0)
        else:
            right_img = np.full((panel_h, panel_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
            cv2.putText(right_img, "No person detected", (panel_w // 4, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

        # Top HUD
        hud_img = make_header_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                                  result.detector_ms, result.hmr_ms, t_tot_ms, rolling_fps, args.device)

        full_canvas = np.vstack([hud_img, np.hstack([left_img, right_img])])
        out_writer.write(full_canvas)

        if idx in (30, 90, 150, 200):
            sample_path = frames_dir / f"yolo26n_mps_frame_{idx:03d}.jpg"
            cv2.imwrite(str(sample_path), full_canvas)

        if (idx + 1) % 30 == 0 or idx == total_frames - 1:
            print(f"Frame {idx+1:3d}/{total_frames} | Total {t_tot_ms:5.1f}ms | YOLO26n {result.detector_ms:4.1f}ms | "
                  f"HMR {result.hmr_ms:4.1f}ms | Real-time FPS: {rolling_fps:4.1f}")

    cap.release()
    out_writer.release()

    total_time = time.perf_counter() - t_start_all
    print(f"\nProcessing complete in {total_time:.1f}s (Average overall speed: {total_frames/total_time:.1f} FPS)")

    # Transcode with ffmpeg for QuickTime Player playback
    print("Transcoding video for Apple QuickTime Player compatibility...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw_output),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw_output.exists():
        temp_raw_output.unlink()

    print(f"Final video ready at: {video_output}")

    # Open on screen
    print(f"Launching video on screen via QuickTime Player...")
    subprocess.run(["open", str(video_output)])


if __name__ == "__main__":
    main()
