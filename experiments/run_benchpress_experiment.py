#!/usr/bin/env python3
"""
Bench Press Detection Experiment on IMG_0107.mp4:
Tests 4 configurations:
  1. Baseline: YOLO26n, standard bbox, crop_expand=1.2, no pose envelope
  2. Config A: YOLO26n + Pose Envelope Bounding Box, crop_expand=1.2
  3. Config B: YOLO26n + Pose Envelope Bounding Box + crop_expand=1.4
  4. Config C: YOLO26s + Pose Envelope Bounding Box + crop_expand=1.4

Measures:
  - Average Latency (Detector ms, InstantHMR ms, Total ms)
  - Real-world processing FPS
  - Limb Clipping / Boundary Margin (distance of wrists to crop edges)
  - Saves visual comparison image at peak barbell press
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from instanthmr import PosePipeline
from instanthmr.skeleton import SKELETON_EDGES
from run_v2_video_showcase import draw_keypoints_and_skeleton

VIDEO_PATH = Path("/Users/sidhparikh/Documents/yolo_pose_eval/videos/IMG_0107.mp4")
OUTPUT_DIR = Path("/Users/sidhparikh/Downloads")


def evaluate_config(name: str, detector: str, crop_expand: float, use_pose_envelope: bool,
                    max_frames: int = 200, start_frame: int = 500):
    print(f"\n=======================================================")
    print(f"Testing: {name}")
    print(f"Detector: {detector} | Crop Expand: {crop_expand} | Pose Envelope: {use_pose_envelope}")
    print(f"=======================================================")

    pipeline = PosePipeline(
        "medium",
        device="mps",
        detector=detector,
        detector_stride=4,
        with_mesh=False,
        crop_expand=crop_expand,
        use_pose_envelope=use_pose_envelope
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    det_times = []
    hmr_times = []
    tot_times = []
    wrist_margins = []
    peak_frame_vis = None
    peak_frame_idx = 670 - start_frame

    for i in range(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = pipeline(rgb)
        t_tot = (time.perf_counter() - t0) * 1000

        det_times.append(result.detector_ms)
        hmr_times.append(result.hmr_ms)
        tot_times.append(t_tot)

        # Evaluate wrist margin inside bounding box
        if result.persons:
            p = result.persons[0]
            box = p.bbox
            # 2D wrists from 70 keypoints
            # Joints 20, 21 are wrists; joints 38-69 are hands
            j2d = p.joints_2d
            # Measure top margin: how close the highest hand/wrist joint is to the top of the box
            highest_joint_y = j2d[:, 1].min()
            box_top_y = box[1]
            margin_y = highest_joint_y - box_top_y
            wrist_margins.append(margin_y)

            if i == peak_frame_idx:
                # Render visualization frame
                vis = frame_bgr.copy()
                draw_keypoints_and_skeleton(vis, p)
                # Overlay config label
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (10, 12, 16), -1)
                cv2.putText(vis, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
                cv2.putText(vis, f"Det: {detector} | Expand: {crop_expand}x | Env: {use_pose_envelope}",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 200), 1)
                peak_frame_vis = vis

    cap.release()

    avg_det = np.mean([t for t in det_times if t > 0.1]) if any(t > 0.1 for t in det_times) else 0.0
    avg_det_amortized = np.mean(det_times)
    avg_hmr = np.mean(hmr_times)
    avg_tot = np.mean(tot_times)
    fps = 1000.0 / avg_tot if avg_tot > 0 else 0.0
    min_wrist_margin = np.min(wrist_margins) if wrist_margins else 0.0

    print(f"Results for {name}:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Det Latency (run): {avg_det:.1f} ms | Det Latency (amortized): {avg_det_amortized:.1f} ms")
    print(f"  HMR Latency: {avg_hmr:.1f} ms | Total Latency: {avg_tot:.1f} ms")
    print(f"  Min Arm Clearance inside Box: {min_wrist_margin:.1f} px ({'CLIPPED/OUTSIDE' if min_wrist_margin < 0 else 'SAFE'})")

    return {
        "name": name,
        "detector": detector,
        "crop_expand": crop_expand,
        "use_pose_envelope": use_pose_envelope,
        "fps": fps,
        "avg_det": avg_det,
        "avg_det_amortized": avg_det_amortized,
        "avg_hmr": avg_hmr,
        "avg_tot": avg_tot,
        "min_wrist_margin": min_wrist_margin,
        "peak_frame_vis": peak_frame_vis,
    }


def main():
    configs = [
        ("1. Baseline (YOLO26n, 1.2x, No Env)", "yolo26n", 1.2, False),
        ("2. YOLO26n + Pose Envelope (1.2x)", "yolo26n", 1.2, True),
        ("3. YOLO26n + Pose Envelope (1.4x)", "yolo26n", 1.4, True),
        ("4. YOLO26s + Pose Envelope (1.4x)", "yolo26s", 1.4, True),
    ]

    results = []
    for name, det, expand, env in configs:
        res = evaluate_config(name, det, expand, env, max_frames=180, start_frame=500)
        results.append(res)

    # Combine peak frames into a 2x2 grid
    vis_frames = [r["peak_frame_vis"] for r in results if r["peak_frame_vis"] is not None]
    if len(vis_frames) == 4:
        top_row = np.hstack([vis_frames[0], vis_frames[1]])
        bot_row = np.hstack([vis_frames[2], vis_frames[3]])
        grid = np.vstack([top_row, bot_row])
        grid_path = OUTPUT_DIR / "benchpress_detection_comparison_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"\nSaved 2x2 visual comparison grid to: {grid_path}")

    print("\n" + "="*80)
    print(f"{'Configuration':<36} | {'FPS':<6} | {'Det (ms)':<9} | {'HMR (ms)':<9} | {'Wrist Margin'}")
    print("="*80)
    for r in results:
        status = f"{r['min_wrist_margin']:+.1f} px (" + ("SAFE" if r['min_wrist_margin'] >= 0 else "CLIPPED") + ")"
        print(f"{r['name']:<36} | {r['fps']:<6.1f} | {r['avg_det_amortized']:<9.1f} | {r['avg_hmr']:<9.1f} | {status}")
    print("="*80)


if __name__ == "__main__":
    main()
