#!/usr/bin/env python3
"""
InstantHMR Video Player & Keypoints Visualizer
==============================================
Stand-alone test script to run InstantHMR on a video and view/export
the rendered keypoints video without modifying the main data pipeline.

Usage:
  python view_video_keypoints.py --video path/to/video.mp4
  python view_video_keypoints.py --video path/to/video.mp4 --device cpu --save-video output.mp4
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.pipeline import PosePipeline
from instanthmr.skeleton import edges_for


def render_and_view(
    video_path: str,
    device: str = "cpu",
    detector_stride: int = 3,
    save_video_path: str = None,
    show_window: bool = True
):
    if not os.path.exists(video_path):
        sys.exit(f"Error: Video not found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Loaded Video: {os.path.basename(video_path)}")
    print(f"Info: {total_frames} frames @ {fps:.1f} fps ({w}x{h})")

    # Locate models
    onnx_path = os.path.join(REPO_ROOT, "models", "instanthmr.onnx")
    yolo_model = os.path.join(REPO_ROOT, "models", "yolov8n.pt")
    if not os.path.exists(yolo_model):
        yolo_model = "yolov8n.pt"

    print("Initializing InstantHMR + YOLOv8n pipeline...")
    pipeline = PosePipeline(
        onnx_path=onnx_path,
        device=device,
        detector_type="yolo",
        detector_variant=yolo_model,
        det_confidence=0.35,
        max_persons=1,
        detector_stride=detector_stride,
        batch_persons=True,
    )

    out_writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (w, h))
        print(f"Saving rendered video to: {save_video_path}")

    frame_idx = 0
    t_start = time.perf_counter()

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        result = pipeline.predict(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        annotated = bgr.copy()

        if result.persons:
            person = result.persons[0]
            j2d = person.joints_2d # (70, 2)

            # Draw Bounding Box
            x1, y1, x2, y2 = person.bbox.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 128), 2)

            # Draw 70 Skeleton Edges
            valid = (j2d[:, 0] >= 0) & (j2d[:, 0] < w) & (j2d[:, 1] >= 0) & (j2d[:, 1] < h)
            for i, j in edges_for(j2d.shape[0]):
                if valid[i] and valid[j]:
                    pt1 = tuple(j2d[i].astype(int))
                    pt2 = tuple(j2d[j].astype(int))
                    cv2.line(annotated, pt1, pt2, (255, 230, 0), 2)

            # Draw 70 Joints
            for k, (x, y) in enumerate(j2d):
                if valid[k]:
                    col = (0, 255, 200) if k >= 25 else (0, 255, 0)
                    cv2.circle(annotated, (int(x), int(y)), 4, col, -1)
                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 0), 1)

        # Draw HUD
        cur_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        hud = f"InstantHMR (YOLOv8n) | Frame {frame_idx+1:>4d}/{total_frames} | Det: {result.detector_ms:4.1f}ms | HMR: {result.hmr_ms:4.1f}ms | FPS: {cur_fps:4.1f}"
        cv2.rectangle(annotated, (10, 10), (10 + 720, 48), (0, 0, 0), -1)
        cv2.putText(annotated, hud, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        if out_writer:
            out_writer.write(annotated)

        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"  Processed {frame_idx:>4d}/{total_frames} frames ({cur_fps:.1f} FPS)")

    cap.release()
    if out_writer:
        out_writer.release()

    total_time = time.perf_counter() - t_start
    print(f"\nDone! Processed {frame_idx} frames in {total_time:.2f}s ({frame_idx/total_time:.1f} FPS).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View InstantHMR 2D keypoints on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "coreml"], help="Compute device")
    parser.add_argument("--detector-stride", type=int, default=3, help="YOLO detection stride")
    parser.add_argument("--save-video", type=str, default=None, help="Optional output path to save rendered video")
    args = parser.parse_args()

    render_and_view(
        video_path=args.video,
        device=args.device,
        detector_stride=args.detector_stride,
        save_video_path=args.save_video,
    )
