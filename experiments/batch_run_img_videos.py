#!/usr/bin/env python3
"""
Batch Runner for all IMG_****.mp4 videos:
1. Spawns the interactive 3D Rerun viewer on IMG_0101.mp4 so the user can spin around the 3D points.
2. In parallel/sequence, rapidly processes all IMG_****.mp4 videos with InstantHMR v2 (MPS + YOLO26n stride 4, keypoints only).
3. Transcodes and opens the resulting videos on the user's screen in QuickTime.
"""

from __future__ import annotations

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
from instanthmr.skeleton import SKELETON_EDGES, NUM_JOINTS
from instanthmr.smoothing import TemporalSmoother
from run_v2_video_showcase import (
    draw_keypoints_and_skeleton,
    render_3d_skeleton_studio,
    make_header_hud,
)

INPUT_VIDEOS = [
    Path("/Users/sidhparikh/Documents/yolo_pose_eval/videos/IMG_0101.mp4"),
    Path("/Users/sidhparikh/Documents/yolo_pose_eval/videos/IMG_0102.mp4"),
    Path("/Users/sidhparikh/Documents/yolo_pose_eval/videos/IMG_0107.mp4"),
    Path("/Users/sidhparikh/Documents/mediapipe_video_pose/IMG_0097.mp4"),
]

OUTPUT_DIR = Path("/Users/sidhparikh/Downloads")


def spawn_interactive_3d_rerun(video_path: Path):
    """Spawns the interactive 3D Rerun viewer as a separate background process."""
    print(f"\n>>> Spawning Interactive 3D Window for {video_path.name} (Rerun)...")
    cmd = [
        sys.executable,
        str(REPO_DIR / "demo.py"),
        "--video", str(video_path),
        "--no-mesh",
        "--device", "mps",
        "--detector", "yolo26n",
        "--detector-stride", "4",
        "--smooth",
    ]
    # Launch detached so the GUI appears on screen and runs independently
    proc = subprocess.Popen(cmd, cwd=str(REPO_DIR))
    print(f">>> Interactive 3D viewer launched (PID: {proc.pid})! You can orbit/spin the 3D points in this window.")
    return proc


def process_video(video_path: Path, pipeline: PosePipeline, output_path: Path):
    """Processes a single video at maximum speed and writes a side-by-side showcase video."""
    print(f"\n==================================================================")
    print(f"Processing: {video_path.name}")
    print(f"==================================================================")

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fast downscale target for display video (e.g. max height 640)
    target_h = 640
    target_w = int(round(w_src * (target_h / h_src)))
    if target_w % 2 != 0:
        target_w += 1

    # Frame step: if 60fps, step by 2 to target smooth 30fps and 2x faster processing
    frame_step = 2 if src_fps > 45 else 1
    out_fps = src_fps / frame_step
    frames_to_process = total_src_frames // frame_step

    print(f"Source: {w_src}x{h_src} @ {src_fps:.1f} FPS ({total_src_frames} frames)")
    print(f"Target: {target_w}x{target_h} @ {out_fps:.1f} FPS ({frames_to_process} frames, step {frame_step})")

    hud_h = 60
    panel_w, panel_h = target_w, target_h
    canvas_w = panel_w * 2
    canvas_h = panel_h + hud_h

    temp_raw = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_raw), fourcc, out_fps, (canvas_w, canvas_h))

    smoother = TemporalSmoother(beta_scale=1.2)
    history_fps = []
    t_start = time.perf_counter()

    proc_idx = 0
    for src_idx in range(total_src_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if src_idx % frame_step != 0:
            continue

        # Resize to display target
        if (w_src, h_src) != (target_w, target_h):
            display_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame_bgr

        t_frame_s = proc_idx / out_fps
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = pipeline(rgb)
        persons = smoother(result.persons, t_frame_s) if smoother else result.persons
        t_tot_ms = (time.perf_counter() - t0) * 1000

        current_fps = 1000.0 / max(t_tot_ms, 1.0)
        history_fps.append(current_fps)
        if len(history_fps) > 30:
            history_fps.pop(0)
        rolling_fps = float(np.mean(history_fps))

        # Left Panel: 2D Video + 70 Keypoints
        left_img = display_frame.copy()
        if persons:
            draw_keypoints_and_skeleton(left_img, persons[0])

        cv2.putText(left_img, f"{video_path.stem} | 70 KEYPOINTS", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)

        # Right Panel: 3D Skeleton Studio
        if persons:
            p = persons[0]
            rot_azim = 24.0 + 8.0 * np.sin(proc_idx * 0.05)
            right_img = render_3d_skeleton_studio(p.joints_3d_local, width=panel_w, height=panel_h,
                                                  azim=rot_azim, elev=-8.0)
        else:
            right_img = np.full((panel_h, panel_w, 3), (14, 16, 20), dtype=np.uint8)
            cv2.putText(right_img, "No person detected", (panel_w // 4, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 145, 155), 1)

        # Top HUD
        hud_img = make_header_hud(canvas_w, hud_h, proc_idx + 1, frames_to_process, t_frame_s,
                                  result.detector_ms, result.hmr_ms, t_tot_ms, rolling_fps, "MPS")

        full_canvas = np.vstack([hud_img, np.hstack([left_img, right_img])])
        out_writer.write(full_canvas)

        proc_idx += 1
        if proc_idx % 45 == 0 or proc_idx == frames_to_process:
            print(f"  Frame {proc_idx:3d}/{frames_to_process} | Latency: {t_tot_ms:5.1f}ms | "
                  f"YOLO26n: {result.detector_ms:4.1f}ms | HMR: {result.hmr_ms:4.1f}ms | FPS: {rolling_fps:4.1f}")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"Finished processing in {dur:.1f}s ({proc_idx / max(dur, 0.1):.1f} FPS)")

    # Transcode to standard H.264 for QuickTime Player
    print(f"Transcoding {output_path.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw.exists():
        temp_raw.unlink()

    print(f"Ready: {output_path}")
    return output_path


def main():
    existing_videos = [v for v in INPUT_VIDEOS if v.exists()]
    if not existing_videos:
        sys.exit("No input IMG_*.mp4 videos found!")

    print(f"Found {len(existing_videos)} videos to process:")
    for v in existing_videos:
        print(f" - {v}")

    # 1. Spawn interactive 3D window for the first video
    first_video = existing_videos[0]
    rerun_proc = spawn_interactive_3d_rerun(first_video)

    # 2. Initialize high-speed shared pipeline on MPS
    print("\nInitializing InstantHMR v2 (MPS + YOLO26n stride 4, keypoints only)...")
    pipeline = PosePipeline("medium", device="mps", detector="yolo26n", detector_stride=4, with_mesh=False)

    # 3. Process all videos
    output_videos = []
    for video in existing_videos:
        out_name = f"instanthmr_v2_{video.stem}.mp4"
        out_path = OUTPUT_DIR / out_name
        res_path = process_video(video, pipeline, out_path)
        output_videos.append(res_path)

    # 4. Open all output videos on the user's screen
    print("\n==================================================================")
    print("All videos successfully generated! Opening on screen...")
    print("==================================================================")
    for v in output_videos:
        print(f"Launching {v.name} in QuickTime Player...")
        subprocess.run(["open", str(v)])


if __name__ == "__main__":
    main()
