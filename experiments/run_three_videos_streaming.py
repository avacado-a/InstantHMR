#!/usr/bin/env python3
"""
Process IMG_0195, IMG_0198, and IMG_0199 with InstantHMR v2:
- Detector: YOLO26n, stride 4, no pose envelope
- Crop Expand: 1.4x
- Keypoints only (with_mesh=False)
- Immediately opens each video in QuickTime Player as soon as it finishes transcoding
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from instanthmr import PosePipeline
from instanthmr.smoothing import TemporalSmoother
from run_v2_video_showcase import (
    draw_keypoints_and_skeleton,
    render_3d_skeleton_studio,
    make_header_hud,
)

INPUT_VIDEOS = [
    Path("/Users/sidhparikh/Downloads/IMG_0195.mp4"),
    Path("/Users/sidhparikh/Downloads/IMG_0198.mp4"),
    Path("/Users/sidhparikh/Downloads/IMG_0199.mp4"),
]

OUTPUT_DIR = Path("/Users/sidhparikh/Downloads")


def process_video_and_launch(video_path: Path, pipeline: PosePipeline):
    print(f"\n" + "=" * 70)
    print(f"PROCESSING: {video_path.name}")
    print(f"=" * 70)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Landscape videos (1920x1080) -> Target 640x360 per panel
    target_w = 640
    target_h = int(round(h_src * (target_w / w_src)))
    if target_h % 2 != 0:
        target_h += 1

    out_fps = src_fps
    frames_to_process = total_src_frames

    print(f"Source: {w_src}x{h_src} @ {src_fps:.1f} FPS ({total_src_frames} frames)")
    print(f"Target: {target_w}x{target_h} @ {out_fps:.1f} FPS (Full {frames_to_process} frames)")

    hud_h = 60
    panel_w, panel_h = target_w, target_h
    canvas_w = panel_w * 2
    canvas_h = panel_h + hud_h

    out_name = f"instanthmr_v2_{video_path.stem}.mp4"
    out_path = OUTPUT_DIR / out_name
    temp_raw = out_path.with_suffix(".raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_raw), fourcc, out_fps, (canvas_w, canvas_h))

    smoother = TemporalSmoother(beta_scale=1.2)
    history_fps = []
    t_start = time.perf_counter()

    for proc_idx in range(frames_to_process):
        ok, frame_bgr = cap.read()
        if not ok:
            break

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

        # Left Panel: 2D Video + 70 Keypoints + YOLO26n Box
        left_img = display_frame.copy()
        if persons:
            draw_keypoints_and_skeleton(left_img, persons[0])

        cv2.putText(left_img, f"{video_path.stem} | YOLO26n (stride 4) | 70 KPS", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)

        # Right Panel: 3D Skeleton Studio
        if persons:
            p = persons[0]
            rot_azim = 24.0 + 8.0 * np.sin(proc_idx * 0.05)
            right_img = render_3d_skeleton_studio(p.joints_3d_local, width=panel_w, height=panel_h,
                                                  azim=rot_azim, elev=-8.0)
        else:
            right_img = np.full((panel_h, panel_w, 3), (14, 16, 20), dtype=np.uint8)
            cv2.putText(right_img, "Searching for person...", (panel_w // 4, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 145, 155), 1)

        # Top HUD
        hud_img = make_header_hud(canvas_w, hud_h, proc_idx + 1, frames_to_process, t_frame_s,
                                  result.detector_ms, result.hmr_ms, t_tot_ms, rolling_fps, "MPS")

        full_canvas = np.vstack([hud_img, np.hstack([left_img, right_img])])
        out_writer.write(full_canvas)

        if (proc_idx + 1) % 50 == 0 or (proc_idx + 1) == frames_to_process:
            print(f"  Frame {proc_idx + 1:3d}/{frames_to_process} | Latency: {t_tot_ms:5.1f}ms | "
                  f"YOLO26n: {result.detector_ms:4.1f}ms | HMR: {result.hmr_ms:4.1f}ms | FPS: {rolling_fps:4.1f}")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"Finished {video_path.name} in {dur:.1f}s ({frames_to_process / max(dur, 0.1):.1f} FPS)")

    # Transcode to standard H.264 for QuickTime Player
    print(f"Transcoding {out_name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_raw.exists():
        temp_raw.unlink()

    print(f"\n>>> VIDEO READY: {out_path}")
    print(f">>> LAUNCHING IMMEDIATELY ON SCREEN IN QUICKTIME PLAYER...")
    subprocess.run(["open", str(out_path)])
    return out_path


def main():
    print("Initializing InstantHMR v2 (MPS + YOLO26n stride 4, crop_expand=1.4, no pose envelope)...")
    pipeline = PosePipeline(
        "medium",
        device="mps",
        detector="yolo26n",
        detector_stride=4,
        with_mesh=False,
        crop_expand=1.4,
        use_pose_envelope=False
    )

    for v in INPUT_VIDEOS:
        if v.exists():
            process_video_and_launch(v, pipeline)
        else:
            print(f"Warning: {v} not found!")

    print("\nAll 3 videos processed and launched on screen!")


if __name__ == "__main__":
    main()
