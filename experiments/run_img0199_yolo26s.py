#!/usr/bin/env python3
"""
Process IMG_0199 with YOLO26s (Small object detector on MPS, stride 4):
- Resolves false positive tracking jump on frame 393
- Smooth 70 keypoints + 3D Skeleton Studio
- Opens in QuickTime Player upon completion
"""

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

INPUT_VIDEO = Path("/Users/sidhparikh/Downloads/IMG_0199.mp4")
OUTPUT_DIR = Path("/Users/sidhparikh/Downloads")
OUTPUT_PATH = OUTPUT_DIR / "instanthmr_v2_IMG_0199_yolo26s.mp4"
TEMP_RAW = OUTPUT_PATH.with_suffix(".raw.mp4")


def main():
    print(f"\n=======================================================")
    print(f"Processing IMG_0199 with YOLO26s (Small)")
    print(f"=======================================================")

    pipeline = PosePipeline(
        "medium",
        device="mps",
        detector="yolo26s",
        det_confidence=0.35,
        detector_stride=4,
        with_mesh=False,
        crop_expand=1.4,
        use_pose_envelope=False
    )

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = 640
    target_h = int(round(h_src * (target_w / w_src)))
    if target_h % 2 != 0:
        target_h += 1

    hud_h = 60
    panel_w, panel_h = target_w, target_h
    canvas_w = panel_w * 2
    canvas_h = panel_h + hud_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(TEMP_RAW), fourcc, src_fps, (canvas_w, canvas_h))

    smoother = TemporalSmoother(beta_scale=1.2)
    history_fps = []
    t_start = time.perf_counter()

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        display_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        t_frame_s = idx / src_fps
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

        # Left: 2D Video + YOLO26s Box + 70 KPS
        left_img = display_frame.copy()
        if persons:
            draw_keypoints_and_skeleton(left_img, persons[0])

        cv2.putText(left_img, f"IMG_0199 | YOLO26s (stride 4) | 70 KPS", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)

        # Right: 3D Studio
        if persons:
            p = persons[0]
            rot_azim = 24.0 + 8.0 * np.sin(idx * 0.05)
            right_img = render_3d_skeleton_studio(p.joints_3d_local, width=panel_w, height=panel_h,
                                                  azim=rot_azim, elev=-8.0)
        else:
            right_img = np.full((panel_h, panel_w, 3), (14, 16, 20), dtype=np.uint8)
            cv2.putText(right_img, "Searching for person...", (panel_w // 4, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 145, 155), 1)

        # Top HUD
        hud_img = make_header_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                                  result.detector_ms, result.hmr_ms, t_tot_ms, rolling_fps, "MPS (YOLO26s)")

        full_canvas = np.vstack([hud_img, np.hstack([left_img, right_img])])
        out_writer.write(full_canvas)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
            print(f"  Frame {idx + 1:3d}/{total_frames} | Latency: {t_tot_ms:5.1f}ms | "
                  f"YOLO26s: {result.detector_ms:4.1f}ms | HMR: {result.hmr_ms:4.1f}ms | FPS: {rolling_fps:4.1f}")

    cap.release()
    out_writer.release()

    dur = time.perf_counter() - t_start
    print(f"\nProcessing completed in {dur:.1f}s ({total_frames / max(dur, 0.1):.1f} FPS)")

    print(f"Transcoding to H.264: {OUTPUT_PATH.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(TEMP_RAW),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUTPUT_PATH)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if TEMP_RAW.exists():
        TEMP_RAW.unlink()

    print(f"\n>>> LAUNCHING ON SCREEN IN QUICKTIME PLAYER: {OUTPUT_PATH}")
    subprocess.run(["open", str(OUTPUT_PATH)])


if __name__ == "__main__":
    main()
