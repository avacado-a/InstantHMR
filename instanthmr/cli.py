"""InstantHMR Unified CLI Tool.

Supports:
- Single video, directory of videos, single image, or live webcam.
- Detectors: YOLO26s (default), YOLO26n, YOLO26s-pose, YOLO26n-pose, RF-DETR Nano, YOLOv8n.
- InstantHMR models: Medium (default), Small, Tiny, Nano.
- Modes:
    * 'studio': Split view (2D camera + 3D Skeleton Studio)
    * 'overlay': Full-frame 2D camera overlay
    * 'studio3d': Full-frame 3D Skeleton Studio
    * 'compare': Side-by-side detector comparison benchmark
- Jitter-free stabilized bounding boxes with 1-Euro motion filtering.
- Real-time telemetry HUD with rolling FPS and latencies.
- Universal H.264 transcoding and automatic macOS QuickTime launch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .detector import PersonDetector
from .inference import InstantHMR
from .models import DEFAULT_MODEL, resolve_model
from .smoothing import BoxStabilizer, TemporalSmoother
from .visualizer import (
    COLOR_BG,
    COLOR_GREEN,
    COLOR_ORANGE,
    draw_person_overlay,
    make_compare_hud,
    make_single_hud,
    render_3d_skeleton_studio,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="instanthmr",
        description="InstantHMR v2: Real-time 3D Human Mesh & Pose Estimation Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Input / Output
    src = parser.add_argument_group("Input / Output Options")
    src.add_argument("-i", "--input", type=str, required=True,
                     help="Path to input video file, directory of videos, image, or camera index (e.g. 0)")
    src.add_argument("-o", "--output", type=str, default=None,
                     help="Path to output file or directory (default: auto in Downloads/ or next to input)")
    src.add_argument("--save-data", type=str, nargs="?", const="auto", default=None,
                     help="Save raw extracted motion data (joints_2d, joints_3d, bbox, cam_trans, timestamps) to .npz or .json file")
    src.add_argument("--render-video", dest="render_video", action="store_true", default=True,
                     help="Render and save output video")
    src.add_argument("--no-render", "--no-video", dest="render_video", action="store_false",
                     help="Skip video rendering and encoding for maximum throughput")
    src.add_argument("--no-save", dest="no_save", action="store_true", default=False,
                     help="Pure speed benchmark mode: do not save video or data files to disk")
    src.add_argument("--max-frames", type=int, default=None,
                     help="Process only first N frames (useful for quick previews and tests)")
    src.add_argument("--open", dest="open_player", action="store_true", default=True,
                     help="Automatically launch output video in macOS QuickTime Player upon completion")
    src.add_argument("--no-open", dest="open_player", action="store_false",
                     help="Do not launch video player after saving")

    # 2. Model & Backend
    models = parser.add_argument_group("Model & Backend Options")
    models.add_argument("-d", "--detector", type=str, default="yolo26s",
                        help="Primary detector: 'yolo26s', 'yolo26n', 'yolo26s-pose', 'rfdetr_nano', 'yolov8n'")
    models.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                        help="InstantHMR model variant: 'medium', 'small', 'tiny', 'nano'")
    models.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                        choices=["mps", "cpu", "cuda", "coreml"],
                        help="Compute backend device for InstantHMR")
    models.add_argument("--detector-device", type=str, default=None,
                        help="Specific device for detector (defaults to --device)")
    models.add_argument("--det-conf", type=float, default=0.25,
                        help="Detector minimum confidence threshold")
    models.add_argument("--max-persons", type=int, default=2,
                        help="Maximum number of people to track and reconstruct per frame")
    models.add_argument("--crop-expand", type=float, default=1.4,
                        help="Bounding box expansion ratio before feeding into InstantHMR")

    # 3. Execution Mode
    mode_group = parser.add_argument_group("Presentation Mode Options")
    mode_group.add_argument("--mode", type=str, default="studio",
                            choices=["studio", "overlay", "studio3d", "compare"],
                            help="Presentation layout: 'studio' (2D+3D), 'overlay' (2D), 'studio3d' (3D), 'compare' (dual detector)")
    mode_group.add_argument("--compare-with", type=str, default="rfdetr_nano",
                            help="Secondary detector to compare against when --mode compare is selected")

    # 4. Smoothing & Jitter Control
    smooth_group = parser.add_argument_group("Temporal Smoothing & Stabilization")
    smooth_group.add_argument("--smooth", dest="smooth", action="store_true", default=True,
                              help="Enable 1-Euro temporal filtering for 2D/3D joints with zero-lag beta tuning")
    smooth_group.add_argument("--no-smooth", dest="smooth", action="store_false",
                              help="Disable smoothing (render 100%% raw frame predictions)")
    smooth_group.add_argument("--stabilize-box", dest="stabilize_box", action="store_true", default=True,
                              help="Enable BoxStabilizer to eliminate 4-frame detector stride jitter")
    smooth_group.add_argument("--no-stabilize-box", dest="stabilize_box", action="store_false",
                              help="Disable bounding box stabilization")
    smooth_group.add_argument("-s", "--detector-stride", type=int, default=4,
                              help="Number of frames between detector runs (1 = detect every frame)")
    smooth_group.add_argument("--async-detector", dest="async_detector", action="store_true", default=True,
                              help="Run detector asynchronously in a background thread to prevent pipeline blocking")
    smooth_group.add_argument("--no-async-detector", dest="async_detector", action="store_false",
                              help="Run detector synchronously inline")
    smooth_group.add_argument("--beta-scale", type=float, default=1.0,
                              help="1-Euro beta scaling factor (higher = more responsive, lower = smoother)")

    # 5. Studio & Visuals
    vis_group = parser.add_argument_group("Studio & Display Options")
    vis_group.add_argument("--hud", dest="show_hud", action="store_true", default=True,
                           help="Display real-time telemetry HUD with live rolling FPS and latency breakdown")
    vis_group.add_argument("--no-hud", dest="show_hud", action="store_false",
                           help="Hide top telemetry HUD")
    vis_group.add_argument("--rotate-speed", type=float, default=0.05,
                           help="Angular rotation speed for 3D studio viewpoint (set to 0 for static angle)")
    vis_group.add_argument("--azim", type=float, default=24.0,
                           help="Base camera azimuth angle for 3D studio in degrees")

    return parser.parse_args(argv)


def resolve_output_path(input_path: Path, output_arg: str | None, mode: str, detector: str,
                        compare_with: str | None = None) -> Path:
    """Determine clean output file path."""
    if output_arg is not None:
        p = Path(output_arg)
        if p.is_dir() or (not p.suffix and not p.exists()):
            p.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem if input_path.exists() else "webcam"
            tag = f"{detector}_vs_{compare_with}" if mode == "compare" else f"{detector}_{mode}"
            return p / f"instanthmr_{stem}_{tag}.mp4"
        return p

    # Default to Downloads or input directory
    downloads_dir = Path("/Users/sidhparikh/Downloads")
    target_dir = downloads_dir if downloads_dir.exists() else input_path.parent
    stem = input_path.stem if input_path.exists() else "stream"
    tag = f"compare_{detector}_vs_{compare_with}" if mode == "compare" else f"{detector}_{mode}"
    return target_dir / f"instanthmr_{stem}_{tag}.mp4"


def resolve_data_path(input_path: Path, data_arg: str | None, mode: str, detector: str,
                      compare_with: str | None = None) -> Path | None:
    """Determine output file path for extracted prediction data (.npz or .json)."""
    if data_arg is None:
        return None

    downloads_dir = Path("/Users/sidhparikh/Downloads")
    target_dir = downloads_dir if downloads_dir.exists() else input_path.parent
    stem = input_path.stem if input_path.exists() else "stream"
    tag = f"compare_{detector}_vs_{compare_with}" if mode == "compare" else f"{detector}_{mode}"

    if data_arg == "auto":
        return target_dir / f"instanthmr_{stem}_{tag}_data.npz"

    p = Path(data_arg)
    if p.is_dir() or (not p.suffix and not p.exists()):
        p.mkdir(parents=True, exist_ok=True)
        return p / f"instanthmr_{stem}_{tag}_data.npz"
    return p


def save_motion_data(data_path: Path, results: list[dict], src_fps: float,
                     w_src: int, h_src: int, meta: dict | None = None) -> None:
    """Save extracted predictions to .npz or .json file."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(results)
    suffix = data_path.suffix.lower()

    if suffix == ".json":
        frames_data = []
        for r in results:
            p = r.get("person")
            f_entry = {
                "frame_idx": int(r["idx"]),
                "time_s": round(float(r["time_s"]), 4),
                "det_ms": round(float(r.get("det_ms", 0.0)), 2),
                "hmr_ms": round(float(r.get("hmr_ms", 0.0)), 2),
                "detected": p is not None,
            }
            if p is not None:
                f_entry["bbox"] = [round(float(x), 2) for x in p.bbox]
                f_entry["confidence"] = round(float(p.confidence), 4)
                f_entry["joints_2d"] = np.round(p.joints_2d, 2).tolist()
                f_entry["joints_3d_local"] = np.round(p.joints_3d_local, 4).tolist()
                f_entry["joints_3d_cam"] = np.round(p.joints_3d_cam, 4).tolist()
                f_entry["cam_trans"] = [round(float(x), 4) for x in p.cam_trans]
            frames_data.append(f_entry)

        payload = {
            "fps": float(src_fps),
            "image_size": [int(w_src), int(h_src)],
            "total_frames": N,
            "metadata": meta or {},
            "frames": frames_data,
        }
        with open(data_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f">>> SAVED MOTION DATA (JSON): {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
    else:
        # Default .npz format
        j2d = np.zeros((N, 70, 2), dtype=np.float32)
        j3d_local = np.zeros((N, 70, 3), dtype=np.float32)
        j3d_cam = np.zeros((N, 70, 3), dtype=np.float32)
        cam_trans = np.zeros((N, 3), dtype=np.float32)
        bbox = np.zeros((N, 4), dtype=np.float32)
        confidence = np.zeros((N,), dtype=np.float32)
        valid = np.zeros((N,), dtype=bool)
        timestamps = np.array([r["time_s"] for r in results], dtype=np.float32)
        det_latencies = np.array([r.get("det_ms", 0.0) for r in results], dtype=np.float32)
        hmr_latencies = np.array([r.get("hmr_ms", 0.0) for r in results], dtype=np.float32)

        for i, r in enumerate(results):
            p = r.get("person")
            if p is not None:
                valid[i] = True
                j2d[i] = p.joints_2d
                j3d_local[i] = p.joints_3d_local
                j3d_cam[i] = p.joints_3d_cam
                cam_trans[i] = p.cam_trans
                bbox[i] = p.bbox
                confidence[i] = p.confidence

        save_dict = {
            "joints_2d": j2d,
            "joints_3d_local": j3d_local,
            "joints_3d_cam": j3d_cam,
            "cam_trans": cam_trans,
            "bbox": bbox,
            "confidence": confidence,
            "valid": valid,
            "timestamps": timestamps,
            "det_latencies_ms": det_latencies,
            "hmr_latencies_ms": hmr_latencies,
            "fps": np.array(src_fps, dtype=np.float32),
            "image_size": np.array([w_src, h_src], dtype=np.int32),
        }
        if meta:
            for k, v in meta.items():
                if isinstance(v, (int, float, str, bool)):
                    save_dict[f"meta_{k}"] = np.array(v)

        np.savez_compressed(str(data_path), **save_dict)
        print(f">>> SAVED MOTION DATA (NPZ): {data_path} ({data_path.stat().st_size / (1024*1024):.2f} MB)")


def save_compare_data(data_path: Path, results: list[dict], src_fps: float,
                      w_src: int, h_src: int, name1: str, name2: str, meta: dict | None = None) -> None:
    """Save dual-detector comparison predictions to .npz or .json file."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(results)
    suffix = data_path.suffix.lower()

    if suffix == ".json":
        frames_data = []
        for r in results:
            p1 = r.get("p1")
            p2 = r.get("p2")
            f_entry = {
                "frame_idx": int(r["idx"]),
                "time_s": round(float(r["time_s"]), 4),
                "det1_ms": round(float(r.get("det1_ms", 0.0)), 2),
                "det2_ms": round(float(r.get("det2_ms", 0.0)), 2),
                "hmr_ms": round(float(r.get("hmr_ms", 0.0)), 2),
            }
            if p1 is not None:
                f_entry[f"{name1}_bbox"] = [round(float(x), 2) for x in p1.bbox]
                f_entry[f"{name1}_confidence"] = round(float(p1.confidence), 4)
                f_entry[f"{name1}_joints_3d"] = np.round(p1.joints_3d_local, 4).tolist()
            if p2 is not None:
                f_entry[f"{name2}_bbox"] = [round(float(x), 2) for x in p2.bbox]
                f_entry[f"{name2}_confidence"] = round(float(p2.confidence), 4)
                f_entry[f"{name2}_joints_3d"] = np.round(p2.joints_3d_local, 4).tolist()
            frames_data.append(f_entry)

        payload = {
            "fps": float(src_fps),
            "image_size": [int(w_src), int(h_src)],
            "total_frames": N,
            "models": [name1, name2],
            "metadata": meta or {},
            "frames": frames_data,
        }
        with open(data_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f">>> SAVED COMPARISON DATA (JSON): {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
    else:
        def pack_person_array(key: str):
            j3d = np.zeros((N, 70, 3), dtype=np.float32)
            bbox = np.zeros((N, 4), dtype=np.float32)
            valid = np.zeros((N,), dtype=bool)
            for i, r in enumerate(results):
                p = r.get(key)
                if p is not None:
                    valid[i] = True
                    j3d[i] = p.joints_3d_local
                    bbox[i] = p.bbox
            return j3d, bbox, valid

        j3d_1, box_1, val_1 = pack_person_array("p1")
        j3d_2, box_2, val_2 = pack_person_array("p2")

        save_dict = {
            f"{name1}_joints_3d": j3d_1,
            f"{name1}_bbox": box_1,
            f"{name1}_valid": val_1,
            f"{name2}_joints_3d": j3d_2,
            f"{name2}_bbox": box_2,
            f"{name2}_valid": val_2,
            "timestamps": np.array([r["time_s"] for r in results], dtype=np.float32),
            "fps": np.array(src_fps, dtype=np.float32),
            "image_size": np.array([w_src, h_src], dtype=np.int32),
        }
        np.savez_compressed(str(data_path), **save_dict)
        print(f">>> SAVED COMPARISON DATA (NPZ): {data_path} ({data_path.stat().st_size / (1024*1024):.2f} MB)")


def transcode_to_h264(raw_path: Path, final_path: Path) -> None:
    """Transcode raw mp4v video to standard H.264 for universal QuickTime playback."""
    print(f"Transcoding to H.264 ({final_path.name})...")
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if raw_path.exists():
        raw_path.unlink()


def process_video_single(cap: cv2.VideoCapture, out_path: Path, data_path: Path | None,
                         args: argparse.Namespace, input_path: Path) -> None:
    """Process video in single-pipeline mode ('studio', 'overlay', or 'studio3d')."""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames and args.max_frames < total_frames:
        total_frames = args.max_frames
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Canvas Dimensions
    if args.mode == "studio":
        canvas_w = 1280
        canvas_h = 720
        hud_h = 70 if args.show_hud else 0
        content_h = canvas_h - hud_h
        v2d_w = int(round(content_h * (w_src / h_src)))
        if v2d_w % 2 != 0:
            v2d_w += 1
        v3d_w = canvas_w - v2d_w
    else:
        # Full frame 720p
        canvas_w = 1280
        canvas_h = 720
        hud_h = 70 if args.show_hud else 0
        content_h = canvas_h - hud_h
        v2d_w = canvas_w
        v3d_w = canvas_w

    # Initialize Models
    det_dev = args.detector_device or ("cpu" if args.async_detector else args.device)
    print(f"Initializing Detector: {args.detector} ({det_dev})...")
    det = PersonDetector(args.detector, confidence=args.det_conf, max_persons=args.max_persons,
                         device=det_dev)

    print(f"Initializing InstantHMR: {args.model} ({args.device}, crop_expand={args.crop_expand})...")
    ckpt = resolve_model(args.model)
    hmr = InstantHMR(ckpt, device=args.device, with_mesh=False, crop_expand=args.crop_expand,
                     max_batch=args.max_persons)
    hmr.warmup(max_batch=args.max_persons)

    smoother = TemporalSmoother(beta_scale=args.beta_scale) if args.smooth else None
    stabilizer = BoxStabilizer(mincutoff=0.8, beta=0.02) if args.stabilize_box else None

    # =========================================================================
    # STAGE 1: PURE INFERENCE PIPELINE (No video encoding overhead)
    # =========================================================================
    print(f"\n[Stage 1/2] Running Pure Inference Pipeline on {total_frames} frames (Async Detector: {args.async_detector})...")
    results = []
    history_fps = []
    det_times = []
    hmr_times = []
    last_det_ms = 0.0

    # Threading setup for asynchronous detector worker
    det_task_queue = queue.Queue(maxsize=1) if args.async_detector else None
    det_res_queue = queue.Queue() if args.async_detector else None
    stop_det_event = threading.Event() if args.async_detector else None

    def async_detector_worker():
        while not stop_det_event.is_set():
            try:
                task = det_task_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if task is None:
                break
            f_rgb, f_time = task
            t0_w = time.perf_counter()
            b, s = det(f_rgb)
            ms_w = (time.perf_counter() - t0_w) * 1000.0
            det_res_queue.put((b, s, ms_w, f_time))

    det_thread = None
    if args.async_detector:
        det_thread = threading.Thread(target=async_detector_worker, daemon=True)
        det_thread.start()

    t_start = time.perf_counter()

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            total_frames = idx
            break

        t_frame_s = idx / src_fps
        frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 1. Detector Step (Async or Sync)
        run_det = (idx % args.detector_stride == 0) or (stabilizer and stabilizer.current_box is None)

        if args.async_detector:
            # Drain any completed async detection
            while not det_res_queue.empty():
                async_b, async_s, async_ms, async_time = det_res_queue.get_nowait()
                last_det_ms = async_ms
                det_times.append(last_det_ms)
                if len(async_b) > 0:
                    if stabilizer:
                        stabilizer.update_detection(async_b[0], async_s[0], async_time)
                else:
                    if stabilizer:
                        stabilizer.current_score = 0.0

            # Submit next frame for detection if interval reached and worker is idle
            if run_det:
                if det_task_queue.empty():
                    det_task_queue.put((rgb.copy(), t_frame_s))
        else:
            if run_det:
                t0 = time.perf_counter()
                boxes, scores = det(rgb)
                last_det_ms = (time.perf_counter() - t0) * 1000.0
                det_times.append(last_det_ms)
                if len(boxes) > 0:
                    if stabilizer:
                        stabilizer.update_detection(boxes[0], scores[0], t_frame_s)
                else:
                    if stabilizer:
                        stabilizer.current_score = 0.0

        curr_box = stabilizer.get_box(t_frame_s) if stabilizer else (boxes[0] if (not args.async_detector and run_det and len(boxes) > 0) else None)
        curr_score = stabilizer.current_score if stabilizer else (scores[0] if (not args.async_detector and run_det and len(scores) > 0) else 0.0)

        # 2. InstantHMR Step
        t0 = time.perf_counter()
        if curr_box is not None:
            preds = hmr.predict(rgb, [curr_box], confidences=[curr_score])
            t_hmr_ms = (time.perf_counter() - t0) * 1000.0
            person = preds[0] if len(preds) > 0 else None
            hmr_times.append(t_hmr_ms)
        else:
            person = None
            t_hmr_ms = 0.0

        # 3. Temporal Smoothing (1-Euro tuned for pixel and 3D metric units)
        if person is not None and smoother:
            s_res = smoother([person], t_frame_s)
            person = s_res[0] if s_res else person

        # 4. Latency & Rolling FPS
        total_lat = (last_det_ms if run_det and not args.async_detector else 0.0) + (t_hmr_ms if person is not None else 0.0)
        history_fps.append(1000.0 / max(total_lat, 1.0))
        if len(history_fps) > 30:
            history_fps.pop(0)
        rolling_fps = float(np.mean(history_fps))

        results.append({
            "idx": idx,
            "time_s": t_frame_s,
            "person": person,
            "det_ms": last_det_ms,
            "hmr_ms": t_hmr_ms,
            "rolling_fps": rolling_fps,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
            elapsed = time.perf_counter() - t_start
            cur_fps = (idx + 1) / max(elapsed, 0.001)
            print(f"  [Inference] Frame {idx + 1:4d}/{total_frames} | "
                  f"Throughput: {cur_fps:4.1f} FPS (Rolling: {rolling_fps:4.1f} FPS) | "
                  f"Det: {last_det_ms:4.1f}ms | HMR: {t_hmr_ms:4.1f}ms")

    if args.async_detector and det_task_queue:
        stop_det_event.set()
        det_task_queue.put(None)
        if det_thread:
            det_thread.join(timeout=1.0)

    dur_inf = time.perf_counter() - t_start
    avg_inf_fps = total_frames / max(dur_inf, 0.001)
    avg_det_ms = float(np.mean(det_times)) if det_times else 0.0
    avg_hmr_ms = float(np.mean(hmr_times)) if hmr_times else 0.0

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE: PURE INFERENCE BENCHMARK")
    print(f"  Total Frames:       {total_frames}")
    print(f"  Inference Duration: {dur_inf:.2f} s")
    print(f"  Avg Inference Speed:{avg_inf_fps:5.1f} FPS")
    print(f"  Avg Detector Time:  {avg_det_ms:5.1f} ms (stride {args.detector_stride})")
    print(f"  Avg InstantHMR Time:{avg_hmr_ms:5.1f} ms")
    print("=" * 70)

    # Save Motion Data (.npz or .json)
    if data_path is not None and not args.no_save:
        save_motion_data(data_path, results, src_fps, w_src, h_src, meta={
            "detector": args.detector,
            "model": args.model,
            "inference_fps": avg_inf_fps,
            "detector_stride": args.detector_stride,
            "smooth": args.smooth,
        })

    # Skip rendering if requested
    if not args.render_video or args.no_save:
        cap.release()
        if args.no_save:
            print("\n>>> PURE BENCHMARK MODE (--no-save): zero disk files created.")
        else:
            print(f"\n>>> VIDEO RENDERING SKIPPED (--no-render). Finished in {dur_inf:.2f}s.")
        return

    # =========================================================================
    # STAGE 2: STUDIO VIDEO RENDERING (Decoupled post-process)
    # =========================================================================
    print(f"\n[Stage 2/2] Rendering {args.mode.upper()} Video ({canvas_w}x{canvas_h})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok_test, _ = cap.read()
    if not ok_test:
        cap.release()
        cap = cv2.VideoCapture(str(input_path))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temp_raw = out_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw), fourcc, src_fps, (canvas_w, canvas_h))

    t_render_start = time.perf_counter()
    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        r = results[idx]
        person = r["person"]
        rolling_fps = r["rolling_fps"]
        last_det_ms = r["det_ms"]
        t_hmr_ms = r["hmr_ms"]
        t_frame_s = r["time_s"]

        frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)

        # Compose Output Content
        azim = args.azim + args.rotate_speed * math.degrees(math.sin(idx * 0.05)) if args.rotate_speed > 0 else args.azim
        j3d = person.joints_3d_local if person is not None else None
        z_depth = person.cam_trans[2] if person is not None else None

        if args.mode == "studio":
            v2d = frame_resized.copy()
            if person is not None:
                draw_person_overlay(v2d, person, args.detector, COLOR_GREEN, (50, 180, 80))
            cv2.putText(v2d, f"{args.detector.upper()} TRACKING", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GREEN, 1, cv2.LINE_AA)
            v3d = render_3d_skeleton_studio(j3d, v3d_w, content_h, azim=azim, z_depth=z_depth, model_tag=args.detector)
            content = np.hstack([v2d, v3d])
            cv2.line(content, (v2d_w, 0), (v2d_w, content_h), (50, 58, 70), 2)
        elif args.mode == "overlay":
            content = frame_resized.copy()
            if person is not None:
                draw_person_overlay(content, person, args.detector, COLOR_GREEN, (50, 180, 80))
        elif args.mode == "studio3d":
            content = render_3d_skeleton_studio(j3d, canvas_w, content_h, azim=azim, z_depth=z_depth, model_tag=args.detector)

        # Top Telemetry HUD
        if args.show_hud:
            hud = make_single_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                                  rolling_fps, last_det_ms, t_hmr_ms, args.detector,
                                  is_smoothed=args.smooth)
            full_frame = np.vstack([hud, content])
        else:
            full_frame = content

        writer.write(full_frame)

        if (idx + 1) % 200 == 0 or (idx + 1) == total_frames:
            r_dur = time.perf_counter() - t_render_start
            r_fps = (idx + 1) / max(r_dur, 0.001)
            print(f"  [Rendering] Frame {idx + 1:4d}/{total_frames} | Speed: {r_fps:4.1f} FPS")

    cap.release()
    writer.release()

    dur_render = time.perf_counter() - t_render_start
    print(f"\nRendering finished in {dur_render:.1f}s ({total_frames / max(dur_render, 0.1):.1f} FPS)")

    transcode_to_h264(temp_raw, out_path)
    print(f"\n>>> SAVED VIDEO: {out_path}")

    if args.open_player and sys.platform == "darwin":
        print(">>> Launching in QuickTime Player...")
        subprocess.run(["open", str(out_path)])


def process_video_compare(cap: cv2.VideoCapture, out_path: Path, data_path: Path | None,
                          args: argparse.Namespace, input_path: Path) -> None:
    """Process video in dual-detector comparison mode ('compare')."""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames and args.max_frames < total_frames:
        total_frames = args.max_frames
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas_w = 1280
    canvas_h = 720
    hud_h = 70 if args.show_hud else 0
    content_h = canvas_h - hud_h
    half_w = canvas_w // 2

    v2d_w = int(round(content_h * (w_src / h_src)))
    if v2d_w % 2 != 0:
        v2d_w += 1
    v3d_w = half_w - v2d_w

    name1 = args.detector
    name2 = args.compare_with

    print(f"Loading Detector 1: {name1}...")
    det1 = PersonDetector(name1, confidence=args.det_conf, max_persons=1,
                          device=args.detector_device or args.device)
    print(f"Loading Detector 2: {name2}...")
    det2 = PersonDetector(name2, confidence=args.det_conf, max_persons=1,
                          device=args.detector_device or args.device)

    print(f"Loading InstantHMR: {args.model} (Batched N=2)...")
    ckpt = resolve_model(args.model)
    hmr = InstantHMR(ckpt, device=args.device, with_mesh=False, crop_expand=args.crop_expand, max_batch=2)
    hmr.warmup(max_batch=2)

    smoother1 = TemporalSmoother(beta_scale=args.beta_scale) if args.smooth else None
    smoother2 = TemporalSmoother(beta_scale=args.beta_scale) if args.smooth else None

    stab1 = BoxStabilizer(mincutoff=0.8, beta=0.02) if args.stabilize_box else None
    stab2 = BoxStabilizer(mincutoff=0.8, beta=0.02) if args.stabilize_box else None

    # =========================================================================
    # STAGE 1: DUAL INFERENCE BENCHMARK
    # =========================================================================
    print(f"\n[Stage 1/2] Running Dual Inference Pipeline on {total_frames} frames ({name1} vs {name2})...")
    results = []
    hist_fps1, hist_fps2 = [], []
    det1_ms, det2_ms = 0.0, 0.0
    det1_times, det2_times = [], []
    hmr_times = []
    t_start = time.perf_counter()

    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            total_frames = idx
            break

        t_frame_s = idx / src_fps
        frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 1. Dual Detector Execution
        run_det = (idx % args.detector_stride == 0) or (stab1 and stab1.current_box is None) or (stab2 and stab2.current_box is None)
        if run_det:
            t0 = time.perf_counter()
            b1, s1 = det1(rgb)
            det1_ms = (time.perf_counter() - t0) * 1000.0
            det1_times.append(det1_ms)
            if len(b1) > 0 and stab1:
                stab1.update_detection(b1[0], s1[0], t_frame_s)

            t0 = time.perf_counter()
            b2, s2 = det2(rgb)
            det2_ms = (time.perf_counter() - t0) * 1000.0
            det2_times.append(det2_ms)
            if len(b2) > 0 and stab2:
                stab2.update_detection(b2[0], s2[0], t_frame_s)

        box1 = stab1.get_box(t_frame_s) if stab1 else None
        box2 = stab2.get_box(t_frame_s) if stab2 else None

        # 2. Batched InstantHMR Step (N=2)
        boxes_for_hmr = []
        conf_for_hmr = []
        mapping = {}

        if box1 is not None:
            mapping["d1"] = len(boxes_for_hmr)
            boxes_for_hmr.append(box1)
            conf_for_hmr.append(stab1.current_score if stab1 else 1.0)
        if box2 is not None:
            mapping["d2"] = len(boxes_for_hmr)
            boxes_for_hmr.append(box2)
            conf_for_hmr.append(stab2.current_score if stab2 else 1.0)

        t0 = time.perf_counter()
        if len(boxes_for_hmr) > 0:
            preds = hmr.predict(rgb, boxes_for_hmr, confidences=conf_for_hmr)
            t_hmr_ms = (time.perf_counter() - t0) * 1000.0
            hmr_times.append(t_hmr_ms)
        else:
            preds = []
            t_hmr_ms = 0.0

        p1 = preds[mapping["d1"]] if "d1" in mapping else None
        p2 = preds[mapping["d2"]] if "d2" in mapping else None

        # 3. Temporal Smoothing (1-Euro tuned for pixel and 3D metric units)
        if p1 is not None and smoother1:
            p1 = smoother1([p1], t_frame_s)[0]

        if p2 is not None and smoother2:
            p2 = smoother2([p2], t_frame_s)[0]

        # 4. Latencies & Rolling FPS
        lat1 = (det1_ms if run_det else 0.0) + (t_hmr_ms if p1 is not None else 0.0)
        lat2 = (det2_ms if run_det else 0.0) + (t_hmr_ms if p2 is not None else 0.0)
        hist_fps1.append(1000.0 / max(lat1, 1.0))
        hist_fps2.append(1000.0 / max(lat2, 1.0))
        if len(hist_fps1) > 30:
            hist_fps1.pop(0)
            hist_fps2.pop(0)
        fps1 = float(np.mean(hist_fps1))
        fps2 = float(np.mean(hist_fps2))

        results.append({
            "idx": idx,
            "time_s": t_frame_s,
            "p1": p1,
            "p2": p2,
            "det1_ms": det1_ms,
            "det2_ms": det2_ms,
            "hmr_ms": t_hmr_ms,
            "fps1": fps1,
            "fps2": fps2,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
            elapsed = time.perf_counter() - t_start
            print(f"  [Dual Inference] Frame {idx + 1:4d}/{total_frames} | "
                  f"{name1}: {det1_ms:4.1f}ms ({fps1:4.1f} FPS) | "
                  f"{name2}: {det2_ms:4.1f}ms ({fps2:4.1f} FPS) | HMR(x2): {t_hmr_ms:4.1f}ms")

    dur_inf = time.perf_counter() - t_start
    avg_inf_fps = total_frames / max(dur_inf, 0.001)

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE: DUAL INFERENCE BENCHMARK")
    print(f"  Total Frames:       {total_frames}")
    print(f"  Dual Inference Time:{dur_inf:.2f} s ({avg_inf_fps:5.1f} dual FPS)")
    print(f"  Avg {name1} Latency:{np.mean(det1_times) if det1_times else 0.0:5.1f} ms")
    print(f"  Avg {name2} Latency:{np.mean(det2_times) if det2_times else 0.0:5.1f} ms")
    print(f"  Avg Batched HMR:    {np.mean(hmr_times) if hmr_times else 0.0:5.1f} ms")
    print("=" * 70)

    # Save Comparison Data
    if data_path is not None and not args.no_save:
        save_compare_data(data_path, results, src_fps, w_src, h_src, name1, name2, meta={
            "detector1": name1,
            "detector2": name2,
            "model": args.model,
            "dual_fps": avg_inf_fps,
        })

    # Skip rendering if requested
    if not args.render_video or args.no_save:
        cap.release()
        if args.no_save:
            print("\n>>> PURE BENCHMARK MODE (--no-save): zero disk files created.")
        else:
            print(f"\n>>> VIDEO RENDERING SKIPPED (--no-render). Finished in {dur_inf:.2f}s.")
        return

    # =========================================================================
    # STAGE 2: DUAL STUDIO VIDEO RENDERING
    # =========================================================================
    print(f"\n[Stage 2/2] Rendering Side-by-Side Comparison Video ({canvas_w}x{canvas_h})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok_test, _ = cap.read()
    if not ok_test:
        cap.release()
        cap = cv2.VideoCapture(str(input_path))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temp_raw = out_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw), fourcc, src_fps, (canvas_w, canvas_h))

    t_render_start = time.perf_counter()
    for idx in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        r = results[idx]
        p1 = r["p1"]
        p2 = r["p2"]
        det1_ms = r["det1_ms"]
        det2_ms = r["det2_ms"]
        t_hmr_ms = r["hmr_ms"]
        fps1 = r["fps1"]
        fps2 = r["fps2"]
        t_frame_s = r["time_s"]

        frame_resized = cv2.resize(frame_bgr, (v2d_w, content_h), interpolation=cv2.INTER_LINEAR)
        azim = args.azim + args.rotate_speed * math.degrees(math.sin(idx * 0.05)) if args.rotate_speed > 0 else args.azim

        # Left: Detector 1
        d1_2d = frame_resized.copy()
        if p1 is not None:
            draw_person_overlay(d1_2d, p1, name1, COLOR_ORANGE, (30, 100, 200))
        cv2.putText(d1_2d, f"{name1.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ORANGE, 1, cv2.LINE_AA)
        d1_3d = render_3d_skeleton_studio(p1.joints_3d_local if p1 else None, v3d_w, content_h, azim=azim,
                                          z_depth=p1.cam_trans[2] if p1 else None, model_tag=name1)
        panel_left = np.hstack([d1_2d, d1_3d])

        # Right: Detector 2
        d2_2d = frame_resized.copy()
        if p2 is not None:
            draw_person_overlay(d2_2d, p2, name2, COLOR_GREEN, (50, 180, 80))
        cv2.putText(d2_2d, f"{name2.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GREEN, 1, cv2.LINE_AA)
        d2_3d = render_3d_skeleton_studio(p2.joints_3d_local if p2 else None, v3d_w, content_h, azim=azim,
                                          z_depth=p2.cam_trans[2] if p2 else None, model_tag=name2)
        panel_right = np.hstack([d2_2d, d2_3d])

        # Combine with Divider & HUD
        body = np.hstack([panel_left, panel_right])
        cv2.line(body, (half_w, 0), (half_w, content_h), (50, 58, 70), 2)

        if args.show_hud:
            hud = make_compare_hud(canvas_w, hud_h, idx + 1, total_frames, t_frame_s,
                                   fps1, det1_ms, name1, fps2, det2_ms, name2, t_hmr_ms)
            full_frame = np.vstack([hud, body])
        else:
            full_frame = body

        writer.write(full_frame)

        if (idx + 1) % 200 == 0 or (idx + 1) == total_frames:
            r_dur = time.perf_counter() - t_render_start
            r_fps = (idx + 1) / max(r_dur, 0.001)
            print(f"  [Rendering] Frame {idx + 1:4d}/{total_frames} | Speed: {r_fps:4.1f} FPS")

    cap.release()
    writer.release()

    dur_render = time.perf_counter() - t_render_start
    print(f"\nBenchmark rendering finished in {dur_render:.1f}s ({total_frames / max(dur_render, 0.1):.1f} FPS)")

    transcode_to_h264(temp_raw, out_path)
    print(f"\n>>> SAVED BENCHMARK VIDEO: {out_path}")

    if args.open_player and sys.platform == "darwin":
        print(">>> Launching in QuickTime Player...")
        subprocess.run(["open", str(out_path)])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate input source
    input_str = args.input
    is_cam = input_str.isdigit()

    if is_cam:
        cap = cv2.VideoCapture(int(input_str))
        input_path = Path(f"webcam_{input_str}")
    else:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"Error: input path not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Directory mode: recursively find videos
        if input_path.is_dir():
            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            vids = sorted([p for p in input_path.iterdir() if p.suffix.lower() in video_exts])
            if not vids:
                print(f"No video files found in directory: {input_path}", file=sys.stderr)
                sys.exit(1)
            print(f"Found {len(vids)} videos in {input_path.name}. Processing sequentially...")
            for v in vids:
                sub_args = argparse.Namespace(**vars(args))
                sub_args.input = str(v)
                main(sys.argv[1:])  # recurse
            return

        cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"Error: failed to open video stream from {input_str}", file=sys.stderr)
        sys.exit(1)

    out_path = resolve_output_path(input_path, args.output, args.mode, args.detector,
                                   compare_with=args.compare_with if args.mode == "compare" else None)
    data_path = resolve_data_path(input_path, args.save_data, args.mode, args.detector,
                                  compare_with=args.compare_with if args.mode == "compare" else None)

    print("=" * 70)
    print("INSTANTHMR v2 UNIFIED STUDIO")
    print(f"Input:        {input_str}")
    if args.no_save:
        print("Video Render: DISABLED (--no-save benchmark mode)")
        print("Data Export:  DISABLED (--no-save benchmark mode)")
    else:
        print(f"Video Render: {'Enabled (' + out_path.name + ')' if args.render_video else 'DISABLED (--no-render)'}")
        print(f"Data Export:  {'Enabled (' + data_path.name + ')' if data_path else 'Disabled'}")
    print(f"Mode:         {args.mode.upper()}")
    print(f"Detector:     {args.detector} (stride: {args.detector_stride}, device: {args.detector_device or args.device})")
    if args.mode == "compare":
        print(f"Compare:      {args.compare_with}")
    print(f"HMR Model:    {args.model} ({args.device})")
    print(f"Smoothing:    {'1-Euro (unit-calibrated beta=0.02, lag=0)' if args.smooth else 'RAW (disabled)'}")
    print(f"BoxStab:      {'Enabled (Jitter-Free)' if args.stabilize_box else 'Disabled'}")
    print("=" * 70)

    if args.mode == "compare":
        process_video_compare(cap, out_path, data_path, args, input_path)
    else:
        process_video_single(cap, out_path, data_path, args, input_path)


if __name__ == "__main__":
    main()
