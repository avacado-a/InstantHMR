#!/usr/bin/env python3
"""InstantHMR v2 - Model Weights Downloader.

Downloads the required ONNX checkpoints and YOLO detector weights into the `models/` directory.

Usage:
    python3 download_models.py               # Download default core models (medium, nano, mhr-body, yolo26s, yolo26n)
    python3 download_models.py --all         # Download all model variants (including small, tiny, pose models)
    python3 download_models.py --model nano  # Download a specific model
    python3 download_models.py --list        # List available models and check local status
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"

# HuggingFace repository for InstantHMR models
HF_BASE_URL = "https://huggingface.co/momolesang/InstantHMR/resolve/main"

# Ultralytics asset release for YOLO models
YOLO_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

MODELS_REGISTRY = {
    # Core InstantHMR MHR models
    "mhr-body.onnx": {
        "url": f"{HF_BASE_URL}/mhr-body.onnx",
        "description": "MHR Body Kinematic Mesh & Skeleton Graph",
        "size_mb": 18.0,
        "default": True,
    },
    "instanthmr-medium.onnx": {
        "url": f"{HF_BASE_URL}/instanthmr-medium.onnx",
        "description": "InstantHMR Medium Model (Highest Accuracy, Default)",
        "size_mb": 160.0,
        "default": True,
    },
    "instanthmr-nano.onnx": {
        "url": f"{HF_BASE_URL}/instanthmr-nano.onnx",
        "description": "InstantHMR Nano Model (High Speed / Real-Time)",
        "size_mb": 81.0,
        "default": True,
    },
    "instanthmr-small.onnx": {
        "url": f"{HF_BASE_URL}/instanthmr-small.onnx",
        "description": "InstantHMR Small Model",
        "size_mb": 115.0,
        "default": False,
    },
    "instanthmr-tiny.onnx": {
        "url": f"{HF_BASE_URL}/instanthmr-tiny.onnx",
        "description": "InstantHMR Tiny Model",
        "size_mb": 95.0,
        "default": False,
    },
    # YOLO Person Detectors
    "yolo26s.pt": {
        "url": f"{YOLO_BASE_URL}/yolo26s.pt",
        "description": "YOLO26s Person Detector (Balanced Speed & Accuracy, Default)",
        "size_mb": 20.4,
        "default": True,
    },
    "yolo26n.pt": {
        "url": f"{YOLO_BASE_URL}/yolo26n.pt",
        "description": "YOLO26n Person Detector (Ultra-fast / Embedded)",
        "size_mb": 5.5,
        "default": True,
    },
    "yolo26s-pose.pt": {
        "url": f"{YOLO_BASE_URL}/yolo26s-pose.pt",
        "description": "YOLO26s Pose Model (with 17-Keypoint Envelope)",
        "size_mb": 20.4,
        "default": False,
    },
    "yolo26n-pose.pt": {
        "url": f"{YOLO_BASE_URL}/yolo26n-pose.pt",
        "description": "YOLO26n Pose Model (Fast Pose Envelope)",
        "size_mb": 6.8,
        "default": False,
    },
    # Mobile NPU / CPU ONNX Detectors
    "yolo26n.onnx": {
        "url": f"{YOLO_BASE_URL}/yolo26n.onnx",
        "description": "YOLO26n Detector (Mobile NPU/CPU ONNX)",
        "size_mb": 9.5,
        "default": False,
    },
    "yolo26s.onnx": {
        "url": f"{YOLO_BASE_URL}/yolo26s.onnx",
        "description": "YOLO26s Detector (Mobile NPU/CPU ONNX)",
        "size_mb": 36.5,
        "default": False,
    },
    "yolo26n-pose.onnx": {
        "url": f"{YOLO_BASE_URL}/yolo26n-pose.onnx",
        "description": "YOLO26n Pose Envelope (Mobile NPU/CPU ONNX)",
        "size_mb": 11.6,
        "default": False,
    },
    "yolo26s-pose.onnx": {
        "url": f"{YOLO_BASE_URL}/yolo26s-pose.onnx",
        "description": "YOLO26s Pose Envelope (Mobile NPU/CPU ONNX)",
        "size_mb": 39.9,
        "default": False,
    },
    "yolov8n.onnx": {
        "url": f"{YOLO_BASE_URL}/yolov8n.onnx",
        "description": "YOLOv8n Detector (Mobile NPU/CPU ONNX)",
        "size_mb": 12.3,
        "default": False,
    },
    "rfdetr_nano.onnx": {
        "url": "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_quantized.onnx",
        "description": "RF-DETR Nano Detector (Mobile NPU/CPU ONNX)",
        "size_mb": 27.5,
        "default": False,
    },
}

ALIAS_MAP = {
    "body": "mhr-body.onnx",
    "mhr": "mhr-body.onnx",
    "medium": "instanthmr-medium.onnx",
    "nano": "instanthmr-nano.onnx",
    "small": "instanthmr-small.onnx",
    "tiny": "instanthmr-tiny.onnx",
    "yolo": "yolo26s.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s-pose": "yolo26s-pose.pt",
    "yolo26n-pose": "yolo26n-pose.pt",
    "yolo26n-onnx": "yolo26n.onnx",
    "yolo26s-onnx": "yolo26s.onnx",
    "rfdetr-onnx": "rfdetr_nano.onnx",
}


def download_file_with_progress(url: str, dest_path: Path, expected_mb: float) -> None:
    """Download a file with an interactive CLI progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".download")

    headers = {
        "User-Agent": "InstantHMR-Downloader/2.0"
    }
    req = urllib.request.Request(url, headers=headers)

    print(f"Downloading {dest_path.name} (~{expected_mb:.1f} MB)...")
    start_time = time.time()

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = response.info().get("Content-Length")
            total_bytes = int(total_size) if total_size else int(expected_mb * 1024 * 1024)
            downloaded = 0
            block_size = 1024 * 64  # 64 KB

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

                elapsed = max(time.time() - start_time, 0.001)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                percent = min(100.0, (downloaded / max(total_bytes, 1)) * 100)
                bar_len = 30
                filled = int(round(bar_len * (percent / 100.0)))
                bar = "█" * filled + "░" * (bar_len - filled)

                sys.stdout.write(
                    f"\r  [{bar}] {percent:5.1f}% | {downloaded / (1024*1024):5.1f}/{total_bytes / (1024*1024):5.1f} MB | {speed_mb:5.1f} MB/s"
                )
                sys.stdout.flush()

        sys.stdout.write("\n")
        temp_path.replace(dest_path)
        print(f"  ✓ Successfully saved: {dest_path.name}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\n  ✗ Error downloading {dest_path.name}: {e}")
        raise


def check_and_download(filename: str, force: bool = False) -> bool:
    """Check if model exists, download if missing or forced."""
    if filename not in MODELS_REGISTRY:
        print(f"Unknown model: {filename}")
        return False

    info = MODELS_REGISTRY[filename]
    dest = MODELS_DIR / filename

    if dest.exists() and not force:
        # Check size > 100KB to ensure not empty stub
        sz_mb = dest.stat().st_size / (1024 * 1024)
        if sz_mb > 0.1:
            print(f"✓ {filename:<25} already exists ({sz_mb:5.1f} MB)")
            return True

    download_file_with_progress(info["url"], dest, info["size_mb"])
    return True


def list_models() -> None:
    """Print table of all available models and their download status."""
    print("=" * 80)
    print(f"{'Filename':<26} {'Size':<10} {'Status':<12} {'Description'}")
    print("-" * 80)
    for filename, info in MODELS_REGISTRY.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            sz = f"{dest.stat().st_size / (1024*1024):.1f} MB"
            status = "Installed"
        else:
            sz = f"~{info['size_mb']:.1f} MB"
            status = "Missing"
        print(f"{filename:<26} {sz:<10} {status:<12} {info['description']}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_models",
        description="InstantHMR v2: Model Checkpoint & Weights Downloader",
    )
    parser.add_argument("--all", action="store_true",
                        help="Download all models including optional small/tiny and pose variants")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Download a specific model by name or alias (e.g. 'nano', 'medium', 'yolo26s')")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-download even if files already exist")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List all models and their local install status")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INSTANTHMR v2 - MODEL WEIGHTS DOWNLOADER")
    print(f"Target Directory: {MODELS_DIR}")
    print("=" * 70)

    if args.model:
        m_name = ALIAS_MAP.get(args.model.lower(), args.model)
        if m_name not in MODELS_REGISTRY:
            print(f"Error: model '{args.model}' not found in registry.")
            print(f"Available options: {', '.join(sorted(set(list(MODELS_REGISTRY.keys()) + list(ALIAS_MAP.keys()))))}")
            sys.exit(1)
        targets = [m_name]
    elif args.all:
        targets = list(MODELS_REGISTRY.keys())
    else:
        # Default core set
        targets = [k for k, v in MODELS_REGISTRY.items() if v["default"]]

    print(f"Checking {len(targets)} models...\n")
    success_count = 0
    for target in targets:
        try:
            if check_and_download(target, force=args.force):
                success_count += 1
        except Exception as e:
            print(f"Failed to fetch {target}: {e}")

    print("\n" + "=" * 70)
    print(f"Done! {success_count}/{len(targets)} models ready in: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
