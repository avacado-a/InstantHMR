#!/usr/bin/env python3
"""Prepare & Export ONNX Models for InstantHMR Mobile (Android & iOS).

Downloads or verifies the InstantHMR ONNX models (`mhr-body.onnx`, `instanthmr-nano.onnx`,
`instanthmr-tiny.onnx`, `instanthmr-small.onnx`, `instanthmr-medium.onnx`, `instanthmr-medium-commercial.onnx`)
and exports YOLO26 / YOLOv8 / Pose detector checkpoints (`.pt` -> `.onnx`) with static shapes
optimized for mobile NPUs (Apple CoreML Neural Engine and Android NNAPI / Qualcomm QNN).

Usage:
    python3 mobile/scripts/prepare_mobile_models.py               # Prepare default mobile models (nano, mhr-body, yolo26n)
    python3 mobile/scripts/prepare_mobile_models.py --all         # Prepare all HMR sizes and detectors
    python3 mobile/scripts/prepare_mobile_models.py --bundle      # Copy prepared ONNX models into mobile/assets/models/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOBILE_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOBILE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
MOBILE_ASSETS_MODELS_DIR = MOBILE_DIR / "assets" / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download_models import MODELS_REGISTRY, check_and_download  # noqa: E402


YOLO_EXPORT_SPECS = {
    "yolo26n.pt": ("yolo26n.onnx", 384),
    "yolo26s.pt": ("yolo26s.onnx", 480),
    "yolo26n-pose.pt": ("yolo26n-pose.onnx", 384),
    "yolo26s-pose.pt": ("yolo26s-pose.onnx", 480),
}


def export_yolo_to_mobile_onnx(pt_filename: str, onnx_filename: str, imgsz: int) -> Path | None:
    """Export a PyTorch YOLO `.pt` checkpoint to an NPU-friendly static-shape `.onnx` graph."""
    pt_path = MODELS_DIR / pt_filename
    out_onnx = MODELS_DIR / onnx_filename
    if out_onnx.exists() and out_onnx.stat().st_size > 100_000:
        print(f"  ✓ Mobile ONNX detector already exists: {out_onnx.name} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
        return out_onnx

    if not pt_path.exists():
        check_and_download(pt_filename)

    try:
        from ultralytics import YOLO

        print(f"  → Exporting {pt_filename} to mobile NPU ONNX ({imgsz}x{imgsz}, opset=17)...")
        model = YOLO(str(pt_path))
        exported_path = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
        if exported_path and Path(exported_path).exists():
            shutil.move(str(exported_path), str(out_onnx))
            print(f"  ✓ Exported {out_onnx.name} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
            return out_onnx
    except Exception as exc:
        print(f"  ! Live export unavailable ({exc}); downloading pre-exported {onnx_filename} directly...")
        if check_and_download(onnx_filename):
            return out_onnx
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Download and prepare all model variants")
    parser.add_argument("--bundle", action="store_true", help="Copy ONNX files into mobile/assets/models/ for offline bundling")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("INSTANTHMR MOBILE STUDIO - MODEL & NPU ASSET PREPARATION")
    print("=" * 72)

    hmr_targets = (
        ["mhr-body.onnx", "instanthmr-nano.onnx", "instanthmr-tiny.onnx", "instanthmr-small.onnx", "instanthmr-medium.onnx"]
        if args.all
        else ["mhr-body.onnx", "instanthmr-nano.onnx"]
    )

    for target in hmr_targets:
        if target in MODELS_REGISTRY:
            check_and_download(target)

    yolo_targets = list(YOLO_EXPORT_SPECS.keys()) if args.all else ["yolo26n.pt"]
    for pt_name in yolo_targets:
        onnx_name, imgsz = YOLO_EXPORT_SPECS[pt_name]
        export_yolo_to_mobile_onnx(pt_name, onnx_name, imgsz)

    if args.bundle:
        MOBILE_ASSETS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for onnx_file in MODELS_DIR.glob("*.onnx"):
            dest = MOBILE_ASSETS_MODELS_DIR / onnx_file.name
            shutil.copy2(onnx_file, dest)
            copied += 1
            print(f"  ✓ Bundled into mobile assets: {dest.relative_to(MOBILE_DIR)}")
        print(f"\nBundled {copied} ONNX model(s) into {MOBILE_ASSETS_MODELS_DIR}")

    print("\nDone! Mobile models are ready.")


if __name__ == "__main__":
    main()
