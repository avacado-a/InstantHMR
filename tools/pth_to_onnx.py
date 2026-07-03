#!/usr/bin/env python3
"""
Convert a trained InstantHMR student checkpoint (.pth) into the ONNX file the
demo expects at ``models/instanthmr.onnx``.

It reuses the exact model architecture and deploy wrapper from
``instanthmr_distill_train/train_distill.py`` and the FP32 export recipe from
notebook Cell 15, so the graph I/O matches ``instanthmr/inference.py``:

    inputs : image (N, 3, 224, 224), cliff_cond (N, 3)
    outputs: mhr_params, shape_params, cam_trans, joints_2d, joints_3d   (in this order)

Usage
-----
    # default: convert the EMA checkpoint -> models/instanthmr.onnx
    python tools/pth_to_onnx.py

    # or point at a specific checkpoint / output
    python tools/pth_to_onnx.py \
        --ckpt   instanthmr_distill_train/runs/distill_repvit_cliff_v2/best_student_model_ema.pth \
        --output models/instanthmr.onnx
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO / "instanthmr_distill_train"

# The model class + deploy wrapper live in the standalone training script.
sys.path.insert(0, str(TRAIN_DIR))
import train_distill as T  # noqa: E402  (InstantHMRStudent, DistillConfig, HMRDeployWrapper)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path,
                    default=TRAIN_DIR / "runs/distill_repvit_cliff_v2/best_student_model_ema.pth",
                    help="Path to the .pth checkpoint (default: the EMA checkpoint).")
    ap.add_argument("--output", type=Path, default=REPO / "models" / "instanthmr.onnx",
                    help="Where to write the ONNX (default: models/instanthmr.onnx).")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"[error] checkpoint not found: {args.ckpt}")

    device = torch.device("cpu")   # export on CPU — portable, no CUDA needed
    cfg = T.DistillConfig()        # default architecture (matches training)

    # --- Build the model and load weights ---
    model = T.InstantHMRStudent(cfg, pretrained=False).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}  # strip torch.compile prefix
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.ckpt.name} | epoch {ckpt.get('epoch', '?')} "
          f"| PA-MPJPE {ckpt.get('val_pa_mpjpe', '?')} | source {ckpt.get('source', '?')}")
    if missing:
        print(f"  missing keys   : {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")

    deploy = T.HMRDeployWrapper(model).eval()

    # --- Export FP32 ONNX (Cell 15 recipe) ---
    output_keys = ["mhr_params", "shape_params", "cam_trans", "joints_2d", "joints_3d"]
    dummy_img = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    dummy_cliff = torch.randn(1, 3, device=device)
    dynamic_axes = {"image": {0: "batch"}, "cliff_cond": {0: "batch"}}
    for k in output_keys:
        dynamic_axes[k] = {0: "batch"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deploy, (dummy_img, dummy_cliff), str(args.output),
        input_names=["image", "cliff_cond"], output_names=output_keys,
        dynamic_axes=dynamic_axes, opset_version=args.opset,
    )
    print(f"\n✅ Wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")

    # --- Sanity check: run once through onnxruntime and verify I/O ---
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        in_names = [i.name for i in sess.get_inputs()]
        outs = sess.run(None, {in_names[0]: dummy_img.numpy(), in_names[1]: dummy_cliff.numpy()})
        print(f"  onnxruntime OK | inputs {in_names}")
        for o, v in zip(sess.get_outputs(), outs):
            print(f"    {o.name:14s} {list(np.asarray(v).shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  (onnxruntime sanity check skipped: {e})")


if __name__ == "__main__":
    main()
