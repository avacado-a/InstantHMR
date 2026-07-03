# InstantHMR Distillation — Standalone Cloud Training

Standalone version of `notebooks/distill_transformer_decoder.ipynb` for running
on the `pfcalcul` SLURM cluster. It bundles notebook cells **0, 1, 3, 7, 8, 10,
11 and 15** into a single script: setup, config, dataset, student architecture,
distillation loss, HMR metrics, the EMA + early-stopping training loop, and ONNX
export/quantization.

## Folder layout (on the cluster)

Project root: `/pfcalcul/work/kchalabi/envs/lstm/instanthmr_distill_train/`

```
instanthmr_distill_train/
├── train_distill.py          # the standalone training script
├── submit_train_distill.sh   # SLURM launcher (sbatch)
├── requirements.txt          # pip deps (auto-installed by the platform)
├── checkpoints/
│   └── mhr_model.pt          # <-- YOU must place this here (≈700 MB, not in git)
└── runs/                     # created at runtime: checkpoints, logs, exports
```

## Dataset layout

The script takes a **single entry-point folder** (`--data_root`) that contains
one or more sub-folders, each with its own `annotations/` and `images/`:

```
instanthmr_data/
├── sam3d_distill_coco/
│   ├── annotations/   *.npz
│   └── images/        *.jpg | *.png
├── sam3d_gt_coco/
│   ├── annotations/
│   └── images/
└── ...
```

All sub-folders are loaded and concatenated. File names may collide across
sub-folders — that is fine, each `(image, npz)` pair is stored as a full path,
so identically-named files in different sub-folders are kept as distinct
samples.

## Before launching

1. Copy `mhr_model.pt` into `checkpoints/` (or pass `--mhr_model_path`).
2. Make sure the dataset is registered as `/datasets/instanthmr_data` so that
   `datasynch_perso` can sync it onto the node.

## Launch

From `/pfcalcul/work/kchalabi/envs/lstm/`:

```sh
sh instanthmr_distill_train/submit_train_distill.sh
```

The launcher syncs `/datasets/instanthmr_data`, `cd`s into the project folder,
and runs `python3 train_distill.py --data_root ../instanthmr_data`. If
`datasynch_perso` places the data somewhere else, edit the `--data_root` value
(and the `datasynch_perso` line) in `submit_train_distill.sh`.

## Useful flags

```
python3 train_distill.py \
    --data_root ../instanthmr_data \
    --output_dir runs/distill_repvit_cliff_v2 \
    --epochs 400 --batch_size 64 --lr 3e-4 --num_workers 8 \
    --self-test     # run arch + perfect-student loss sanity checks first
    --no-resume     # ignore any existing checkpoint, train from scratch
    --no-export     # skip ONNX export/quantization after training
```

Training auto-resumes from `runs/<name>/best_student_model_v3.pth` if present.
Outputs (best checkpoints + `export/*.onnx`) land under `--output_dir`.
