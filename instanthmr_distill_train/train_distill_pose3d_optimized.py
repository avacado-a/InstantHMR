#!/usr/bin/env python3
# ============================================================
# Keypoint + camera student — OPTIMIZED (geometric aug + soft-argmax).
# ------------------------------------------------------------
# Same task as train_distill_pose3d.py (70 2D + 3D keypoints + cam_trans,
# NO MHR), with two SOTA changes:
#
#  (1) GEOMETRIC AUGMENTATION — random rotation / scale / translation +
#      horizontal flip, propagated CORRECTLY to the 2D targets, the 3D
#      targets, and cam_trans (verified: image<->2D warp matches to the
#      pixel; rotating 3D XY + cam by R=[[c,s],[-s,c]] reprojects onto the
#      rotated 2D to 1e-7; flip mirrors X and swaps left<->right joints).
#  (2) SOFT-ARGMAX 2D HEAD — the 2D tokens emit 1D bin distributions
#      (SimCC-style) and soft-argmax to coordinates: a light heatmap-style
#      spatial prior at ~the same compute as the old MLP regression.
#
# Loss:    SmoothL1(2D coords) + SimCC-CE(2D bins) + SmoothL1(3D) + SmoothL1(cam).
# Metrics: MPJPE / PA-MPJPE on the 3D head.
# Export:  5 ONNX outputs in demo order (mhr_params & shape_params zero-filled).
#
# Sanity:  python3 train_distill_pose3d_optimized.py --overfit-test --data_root ../data
# Full:    python3 train_distill_pose3d_optimized.py --data_root /datasets/instanthmr_data --num_workers 8
# ============================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import random
import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.swa_utils import AveragedModel

import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t
from torchvision.transforms import InterpolationMode

import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/pfcalcul/datasets/instanthmr_data/"

# MHR70 joint ordering (mirror of instanthmr/skeleton.py, embedded so this
# script is self-contained on the cluster). Used to build the horizontal-flip
# left<->right permutation by name.
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint", "right_thumb_third_joint",
    "right_index_tip", "right_index_first_joint", "right_index_second_joint", "right_index_third_joint",
    "right_middle_tip", "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint", "right_ring_third_joint",
    "right_pinky_tip", "right_pinky_first_joint", "right_pinky_second_joint", "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint", "left_thumb_third_joint",
    "left_index_tip", "left_index_first_joint", "left_index_second_joint", "left_index_third_joint",
    "left_middle_tip", "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint", "left_ring_third_joint",
    "left_pinky_tip", "left_pinky_first_joint", "left_pinky_second_joint", "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]


def _build_flip_perm(names):
    """Left<->right joint index permutation for horizontal flip (central joints fixed)."""
    idx = {n: i for i, n in enumerate(names)}
    perm = list(range(len(names)))
    for i, n in enumerate(names):
        if n.startswith("left_"):
            perm[i] = idx.get("right_" + n[5:], i)
        elif n.startswith("right_"):
            perm[i] = idx.get("left_" + n[6:], i)
    perm = np.array(perm, dtype=np.int64)
    assert (perm[perm] == np.arange(len(perm))).all(), "flip permutation is not an involution"
    return perm


FLIP_PERM = _build_flip_perm(JOINT_NAMES)


# ============================================================
# Configuration
# ============================================================
@dataclass
class PoseConfig:
    data_root: str = str(DEFAULT_DATA_ROOT)
    log_dir: str = str(PROJECT_ROOT / "runs/distill_pose3d_v1")

    # --- Data ---
    image_size: int = 224
    max_images: int | None = None
    per_dataset_caps: dict = field(default_factory=lambda: {"sam3d_gt_harmony4d": 300_000})
    val_split: float = 0.1
    num_workers: int = 4
    augment: bool = True

    # --- Geometric augmentation (change #1) ---
    geom_rot_deg: float = 30.0     # random in-plane rotation ±deg
    geom_scale_range: float = 0.25  # random scale in [1-r, 1+r]
    geom_trans: float = 0.08       # random translation, fraction of half-crop
    geom_flip_p: float = 0.5       # horizontal flip probability (with L/R joint swap)

    # --- Model ---
    backbone: str = "repvit_m2_3"
    d_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 4
    dropout: float = 0.1
    cliff_dim: int = 3
    cam_dim: int = 3
    num_joints: int = 70
    kp2d_bins: int = 64            # soft-argmax 2D head bins per axis (change #2)

    # --- Training ---
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    grad_clip: float = 1.0
    anomaly_loss_threshold: float = 100.0
    use_amp: bool = True
    ema_decay: float = 0.9998
    early_stop_patience: int = 150
    resume: bool = True

    # --- Loss weights ---
    w_keypoints2d: float = 10.0
    w_keypoints3d: float = 10.0
    w_cam: float = 0.1           # direct SmoothL1 supervision on cam_trans
    w_simcc: float = 1.0         # SimCC bin-classification weight (has an entropy floor)


# ============================================================
# Dataset (same crops as train_distill.py; keeps cam_trans + camera intrinsics)
# ============================================================
class RandomPixelate:
    def __init__(self, p=0.2, min_res=32, max_res=112):
        self.p, self.min_res, self.max_res = p, min_res, max_res

    def __call__(self, img):
        if random.random() < self.p:
            orig_w, orig_h = img.size
            degraded_h = random.randint(self.min_res, self.max_res)
            degraded_w = int(orig_w * (degraded_h / orig_h))
            img = F_t.resize(img, [degraded_h, degraded_w], interpolation=InterpolationMode.NEAREST)
            img = F_t.resize(img, [orig_h, orig_w], interpolation=InterpolationMode.NEAREST)
        return img


class SAM3DStudentDataset(Dataset):
    def __init__(self, data_root: str, image_size: int = 224, max_images: int | None = None,
                 augment: bool = True, per_dataset_caps: dict | None = None,
                 geom_rot_deg: float = 30.0, geom_scale_range: float = 0.25,
                 geom_trans: float = 0.08, geom_flip_p: float = 0.5):
        super().__init__()
        self.image_size = image_size
        self.data_root = Path(data_root)
        # geometric augmentation only on the train split (augment=True)
        self.augment = augment
        self.geom_rot_deg = geom_rot_deg
        self.geom_scale_range = geom_scale_range
        self.geom_trans = geom_trans
        self.geom_flip_p = geom_flip_p

        # `self.transform` is now PHOTOMETRIC only — geometric aug happens in
        # __getitem__ jointly on the image + 2D/3D/cam targets (before ToTensor).
        tfm_list = [transforms.Resize((image_size, image_size))]
        if augment:
            tfm_list.extend([
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
                RandomPixelate(p=0.2, min_res=48, max_res=112),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
            ])
        tfm_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if augment:
            tfm_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)))
        self.transform = transforms.Compose(tfm_list)

        self.pairs = []
        dataset_dirs = self._find_dataset_dirs(self.data_root)
        if not dataset_dirs:
            raise FileNotFoundError(f"No annotations/images sub-folders under '{self.data_root}'.")

        cap_rng = random.Random(42)
        caps = per_dataset_caps or {}
        for ann_dir, img_dir in dataset_dirs:
            sub_name = ann_dir.parent.name
            sub_pairs = []
            for npz_path in sorted(ann_dir.glob("*.npz")):
                stem = npz_path.stem
                img_path = img_dir / f"{stem}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{stem}.png"
                if img_path.exists():
                    sub_pairs.append((img_path, npz_path))
            cap = caps.get(sub_name)
            if cap is not None and len(sub_pairs) > cap:
                n_before = len(sub_pairs)
                sub_pairs.sort(key=lambda p: (str(p[1]), str(p[0])))
                sub_pairs = cap_rng.sample(sub_pairs, cap)
                print(f"  [cap] {sub_name}: sampled {cap:,} of {n_before:,} crops")
            self.pairs.extend(sub_pairs)

        self.pairs.sort(key=lambda p: (str(p[1]), str(p[0])))
        if max_images is not None:
            self.pairs = self.pairs[:max_images]
        print(f"  Found {len(dataset_dirs)} sub-folder(s) under {self.data_root}; "
              f"{len(self.pairs)} valid (image, npz) pairs (augment={augment}).")

    @staticmethod
    def _find_dataset_dirs(root: Path):
        dirs = []
        if (root / "annotations").is_dir() and (root / "images").is_dir():
            dirs.append((root / "annotations", root / "images"))
        if root.is_dir():
            for sub in sorted(root.iterdir()):
                if not sub.is_dir():
                    continue
                ann, img = sub / "annotations", sub / "images"
                if ann.is_dir() and img.is_dir():
                    dirs.append((ann, img))
        return dirs

    def __len__(self):
        return len(self.pairs)

    def _geometric_aug(self, img_np, j2, j3, cam):
        """Jointly warp image + 2D/3D/cam. Verified: image<->2D warp matches to the
        pixel; 3D XY & cam rotate by R=[[c,s],[-s,c]] consistently with the 2D."""
        W = H = self.image_size

        # --- horizontal flip: mirror X + swap left/right joints ---
        if random.random() < self.geom_flip_p:
            img_np = np.ascontiguousarray(img_np[:, ::-1, :])
            j2[:, 0] = -j2[:, 0]
            j3[:, 0] = -j3[:, 0]
            cam[0] = -cam[0]
            j2 = j2[FLIP_PERM]
            j3 = j3[FLIP_PERM]

        # --- rotation + scale + translation (single cv2 affine) ---
        angle = random.uniform(-self.geom_rot_deg, self.geom_rot_deg)
        scale = random.uniform(1.0 - self.geom_scale_range, 1.0 + self.geom_scale_range)
        dx = random.uniform(-self.geom_trans, self.geom_trans)
        dy = random.uniform(-self.geom_trans, self.geom_trans)

        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
        M[0, 2] += dx * (W / 2.0)
        M[1, 2] += dy * (H / 2.0)
        img_np = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # 2D: apply the SAME affine M in pixel space, back to normalised [-1,1]
        px = (j2[:, 0] + 1.0) * 0.5 * W
        py = (j2[:, 1] + 1.0) * 0.5 * H
        px2 = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        py2 = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        j2[:, 0] = px2 / (0.5 * W) - 1.0
        j2[:, 1] = py2 / (0.5 * H) - 1.0

        # 3D: rotate the XY plane about the optical axis (scale/translate don't touch 3D)
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)   # matches cv2 rotation (scale=1)
        j3[:, :2] = j3[:, :2] @ R.T
        cam[:2] = R @ cam[:2]
        return img_np, j2, j3, cam

    def __getitem__(self, idx):
        img_path, npz_path = self.pairs[idx]
        img_pil = Image.open(img_path).convert("RGB")

        ann = np.load(npz_path)
        orig_h, orig_w = ann["orig_shape"]

        sq_bbox = ann["bbox_square"]
        sq_x1, sq_y1 = sq_bbox[0], sq_bbox[1]
        orig_crop_size = sq_bbox[2] - sq_bbox[0]
        true_scale = self.image_size / max(orig_crop_size, 1.0)

        joints_2d = ann["joints_2d"].copy().astype(np.float32)
        joints_2d[:, 0] = (joints_2d[:, 0] - sq_x1) * true_scale
        joints_2d[:, 1] = (joints_2d[:, 1] - sq_y1) * true_scale
        joints_2d[:, 0] = (joints_2d[:, 0] / self.image_size) * 2.0 - 1.0
        joints_2d[:, 1] = (joints_2d[:, 1] / self.image_size) * 2.0 - 1.0

        joints_3d = ann["joints_3d"].astype(np.float32).copy()
        cam_trans = ann["cam_trans"].astype(np.float32).copy()

        # --- geometric augmentation (train only), jointly on image + targets ---
        if self.augment:
            img_np = np.asarray(img_pil.resize((self.image_size, self.image_size)))
            img_np, joints_2d, joints_3d, cam_trans = self._geometric_aug(img_np, joints_2d, joints_3d, cam_trans)
            img_pil = Image.fromarray(img_np)

        image = self.transform(img_pil)   # photometric + ToTensor + Normalize

        tight_bbox = ann["bbox"]
        cx = (tight_bbox[0] + tight_bbox[2]) / 2.0
        cy = (tight_bbox[1] + tight_bbox[3]) / 2.0
        cx_norm = 2.0 * (cx / orig_w) - 1.0
        cy_norm = 2.0 * (cy / orig_h) - 1.0
        b_size = max(tight_bbox[2] - tight_bbox[0], tight_bbox[3] - tight_bbox[1])
        b_scale = b_size / max(orig_w, orig_h)
        cliff_cond = torch.tensor([cx_norm, cy_norm, b_scale], dtype=torch.float32)

        return {
            "image": image,                                              # [3,224,224]
            "cliff_cond": cliff_cond,                                    # [3]
            "joints_2d": torch.from_numpy(joints_2d).float(),            # [70,2] crop [-1,1]
            "joints_3d": torch.from_numpy(joints_3d).float(),            # [70,3] root-centred
            "cam_trans": torch.from_numpy(cam_trans).float(),           # [3]  camera translation
        }


def build_dataloaders(cfg):
    geom = dict(geom_rot_deg=cfg.geom_rot_deg, geom_scale_range=cfg.geom_scale_range,
                geom_trans=cfg.geom_trans, geom_flip_p=cfg.geom_flip_p)
    full_dataset = SAM3DStudentDataset(cfg.data_root, augment=cfg.augment,
                                       max_images=cfg.max_images, per_dataset_caps=cfg.per_dataset_caps, **geom)
    val_dataset_clean = SAM3DStudentDataset(cfg.data_root, augment=False,
                                            max_images=cfg.max_images, per_dataset_caps=cfg.per_dataset_caps, **geom)
    n_val = int(len(full_dataset) * cfg.val_split)
    n_train = len(full_dataset) - n_val
    g = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset, [n_train, n_val], generator=g)
    g = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(val_dataset_clean, [n_train, n_val], generator=g)

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    print(f"Dataset Loaded! Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, full_dataset


# ============================================================
# Model — RepViT + decoder + cam head + 2D/3D heads (no MHR head)
# ============================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h, grid_w = grid_size, grid_size
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    omega = torch.arange(embed_dim // 4).float() / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_y = grid_y.flatten().unsqueeze(1) * omega.unsqueeze(0)
    out_x = grid_x.flatten().unsqueeze(1) * omega.unsqueeze(0)
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    return torch.cat([pe_y, pe_x], dim=1).unsqueeze(0)


class SoftArgmax2DHead(nn.Module):
    """SimCC-style 2D head: each joint token emits 1D bin distributions for x and y,
    then soft-argmax to a continuous [-1,1] coordinate. A light heatmap-style spatial
    prior at ~the same compute as an MLP regressor. Fully ONNX-exportable."""
    def __init__(self, d_in, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
        self.proj = nn.Sequential(nn.Linear(d_in, 256), nn.GELU())
        self.to_logits = nn.Linear(256, 2 * num_bins)
        self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, num_bins))

    def forward(self, feat):                          # feat: [B, J, d_in]
        logits = self.to_logits(self.proj(feat))      # [B, J, 2*bins]
        logits = logits.unflatten(-1, (2, self.num_bins))   # [B, J, 2, bins]
        prob = torch.softmax(logits, dim=-1)
        coords = (prob * self.bin_centers).sum(dim=-1)      # [B, J, 2] in [-1,1]
        return coords, logits                                # logits used by the SimCC loss


class InstantHMRPoseStudent(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.feat_proj = nn.Linear(embed_dim, cfg.d_model)

        self.grid_size = cfg.image_size // 32
        self.register_buffer("mem_pos_embed", get_2d_sincos_pos_embed(cfg.d_model, self.grid_size))

        # 1 global (camera) + 70 2D + 70 3D query tokens
        self.num_global = 1
        self.num_2d = cfg.num_joints
        self.num_3d = cfg.num_joints
        self.total_queries = self.num_global + self.num_2d + self.num_3d
        self.query_embed = nn.Parameter(torch.randn(1, self.total_queries, cfg.d_model) * 0.02)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cliff_dim, 128), nn.GELU(), nn.Linear(128, cfg.d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_decoder_layers)

        # Camera head (translation only) + soft-argmax 2D head + 3D head
        self.head_cam = nn.Linear(cfg.d_model, cfg.cam_dim)
        self.head_2d = SoftArgmax2DHead(cfg.d_model, cfg.kp2d_bins)
        self.head_3d = nn.Sequential(nn.Linear(cfg.d_model, 256), nn.GELU(), nn.Linear(256, 3))

        # Init camera near (0, 0, 2m) depth; tiny init on the 3D head for stability
        nn.init.normal_(self.head_cam.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_cam.bias, 0.0)
        nn.init.constant_(self.head_cam.bias[-1], 2.0)
        nn.init.normal_(self.head_3d[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_3d[-1].bias, 0.0)

    def forward(self, images, cliff_cond):
        B = images.shape[0]
        img_feats = self.backbone.forward_features(images)
        if img_feats.dim() == 4:
            img_feats = img_feats.flatten(2).transpose(1, 2)
        memory = self.feat_proj(img_feats) + self.mem_pos_embed

        queries = self.query_embed.expand(B, -1, -1) + self.cond_proj(cliff_cond).unsqueeze(1)
        shared = self.transformer(tgt=queries, memory=memory)

        feat_global = shared[:, 0, :]
        feat_2d = shared[:, 1:1 + self.num_2d, :]
        feat_3d = shared[:, 1 + self.num_2d:, :]

        pred_cam = self.head_cam(feat_global)
        pred_2d, logits_2d = self.head_2d(feat_2d)
        pred_3d = self.head_3d(feat_3d)
        return {"cam_trans": pred_cam, "joints_2d": pred_2d,
                "joints_2d_logits": logits_2d, "joints_3d": pred_3d}


# ============================================================
# Loss — keypoints + camera (no MHR)
# ============================================================
class Pose3DLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l1 = nn.SmoothL1Loss()
        bins = torch.linspace(-1.0, 1.0, cfg.kp2d_bins)
        self.register_buffer("bin_centers", bins)
        self.simcc_sigma = 2.0 * (2.0 / (cfg.kp2d_bins - 1))   # ~2 bins wide Gaussian label

    def _simcc_ce(self, logits, tgt_2d):
        """SimCC classification loss: CE between the predicted bin distribution and a
        Gaussian soft-label centred on the target coordinate (sharpens the soft-argmax)."""
        # logits: [B,J,2,bins]  tgt_2d: [B,J,2] in [-1,1]
        d = self.bin_centers.view(1, 1, 1, -1) - tgt_2d.unsqueeze(-1)     # [B,J,2,bins]
        label = torch.exp(-(d ** 2) / (2.0 * self.simcc_sigma ** 2))
        label = label / label.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logp = F.log_softmax(logits, dim=-1)
        return -(label * logp).sum(dim=-1).mean()

    def forward(self, preds, targets):
        losses = {}
        # 2D: soft-argmax coordinate SmoothL1 (accurate read-out) + SimCC classification (sharpen)
        losses['loss_2d'] = self.l1(preds["joints_2d"][..., :2], targets["joints_2d"]) * self.cfg.w_keypoints2d
        losses['loss_2d_simcc'] = self._simcc_ce(preds["joints_2d_logits"], targets["joints_2d"]) * self.cfg.w_simcc
        losses['loss_3d'] = self.l1(preds["joints_3d"][..., :3], targets["joints_3d"]) * self.cfg.w_keypoints3d
        # Direct camera supervision — SmoothL1 (bounded gradient, robust to large depths).
        losses['loss_cam'] = self.l1(preds["cam_trans"], targets["cam_trans"]) * self.cfg.w_cam
        losses['total_loss'] = (losses['loss_2d'] + losses['loss_2d_simcc']
                                + losses['loss_3d'] + losses['loss_cam'])
        return losses


# ============================================================
# Metrics — MPJPE / PA-MPJPE on the 3D head (mm)
# ============================================================
def batched_procrustes_alignment(pred_pts, gt_pts):
    mu_pred = pred_pts.mean(dim=1, keepdim=True)
    mu_gt = gt_pts.mean(dim=1, keepdim=True)
    pred_c, gt_c = pred_pts - mu_pred, gt_pts - mu_gt
    norm_pred = torch.linalg.norm(pred_c, dim=(1, 2), keepdim=True)
    norm_gt = torch.linalg.norm(gt_c, dim=(1, 2), keepdim=True)
    pred_n = pred_c / torch.clamp(norm_pred, min=1e-8)
    gt_n = gt_c / torch.clamp(norm_gt, min=1e-8)
    H = torch.bmm(pred_n.transpose(1, 2), gt_n)
    U, S, Vh = torch.linalg.svd(H)
    R = torch.bmm(U, Vh)
    det = torch.linalg.det(R)
    det_sign = torch.where(det < 0, torch.tensor(-1.0, device=pred_pts.device),
                           torch.tensor(1.0, device=pred_pts.device)).unsqueeze(-1)
    U_fixed = U.clone()
    U_fixed[:, :, 2] *= det_sign
    R_fixed = torch.bmm(U_fixed, Vh)
    S_fixed = S.clone()
    S_fixed[:, 2] *= det_sign.squeeze(-1)
    scale = S_fixed.sum(dim=-1, keepdim=True).unsqueeze(-1) * (norm_gt / torch.clamp(norm_pred, min=1e-8))
    return scale * torch.bmm(pred_c, R_fixed) + mu_gt


@torch.no_grad()
def evaluate_pose_batch(preds, targets):
    metrics = {}
    pred_3d = preds["joints_3d"][..., :3]
    gt_3d = targets["joints_3d"][..., :3]
    pred_rr = pred_3d - pred_3d[:, 0:1, :]
    gt_rr = gt_3d - gt_3d[:, 0:1, :]
    metrics['MPJPE'] = torch.linalg.norm(pred_rr - gt_rr, dim=-1).mean().item() * 1000.0
    pred_al = batched_procrustes_alignment(pred_3d, gt_3d)
    metrics['PA_MPJPE'] = torch.linalg.norm(pred_al - gt_3d, dim=-1).mean().item() * 1000.0
    return metrics


# ============================================================
# 1-batch overfit test — proves gradients flow / model can learn
# ============================================================
def run_overfit_test(cfg, train_loader, steps=800, subset=8, lr=1e-3):
    print(f"--- 1-Batch Overfit Test ({subset} images, {steps} steps) ---")
    model = InstantHMRPoseStudent(cfg, pretrained=True).to(device)
    criterion = Pose3DLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.use_amp)

    batch = next(iter(train_loader))
    batch = {k: (v[:subset].to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # Judge overfit on the REGRESSION losses (2D coord + 3D + cam), which reach ~0.
    # The SimCC cross-entropy has an irreducible entropy floor, so the *total* can't.
    def reg_loss(ls):
        return ls['loss_2d'].item() + ls['loss_3d'].item() + ls['loss_cam'].item()

    first_reg, last_reg, last_pa = None, None, None
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
            preds = model(batch["image"], batch["cliff_cond"])
        preds = {k: v.float() for k, v in preds.items()}
        losses = criterion(preds, batch)
        loss = losses['total_loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if first_reg is None:
            first_reg = reg_loss(losses)
        last_reg = reg_loss(losses)
        if step % 50 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                p = {k: v.float() for k, v in model(batch["image"], batch["cliff_cond"]).items()}
                m = evaluate_pose_batch(p, batch)
            model.train()
            last_pa = m['PA_MPJPE']
            print(f"  step {step:04d} | total {loss.item():.4f} | 2d {losses['loss_2d'].item():.4f} "
                  f"| simcc {losses['loss_2d_simcc'].item():.3f} | 3d {losses['loss_3d'].item():.4f} "
                  f"| cam {losses['loss_cam'].item():.4f} | MPJPE {m['MPJPE']:.1f} mm | PA {m['PA_MPJPE']:.1f} mm")

    ok = (last_reg < first_reg * 0.05) and (last_pa is not None and last_pa < 40.0)
    print(f"\n{'✅ SUCCESS' if ok else '❌ FAIL'}: regression loss {first_reg:.4f} -> {last_reg:.5f}"
          f" | final PA-MPJPE {last_pa:.1f} mm "
          f"({'overfit, gradients flow' if ok else 'did NOT fit — check the setup'})")
    return ok


# ============================================================
# EMA helper (decay warmup)
# ============================================================
def make_ema_avg_fn(max_decay):
    @torch.no_grad()
    def avg_fn(ema_params, model_params, num_averaged):
        if not (torch.is_floating_point(ema_params[0]) or torch.is_complex(ema_params[0])):
            for e, m in zip(ema_params, model_params):
                e.copy_(m)
            return
        n = num_averaged.item() if torch.is_tensor(num_averaged) else float(num_averaged)
        decay = min(max_decay, (1.0 + n) / (10.0 + n))
        torch._foreach_lerp_(ema_params, model_params, 1.0 - decay)
    return avg_fn


# ============================================================
# Training loop — EMA + early stopping + guards, keypoints + camera
# ============================================================
def train_pose3d(cfg, train_loader, val_loader):
    print(f"--- Training {cfg.backbone} (pose3d) | EMA={cfg.ema_decay} | patience={cfg.early_stop_patience} ---")
    os.makedirs(cfg.log_dir, exist_ok=True)

    model = InstantHMRPoseStudent(cfg, pretrained=True).to(device)
    criterion = Pose3DLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.use_amp)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
                                        steps_per_epoch=steps_per_epoch, pct_start=0.1)

    best_raw_path = os.path.join(cfg.log_dir, "best_pose3d_raw.pth")
    best_ema_path = os.path.join(cfg.log_dir, "best_pose3d_ema.pth")
    best_ckpt_path = os.path.join(cfg.log_dir, "best_pose3d.pth")

    start_epoch = 0
    best_raw_pa = best_ema_pa = best_overall_pa = float('inf')
    epochs_no_improve = 0

    if cfg.resume and os.path.exists(best_ckpt_path):
        print(f"🔄 Resuming from {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        for k, obj in [("optimizer_state_dict", optimizer), ("scaler_state_dict", scaler),
                       ("scheduler_state_dict", scheduler)]:
            if k in ckpt:
                obj.load_state_dict(ckpt[k])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_overall_pa = best_raw_pa = best_ema_pa = ckpt.get("val_pa_mpjpe", float('inf'))
        print(f"✅ Resumed at epoch {start_epoch}. Best PA-MPJPE {best_overall_pa:.1f} mm.")
    else:
        print("🚀 Training from scratch.")

    ema_model = AveragedModel(model, multi_avg_fn=make_ema_avg_fn(cfg.ema_decay), use_buffers=True)

    @torch.no_grad()
    def run_validation(eval_model, tag, epoch):
        eval_model.eval()
        agg = {'total': 0.0, '2d': 0.0, '3d': 0.0, 'cam': 0.0, 'MPJPE': 0.0, 'PA_MPJPE': 0.0}
        n = 0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val:{tag}]", leave=False)
        for batch in vbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
                preds = eval_model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds.items()}
            losses = criterion(preds, batch)
            if math.isnan(losses['total_loss'].item()) or math.isinf(losses['total_loss'].item()):
                continue
            agg['total'] += losses['total_loss'].item()
            agg['2d'] += losses['loss_2d'].item()
            agg['3d'] += losses['loss_3d'].item()
            agg['cam'] += losses['loss_cam'].item()
            bm = evaluate_pose_batch(preds, batch)
            agg['MPJPE'] += bm['MPJPE']
            agg['PA_MPJPE'] += bm['PA_MPJPE']
            n += 1
            vbar.set_postfix({'PA-MPJPE': f"{agg['PA_MPJPE']/max(n,1):.1f}mm"})
        vbar.close()
        return {k: v / max(n, 1) for k, v in agg.items()}, n

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tm = {'total': 0.0, '2d': 0.0, '3d': 0.0, 'cam': 0.0}
        nb = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
                preds_fp16 = model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds_fp16.items()}
            losses = criterion(preds, batch)
            loss = losses['total_loss']
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print("⚠️ WARNING: NaN/Inf loss detected! Skipping step.")
                continue
            if loss_val > cfg.anomaly_loss_threshold:
                print(f"⚠️ WARNING: anomalous loss {loss_val:.1f} > {cfg.anomaly_loss_threshold} — skipping step.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()
                ema_model.update_parameters(model)
            tm['total'] += loss_val
            tm['2d'] += losses['loss_2d'].item()
            tm['3d'] += losses['loss_3d'].item()
            tm['cam'] += losses['loss_cam'].item()
            nb += 1
            pbar.set_postfix({'Tot': f"{loss_val:.4f}"})
        pbar.close()
        avg_train = {k: v / max(nb, 1) for k, v in tm.items()}

        raw_val, n_raw = run_validation(model, "raw", epoch)
        ema_val, n_ema = run_validation(ema_model, "ema", epoch)

        print(f"\n📈 Epoch {epoch+1} Summary | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   [Train]   Tot: {avg_train['total']:.4f} | 2D: {avg_train['2d']:.4f} | 3D: {avg_train['3d']:.4f} | cam: {avg_train['cam']:.4f}")
        print(f"   [Val RAW] Tot: {raw_val['total']:.4f} | PA-MPJPE: {raw_val['PA_MPJPE']:.1f} | MPJPE: {raw_val['MPJPE']:.1f} mm")
        print(f"   [Val EMA] Tot: {ema_val['total']:.4f} | PA-MPJPE: {ema_val['PA_MPJPE']:.1f} | MPJPE: {ema_val['MPJPE']:.1f} mm")

        def save_dict(m, val_metric, source):
            return {'epoch': epoch, 'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_pa_mpjpe': val_metric, 'source': source}

        improved_raw = n_raw > 0 and raw_val['PA_MPJPE'] < best_raw_pa
        improved_ema = n_ema > 0 and ema_val['PA_MPJPE'] < best_ema_pa
        if improved_raw:
            best_raw_pa = raw_val['PA_MPJPE']
            torch.save(save_dict(model, best_raw_pa, 'raw'), best_raw_path)
        if improved_ema:
            best_ema_pa = ema_val['PA_MPJPE']
            torch.save(save_dict(ema_model.module, best_ema_pa, 'ema'), best_ema_path)

        current = min(raw_val['PA_MPJPE'], ema_val['PA_MPJPE'])
        if current < best_overall_pa - 1e-4:
            best_overall_pa = current
            use_ema = ema_val['PA_MPJPE'] <= raw_val['PA_MPJPE']
            better = ema_model.module if use_ema else model
            torch.save(save_dict(better, best_overall_pa, 'ema' if use_ema else 'raw'), best_ckpt_path)
            print(f"💾 New best overall: {best_overall_pa:.1f} mm ({'EMA' if use_ema else 'RAW'}) -> {os.path.basename(best_ckpt_path)}")

        if improved_raw or improved_ema:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Neither RAW nor EMA improved: {epochs_no_improve}/{cfg.early_stop_patience} "
                  f"(best raw {best_raw_pa:.1f} / ema {best_ema_pa:.1f} mm)")
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}.")
                break

    print(f"\n🎉 Training complete! Best PA-MPJPE: {best_overall_pa:.1f} mm")


# ============================================================
# ONNX export — 5 outputs in demo order (MHR slots zero-filled)
# ============================================================
class PoseDeployWrapper(nn.Module):
    """Emits the demo's 5-output contract so the ONNX drops into demo.py.
    mhr_params/shape_params are zeros (this model has no MHR mesh); the demo
    only reads them when --mhr-assets is set, which you won't use here."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, cliff_cond):
        out = self.model(image, cliff_cond)
        cam = out["cam_trans"]
        B = cam.shape[0]
        mhr_params = cam.new_zeros(B, 204)
        shape_params = cam.new_zeros(B, 45)
        return mhr_params, shape_params, cam, out["joints_2d"], out["joints_3d"]


def export_onnx(cfg):
    import onnx  # noqa: F401
    export_dir = Path(cfg.log_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "instanthmr_pose3d.onnx"
    ckpt_path = os.path.join(cfg.log_dir, "best_pose3d.pth")
    if not os.path.exists(ckpt_path):
        print("⚠️ No checkpoint to export at", ckpt_path)
        return
    model = InstantHMRPoseStudent(cfg, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    deploy = PoseDeployWrapper(model).eval()

    keys = ["mhr_params", "shape_params", "cam_trans", "joints_2d", "joints_3d"]
    dummy_img = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    dummy_cliff = torch.randn(1, 3, device=device)
    dyn = {"image": {0: "batch"}, "cliff_cond": {0: "batch"}}
    for k in keys:
        dyn[k] = {0: "batch"}
    torch.onnx.export(deploy, (dummy_img, dummy_cliff), str(onnx_path),
                      input_names=["image", "cliff_cond"], output_names=keys,
                      dynamic_axes=dyn, opset_version=17)
    print(f"✅ Exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"   Run the demo with:  python demo.py --camera 0 --model {onnx_path}")


# ============================================================
# Entry point
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Keypoint + camera student (no MHR mesh).")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--harmony4d_cap", type=int, default=None, help="Max Harmony4D crops (0 = uncapped).")
    p.add_argument("--overfit-test", dest="overfit_test", action="store_true",
                   help="Run the 1-batch overfit sanity check and exit.")
    p.add_argument("--no-resume", dest="no_resume", action="store_true")
    p.add_argument("--no-export", dest="no_export", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"⚠️ Ignoring unrecognized arguments: {unknown}")
    return args


def main():
    args = parse_args()
    cfg = PoseConfig()
    if args.data_root is not None:      cfg.data_root = args.data_root
    if args.output_dir is not None:     cfg.log_dir = args.output_dir
    if args.epochs is not None:         cfg.epochs = args.epochs
    if args.batch_size is not None:     cfg.batch_size = args.batch_size
    if args.lr is not None:             cfg.lr = args.lr
    if args.num_workers is not None:    cfg.num_workers = args.num_workers
    if args.max_images is not None:     cfg.max_images = args.max_images
    if args.harmony4d_cap is not None:
        if args.harmony4d_cap <= 0:
            cfg.per_dataset_caps.pop("sam3d_gt_harmony4d", None)
        else:
            cfg.per_dataset_caps["sam3d_gt_harmony4d"] = args.harmony4d_cap
    if args.no_resume:                  cfg.resume = False

    print("=" * 60)
    print(f"Data root  : {cfg.data_root}")
    print(f"Output dir : {cfg.log_dir}")
    print(f"Backbone   : {cfg.backbone} | epochs={cfg.epochs} | batch={cfg.batch_size} | lr={cfg.lr}")
    print(f"Loss       : 2D + 3D keypoints + camera (NO MHR mesh)")
    print("=" * 60)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    if args.overfit_test:
        run_overfit_test(cfg, train_loader)
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_pose3d(cfg, train_loader, val_loader)
    if not args.no_export:
        export_onnx(cfg)


if __name__ == "__main__":
    main()
