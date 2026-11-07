"""Reusable visualization and rendering tools for InstantHMR.

Includes:
- 2D camera overlay with 70 MHR keypoints, bone links, stabilized bounding box, and depth tag.
- 3D Skeleton Studio with rotatable camera, floor perspective grid, and depth indicators.
- Real-time telemetry HUD displaying rolling pipeline FPS, detector latency, and HMR latency.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from .skeleton import SKELETON_EDGES, NUM_JOINTS
from .inference import HMRPrediction
from typing import Optional, Sequence

# Modern UI Color Palette (BGR)
COLOR_BG = (20, 22, 26)
COLOR_PANEL_BG = (14, 16, 20)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 145, 155)
COLOR_CYAN = (240, 200, 0)
COLOR_GREEN = (80, 225, 100)
COLOR_ORANGE = (40, 140, 255)
COLOR_PURPLE = (220, 100, 180)
COLOR_BLUE = (255, 160, 40)
COLOR_DARK_BORDER = (45, 50, 60)


def get_bone_color(idx1: int, idx2: int) -> tuple[int, int, int]:
    """Color-code bones by anatomical section (head, torso, arms, legs, hands)."""
    # Head / Neck
    if idx1 in (0, 1, 2, 3, 4, 69) or idx2 in (0, 1, 2, 3, 4, 69):
        return (255, 220, 60)
    # Right Hand
    if idx1 in range(21, 41) or idx2 in range(21, 41):
        return (60, 240, 255)
    # Left Hand
    if idx1 in range(42, 62) or idx2 in range(42, 62):
        return (240, 80, 220)
    # Arms
    if idx1 in (5, 7, 41, 63, 65, 67) or idx2 in (5, 7, 41, 63, 65, 67) or \
       idx1 in (6, 8, 62, 64, 66, 68) or idx2 in (6, 8, 62, 64, 66, 68):
        return (40, 160, 255)
    # Torso / Spine / Hips
    if idx1 in range(9, 21) or idx2 in range(9, 21):
        return (255, 110, 50)
    # Legs
    return (80, 230, 120)


def draw_person_overlay(img: np.ndarray, person, model_name: str = "InstantHMR",
                        box_color: tuple[int, int, int] = COLOR_GREEN,
                        tag_color: tuple[int, int, int] = (50, 180, 80),
                        draw_box: bool = True) -> None:
    """Draw 2D skeleton, keypoints, and stabilized bounding box overlay on frame in-place."""
    h, w = img.shape[:2]
    joints_2d = person.joints_2d

    # 1. Bones
    for a, b in SKELETON_EDGES:
        if a < len(joints_2d) and b < len(joints_2d):
            pt1 = (int(round(joints_2d[a, 0])), int(round(joints_2d[a, 1])))
            pt2 = (int(round(joints_2d[b, 0])), int(round(joints_2d[b, 1])))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                color = get_bone_color(a, b)
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # 2. Keypoints (70 joints)
    for i in range(min(len(joints_2d), NUM_JOINTS)):
        pt = (int(round(joints_2d[i, 0])), int(round(joints_2d[i, 1])))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            radius = 2 if (21 <= i <= 61) else 3
            color = (255, 255, 255) if (21 <= i <= 61) else ((0, 240, 255) if i in (0, 1, 2, 3, 4, 69) else (0, 255, 120))
            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    # 3. Stabilized Box & Depth Tag
    if draw_box and hasattr(person, "bbox") and person.bbox is not None:
        bbox = np.asarray(person.bbox, dtype=int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3])
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
            corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), box_color, 2)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), box_color, 2)
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), box_color, 2)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), box_color, 2)
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), box_color, 2)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), box_color, 2)
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, 2)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), box_color, 2)

            z_depth = person.cam_trans[2] if hasattr(person, "cam_trans") else 0.0
            conf = getattr(person, "confidence", 1.0)
            tag = f"{model_name}: {conf * 100:.0f}% | Z:{z_depth:.1f}m"
            (tw, th_text), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            tag_y1 = max(0, y1 - th_text - 6)
            tag_y2 = max(th_text + 6, y1)
            cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 8, tag_y2), tag_color, -1)
            cv2.putText(img, tag, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


def render_3d_skeleton_studio(joints_3d: np.ndarray | None, width: int, height: int,
                               azim: float = 24.0, elev: float = -8.0,
                               z_depth: float | None = None, model_tag: str = "InstantHMR") -> np.ndarray:
    """Render 3D articulated skeleton in studio environment with floor grid and depth."""
    canvas = np.full((height, width, 3), COLOR_PANEL_BG, dtype=np.uint8)

    # Floor grid
    cy_floor = int(height * 0.88)
    for gx in range(20, width, max(20, width // 12)):
        cv2.line(canvas, (gx, cy_floor), (gx + (gx - width // 2) // 2, height - 6), (28, 32, 40), 1)
    for gy in range(cy_floor, height - 6, max(12, (height - cy_floor) // 5)):
        cv2.line(canvas, (10, gy), (width - 10, gy), (28, 32, 40), 1)

    if joints_3d is None or len(joints_3d) == 0:
        cv2.putText(canvas, "Searching for subject...", (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
        return canvas

    rad_az = np.radians(azim)
    rad_el = np.radians(elev)
    R_az = np.array([[np.cos(rad_az), 0, np.sin(rad_az)],
                     [0, 1, 0],
                     [-np.sin(rad_az), 0, np.cos(rad_az)]])
    R_el = np.array([[1, 0, 0],
                     [0, np.cos(rad_el), -np.sin(rad_el)],
                     [0, np.sin(rad_el), np.cos(rad_el)]])
    R = R_el @ R_az

    rot_j = joints_3d @ R.T
    center = rot_j.mean(axis=0)
    j_c = rot_j - center

    span_y = j_c[:, 1].max() - j_c[:, 1].min()
    scale = (height * 0.68) / max(span_y, 0.1)

    cx, cy = width / 2.0, height * 0.46
    u_j = (j_c[:, 0] * scale + cx).astype(np.int32)
    v_j = (j_c[:, 1] * scale + cy).astype(np.int32)
    pts_joints = np.stack([u_j, v_j], axis=1)

    # 3D Bones
    for a, b in SKELETON_EDGES:
        if a < len(pts_joints) and b < len(pts_joints):
            pt1 = tuple(pts_joints[a])
            pt2 = tuple(pts_joints[b])
            if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                color = get_bone_color(a, b)
                cv2.line(canvas, pt1, pt2, (color[0] // 2, color[1] // 2, color[2] // 2), 4, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

    # 3D Joints
    for i in range(min(len(pts_joints), NUM_JOINTS)):
        pt = tuple(pts_joints[i])
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            radius = 2 if (21 <= i <= 61) else 3
            color = (255, 255, 255) if (21 <= i <= 61) else (0, 240, 255)
            cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Title & Depth
    cv2.putText(canvas, f"3D SKELETON STUDIO (MHR 70)", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 255), 1, cv2.LINE_AA)
    depth_str = f"Depth Z: {z_depth:.2f}m" if z_depth is not None else "70 Articulated Joints"
    cv2.putText(canvas, f"{model_tag} | {depth_str}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_GRAY, 1, cv2.LINE_AA)

    # 3D axes widget
    ax_x, ax_y = width - 36, height - 28
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x + 18, ax_y), (80, 80, 240), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x, ax_y - 18), (80, 240, 80), 1, tipLength=0.3)
    cv2.arrowedLine(canvas, (ax_x, ax_y), (ax_x - 10, ax_y + 10), (240, 160, 60), 1, tipLength=0.3)
    cv2.putText(canvas, "3D", (ax_x - 8, ax_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.26, COLOR_WHITE, 1)

    return canvas


def make_single_hud(width: int, height: int, frame_idx: int, total_frames: int, t_frame_s: float,
                    rolling_fps: float, det_ms: float, hmr_ms: float, detector_name: str,
                    is_smoothed: bool = True) -> np.ndarray:
    """Create a telemetry HUD bar for single-pipeline execution."""
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # Title & Video Progress
    cv2.putText(hud, "InstantHMR v2", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)
    progress_str = f"Frame {frame_idx}/{total_frames} ({t_frame_s:.1f}s)" if total_frames > 0 else f"Frame {frame_idx} ({t_frame_s:.1f}s)"
    cv2.putText(hud, progress_str, (160, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # 1. Pipeline FPS Pill
    cv2.rectangle(hud, (16, 34), (270, 62), (24, 38, 28), -1)
    cv2.rectangle(hud, (16, 34), (270, 62), COLOR_GREEN, 1)
    cv2.putText(hud, f"Pipeline: {rolling_fps:4.1f} FPS", (26, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GREEN, 1, cv2.LINE_AA)

    # 2. Detector Latency Pill
    cv2.rectangle(hud, (282, 34), (580, 62), (32, 28, 20), -1)
    cv2.rectangle(hud, (282, 34), (580, 62), COLOR_ORANGE, 1)
    cv2.putText(hud, f"{detector_name}: {det_ms:4.1f}ms", (292, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ORANGE, 1, cv2.LINE_AA)

    # 3. InstantHMR CoreML/MPS Pill
    cv2.rectangle(hud, (592, 34), (880, 62), (32, 28, 42), -1)
    cv2.rectangle(hud, (592, 34), (880, 62), COLOR_PURPLE, 1)
    cv2.putText(hud, f"CoreML HMR: {hmr_ms:4.1f}ms", (602, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 140, 215), 1, cv2.LINE_AA)

    # 4. Filter Mode Badge
    filter_label = "1-Euro + Box Stabilizer (Zero Lag)" if is_smoothed else "RAW (Zero Smoothing)"
    filter_color = COLOR_CYAN if is_smoothed else (100, 100, 255)
    cv2.rectangle(hud, (892, 34), (width - 16, 62), (24, 26, 32), -1)
    cv2.rectangle(hud, (892, 34), (width - 16, 62), (90, 95, 105), 1)
    cv2.putText(hud, filter_label, (902, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, filter_color, 1, cv2.LINE_AA)

    return hud


def make_compare_hud(width: int, height: int, frame_idx: int, total_frames: int, t_frame_s: float,
                     fps1: float, det1_ms: float, name1: str,
                     fps2: float, det2_ms: float, name2: str,
                     hmr_ms: float) -> np.ndarray:
    """Create a telemetry HUD bar for dual-detector comparison execution."""
    hud = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    # Title & Video Progress
    cv2.putText(hud, f"DETECTOR BENCHMARK: {name1} vs {name2}", (16, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)
    progress_str = f"Frame {frame_idx}/{total_frames} ({t_frame_s:.1f}s)" if total_frames > 0 else f"Frame {frame_idx}"
    cv2.putText(hud, progress_str, (520, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # 1. Detector 1 Pill
    cv2.rectangle(hud, (16, 34), (380, 62), (32, 28, 20), -1)
    cv2.rectangle(hud, (16, 34), (380, 62), COLOR_ORANGE, 1)
    cv2.putText(hud, f"{name1}: {fps1:4.1f} FPS | Det: {det1_ms:4.1f}ms", (26, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_ORANGE, 1, cv2.LINE_AA)

    # 2. Detector 2 Pill
    cv2.rectangle(hud, (392, 34), (756, 62), (20, 38, 28), -1)
    cv2.rectangle(hud, (392, 34), (756, 62), COLOR_GREEN, 1)
    cv2.putText(hud, f"{name2}: {fps2:4.1f} FPS | Det: {det2_ms:4.1f}ms", (402, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GREEN, 1, cv2.LINE_AA)

    # 3. Batched CoreML Regressor Pill
    cv2.rectangle(hud, (768, 34), (1030, 62), (32, 28, 42), -1)
    cv2.rectangle(hud, (768, 34), (1030, 62), COLOR_PURPLE, 1)
    cv2.putText(hud, f"HMR CoreML: {hmr_ms:4.1f}ms (Batch N=2)", (778, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 140, 215), 1, cv2.LINE_AA)

    # 4. Speedup Ratio Badge
    cv2.rectangle(hud, (1042, 34), (width - 16, 62), (24, 26, 32), -1)
    cv2.rectangle(hud, (1042, 34), (width - 16, 62), (90, 95, 105), 1)
    faster_tag = f"{name2} is {det1_ms / max(det2_ms, 0.1):.1f}x Faster" if det1_ms > det2_ms else f"{name1} is {det2_ms / max(det1_ms, 0.1):.1f}x Faster"
    cv2.putText(hud, faster_tag, (1052, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1, cv2.LINE_AA)

    return hud


# ---------------------------------------------------------------------------
# Legacy Rerun Visualizer (Preserved for backwards compatibility with demo.py)
# ---------------------------------------------------------------------------

MESH_COLOR = (200, 160, 130)
JOINT_COLOR = (0, 230, 0)
BONE_COLOR = (255, 230, 0)
BOX_COLOR = (0, 200, 255)


class RerunVisualizer:
    def __init__(self, faces: Optional[np.ndarray] = None, mesh_alpha: float = 0.35, spawn: bool = True,
                 save_path: Optional[str] = None):
        import rerun as rr

        self.rr = rr
        self.faces = faces
        self.albedo = (*MESH_COLOR, int(round(255 * min(max(mesh_alpha, 0.0), 1.0))))
        rr.init("instanthmr", spawn=spawn)
        if save_path:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self._n_logged = 0
        self._layout_sent = False

    def log(self, frame: int, image_rgb: np.ndarray, persons: Sequence[HMRPrediction],
            timings: dict[str, float]) -> None:
        rr = self.rr
        rr.set_time("frame", sequence=frame)
        for name, ms in timings.items():
            rr.log(f"timing/{name}", rr.Scalars(float(ms)))
        for stale in range(len(persons), self._n_logged):
            rr.log(f"world/person_{stale}", rr.Clear(recursive=True))
        self._n_logged = len(persons)

        for i, p in enumerate(persons):
            joints = p.joints_3d_cam
            rr.log(f"world/person_{i}/joints", rr.Points3D(joints, radii=0.012, colors=JOINT_COLOR))
            rr.log(f"world/person_{i}/skeleton",
                   rr.LineStrips3D(joints[np.array(SKELETON_EDGES)], colors=BONE_COLOR, radii=0.004))
            if self.faces is not None and p.vertices_local is not None:
                rr.log(f"world/person_{i}/mesh", rr.Mesh3D(vertex_positions=p.vertices_cam,
                                                           triangle_indices=self.faces, albedo_factor=self.albedo))

        h, w = image_rgb.shape[:2]
        rr.log("camera/image", rr.Image(draw_overlay(image_rgb, persons, timings)))
        if persons:
            depth = float(np.median([p.cam_trans[2] for p in persons]))
            rr.log("camera", rr.Pinhole(width=w, height=h, focal_length=persons[0].focal_length,
                                        principal_point=[w / 2, h / 2], image_plane_distance=max(0.25 * depth, 0.2)))
            if not self._layout_sent:
                self._send_layout(persons[0])
                self._layout_sent = True

    def _send_layout(self, first: HMRPrediction) -> None:
        import rerun.blueprint as rrb

        centre = first.joints_3d_cam.mean(0)
        distance = max(4 * float(np.abs(first.joints_3d_cam - centre).max()), 1.5)
        self.rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(rrb.Spatial2DView(origin="camera/image", name="Camera"),
                         rrb.TimeSeriesView(origin="timing", name="Latency (ms)"), row_shares=[3, 1]),
            rrb.Spatial3DView(origin="/", name="3D", background=rrb.Background(color=[25, 25, 25]),
                              eye_controls=rrb.EyeControls3D(
                                  position=[float(centre[0]), float(centre[1]), float(centre[2] - distance)],
                                  look_target=centre.tolist(), eye_up=[0.0, -1.0, 0.0], kind=rrb.Eye3DKind.Orbital)),
            column_shares=[2, 3])))


def draw_overlay(image_rgb: np.ndarray, persons: Sequence[HMRPrediction], timings: dict[str, float]) -> np.ndarray:
    """The frame with each person's box and 2D skeleton, and the stage timings top left."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 400))
    for p in persons:
        x1, y1, x2, y2 = p.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thick)
        pts = p.joints_2d.astype(int)
        for a, b in SKELETON_EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), BONE_COLOR, thick, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(out, tuple(pt), thick + 1, JOINT_COLOR, -1, cv2.LINE_AA)

    lines = [f"{name:<9s}{ms:6.1f} ms" for name, ms in timings.items()]
    total = sum(timings.values())
    if total > 0:
        lines.append(f"{'total':<9s}{total:6.1f} ms ({1000 / total:.0f} fps)")
    scale = max(0.5, min(h, w) / 1200)
    line_h = int(28 * scale)
    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (int(300 * scale) + 16, 16 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)
    for k, line in enumerate(lines):
        cv2.putText(out, line, (16, 8 + line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7,
                    (255, 255, 255), max(1, round(scale * 1.5)), cv2.LINE_AA)
    return out
