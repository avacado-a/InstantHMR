# InstantHMR Mobile Studio (Android & iOS)

Real-time **3D Human Mesh & 70-Keypoint Pose Recovery** on **Android** and **iOS** powered by **Mobile NPU** (Apple CoreML Neural Engine / Android NNAPI & Qualcomm Hexagon QNN) and **Mobile CPU** (ARM XNNPACK SIMD).

---

## Key Features

1. **Full Pre-Launch Studio Configurator (Everything Configurable Before Starting)**:
   - **Camera Optics & Stream Size**: Choose **Front Camera (Selfie)** or **Back Camera (World)**, toggle front-camera coordinate mirroring, select stream resolution (**480p**, **720p**, **1080p**), target frame rate (**30 FPS** or **60 FPS**), and camera focal calibration.
   - **Hardware Compute Accelerator (NPU vs GPU vs CPU)**:
     - **Mobile NPU**: Dispatches FP16 subgraphs onto the **Apple 16-Core Neural Engine (CoreML ANE)** on iOS and **Qualcomm Hexagon HTP / Google Tensor TPU / Exynos NPU (NNAPI / QNN)** on Android (~4.6–5.5 ms/person).
     - **Mobile GPU**: Runs shaders via Apple Metal or Android Vulkan/FP16.
     - **Mobile CPU**: Multi-threaded ARM NEON **XNNPACK SIMD** execution with selectable thread count (**1, 2, 4, 6, or 8 threads**).
     - Independent accelerator selection for both the **InstantHMR 3D Network** and the **Person Detector**.
   - **InstantHMR Model Size & 3D Mesh**:
     - Switch between **`nano`** (20.0M, 4.60 ms NPU), **`tiny`** (20.3M, 4.84 ms NPU), **`small`** (24.4M, 5.38 ms NPU), **`medium`** (32.2M, 5.49 ms NPU), and **`medium-commercial`** (Apache-2.0).
     - Toggle full **18,439-vertex MHR 3D body mesh** decoding vs ultra-fast **70-joint 3D skeleton** mode, and adjust the **CLIFF crop expansion ratio** (`1.1×` to `1.5×`).
   - **Hardware-Aware Person Detector & Stride Schedule**:
     - Choose between **`yolo26n`**, **`yolo26s`**, **`yolo26n-pose`**, **`yolo26s-pose`**, **`rfdetr_nano`**, and **`yolov8n`** — each annotated with device NPU/CPU compatibility and latency estimates.
     - Configure **Detector Frame Stride** (`1` = every frame, `2×`, `3×`, `4×`, `6×`, `8×`, `12×`), **Detector Confidence Threshold** (`25%`–`65%`), **Max Persons** (`1`–`4`), and **17-Keypoint Pose Envelope** box expansion.
   - **Temporal Smoothing & Box Stabilization**:
     - Toggle causal **1-Euro Joint Smoother** ($\beta=0.02\text{ px}$ for 2D, $\beta=4.0\text{ m}$ for 3D) with adjustable **Beta Responsiveness Scale** (`0.5×`–`2.0×`).
     - Toggle **`BoxStabilizer`** to eliminate inter-stride bounding box jitter.
   - **4 Live Studio Presentation Modes**:
     - **`studio`**: Split screen with Live 2D Camera Overlay on top + Interactive 3D MHR70 Skeleton Studio on bottom (drag to orbit 3D view).
     - **`overlay`**: Full-screen 2D Camera Overlay with 70 keypoints, anatomical bones, and metric depth tags (`Z: 2.35m`).
     - **`studio3d`**: Full-screen interactive 3D Skeleton & Mesh Studio with perspective floor grid.
     - **`compare`**: Dual-detector real-time side-by-side benchmark comparing two detectors simultaneously.

2. **On-Demand Model Manager UI**:
   - Inspect, download, or delete ONNX models directly inside the app with live progress bars, or pre-bundle them offline before building.

---

## One-Command Easy Install

From the repository root:

```bash
# 1. Install dependencies & verify pipeline
./mobile/install_mobile.sh

# 2. Build & launch on Android (NNAPI / QNN NPU + XNNPACK CPU)
./mobile/install_mobile.sh --android

# 3. Build & launch on iOS (Apple Neural Engine CoreML + Metal + CPU)
./mobile/install_mobile.sh --ios

# Optional: Pre-download & bundle default ONNX models offline into the app binary
./mobile/install_mobile.sh --bundle
```

Or run directly inside [`mobile/`](file:///home/antigravity/avacado-a/InstantHMR/mobile):

```bash
cd mobile
npm install --legacy-peer-deps

# Prepare or bundle ONNX models (optional — app also downloads on-demand)
npm run prepare-models
npm run bundle-models

# Run on Android or iOS
npm run android
npm run ios
```

---

## Project Structure

- [`App.tsx`](file:///home/antigravity/avacado-a/InstantHMR/mobile/App.tsx) — Root application state & navigation between Pre-Launch Studio Setup, Model Manager, and Live Camera Studio.
- [`plugins/withNpuAcceleration.js`](file:///home/antigravity/avacado-a/InstantHMR/mobile/plugins/withNpuAcceleration.js) — Expo config plugin configuring Android `largeHeap`, hardware acceleration, ONNX native `.so` packaging, and iOS `CADisableMinimumFrameDurationOnPhone` (120Hz ProMotion) + camera permissions.
- [`src/types/config.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/types/config.ts) — Complete TypeScript definitions for all camera, stride, size, detector, NPU/CPU, and smoothing options.
- [`src/constants/hardware.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/constants/hardware.ts) — Runtime SoC/NPU capability detector and `onnxruntime-react-native` execution provider builder (`coreml`, `nnapi`, `xnnpack`, `cpu`).
- [`src/constants/skeleton.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/constants/skeleton.ts) — MHR70 70-joint hierarchy, bone edges, and anatomical color palette.
- [`src/engine/preprocessing.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/engine/preprocessing.ts) — CLIFF camera conditioning, $288\times 288$ bilinear crop extraction, ImageNet normalization, 2D keypoint unprojection, and 17-keypoint pose envelope padding.
- [`src/engine/smoothing.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/engine/smoothing.ts) — Causal 1-Euro filter, `BoxStabilizer`, and multi-person IoU tracker.
- [`src/engine/inferenceEngine.ts`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/engine/inferenceEngine.ts) — End-to-end mobile inference pipeline with stride reuse and dual-detector benchmarking.
- [`src/components/PreLaunchConfigurator.tsx`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/components/PreLaunchConfigurator.tsx) — Pre-launch dashboard exposing every pipeline parameter before starting the camera.
- [`src/components/LiveCameraStudio.tsx`](file:///home/antigravity/avacado-a/InstantHMR/mobile/src/components/LiveCameraStudio.tsx) — Real-time camera & 3D studio screen with instant in-flight toggles.
