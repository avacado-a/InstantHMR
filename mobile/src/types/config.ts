/**
 * Complete configuration, hardware capability, and prediction types for InstantHMR Mobile Studio.
 */

export type CameraPosition = 'front' | 'back';

export type CameraResolutionPreset = '480p' | '720p' | '1080p';

export type TargetFpsPreset = 30 | 60;

export type HmrModelVariant =
  | 'nano'
  | 'tiny'
  | 'small'
  | 'medium'
  | 'medium-commercial';

export type DetectorVariant =
  | 'yolo26n'
  | 'yolo26s'
  | 'yolo26n-pose'
  | 'yolo26s-pose'
  | 'rfdetr_nano'
  | 'yolov8n';

export type ComputeBackend =
  | 'auto'     // Best available NPU -> GPU -> CPU
  | 'npu'      // CoreML ANE (iOS) / NNAPI or QNN HTP (Android)
  | 'gpu'      // Metal (iOS) / Vulkan or NNAPI GPU (Android)
  | 'cpu';     // XNNPACK / SIMD Multi-threaded CPU

export type PresentationMode =
  | 'studio'    // Split view: Live 2D Camera Overlay + Interactive 3D Skeleton/Mesh Studio
  | 'overlay'   // Full-screen 2D Camera with 70-joint MHR skeleton + depth tags
  | 'studio3d'  // Full-screen 3D interactive MHR studio with floor grid
  | 'compare';  // Dual-detector real-time side-by-side benchmark

export interface HardwareAcceleratorInfo {
  id: ComputeBackend;
  name: string;
  providerName: string;
  supported: boolean;
  isNpu: boolean;
  description: string;
  estimatedLatencyMs: Record<HmrModelVariant, number>;
}

export interface DetectorCapabilityInfo {
  id: DetectorVariant;
  name: string;
  shortName: string;
  filename: string;
  sizeMb: number;
  inputSize: number;
  hasPoseEnvelope: boolean;
  supportedOnDevice: boolean;
  supportedOnNpu: boolean;
  recommendedForMobile: boolean;
  estimatedMsNpu: number;
  estimatedMsCpu: number;
  description: string;
}

export interface HmrModelCapabilityInfo {
  id: HmrModelVariant;
  name: string;
  filename: string;
  parametersM: number;
  sizeMb: number;
  inputSize: number;
  emdb1Mm: number;
  threeDpwMm: number;
  s23NpuMs: number;
  cpuMs: number;
  license: string;
  supportedOnDevice: boolean;
  supportedOnNpu: boolean;
  recommended: boolean;
  description: string;
}

export interface SessionConfig {
  // 1. Camera & Stream Settings
  cameraPosition: CameraPosition;
  mirrorFrontCamera: boolean;
  resolution: CameraResolutionPreset;
  targetFps: TargetFpsPreset;
  focalLengthPx: number | null; // null = auto (1.05 * diagonal)

  // 2. Compute & Hardware Acceleration (CPU vs NPU)
  hmrBackend: ComputeBackend;
  detectorBackend: ComputeBackend;
  cpuThreads: number;

  // 3. InstantHMR Model & Mesh Settings
  hmrModel: HmrModelVariant;
  withMesh: boolean;
  meshAlpha: number;
  maxPersons: number;
  cropExpand: number;

  // 4. Person Detector & Stride Settings
  detector: DetectorVariant;
  compareDetector: DetectorVariant;
  detectorStride: number; // 1 to 12
  detConfidence: number;
  usePoseEnvelope: boolean;

  // 5. Temporal Smoothing & Stabilization
  smoothJoints: boolean;
  betaScale: number;
  stabilizeBox: boolean;

  // 6. Presentation & Studio Visuals
  mode: PresentationMode;
  showHud: boolean;
  rotateSpeed: number;
  initialAzim: number;
  initialElev: number;
}

export interface PersonPrediction {
  id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in image pixels
  confidence: number;
  joints2d: Float32Array;       // (70 * 2) flattened [x0, y0, x1, y1, ...] in image pixels
  joints3dLocal: Float32Array;  // (70 * 3) flattened [x0, y0, z0, ...] in meters, body-centered
  joints3dCam: Float32Array;    // (70 * 3) flattened [x0, y0, z0, ...] in meters, camera frame
  camTrans: [number, number, number]; // [tx, ty, tz] in meters
  mhrParams: Float32Array;      // (204,)
  shapeParams: Float32Array;    // (45,)
  verticesLocal?: Float32Array; // (18439 * 3) optional 3D mesh vertices
  focalLength: number;
  imageShape: [number, number]; // [H, W]
}

export interface FrameTelemetry {
  frameIndex: number;
  timestampS: number;
  rollingFps: number;
  detectorMs: number;
  hmrMs: number;
  bodyDecoderMs: number;
  totalPipelineMs: number;
  detectorReused: boolean;
  activeProvider: string;
  activeDetectorProvider: string;
  personCount: number;
  secondaryDetectorMs?: number;
  secondaryPersonCount?: number;
}

export interface FrameInferenceResult {
  persons: PersonPrediction[];
  secondaryPersons?: PersonPrediction[];
  telemetry: FrameTelemetry;
}

export const DEFAULT_SESSION_CONFIG: SessionConfig = {
  cameraPosition: 'back',
  mirrorFrontCamera: true,
  resolution: '720p',
  targetFps: 30,
  focalLengthPx: null,

  hmrBackend: 'cpu',
  detectorBackend: 'cpu',
  cpuThreads: 4,

  hmrModel: 'nano',
  withMesh: false,
  meshAlpha: 0.38,
  maxPersons: 1,
  cropExpand: 1.2,

  detector: 'yolo26n',
  compareDetector: 'yolo26s',
  detectorStride: 3,
  detConfidence: 0.40,
  usePoseEnvelope: false,

  smoothJoints: true,
  betaScale: 1.0,
  stabilizeBox: true,

  mode: 'overlay',
  showHud: false,
  rotateSpeed: 0.04,
  initialAzim: 24.0,
  initialElev: -8.0,
};
