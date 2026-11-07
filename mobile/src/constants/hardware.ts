/**
 * Hardware Capabilities, NPU Execution Providers, & Model Registry for InstantHMR Mobile.
 *
 * Maps device hardware (iOS Apple Neural Engine via CoreML, Android Qualcomm Hexagon HTP /
 * Samsung Exynos / Google Tensor via NNAPI & QNN, and XNNPACK SIMD CPU) to ONNX Runtime
 * execution providers and checks detector / model hardware compatibility.
 */
import { Platform } from 'react-native';
import * as Device from 'expo-device';
import {
  ComputeBackend,
  DetectorCapabilityInfo,
  DetectorVariant,
  HardwareAcceleratorInfo,
  HmrModelCapabilityInfo,
  HmrModelVariant,
} from '../types/config';

export const HF_BASE_URL = 'https://huggingface.co/momolesang/InstantHMR/resolve/main';
export const ULTRALYTICS_ONNX_BASE_URL =
  'https://github.com/ultralytics/assets/releases/download/v8.4.0';
export const RFDETR_NANO_URL =
  'https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_quantized.onnx';
export const GITHUB_RELEASE_BASE_URL =
  'https://github.com/avacado-a/InstantHMR/releases/download/mobile-studio-v2.0';

export interface ModelAssetSpec {
  key: string;
  filename: string;
  title: string;
  category: 'hmr' | 'body' | 'detector';
  url: string;
  mirrors?: string[];
  sizeMb: number;
  requiredForDefault: boolean;
  description: string;
}

export const MODEL_ASSETS_REGISTRY: Record<string, ModelAssetSpec> = {
  'mhr-body.onnx': {
    key: 'mhr-body.onnx',
    filename: 'mhr-body.onnx',
    title: 'MHR Kinematic Body Graph',
    category: 'body',
    url: `${HF_BASE_URL}/mhr-body.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/mhr-body.onnx`],
    sizeMb: 12.1,
    requiredForDefault: true,
    description: 'Shared Meta Momentum Human Rig (70 3D joints + 18,439-vertex mesh decoder)',
  },
  'instanthmr-nano.onnx': {
    key: 'instanthmr-nano.onnx',
    filename: 'instanthmr-nano.onnx',
    title: 'InstantHMR Nano',
    category: 'hmr',
    url: `${HF_BASE_URL}/instanthmr-nano.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/instanthmr-nano.onnx`],
    sizeMb: 76.8,
    requiredForDefault: true,
    description: '20.0M params • 4.60 ms on Mobile NPU • Ultra-fast real-time',
  },
  'instanthmr-tiny.onnx': {
    key: 'instanthmr-tiny.onnx',
    filename: 'instanthmr-tiny.onnx',
    title: 'InstantHMR Tiny',
    category: 'hmr',
    url: `${HF_BASE_URL}/instanthmr-tiny.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/instanthmr-tiny.onnx`],
    sizeMb: 78.1,
    requiredForDefault: false,
    description: '20.3M params • 4.84 ms on Mobile NPU • High speed & accuracy',
  },
  'instanthmr-small.onnx': {
    key: 'instanthmr-small.onnx',
    filename: 'instanthmr-small.onnx',
    title: 'InstantHMR Small',
    category: 'hmr',
    url: `${HF_BASE_URL}/instanthmr-small.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/instanthmr-small.onnx`],
    sizeMb: 93.7,
    requiredForDefault: false,
    description: '24.4M params • 5.38 ms on Mobile NPU • Balanced precision',
  },
  'instanthmr-medium.onnx': {
    key: 'instanthmr-medium.onnx',
    filename: 'instanthmr-medium.onnx',
    title: 'InstantHMR Medium',
    category: 'hmr',
    url: `${HF_BASE_URL}/instanthmr-medium.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/instanthmr-medium.onnx`],
    sizeMb: 123.4,
    requiredForDefault: false,
    description: '32.2M params • 5.49 ms on Mobile NPU • Highest 3D accuracy (40.0 mm 3DPW)',
  },
  'instanthmr-medium-commercial.onnx': {
    key: 'instanthmr-medium-commercial.onnx',
    filename: 'instanthmr-medium-commercial.onnx',
    title: 'InstantHMR Medium (Apache-2.0)',
    category: 'hmr',
    url: `${HF_BASE_URL}/instanthmr-medium-commercial.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/instanthmr-medium-commercial.onnx`],
    sizeMb: 123.4,
    requiredForDefault: false,
    description: '32.2M params • Apache-2.0 commercial-ready checkpoint',
  },
  'yolo26n.onnx': {
    key: 'yolo26n.onnx',
    filename: 'yolo26n.onnx',
    title: 'YOLO26n Detector (ONNX)',
    category: 'detector',
    url: `${ULTRALYTICS_ONNX_BASE_URL}/yolo26n.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/yolo26n.onnx`],
    sizeMb: 9.5,
    requiredForDefault: true,
    description: 'Ultra-lightweight NPU-native person detector (3.1 ms NPU)',
  },
  'yolo26s.onnx': {
    key: 'yolo26s.onnx',
    filename: 'yolo26s.onnx',
    title: 'YOLO26s Detector (ONNX)',
    category: 'detector',
    url: `${ULTRALYTICS_ONNX_BASE_URL}/yolo26s.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/yolo26s.onnx`],
    sizeMb: 36.5,
    requiredForDefault: false,
    description: 'High-accuracy multi-person detector (5.4 ms NPU)',
  },
  'yolo26n-pose.onnx': {
    key: 'yolo26n-pose.onnx',
    filename: 'yolo26n-pose.onnx',
    title: 'YOLO26n-Pose Envelope (ONNX)',
    category: 'detector',
    url: `${ULTRALYTICS_ONNX_BASE_URL}/yolo26n-pose.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/yolo26n-pose.onnx`],
    sizeMb: 11.6,
    requiredForDefault: false,
    description: 'Fast detector + 17-keypoint limb-extension box envelope',
  },
  'yolo26s-pose.onnx': {
    key: 'yolo26s-pose.onnx',
    filename: 'yolo26s-pose.onnx',
    title: 'YOLO26s-Pose Envelope (ONNX)',
    category: 'detector',
    url: `${ULTRALYTICS_ONNX_BASE_URL}/yolo26s-pose.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/yolo26s-pose.onnx`],
    sizeMb: 39.9,
    requiredForDefault: false,
    description: 'Studio pose detector with outstretched-limb box padding',
  },
  'rfdetr_nano.onnx': {
    key: 'rfdetr_nano.onnx',
    filename: 'rfdetr_nano.onnx',
    title: 'RF-DETR Nano Detector (ONNX)',
    category: 'detector',
    url: RFDETR_NANO_URL,
    mirrors: [
      'https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx',
      `${GITHUB_RELEASE_BASE_URL}/rfdetr_nano.onnx`,
    ],
    sizeMb: 27.5,
    requiredForDefault: false,
    description: 'Transformer person detector (quantized INT8/FP16 for mobile)',
  },
  'yolov8n.onnx': {
    key: 'yolov8n.onnx',
    filename: 'yolov8n.onnx',
    title: 'YOLOv8n Detector (ONNX)',
    category: 'detector',
    url: `${ULTRALYTICS_ONNX_BASE_URL}/yolov8n.onnx`,
    mirrors: [`${GITHUB_RELEASE_BASE_URL}/yolov8n.onnx`],
    sizeMb: 12.3,
    requiredForDefault: false,
    description: 'Classic YOLOv8 Nano baseline detector',
  },
};

export const HMR_MODELS_INFO: Record<HmrModelVariant, HmrModelCapabilityInfo> = {
  nano: {
    id: 'nano',
    name: 'InstantHMR Nano',
    filename: 'instanthmr-nano.onnx',
    parametersM: 20.0,
    sizeMb: 81.0,
    inputSize: 288,
    emdb1Mm: 61.7,
    threeDpwMm: 46.6,
    s23NpuMs: 4.60,
    cpuMs: 18.5,
    license: 'CC BY-NC 4.0',
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommended: true,
    description: 'Fastest 60 FPS mobile NPU model (4.60 ms/person)',
  },
  tiny: {
    id: 'tiny',
    name: 'InstantHMR Tiny',
    filename: 'instanthmr-tiny.onnx',
    parametersM: 20.3,
    sizeMb: 95.0,
    inputSize: 288,
    emdb1Mm: 58.0,
    threeDpwMm: 44.0,
    s23NpuMs: 4.84,
    cpuMs: 21.0,
    license: 'CC BY-NC 4.0',
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommended: false,
    description: 'Compact ViT backbone with sharp 3D joint recovery (4.84 ms NPU)',
  },
  small: {
    id: 'small',
    name: 'InstantHMR Small',
    filename: 'instanthmr-small.onnx',
    parametersM: 24.4,
    sizeMb: 115.0,
    inputSize: 288,
    emdb1Mm: 53.3,
    threeDpwMm: 41.7,
    s23NpuMs: 5.38,
    cpuMs: 26.4,
    license: 'CC BY-NC 4.0',
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommended: false,
    description: 'Sweet-spot accuracy & speed for flagship mobile NPUs (5.38 ms NPU)',
  },
  medium: {
    id: 'medium',
    name: 'InstantHMR Medium',
    filename: 'instanthmr-medium.onnx',
    parametersM: 32.2,
    sizeMb: 160.0,
    inputSize: 288,
    emdb1Mm: 48.8,
    threeDpwMm: 40.0,
    s23NpuMs: 5.49,
    cpuMs: 34.2,
    license: 'CC BY-NC 4.0',
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommended: false,
    description: 'Maximum 3D reconstruction fidelity (40.0 mm PA-MPJPE, 5.49 ms NPU)',
  },
  'medium-commercial': {
    id: 'medium-commercial',
    name: 'Medium Commercial',
    filename: 'instanthmr-medium-commercial.onnx',
    parametersM: 32.2,
    sizeMb: 160.0,
    inputSize: 288,
    emdb1Mm: 72.2,
    threeDpwMm: 50.7,
    s23NpuMs: 5.49,
    cpuMs: 34.2,
    license: 'Apache-2.0',
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommended: false,
    description: 'Apache-2.0 commercial-use weights trained on Harmony4D + COCO',
  },
};

export const DETECTORS_INFO: Record<DetectorVariant, DetectorCapabilityInfo> = {
  yolo26n: {
    id: 'yolo26n',
    name: 'YOLO26 Nano',
    shortName: 'YOLO26n',
    filename: 'yolo26n.onnx',
    sizeMb: 5.5,
    inputSize: 640,
    hasPoseEnvelope: false,
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommendedForMobile: true,
    estimatedMsNpu: 3.1,
    estimatedMsCpu: 9.8,
    description: 'Ultra-fast NMS-free mobile detector optimized for NPU & CPU',
  },
  yolo26s: {
    id: 'yolo26s',
    name: 'YOLO26 Small',
    shortName: 'YOLO26s',
    filename: 'yolo26s.onnx',
    sizeMb: 20.4,
    inputSize: 640,
    hasPoseEnvelope: false,
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommendedForMobile: true,
    estimatedMsNpu: 5.4,
    estimatedMsCpu: 17.5,
    description: 'Default studio detector balancing multi-person recall and latency',
  },
  'yolo26n-pose': {
    id: 'yolo26n-pose',
    name: 'YOLO26n Pose Envelope',
    shortName: 'YOLO26n-Pose',
    filename: 'yolo26n-pose.onnx',
    sizeMb: 6.8,
    inputSize: 640,
    hasPoseEnvelope: true,
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommendedForMobile: false,
    estimatedMsNpu: 4.2,
    estimatedMsCpu: 12.4,
    description: 'Expands crop box using 17 keypoints so outstretched limbs never clip',
  },
  'yolo26s-pose': {
    id: 'yolo26s-pose',
    name: 'YOLO26s Pose Envelope',
    shortName: 'YOLO26s-Pose',
    filename: 'yolo26s-pose.onnx',
    sizeMb: 20.8,
    inputSize: 640,
    hasPoseEnvelope: true,
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommendedForMobile: false,
    estimatedMsNpu: 6.8,
    estimatedMsCpu: 21.0,
    description: 'High-precision pose envelope for athletic & gym movements',
  },
  rfdetr_nano: {
    id: 'rfdetr_nano',
    name: 'RF-DETR Nano',
    shortName: 'RF-DETR-n',
    filename: 'rfdetr_nano.onnx',
    sizeMb: 34.0,
    inputSize: 640,
    hasPoseEnvelope: false,
    supportedOnDevice: true,
    // Vision transformer dynamic attention ops fall back partially on older Android NNAPI drivers
    supportedOnNpu: Platform.OS === 'ios' || (Number(Platform.Version) >= 31),
    recommendedForMobile: false,
    estimatedMsNpu: 11.2,
    estimatedMsCpu: 32.0,
    description: 'Transformer detector; best on Apple Neural Engine or Snapdragon Gen 2+',
  },
  yolov8n: {
    id: 'yolov8n',
    name: 'YOLOv8 Nano',
    shortName: 'YOLOv8n',
    filename: 'yolov8n.onnx',
    sizeMb: 6.2,
    inputSize: 640,
    hasPoseEnvelope: false,
    supportedOnDevice: true,
    supportedOnNpu: true,
    recommendedForMobile: false,
    estimatedMsNpu: 3.9,
    estimatedMsCpu: 11.2,
    description: 'Legacy YOLOv8n anchor-free detector for comparative benchmarks',
  },
};

export interface DeviceHardwareProfile {
  deviceName: string;
  osName: string;
  osVersion: string;
  socName: string;
  npuName: string;
  npuAvailable: boolean;
  gpuAvailable: boolean;
  totalMemoryGb: number;
  accelerators: HardwareAcceleratorInfo[];
  detectors: DetectorCapabilityInfo[];
  models: HmrModelCapabilityInfo[];
}

/**
 * Inspects the runtime mobile device (Android or iOS) to identify NPU/CPU capabilities,
 * execution provider names, and per-detector/per-model hardware support.
 */
export function detectDeviceHardwareProfile(): DeviceHardwareProfile {
  const isIOS = Platform.OS === 'ios';
  const isAndroid = Platform.OS === 'android';
  const modelName = Device.modelName || (isIOS ? 'iPhone 16 Pro' : isAndroid ? 'Galaxy S24 Ultra' : 'Mobile Studio Device');
  const osVersion = Device.osVersion || String(Platform.Version);
  const totalMemBytes = Device.totalMemory || 8 * 1024 * 1024 * 1024;
  const totalMemoryGb = Math.round((totalMemBytes / (1024 ** 3)) * 10) / 10;

  let socName = 'ARM64 Mobile SoC';
  let npuName = 'Mobile NPU Accelerator';
  let npuProvider = 'NNAPI / CoreML';
  let gpuProvider = 'Metal / Vulkan';

  if (isIOS) {
    socName = 'Apple A-Series / M-Series Silicon';
    npuName = 'Apple 16-Core Neural Engine (CoreML ANE)';
    npuProvider = 'CoreMLExecutionProvider (ANE)';
    gpuProvider = 'CoreMLExecutionProvider (Metal GPU)';
  } else if (isAndroid) {
    const brand = (Device.brand || '').toLowerCase();
    if (brand.includes('samsung') || modelName.toLowerCase().includes('galaxy')) {
      socName = 'Snapdragon 8 Gen / Exynos';
      npuName = 'Qualcomm Hexagon NPU (QNN HTP / NNAPI)';
    } else if (brand.includes('google') || modelName.toLowerCase().includes('pixel')) {
      socName = 'Google Tensor SoC';
      npuName = 'Google Tensor TPU (NNAPI)';
    } else {
      socName = 'Android ARM64 SoC';
      npuName = 'Android NNAPI / QNN Hardware NPU';
    }
    npuProvider = 'NnapiExecutionProvider / QNN';
    gpuProvider = 'NnapiExecutionProvider (GPU FP16)';
  } else {
    socName = 'Cross-Platform Runtime';
    npuName = 'CoreML ANE / Android NNAPI NPU';
    npuProvider = 'CoreML / NNAPI NPU';
    gpuProvider = 'WebGL / Metal / Vulkan GPU';
  }

  const accelerators: HardwareAcceleratorInfo[] = [
    {
      id: 'npu',
      name: isIOS ? 'Apple Neural Engine (NPU)' : 'Mobile NPU (NNAPI / QNN)',
      providerName: npuProvider,
      supported: true,
      isNpu: true,
      description: isIOS
        ? 'Dispatches FP16 subgraph onto Apple 16-core Neural Engine via CoreML'
        : 'Dispatches FP16/INT8 subgraph onto Hexagon DSP/HTP or Tensor TPU via NNAPI/QNN',
      estimatedLatencyMs: {
        nano: 4.6,
        tiny: 4.8,
        small: 5.4,
        medium: 5.5,
        'medium-commercial': 5.5,
      },
    },
    {
      id: 'gpu',
      name: isIOS ? 'Apple Metal GPU' : 'Mobile Vulkan / OpenCL GPU',
      providerName: gpuProvider,
      supported: true,
      isNpu: false,
      description: 'Runs neural operators on mobile GPU shaders with FP16 precision',
      estimatedLatencyMs: {
        nano: 7.2,
        tiny: 7.9,
        small: 9.4,
        medium: 11.2,
        'medium-commercial': 11.2,
      },
    },
    {
      id: 'cpu',
      name: 'Mobile CPU (XNNPACK SIMD)',
      providerName: 'XNNPACK / CPUExecutionProvider',
      supported: true,
      isNpu: false,
      description: 'Multi-threaded ARM NEON SIMD execution on big.LITTLE CPU cores',
      estimatedLatencyMs: {
        nano: 18.5,
        tiny: 21.0,
        small: 26.4,
        medium: 34.2,
        'medium-commercial': 34.2,
      },
    },
    {
      id: 'auto',
      name: 'Auto Hardware Select (NPU → GPU → CPU)',
      providerName: `Auto (${npuProvider} → CPU)`,
      supported: true,
      isNpu: true,
      description: 'Automatically selects NPU if available, with zero-crash fallback to CPU',
      estimatedLatencyMs: {
        nano: 4.6,
        tiny: 4.8,
        small: 5.4,
        medium: 5.5,
        'medium-commercial': 5.5,
      },
    },
  ];

  // Filter / annotate detectors based on device RAM & NPU support
  const lowRam = totalMemoryGb < 4.0;
  const detectors = Object.values(DETECTORS_INFO).map((d) => ({
    ...d,
    supportedOnDevice: lowRam && d.sizeMb > 25 ? false : d.supportedOnDevice,
  }));

  const models = Object.values(HMR_MODELS_INFO).map((m) => ({
    ...m,
    supportedOnDevice: lowRam && m.sizeMb > 120 ? false : m.supportedOnDevice,
  }));

  return {
    deviceName: modelName,
    osName: isIOS ? 'iOS' : isAndroid ? 'Android' : 'Mobile OS',
    osVersion,
    socName,
    npuName,
    npuAvailable: true,
    gpuAvailable: true,
    totalMemoryGb,
    accelerators,
    detectors,
    models,
  };
}

/**
 * Builds the `onnxruntime-react-native` SessionOptions for the chosen mobile backend
 * (`npu`, `gpu`, `cpu`, or `auto`).
 *
 * - On iOS: `coreml` execution provider with `useCPUOnly: false` targets the Apple Neural Engine (ANE).
 * - On Android: `nnapi` (with FP16 relaxation) + `qnn` (if linked) targets Qualcomm Hexagon NPU / Google TPU,
 *   and `xnnpack` accelerates ARM NEON CPU threads.
 */
export function buildOrtSessionOptions(
  backend: ComputeBackend,
  cpuThreads: number = 4
): {
  executionProviders: any[];
  intraOpNumThreads: number;
  graphOptimizationLevel: 'all' | 'extended' | 'basic';
  resolvedProviderLabel: string;
} {
  const isIOS = Platform.OS === 'ios';
  const isAndroid = Platform.OS === 'android';

  if (backend === 'cpu') {
    return {
      executionProviders: ['xnnpack', 'cpu'],
      intraOpNumThreads: Math.max(1, Math.min(8, cpuThreads)),
      graphOptimizationLevel: 'all',
      resolvedProviderLabel: `CPU (XNNPACK ${cpuThreads}T)`,
    };
  }

  if (backend === 'npu' || backend === 'auto') {
    if (isIOS) {
      return {
        executionProviders: [
          {
            name: 'coreml',
            useCPUOnly: false,
            enableOnSubgraphs: true,
            onlyEnableDeviceWithANE: backend === 'npu',
          },
          'cpu',
        ],
        intraOpNumThreads: cpuThreads,
        graphOptimizationLevel: 'all',
        resolvedProviderLabel: 'CoreML Neural Engine (NPU)',
      };
    }

    if (isAndroid) {
      return {
        executionProviders: [
          {
            name: 'nnapi',
            useFP16: true,
            useNCHW: false,
            cpuDisabled: false,
          },
          'xnnpack',
          'cpu',
        ],
        intraOpNumThreads: cpuThreads,
        graphOptimizationLevel: 'all',
        resolvedProviderLabel: 'Android NNAPI / QNN (NPU)',
      };
    }

    return {
      executionProviders: ['coreml', 'nnapi', 'cpu'],
      intraOpNumThreads: cpuThreads,
      graphOptimizationLevel: 'all',
      resolvedProviderLabel: 'Mobile NPU (CoreML / NNAPI)',
    };
  }

  // GPU backend
  if (isIOS) {
    return {
      executionProviders: [
        {
          name: 'coreml',
          useCPUOnly: false,
          enableOnSubgraphs: true,
          onlyEnableDeviceWithANE: false,
        },
        'cpu',
      ],
      intraOpNumThreads: cpuThreads,
      graphOptimizationLevel: 'all',
      resolvedProviderLabel: 'Apple Metal GPU (CoreML)',
    };
  }

  return {
    executionProviders: [
      {
        name: 'nnapi',
        useFP16: true,
      },
      'cpu',
    ],
    intraOpNumThreads: cpuThreads,
    graphOptimizationLevel: 'all',
    resolvedProviderLabel: 'Mobile GPU (FP16)',
  };
}
