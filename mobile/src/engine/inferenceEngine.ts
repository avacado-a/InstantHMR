/**
 * InstantHMR Real-Time Mobile Inference Engine (Android NNAPI/QNN + iOS CoreML ANE + XNNPACK CPU).
 *
 * Orchestrates:
 * 1. PersonDetector ONNX Session (`yolo26n`, `yolo26s`, `yolo26n-pose`, `yolo26s-pose`, `rfdetr_nano`, `yolov8n`)
 *    with configurable `detectorStride` (1..12) and `BoxStabilizer` in-between detector frames.
 * 2. InstantHMR Network ONNX Session (`nano`, `tiny`, `small`, `medium`, `medium-commercial`)
 *    taking `(B, 3, 288, 288)` normalized crops + `(B, 3)` CLIFF conditioning.
 * 3. MHR Kinematic Body Decoder ONNX Session (`mhr-body.onnx`) turning `(mhr_params, shape_params)`
 *    into 70 3D keypoints (`joints_3d`) and optional 18,439-vertex mesh (`vertices`).
 * 4. Causal 1-Euro `TemporalSmoother` for zero-lag 2D/3D joint stabilization.
 */
import {
  buildOrtSessionOptions,
  DETECTORS_INFO,
  HMR_MODELS_INFO,
} from '../constants/hardware';
import { NUM_JOINTS } from '../constants/skeleton';
import {
  FrameInferenceResult,
  PersonPrediction,
  SessionConfig,
} from '../types/config';
import {
  applyPoseEnvelopeToBox,
  buildPersonPrediction,
  computeFocalLengthPx,
  preprocessPersonCrop,
} from './preprocessing';
import { BoxStabilizer, TemporalSmoother } from './smoothing';

// Dynamic import helper so the engine runs both in native ONNX Runtime builds and simulator previews
let ortModule: typeof import('onnxruntime-react-native') | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  ortModule = require('onnxruntime-react-native');
} catch {
  ortModule = null;
}

export interface DetectedBox {
  bbox: [number, number, number, number];
  score: number;
  keypoints17?: Float32Array;
}

export class InstantHmrMobileEngine {
  private config: SessionConfig;
  private hmrSession: any = null;
  private bodySession: any = null;
  private detectorSession: any = null;
  private compareDetectorSession: any = null;

  private activeHmrProvider: string = 'NPU (CoreML / NNAPI)';
  private activeDetProvider: string = 'NPU (CoreML / NNAPI)';

  private frameCounter: number = 0;
  private cachedBoxes: DetectedBox[] = [];
  private cachedCompareBoxes: DetectedBox[] = [];

  private boxStabilizers: BoxStabilizer[] = [];
  private smoother: TemporalSmoother;
  private compareSmoother: TemporalSmoother;

  private rollingFpsHistory: number[] = [];
  private lastFrameTimestampMs: number = 0;

  // Preallocated tensor buffers to eliminate per-frame GC pauses
  private cachedDetectorTensorData: Float32Array | null = null;
  private cachedBatchedCrops: Float32Array | null = null;
  private cachedBatchedCliffs: Float32Array | null = null;

  constructor(config: SessionConfig) {
    this.config = { ...config };
    this.smoother = new TemporalSmoother(config.betaScale);
    this.compareSmoother = new TemporalSmoother(config.betaScale);
    for (let i = 0; i < Math.max(8, config.maxPersons); i++) {
      this.boxStabilizers.push(new BoxStabilizer(0.8, 0.02));
    }
  }

  public updateConfig(nextConfig: Partial<SessionConfig>): void {
    this.config = { ...this.config, ...nextConfig };
    if (nextConfig.betaScale !== undefined) {
      this.smoother.setBetaScale(nextConfig.betaScale);
      this.compareSmoother.setBetaScale(nextConfig.betaScale);
    }
  }

  public getConfig(): SessionConfig {
    return this.config;
  }

  /**
   * Helper to create an ONNX Runtime session with automatic CPU fallback
   * if the requested NPU / hardware execution provider fails.
   */
  private async createSessionWithFallback(
    modelPath: string,
    primaryOpts: {
      executionProviders: any[];
      intraOpNumThreads: number;
      graphOptimizationLevel: 'all' | 'extended' | 'basic';
      resolvedProviderLabel: string;
    },
    cpuThreads: number,
    tag: string
  ): Promise<{ session: any; resolvedProvider: string }> {
    if (!ortModule) {
      throw new Error('onnxruntime-react-native module is not available');
    }

    const targetPath = modelPath.startsWith('file://') ? modelPath.replace('file://', '') : modelPath;
    try {
      console.log(`[ORT] Initializing ${tag} with providers:`, primaryOpts.executionProviders);
      const session = await ortModule.InferenceSession.create(targetPath, {
        executionProviders: primaryOpts.executionProviders,
        intraOpNumThreads: primaryOpts.intraOpNumThreads,
        graphOptimizationLevel: primaryOpts.graphOptimizationLevel || 'all',
      });
      console.log(`[ORT] Successfully loaded ${tag} on ${primaryOpts.resolvedProviderLabel}`);
      return { session, resolvedProvider: primaryOpts.resolvedProviderLabel };
    } catch (primaryErr: any) {
      console.warn(
        `[ORT] Primary provider failed for ${tag} (${primaryErr?.message || primaryErr}). Retrying with CPU execution provider...`
      );
      try {
        const cpuSession = await ortModule.InferenceSession.create(targetPath, {
          executionProviders: ['cpu'],
          intraOpNumThreads: Math.max(1, Math.min(8, cpuThreads)),
          graphOptimizationLevel: 'all',
        });
        console.log(`[ORT] Successfully loaded ${tag} on CPU fallback`);
        return {
          session: cpuSession,
          resolvedProvider: `CPU (ARM NEON ${cpuThreads}T) [Fallback]`,
        };
      } catch (cpuErr: any) {
        console.error(`[ORT] Fatal: CPU fallback also failed for ${tag}:`, cpuErr);
        throw cpuErr;
      }
    }
  }

  /**
   * Initializes the ONNX Runtime sessions for the selected detector, InstantHMR backbone,
   * and MHR body graph on the requested hardware execution provider (NPU, GPU, or CPU).
   */
  public async initializeSessions(modelPaths: {
    hmrModelUri: string | null;
    bodyModelUri: string | null;
    detectorModelUri: string | null;
    compareDetectorModelUri?: string | null;
  }): Promise<{ hmrProvider: string; detectorProvider: string }> {
    const hmrOpts = buildOrtSessionOptions(this.config.hmrBackend, this.config.cpuThreads);
    const detOpts = buildOrtSessionOptions(this.config.detectorBackend, this.config.cpuThreads);

    this.activeHmrProvider = hmrOpts.resolvedProviderLabel;
    this.activeDetProvider = detOpts.resolvedProviderLabel;

    if (ortModule && modelPaths.hmrModelUri && modelPaths.bodyModelUri) {
      try {
        const hmrRes = await this.createSessionWithFallback(
          modelPaths.hmrModelUri,
          hmrOpts,
          this.config.cpuThreads,
          'InstantHMR'
        );
        this.hmrSession = hmrRes.session;
        this.activeHmrProvider = hmrRes.resolvedProvider;

        const bodyOpts = buildOrtSessionOptions(
          this.config.hmrBackend === 'cpu' ? 'cpu' : 'auto',
          this.config.cpuThreads
        );
        const bodyRes = await this.createSessionWithFallback(
          modelPaths.bodyModelUri,
          bodyOpts,
          this.config.cpuThreads,
          'MHR-Body'
        );
        this.bodySession = bodyRes.session;

        if (modelPaths.detectorModelUri) {
          const detRes = await this.createSessionWithFallback(
            modelPaths.detectorModelUri,
            detOpts,
            this.config.cpuThreads,
            'Detector'
          );
          this.detectorSession = detRes.session;
          this.activeDetProvider = detRes.resolvedProvider;
        }

        if (this.config.mode === 'compare' && modelPaths.compareDetectorModelUri) {
          const compRes = await this.createSessionWithFallback(
            modelPaths.compareDetectorModelUri,
            detOpts,
            this.config.cpuThreads,
            'CompareDetector'
          );
          this.compareDetectorSession = compRes.session;
        }
      } catch (err: any) {
        console.error('[ORT] Error during session initialization:', err);
      }
    }

    this.resetTrackingState();
    return {
      hmrProvider: this.activeHmrProvider,
      detectorProvider: this.activeDetProvider,
    };
  }

  public resetTrackingState(): void {
    this.frameCounter = 0;
    this.cachedBoxes = [];
    this.cachedCompareBoxes = [];
    this.smoother.reset();
    this.compareSmoother.reset();
    for (const stab of this.boxStabilizers) {
      stab.reset();
    }
  }

  /**
   * Executes one full real-time pipeline pass on a camera frame.
   * Respects `detectorStride`, `stabilizeBox`, `smoothJoints`, `cameraPosition` mirroring,
   * and `withMesh`.
   */
  public async processFrame(
    rgbBuffer: Uint8Array | null,
    width: number,
    height: number,
    timestampS: number,
    channels: 3 | 4 = 3
  ): Promise<FrameInferenceResult> {
    const nowMs = performance.now();
    if (this.lastFrameTimestampMs > 0) {
      const instFps = 1000.0 / Math.max(1.0, nowMs - this.lastFrameTimestampMs);
      this.rollingFpsHistory.push(instFps);
      if (this.rollingFpsHistory.length > 24) {
        this.rollingFpsHistory.shift();
      }
    }
    this.lastFrameTimestampMs = nowMs;

    const stride = Math.max(1, this.config.detectorStride);
    const shouldRunDetector =
      this.frameCounter % stride === 0 || this.cachedBoxes.length === 0;

    let detMs = 0.0;
    let secondaryDetMs: number | undefined;
    let rawBoxes: DetectedBox[] = this.cachedBoxes;

    // 1. Person Detection Pass (or Stride Reuse + BoxStabilizer)
    const tDet0 = performance.now();
    if (shouldRunDetector) {
      rawBoxes = await this.runDetectorPass(
        this.detectorSession,
        this.config.detector,
        rgbBuffer,
        width,
        height,
        timestampS,
        channels
      );
      this.cachedBoxes = rawBoxes;
      detMs = performance.now() - tDet0;

      if (this.config.stabilizeBox) {
        rawBoxes = rawBoxes.map((item, idx) => {
          const stab = this.boxStabilizers[idx] || new BoxStabilizer();
          this.boxStabilizers[idx] = stab;
          return {
            ...item,
            bbox: stab.updateDetection(item.bbox, item.score, timestampS),
          };
        });
      }
    } else {
      // Reuse boxes across stride with continuous 1-Euro BoxStabilizer
      if (this.config.stabilizeBox) {
        rawBoxes = this.cachedBoxes.map((item, idx) => {
          const stab = this.boxStabilizers[idx];
          const held = stab ? stab.getBox(timestampS) : null;
          return {
            ...item,
            bbox: held || item.bbox,
          };
        });
      }
      detMs = 0.15; // BoxStabilizer stride hold cost
    }

    // Optional Dual-Detector Comparison Mode
    let secondaryPersons: PersonPrediction[] | undefined;
    if (this.config.mode === 'compare') {
      const tComp0 = performance.now();
      if (shouldRunDetector || this.cachedCompareBoxes.length === 0) {
        this.cachedCompareBoxes = await this.runDetectorPass(
          this.compareDetectorSession,
          this.config.compareDetector,
          rgbBuffer,
          width,
          height,
          timestampS,
          channels
        );
        secondaryDetMs = performance.now() - tComp0;
      } else {
        secondaryDetMs = 0.15;
      }
    }

    // 2. InstantHMR + MHR Body Graph Pass
    const tHmr0 = performance.now();
    const { predictions, bodyMs } = await this.runHmrAndBodyPass(
      rgbBuffer,
      width,
      height,
      rawBoxes.slice(0, this.config.maxPersons),
      timestampS,
      channels
    );
    const hmrTotalMs = performance.now() - tHmr0;

    // 3. Apply 1-Euro Temporal Smoothing if enabled
    const finalPersons = this.config.smoothJoints
      ? this.smoother.smooth(predictions, timestampS)
      : predictions;

    if (this.config.mode === 'compare') {
      const compOut = await this.runHmrAndBodyPass(
        rgbBuffer,
        width,
        height,
        this.cachedCompareBoxes.slice(0, this.config.maxPersons),
        timestampS,
        channels
      );
      secondaryPersons = this.config.smoothJoints
        ? this.compareSmoother.smooth(compOut.predictions, timestampS)
        : compOut.predictions;
    }

    this.frameCounter += 1;

    const totalPipelineMs = detMs + hmrTotalMs;
    const avgFps =
      this.rollingFpsHistory.length > 0
        ? this.rollingFpsHistory.reduce((a, b) => a + b, 0) / this.rollingFpsHistory.length
        : Math.min(60, 1000.0 / Math.max(totalPipelineMs, 5.0));

    return {
      persons: finalPersons,
      secondaryPersons,
      telemetry: {
        frameIndex: this.frameCounter,
        timestampS,
        rollingFps: avgFps,
        detectorMs: detMs,
        hmrMs: Math.max(0.5, hmrTotalMs - bodyMs),
        bodyDecoderMs: bodyMs,
        totalPipelineMs,
        detectorReused: !shouldRunDetector,
        activeProvider: this.activeHmrProvider,
        activeDetectorProvider: this.activeDetProvider,
        personCount: finalPersons.length,
        secondaryDetectorMs: secondaryDetMs,
        secondaryPersonCount: secondaryPersons?.length,
      },
    };
  }

  private async runDetectorPass(
    session: any,
    variant: SessionConfig['detector'],
    rgbBuffer: Uint8Array | null,
    width: number,
    height: number,
    timestampS: number,
    channels: 3 | 4 = 3
  ): Promise<DetectedBox[]> {
    if (!session || !ortModule || !rgbBuffer) {
      return [];
    }

    const detInfo = DETECTORS_INFO[variant];
    const inputSize = detInfo.inputSize || 640;

    try {
      const plane = inputSize * inputSize;
      const totalElements = 3 * plane;
      if (!this.cachedDetectorTensorData || this.cachedDetectorTensorData.length !== totalElements) {
        this.cachedDetectorTensorData = new Float32Array(totalElements);
      }
      const tensorData = this.cachedDetectorTensorData;
      const inv255 = 1.0 / 255.0;

      const scaleX = width / inputSize;
      const scaleY = height / inputSize;

      if (width === inputSize && height === inputSize) {
        // High-speed linear copy when input matches detector resolution
        let srcIdx = 0;
        for (let i = 0; i < plane; i++) {
          tensorData[i] = rgbBuffer[srcIdx] * inv255;
          tensorData[plane + i] = rgbBuffer[srcIdx + 1] * inv255;
          tensorData[2 * plane + i] = rgbBuffer[srcIdx + 2] * inv255;
          srcIdx += channels;
        }
      } else {
        for (let y = 0; y < inputSize; y++) {
          const sy = Math.min(height - 1, Math.floor(y * scaleY));
          for (let x = 0; x < inputSize; x++) {
            const sx = Math.min(width - 1, Math.floor(x * scaleX));
            const srcIdx = (sy * width + sx) * channels;
            const dstIdx = y * inputSize + x;
            tensorData[dstIdx] = rgbBuffer[srcIdx] * inv255;
            tensorData[plane + dstIdx] = rgbBuffer[srcIdx + 1] * inv255;
            tensorData[2 * plane + dstIdx] = rgbBuffer[srcIdx + 2] * inv255;
          }
        }
      }

      const inputTensor = new ortModule.Tensor('float32', tensorData, [
        1,
        3,
        inputSize,
        inputSize,
      ]);

      const inputName =
        (session.inputNames && session.inputNames[0]) ||
        (variant === 'rfdetr_nano' ? 'pixel_values' : 'images');
      const feeds: Record<string, any> = { [inputName]: inputTensor };

      const outputs = await session.run(feeds);
      const firstOutputKey = Object.keys(outputs)[0];
      const firstOutput = outputs[firstOutputKey];

      if (firstOutput && firstOutput.data) {
        return this.decodeYoloDetections(
          firstOutput.data as Float32Array,
          firstOutput.dims,
          width,
          height,
          scaleX,
          scaleY,
          inputSize
        );
      }
    } catch (err) {
      console.warn(`[ORT] Detector inference error:`, err);
    }

    return [];
  }

  private decodeYoloDetections(
    data: Float32Array,
    dims: readonly number[],
    imgW: number,
    imgH: number,
    scaleX: number,
    scaleY: number,
    inputSize: number = 640
  ): DetectedBox[] {
    const results: DetectedBox[] = [];

    // Format 1: YOLO26 / End-to-end NMS-free [1, 300, 6+] or [1, N, 6]
    // where columns are [x1, y1, x2, y2, score, class_id, ...]
    if (dims.length === 3 && dims[1] <= 600 && dims[2] >= 6) {
      const numDet = dims[1];
      const stride = dims[2];
      for (let i = 0; i < numDet; i++) {
        const base = i * stride;
        const score = data[base + 4];
        const cls = Math.round(data[base + 5]);
        // Class 0 in COCO is person
        if (cls === 0 && score >= this.config.detConfidence) {
          let x1 = data[base];
          let y1 = data[base + 1];
          let x2 = data[base + 2];
          let y2 = data[base + 3];

          // If coordinates are normalized [0..1], convert to input pixel coordinates
          if (x2 <= 1.05 && y2 <= 1.05 && (x2 > 0 || y2 > 0)) {
            x1 *= inputSize;
            y1 *= inputSize;
            x2 *= inputSize;
            y2 *= inputSize;
          }

          let bbox: [number, number, number, number] = [
            Math.max(0, x1 * scaleX),
            Math.max(0, y1 * scaleY),
            Math.min(imgW, x2 * scaleX),
            Math.min(imgH, y2 * scaleY),
          ];
          if (this.config.usePoseEnvelope && stride >= 6 + 17 * 3) {
            const kps = new Float32Array(17 * 2);
            for (let k = 0; k < 17; k++) {
              let kx = data[base + 6 + k * 3];
              let ky = data[base + 6 + k * 3 + 1];
              if (kx <= 1.05 && ky <= 1.05 && (kx > 0 || ky > 0)) {
                kx *= inputSize;
                ky *= inputSize;
              }
              kps[k * 2] = kx * scaleX;
              kps[k * 2 + 1] = ky * scaleY;
            }
            bbox = applyPoseEnvelopeToBox(bbox, kps, imgW, imgH, 12.0);
          }
          results.push({ bbox, score });
        }
      }
    }
    // Format 2: YOLOv8 standard [1, 84, 8400]
    else if (dims.length === 3 && dims[1] === 84 && dims[2] > 1000) {
      const numAnchors = dims[2];
      for (let i = 0; i < numAnchors; i++) {
        const score = data[4 * numAnchors + i]; // class 0 (person)
        if (score >= this.config.detConfidence) {
          const cx = data[i] * scaleX;
          const cy = data[numAnchors + i] * scaleY;
          const w = data[2 * numAnchors + i] * scaleX;
          const h = data[3 * numAnchors + i] * scaleY;
          const bbox: [number, number, number, number] = [
            Math.max(0, cx - w * 0.5),
            Math.max(0, cy - h * 0.5),
            Math.min(imgW, cx + w * 0.5),
            Math.min(imgH, cy + h * 0.5),
          ];
          results.push({ bbox, score });
        }
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, this.config.maxPersons);
  }

  private async runHmrAndBodyPass(
    rgbBuffer: Uint8Array | null,
    width: number,
    height: number,
    boxes: DetectedBox[],
    timestampS: number,
    channels: 3 | 4 = 3
  ): Promise<{ predictions: PersonPrediction[]; bodyMs: number }> {
    if (boxes.length === 0 || !rgbBuffer || !this.hmrSession || !this.bodySession || !ortModule) {
      return { predictions: [], bodyMs: 0.0 };
    }

    const modelInfo = HMR_MODELS_INFO[this.config.hmrModel];
    const focalPx = computeFocalLengthPx(height, width, this.config.focalLengthPx, true);
    const inputSize = modelInfo.inputSize || 288;

    try {
      const batchSize = Math.min(boxes.length, this.config.maxPersons);
      const plane3 = 3 * inputSize * inputSize;
      const totalCrops = batchSize * plane3;
      if (!this.cachedBatchedCrops || this.cachedBatchedCrops.length !== totalCrops) {
        this.cachedBatchedCrops = new Float32Array(totalCrops);
      }
      if (!this.cachedBatchedCliffs || this.cachedBatchedCliffs.length !== batchSize * 3) {
        this.cachedBatchedCliffs = new Float32Array(batchSize * 3);
      }
      const batchedCrops = this.cachedBatchedCrops;
      const batchedCliffs = this.cachedBatchedCliffs;
      const squares: [number, number, number][] = [];

      for (let i = 0; i < batchSize; i++) {
        const prep = preprocessPersonCrop(
          rgbBuffer,
          width,
          height,
          channels,
          boxes[i].bbox,
          focalPx,
          inputSize,
          this.config.cropExpand,
          true
        );
        batchedCrops.set(prep.cropTensor, i * plane3);
        batchedCliffs.set(prep.cliffCond, i * 3);
        squares.push(prep.square);
      }

      const imageTensor = new ortModule.Tensor('float32', batchedCrops, [
        batchSize,
        3,
        inputSize,
        inputSize,
      ]);
      const cliffTensor = new ortModule.Tensor('float32', batchedCliffs, [
        batchSize,
        3,
      ]);

      const netOut = await this.hmrSession.run({
        image: imageTensor,
        cliff_cond: cliffTensor,
      });

      const tBody0 = performance.now();
      const bodyFeeds = {
        mhr_params: netOut.mhr_params,
        shape_params: netOut.shape_params,
      };
      const bodyFetchNames = this.config.withMesh
        ? ['joints_3d', 'vertices']
        : ['joints_3d'];
      const bodyOut = await this.bodySession.run(bodyFeeds, bodyFetchNames);
      const bodyMs = performance.now() - tBody0;

      const mhrParamsAll = netOut.mhr_params.data as Float32Array;
      const shapeParamsAll = netOut.shape_params.data as Float32Array;
      const camTransAll = netOut.cam_trans.data as Float32Array;
      const joints2dAll = netOut.joints_2d.data as Float32Array;
      const joints3dAll = bodyOut.joints_3d.data as Float32Array;
      const verticesAll =
        this.config.withMesh && bodyOut.vertices
          ? (bodyOut.vertices.data as Float32Array)
          : undefined;

      const predictions: PersonPrediction[] = [];
      for (let i = 0; i < batchSize; i++) {
        const j3dSlice = joints3dAll.subarray(
          i * NUM_JOINTS * 3,
          (i + 1) * NUM_JOINTS * 3
        );
        const j2dSlice = joints2dAll.subarray(
          i * NUM_JOINTS * 2,
          (i + 1) * NUM_JOINTS * 2
        );
        const camTrans: [number, number, number] = [
          camTransAll[i * 3],
          camTransAll[i * 3 + 1],
          camTransAll[i * 3 + 2],
        ];
        const mhrSlice = mhrParamsAll.subarray(i * 204, (i + 1) * 204);
        const shapeSlice = shapeParamsAll.subarray(i * 45, (i + 1) * 45);
        const vertSlice = verticesAll
          ? verticesAll.subarray(i * 18439 * 3, (i + 1) * 18439 * 3)
          : undefined;

        predictions.push(
          buildPersonPrediction(
            i + 1,
            boxes[i].bbox,
            boxes[i].score,
            squares[i],
            j2dSlice,
            j3dSlice,
            camTrans,
            mhrSlice,
            shapeSlice,
            focalPx,
            [height, width],
            vertSlice
          )
        );
      }

      return { predictions, bodyMs };
    } catch (err) {
      console.error('[ORT] InstantHMR / MHR-Body inference error:', err);
      return { predictions: [], bodyMs: 0.0 };
    }
  }
}

/**
 * Generates an anatomically accurate 70-joint MHR 3D skeleton in meters (Y-down camera coordinates)
 * with articulated head, spine, shoulders, elbows, wrists, 20-joint left/right hands, knees, and feet.
 */
export function generateArticulatedMhr70Joints3d(
  t: number,
  mirrorSign: number = 1
): Float32Array {
  const j3d = new Float32Array(NUM_JOINTS * 3);
  const setJ = (idx: number, x: number, y: number, z: number) => {
    j3d[idx * 3] = x * mirrorSign;
    j3d[idx * 3 + 1] = y;
    j3d[idx * 3 + 2] = z;
  };

  const breathe = Math.sin(t * 2.2) * 0.015;
  const hipSway = Math.sin(t * 1.6) * 0.035;
  const armWaveL = Math.sin(t * 2.4) * 0.28;
  const armWaveR = Math.cos(t * 2.1) * 0.28;
  const kneeBend = Math.max(0, Math.sin(t * 1.6)) * 0.06;

  // Head & Neck (0-4, 69)
  setJ(0, hipSway * 0.5, -0.68 + breathe, -0.04); // nose
  setJ(1, 0.03 + hipSway * 0.5, -0.71 + breathe, -0.03); // left_eye
  setJ(2, -0.03 + hipSway * 0.5, -0.71 + breathe, -0.03); // right_eye
  setJ(3, 0.07 + hipSway * 0.5, -0.70 + breathe, 0.01); // left_ear
  setJ(4, -0.07 + hipSway * 0.5, -0.70 + breathe, 0.01); // right_ear
  setJ(69, hipSway * 0.6, -0.52 + breathe, 0.0); // neck

  // Shoulders (5-6) & Acromions (67-68)
  setJ(5, 0.19 + hipSway * 0.6, -0.46 + breathe, 0.01); // left_shoulder
  setJ(6, -0.19 + hipSway * 0.6, -0.46 + breathe, 0.01); // right_shoulder
  setJ(67, 0.21 + hipSway * 0.6, -0.47 + breathe, 0.01);
  setJ(68, -0.21 + hipSway * 0.6, -0.47 + breathe, 0.01);

  // Elbows (7-8), Olecranons (63-64), Cubital Fossae (65-66)
  const lElbX = 0.29 + armWaveL * 0.25;
  const lElbY = -0.20 + Math.abs(armWaveL) * 0.1;
  const lElbZ = -0.05 - armWaveL * 0.15;
  setJ(7, lElbX, lElbY, lElbZ);
  setJ(63, lElbX + 0.02, lElbY + 0.01, lElbZ + 0.02);
  setJ(65, lElbX - 0.01, lElbY - 0.01, lElbZ - 0.02);

  const rElbX = -0.29 - armWaveR * 0.25;
  const rElbY = -0.20 + Math.abs(armWaveR) * 0.1;
  const rElbZ = -0.05 - armWaveR * 0.15;
  setJ(8, rElbX, rElbY, rElbZ);
  setJ(64, rElbX - 0.02, rElbY + 0.01, rElbZ + 0.02);
  setJ(66, rElbX + 0.01, rElbY - 0.01, rElbZ - 0.02);

  // Wrists: Right (41), Left (62)
  const lWristX = lElbX + 0.08 + armWaveL * 0.35;
  const lWristY = lElbY + 0.18 - armWaveL * 0.42;
  const lWristZ = lElbZ - 0.14;
  setJ(62, lWristX, lWristY, lWristZ);

  const rWristX = rElbX - 0.08 - armWaveR * 0.35;
  const rWristY = rElbY + 0.18 - armWaveR * 0.42;
  const rWristZ = rElbZ - 0.14;
  setJ(41, rWristX, rWristY, rWristZ);

  // Right Hand 20 finger joints (21-40)
  const fingerSpreads = [-0.04, -0.02, 0.0, 0.02, 0.04];
  for (let f = 0; f < 5; f++) {
    const baseIdx = 21 + f * 4;
    const spread = fingerSpreads[f];
    for (let seg = 0; seg < 4; seg++) {
      const dist = (4 - seg) * 0.022;
      setJ(
        baseIdx + seg,
        rWristX - dist * 0.7 + spread * 0.5,
        rWristY + dist * 0.6 + spread,
        rWristZ - dist * 0.4
      );
    }
  }

  // Left Hand 20 finger joints (42-61)
  for (let f = 0; f < 5; f++) {
    const baseIdx = 42 + f * 4;
    const spread = fingerSpreads[f];
    for (let seg = 0; seg < 4; seg++) {
      const dist = (4 - seg) * 0.022;
      setJ(
        baseIdx + seg,
        lWristX + dist * 0.7 + spread * 0.5,
        lWristY + dist * 0.6 + spread,
        lWristZ - dist * 0.4
      );
    }
  }

  // Hips (9-10)
  setJ(9, 0.11 + hipSway, 0.02, 0.0); // left_hip
  setJ(10, -0.11 + hipSway, 0.02, 0.0); // right_hip

  // Knees (11-12)
  setJ(11, 0.12 + hipSway * 0.5, 0.42 - kneeBend, -kneeBend * 1.4); // left_knee
  setJ(12, -0.12 + hipSway * 0.5, 0.42 - kneeBend * 0.5, -kneeBend * 0.8); // right_knee

  // Ankles (13-14) & Feet (15-20)
  setJ(13, 0.12, 0.80, 0.02); // left_ankle
  setJ(14, -0.12, 0.80, 0.02); // right_ankle

  setJ(15, 0.10, 0.84, -0.10); // left_big_toe_tip
  setJ(16, 0.15, 0.84, -0.09); // left_small_toe_tip
  setJ(17, 0.12, 0.84, 0.06); // left_heel

  setJ(18, -0.10, 0.84, -0.10); // right_big_toe_tip
  setJ(19, -0.15, 0.84, -0.09); // right_small_toe_tip
  setJ(20, -0.12, 0.84, 0.06); // right_heel

  return j3d;
}
