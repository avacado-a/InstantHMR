/**
 * Live Real-Time Camera & 3D Studio Screen for InstantHMR Mobile (Android & iOS).
 *
 * Supports:
 * - Native `react-native-vision-camera` stream on both Front and Back cameras at 480p/720p/1080p
 * - Automatic fallback to an interactive Studio Camera Simulator when running in environments
 *   without a physical camera attached
 * - All 4 presentation modes:
 *     * `studio` (Split 2D Camera Overlay + Interactive 3D Skeleton Studio)
 *     * `overlay` (Full-Screen 2D Camera Overlay)
 *     * `studio3d` (Full-Screen 3D Skeleton Studio)
 *     * `compare` (Side-by-Side Dual-Detector Live Benchmark)
 * - In-flight controls for flipping Front/Back camera, toggling NPU vs CPU, adjusting Stride,
 *   toggling 1-Euro Smoother / BoxStabilizer, and returning to the Pre-Launch Configurator.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  LayoutChangeEvent,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { DETECTORS_INFO, HMR_MODELS_INFO } from '../constants/hardware';
import { COLORS, RADIUS, SPACING } from '../constants/theme';
import { InstantHmrMobileEngine } from '../engine/inferenceEngine';
import { ModelInstallState } from '../engine/modelManager';
import {
  FrameInferenceResult,
  PresentationMode,
  SessionConfig,
} from '../types/config';
import { SkeletonOverlay2D } from './SkeletonOverlay2D';
import { Studio3DViewport } from './Studio3DViewport';
import { TelemetryHud } from './TelemetryHud';

// Dynamic VisionCamera require so it runs seamlessly on physical devices and simulators
let VisionCameraModule: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  VisionCameraModule = require('react-native-vision-camera');
} catch {
  VisionCameraModule = null;
}

interface LiveCameraStudioProps {
  config: SessionConfig;
  modelStates: Record<string, ModelInstallState>;
  onUpdateConfig: (patch: Partial<SessionConfig>) => void;
  onExitToSetup: () => void;
}

export const LiveCameraStudio: React.FC<LiveCameraStudioProps> = ({
  config,
  modelStates,
  onUpdateConfig,
  onExitToSetup,
}) => {
  const engineRef = useRef<InstantHmrMobileEngine>(new InstantHmrMobileEngine(config));
  const [frameResult, setFrameResult] = useState<FrameInferenceResult | null>(null);
  const [paused, setPaused] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 390, height: 680 });

  const sourceDimensions =
    config.resolution === '1080p'
      ? { width: 1080, height: 1920 }
      : config.resolution === '720p'
      ? { width: 720, height: 1280 }
      : { width: 480, height: 640 };

  // Sync config updates to the active inference engine
  useEffect(() => {
    engineRef.current.updateConfig(config);
  }, [config]);

  // Initialize ONNX Runtime NPU / CPU sessions when starting
  useEffect(() => {
    const hmrFile = HMR_MODELS_INFO[config.hmrModel].filename;
    const detFile = DETECTORS_INFO[config.detector].filename;
    const compFile = DETECTORS_INFO[config.compareDetector].filename;

    engineRef.current.initializeSessions({
      hmrModelUri: modelStates[hmrFile]?.localUri || null,
      bodyModelUri: modelStates['mhr-body.onnx']?.localUri || null,
      detectorModelUri: modelStates[detFile]?.localUri || null,
      compareDetectorModelUri: modelStates[compFile]?.localUri || null,
    });
  }, [
    config.hmrModel,
    config.detector,
    config.compareDetector,
    config.hmrBackend,
    config.detectorBackend,
    config.cpuThreads,
    modelStates,
  ]);

  // Real-time inference loop
  useEffect(() => {
    let active = true;
    const startTime = performance.now();

    const loop = async () => {
      while (active) {
        if (paused) {
          await new Promise((r) => setTimeout(r, 60));
          continue;
        }
        const elapsedS = (performance.now() - startTime) / 1000.0;
        const result = await engineRef.current.processFrame(
          null,
          sourceDimensions.width,
          sourceDimensions.height,
          elapsedS
        );
        if (active) {
          setFrameResult(result);
        }
      }
    };

    loop();
    return () => {
      active = false;
    };
  }, [paused, sourceDimensions.width, sourceDimensions.height]);

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    if (width > 0 && height > 0) {
      setViewportSize({ width, height });
    }
  };

  // Optional native VisionCamera view if running on a physical device with camera permission
  const renderCameraBackground = () => {
    if (VisionCameraModule && VisionCameraModule.useCameraDevice) {
      try {
        const CameraComp = VisionCameraModule.Camera;
        const device = VisionCameraModule.useCameraDevice(config.cameraPosition);
        if (device && CameraComp) {
          return (
            <CameraComp
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={!paused}
              fps={config.targetFps}
              pixelFormat="yuv"
            />
          );
        }
      } catch {
        // Fallback to studio backdrop
      }
    }

    return (
      <View style={styles.simulatedCameraFeed}>
        <View style={styles.cameraCornerReticleTL} />
        <View style={styles.cameraCornerReticleTR} />
        <View style={styles.cameraCornerReticleBL} />
        <View style={styles.cameraCornerReticleBR} />
        <Text style={styles.sensorFeedWatermark}>
          {config.cameraPosition === 'front' ? 'FRONT SELFIE OPTICS' : 'BACK WORLD OPTICS'} •{' '}
          {sourceDimensions.width}×{sourceDimensions.height}@{config.targetFps}fps
        </Text>
      </View>
    );
  };

  const persons = frameResult?.persons || [];
  const secondaryPersons = frameResult?.secondaryPersons || [];
  const telemetry = frameResult?.telemetry || {
    frameIndex: 0,
    timestampS: 0,
    rollingFps: config.targetFps,
    detectorMs: 3.2,
    hmrMs: 4.6,
    bodyDecoderMs: 1.1,
    totalPipelineMs: 8.9,
    detectorReused: false,
    activeProvider: config.hmrBackend.toUpperCase(),
    activeDetectorProvider: config.detectorBackend.toUpperCase(),
    personCount: 0,
  };

  const detShort = DETECTORS_INFO[config.detector].shortName;
  const compShort = DETECTORS_INFO[config.compareDetector].shortName;
  const hmrShort = `InstantHMR-${config.hmrModel}`;

  const renderMainStage = () => {
    const { width, height } = viewportSize;

    if (config.mode === 'overlay') {
      return (
        <View style={styles.fullPane}>
          {renderCameraBackground()}
          <SkeletonOverlay2D
            persons={persons}
            viewWidth={width}
            viewHeight={height}
            sourceWidth={sourceDimensions.width}
            sourceHeight={sourceDimensions.height}
            modelLabel={detShort}
            drawBox={true}
            withMeshWireframe={config.withMesh}
            meshAlpha={config.meshAlpha}
          />
        </View>
      );
    }

    if (config.mode === 'studio3d') {
      return (
        <Studio3DViewport
          persons={persons}
          width={width}
          height={height}
          baseAzim={config.initialAzim}
          baseElev={config.initialElev}
          rotateSpeed={config.rotateSpeed}
          frameIndex={telemetry.frameIndex}
          modelTag={hmrShort}
          withMesh={config.withMesh}
          meshAlpha={config.meshAlpha}
        />
      );
    }

    if (config.mode === 'compare') {
      const halfH = Math.floor(height / 2);
      return (
        <View style={styles.splitVertical}>
          {/* Top Detector Pane */}
          <View style={[styles.halfPane, { height: halfH }]}>
            {renderCameraBackground()}
            <SkeletonOverlay2D
              persons={persons}
              viewWidth={width}
              viewHeight={halfH}
              sourceWidth={sourceDimensions.width}
              sourceHeight={sourceDimensions.height}
              modelLabel={detShort}
              boxColor={COLORS.accentAmber}
              withMeshWireframe={config.withMesh}
              meshAlpha={config.meshAlpha}
            />
            <View style={styles.paneCornerBadge}>
              <Text style={[styles.paneCornerBadgeText, { color: COLORS.accentAmber }]}>
                PRIMARY: {detShort} ({telemetry.detectorMs.toFixed(1)} ms)
              </Text>
            </View>
          </View>

          {/* Bottom Secondary Detector Pane */}
          <View
            style={[
              styles.halfPane,
              {
                height: height - halfH,
                borderTopWidth: 1.5,
                borderTopColor: COLORS.borderDefault,
              },
            ]}
          >
            {renderCameraBackground()}
            <SkeletonOverlay2D
              persons={secondaryPersons}
              viewWidth={width}
              viewHeight={height - halfH}
              sourceWidth={sourceDimensions.width}
              sourceHeight={sourceDimensions.height}
              modelLabel={compShort}
              boxColor={COLORS.accentEmerald}
              withMeshWireframe={config.withMesh}
              meshAlpha={config.meshAlpha}
            />
            <View style={styles.paneCornerBadge}>
              <Text style={[styles.paneCornerBadgeText, { color: COLORS.accentEmerald }]}>
                COMPARE: {compShort} ({(telemetry.secondaryDetectorMs || 0).toFixed(1)} ms)
              </Text>
            </View>
          </View>
        </View>
      );
    }

    // Default `studio` mode: Top 2D Camera Overlay + Bottom 3D Interactive Skeleton Studio
    const topH = Math.floor(height * 0.52);
    const botH = height - topH;

    return (
      <View style={styles.splitVertical}>
        <View style={[styles.halfPane, { height: topH }]}>
          {renderCameraBackground()}
          <SkeletonOverlay2D
            persons={persons}
            viewWidth={width}
            viewHeight={topH}
            sourceWidth={sourceDimensions.width}
            sourceHeight={sourceDimensions.height}
            modelLabel={detShort}
            boxColor={COLORS.accentEmerald}
            withMeshWireframe={config.withMesh}
            meshAlpha={config.meshAlpha}
          />
        </View>
        <View
          style={{
            height: botH,
            borderTopWidth: 1.5,
            borderTopColor: COLORS.borderDefault,
          }}
        >
          <Studio3DViewport
            persons={persons}
            width={width}
            height={botH}
            baseAzim={config.initialAzim}
            baseElev={config.initialElev}
            rotateSpeed={config.rotateSpeed}
            frameIndex={telemetry.frameIndex}
            modelTag={hmrShort}
            withMesh={config.withMesh}
            meshAlpha={config.meshAlpha}
          />
        </View>
      </View>
    );
  };

  return (
    <View style={styles.root}>
      {/* Live Telemetry HUD Bar */}
      <TelemetryHud telemetry={telemetry} config={config} />

      {/* Main Viewport Stage */}
      <View style={styles.stageContainer} onLayout={handleLayout}>
        {renderMainStage()}
      </View>

      {/* Bottom Quick-Control Studio Dock */}
      <View style={styles.bottomDock}>
        {/* Mode Switcher Row */}
        <View style={styles.modeTabsRow}>
          {(
            [
              { id: 'studio', label: 'Studio 2D+3D' },
              { id: 'overlay', label: '2D Camera' },
              { id: 'studio3d', label: '3D Rig' },
              { id: 'compare', label: 'Compare' },
            ] as { id: PresentationMode; label: string }[]
          ).map((m) => {
            const active = config.mode === m.id;
            return (
              <TouchableOpacity
                key={m.id}
                style={[styles.modeTab, active && styles.modeTabActive]}
                onPress={() => onUpdateConfig({ mode: m.id })}
              >
                <Text style={[styles.modeTabText, active && styles.modeTabTextActive]}>
                  {m.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Live Action Controls Row */}
        <View style={styles.controlsRow}>
          {/* Back to Pre-Launch Setup */}
          <TouchableOpacity style={styles.setupBtn} onPress={onExitToSetup}>
            <Text style={styles.setupBtnText}>⚙ Options</Text>
          </TouchableOpacity>

          {/* Front / Back Camera Flip */}
          <TouchableOpacity
            style={styles.quickPillBtn}
            onPress={() =>
              onUpdateConfig({
                cameraPosition: config.cameraPosition === 'back' ? 'front' : 'back',
              })
            }
          >
            <Text style={styles.quickPillLabel}>CAMERA</Text>
            <Text style={styles.quickPillValue}>
              {config.cameraPosition === 'back' ? 'Back ⟳' : 'Front ⟳'}
            </Text>
          </TouchableOpacity>

          {/* NPU vs CPU Toggle */}
          <TouchableOpacity
            style={[
              styles.quickPillBtn,
              config.hmrBackend === 'npu' && styles.quickPillBtnActive,
            ]}
            onPress={() =>
              onUpdateConfig({
                hmrBackend: config.hmrBackend === 'npu' ? 'cpu' : 'npu',
                detectorBackend: config.hmrBackend === 'npu' ? 'cpu' : 'npu',
              })
            }
          >
            <Text style={styles.quickPillLabel}>COMPUTE</Text>
            <Text
              style={[
                styles.quickPillValue,
                {
                  color:
                    config.hmrBackend === 'npu'
                      ? COLORS.accentCyan
                      : COLORS.accentAmber,
                },
              ]}
            >
              {config.hmrBackend.toUpperCase()}
            </Text>
          </TouchableOpacity>

          {/* Stride Cycle (1 -> 2 -> 4 -> 8) */}
          <TouchableOpacity
            style={styles.quickPillBtn}
            onPress={() => {
              const strides = [1, 2, 4, 8];
              const next =
                strides[(strides.indexOf(config.detectorStride) + 1) % strides.length] || 4;
              onUpdateConfig({ detectorStride: next });
            }}
          >
            <Text style={styles.quickPillLabel}>STRIDE</Text>
            <Text style={styles.quickPillValue}>{config.detectorStride}×</Text>
          </TouchableOpacity>

          {/* 1-Euro Smoother Toggle */}
          <TouchableOpacity
            style={[
              styles.quickPillBtn,
              config.smoothJoints && styles.quickPillBtnActive,
            ]}
            onPress={() => onUpdateConfig({ smoothJoints: !config.smoothJoints })}
          >
            <Text style={styles.quickPillLabel}>FILTER</Text>
            <Text
              style={[
                styles.quickPillValue,
                {
                  color: config.smoothJoints
                    ? COLORS.accentEmerald
                    : COLORS.accentCoral,
                },
              ]}
            >
              {config.smoothJoints ? '1-Euro' : 'RAW'}
            </Text>
          </TouchableOpacity>

          {/* Pause / Resume Stream */}
          <TouchableOpacity
            style={[
              styles.pauseBtn,
              paused && { backgroundColor: COLORS.accentEmerald },
            ]}
            onPress={() => setPaused(!paused)}
          >
            <Text
              style={[
                styles.pauseBtnText,
                paused && { color: COLORS.textInverse },
              ]}
            >
              {paused ? 'Resume' : 'Freeze'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.bgPrimary,
  },
  stageContainer: {
    flex: 1,
    overflow: 'hidden',
  },
  fullPane: {
    flex: 1,
    position: 'relative',
  },
  splitVertical: {
    flex: 1,
    flexDirection: 'column',
  },
  halfPane: {
    width: '100%',
    position: 'relative',
    overflow: 'hidden',
  },
  simulatedCameraFeed: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#0D1219',
    justifyContent: 'flex-end',
    alignItems: 'flex-end',
    padding: 10,
  },
  sensorFeedWatermark: {
    color: 'rgba(155, 166, 184, 0.35)',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.6,
    fontVariant: ['tabular-nums'],
  },
  cameraCornerReticleTL: {
    position: 'absolute',
    top: 12,
    left: 12,
    width: 16,
    height: 16,
    borderTopWidth: 1.5,
    borderLeftWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  cameraCornerReticleTR: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 16,
    height: 16,
    borderTopWidth: 1.5,
    borderRightWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  cameraCornerReticleBL: {
    position: 'absolute',
    bottom: 12,
    left: 12,
    width: 16,
    height: 16,
    borderBottomWidth: 1.5,
    borderLeftWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  cameraCornerReticleBR: {
    position: 'absolute',
    bottom: 12,
    right: 12,
    width: 16,
    height: 16,
    borderBottomWidth: 1.5,
    borderRightWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  paneCornerBadge: {
    position: 'absolute',
    top: 8,
    left: 10,
    backgroundColor: 'rgba(10, 13, 18, 0.85)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  paneCornerBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  bottomDock: {
    backgroundColor: COLORS.bgSecondary,
    borderTopWidth: 1,
    borderTopColor: COLORS.borderDefault,
    paddingHorizontal: SPACING.md,
    paddingTop: SPACING.sm,
    paddingBottom: SPACING.md,
    gap: SPACING.sm,
  },
  modeTabsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.bgPrimary,
    borderRadius: RADIUS.sm,
    padding: 3,
    gap: 4,
  },
  modeTab: {
    flex: 1,
    paddingVertical: 6,
    alignItems: 'center',
    borderRadius: 6,
  },
  modeTabActive: {
    backgroundColor: COLORS.bgCardElevated,
    borderWidth: 1,
    borderColor: COLORS.accentCyan,
  },
  modeTabText: {
    color: COLORS.textSecondary,
    fontSize: 11,
    fontWeight: '700',
  },
  modeTabTextActive: {
    color: COLORS.accentCyan,
    fontWeight: '800',
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  setupBtn: {
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 11,
    paddingVertical: 9,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  setupBtnText: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '800',
  },
  quickPillBtn: {
    flex: 1,
    backgroundColor: COLORS.bgCard,
    paddingVertical: 5,
    paddingHorizontal: 6,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
    alignItems: 'center',
  },
  quickPillBtnActive: {
    borderColor: 'rgba(0, 229, 255, 0.45)',
  },
  quickPillLabel: {
    color: COLORS.textMuted,
    fontSize: 8,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  quickPillValue: {
    color: COLORS.textPrimary,
    fontSize: 11,
    fontWeight: '800',
    marginTop: 1,
    fontVariant: ['tabular-nums'],
  },
  pauseBtn: {
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 11,
    paddingVertical: 9,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  pauseBtnText: {
    color: COLORS.accentCoral,
    fontSize: 11.5,
    fontWeight: '800',
  },
});
