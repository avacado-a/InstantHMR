/**
 * Pre-Launch Studio Configuration Dashboard for InstantHMR Mobile (Android & iOS).
 *
 * Exposes EVERY InstantHMR pipeline option before starting real-time camera execution:
 * 1. Camera Optics & Resolution: Front / Back camera, Mirroring, 480p / 720p / 1080p, 30 / 60 FPS, Focal length.
 * 2. Hardware Compute Accelerator: Mobile NPU (CoreML ANE / Android NNAPI QNN), Mobile GPU, or Mobile CPU (XNNPACK SIMD) + Thread count.
 * 3. InstantHMR Model Size: Nano (20.0M), Tiny (20.3M), Small (24.4M), Medium (32.2M), Medium-Commercial + 18,439-vertex 3D Mesh toggle & Crop Expand.
 * 4. Person Detector & Stride: Hardware-aware detector cards (`yolo26n`, `yolo26s`, `yolo26n-pose`, `yolo26s-pose`, `rfdetr_nano`, `yolov8n`), Detector Stride (1..12), Confidence threshold, Max Persons, Pose Envelope.
 * 5. Temporal Smoothing & Stabilization: 1-Euro Filter (`smooth`), Beta Scale, and `BoxStabilizer`.
 * 6. Studio Presentation Mode: `studio` (2D+3D Split), `overlay` (2D Full), `studio3d` (3D Full), `compare` (Dual Detector Benchmark) + 3D Orbit speed & Azimuth.
 */
import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  DETECTORS_INFO,
  DeviceHardwareProfile,
  HMR_MODELS_INFO,
} from '../constants/hardware';
import { useCameraPermission } from 'react-native-vision-camera';
import { COLORS, RADIUS, SPACING } from '../constants/theme';
import { ModelInstallState } from '../engine/modelManager';
import {
  CameraPosition,
  CameraResolutionPreset,
  ComputeBackend,
  DEFAULT_SESSION_CONFIG,
  DetectorVariant,
  HmrModelVariant,
  PresentationMode,
  SessionConfig,
  TargetFpsPreset,
} from '../types/config';

interface PreLaunchConfiguratorProps {
  config: SessionConfig;
  hardware: DeviceHardwareProfile;
  modelStates: Record<string, ModelInstallState>;
  requiredFiles: string[];
  onChangeConfig: (patch: Partial<SessionConfig>) => void;
  onResetDefaults: () => void;
  onOpenModelManager: () => void;
  onStartRealtimeSession: () => void;
}

export const PreLaunchConfigurator: React.FC<PreLaunchConfiguratorProps> = ({
  config,
  hardware,
  modelStates,
  requiredFiles,
  onChangeConfig,
  onResetDefaults,
  onOpenModelManager,
  onStartRealtimeSession,
}) => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const selectedModelInfo = HMR_MODELS_INFO[config.hmrModel];
  const selectedDetectorInfo = DETECTORS_INFO[config.detector];
  const isNpu = config.hmrBackend === 'npu' || config.hmrBackend === 'auto';

  // Estimate end-to-end latency per frame given current stride, backend, model, and detector
  const estDetMs =
    (isNpu ? selectedDetectorInfo.estimatedMsNpu : selectedDetectorInfo.estimatedMsCpu) /
    Math.max(1, config.detectorStride);
  const estHmrMs = isNpu ? selectedModelInfo.s23NpuMs : selectedModelInfo.cpuMs;
  const estBodyMs = config.withMesh ? 2.4 : 1.1;
  const estTotalMs = estDetMs + estHmrMs + estBodyMs;
  const estFps = Math.min(config.targetFps, Math.round(1000 / Math.max(1, estTotalMs)));

  const readyCount = requiredFiles.filter((f) => modelStates[f]?.installed).length;

  return (
    <View style={styles.root}>
      {/* Sticky Studio Header */}
      <View style={styles.topHeader}>
        <View>
          <View style={styles.titleRow}>
            <Text style={styles.appTitle}>InstantHMR</Text>
            <View style={styles.versionPill}>
              <Text style={styles.versionPillText}>MOBILE STUDIO v2.0</Text>
            </View>
          </View>
          <Text style={styles.deviceSub}>
            {hardware.deviceName} • {hardware.npuName}
          </Text>
        </View>

        <View style={styles.headerRightActions}>
          <TouchableOpacity
            style={styles.modelsBtn}
            onPress={onOpenModelManager}
          >
            <View
              style={[
                styles.statusDot,
                {
                  backgroundColor:
                    readyCount === requiredFiles.length
                      ? COLORS.accentEmerald
                      : COLORS.accentAmber,
                },
              ]}
            />
            <Text style={styles.modelsBtnText}>
              Models ({readyCount}/{requiredFiles.length})
            </Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.resetBtn} onPress={onResetDefaults}>
            <Text style={styles.resetBtnText}>Reset</Text>
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Live Hardware & Performance Readiness Card */}
        <View style={styles.telemetryPreviewCard}>
          <View style={styles.telemetryTopRow}>
            <View>
              <Text style={styles.telemetryCardLabel}>ESTIMATED REAL-TIME THROUGHPUT</Text>
              <Text style={styles.telemetryCardMain}>
                ~{estFps} FPS ({estTotalMs.toFixed(1)} ms/frame)
              </Text>
            </View>
            <View style={styles.npuStatusChip}>
              <Text style={styles.npuStatusChipText}>
                {isNpu ? 'NPU READY' : 'CPU SIMD'}
              </Text>
            </View>
          </View>

          <View style={styles.telemetryBreakdownRow}>
            <View style={styles.telemetryMetricItem}>
              <Text style={styles.telemetryMetricTitle}>Camera Input</Text>
              <Text style={styles.telemetryMetricVal}>
                {config.cameraPosition === 'front' ? 'Front' : 'Back'} • {config.resolution}@{config.targetFps}
              </Text>
            </View>
            <View style={styles.telemetryMetricItem}>
              <Text style={styles.telemetryMetricTitle}>Avg Detector / Frame</Text>
              <Text style={[styles.telemetryMetricVal, { color: COLORS.accentAmber }]}>
                {estDetMs.toFixed(1)} ms (Stride {config.detectorStride})
              </Text>
            </View>
            <View style={styles.telemetryMetricItem}>
              <Text style={styles.telemetryMetricTitle}>InstantHMR + MHR70</Text>
              <Text style={[styles.telemetryMetricVal, { color: COLORS.accentCyan }]}>
                {(estHmrMs + estBodyMs).toFixed(1)} ms
              </Text>
            </View>
          </View>
        </View>

        {/* SECTION 1: CAMERA OPTICS & STREAM */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>01</Text>
            <View>
              <Text style={styles.sectionTitle}>Camera Sensor & Stream Size</Text>
              <Text style={styles.sectionDesc}>
                Select Front or Back camera, stream resolution, frame rate, and focal calibration
              </Text>
            </View>
          </View>

          {!hasPermission && (
            <View style={styles.permissionPromptCard}>
              <View style={{ flex: 1 }}>
                <Text style={styles.permissionPromptTitle}>📷 Camera Access Required</Text>
                <Text style={styles.permissionPromptSub}>
                  Camera permission is needed for live 3D pose estimation and mesh recovery.
                </Text>
              </View>
              <TouchableOpacity
                style={styles.permissionPromptBtn}
                onPress={() => requestPermission()}
              >
                <Text style={styles.permissionPromptBtnText}>Authorize</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Front vs Back Camera Selector */}
          <Text style={styles.fieldLabel}>Active Camera Sensor</Text>
          <View style={styles.twoColGrid}>
            {(['back', 'front'] as CameraPosition[]).map((pos) => {
              const active = config.cameraPosition === pos;
              return (
                <TouchableOpacity
                  key={pos}
                  style={[styles.choiceCard, active && styles.choiceCardActive]}
                  onPress={() => onChangeConfig({ cameraPosition: pos })}
                >
                  <View style={styles.choiceHeaderRow}>
                    <Text style={[styles.choiceTitle, active && styles.choiceTitleActive]}>
                      {pos === 'back' ? 'Back Camera (World)' : 'Front Camera (Selfie)'}
                    </Text>
                    <View style={[styles.radioDot, active && styles.radioDotActive]} />
                  </View>
                  <Text style={styles.choiceSub}>
                    {pos === 'back'
                      ? 'Wide-angle primary lens • Multi-person room tracking'
                      : 'Front-facing lens • Mirrored self-pose & workout studio'}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Resolution / Input Stream Size */}
          <Text style={styles.fieldLabel}>Camera Stream Size & Target Frame Rate</Text>
          <View style={styles.segmentedRow}>
            {(
              [
                { id: '480p', label: '480p Fast (640×480)' },
                { id: '720p', label: '720p HD (1280×720)' },
                { id: '1080p', label: '1080p FHD (1920×1080)' },
              ] as { id: CameraResolutionPreset; label: string }[]
            ).map((res) => {
              const active = config.resolution === res.id;
              return (
                <TouchableOpacity
                  key={res.id}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ resolution: res.id })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {res.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          <View style={[styles.segmentedRow, { marginTop: SPACING.sm }]}>
            {([30, 60] as TargetFpsPreset[]).map((fps) => {
              const active = config.targetFps === fps;
              return (
                <TouchableOpacity
                  key={fps}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ targetFps: fps })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {fps} FPS Target Capture
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {config.cameraPosition === 'front' && (
            <View style={styles.toggleRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.toggleTitle}>Mirror Front Camera Coordinates</Text>
                <Text style={styles.toggleSub}>
                  Horizontally flips 2D & 3D skeleton to act like a natural mirror
                </Text>
              </View>
              <Switch
                value={config.mirrorFrontCamera}
                onValueChange={(v) => onChangeConfig({ mirrorFrontCamera: v })}
                trackColor={{ false: COLORS.borderDefault, true: COLORS.accentCyan }}
              />
            </View>
          )}
        </View>

        {/* SECTION 2: HARDWARE ACCELERATION (MOBILE NPU vs CPU) */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>02</Text>
            <View>
              <Text style={styles.sectionTitle}>Hardware Compute Backend (NPU / GPU / CPU)</Text>
              <Text style={styles.sectionDesc}>
                Configure ONNX Runtime execution providers for InstantHMR and the Detector
              </Text>
            </View>
          </View>

          <Text style={styles.fieldLabel}>InstantHMR 3D Network Accelerator</Text>
          <View style={styles.verticalStack}>
            {hardware.accelerators.map((acc) => {
              const active = config.hmrBackend === acc.id;
              return (
                <TouchableOpacity
                  key={acc.id}
                  style={[
                    styles.acceleratorCard,
                    active && styles.acceleratorCardActive,
                  ]}
                  onPress={() =>
                    onChangeConfig({
                      hmrBackend: acc.id,
                      detectorBackend: acc.id,
                    })
                  }
                >
                  <View style={styles.accTopRow}>
                    <View style={styles.accTitleGroup}>
                      <Text style={[styles.accName, active && styles.accNameActive]}>
                        {acc.name}
                      </Text>
                      {acc.isNpu && (
                        <View style={styles.npuPill}>
                          <Text style={styles.npuPillText}>NPU</Text>
                        </View>
                      )}
                    </View>
                    <Text style={styles.accLatencyTag}>
                      ~{acc.estimatedLatencyMs[config.hmrModel].toFixed(1)} ms / person
                    </Text>
                  </View>
                  <Text style={styles.accProviderText}>{acc.providerName}</Text>
                  <Text style={styles.accDescText}>{acc.description}</Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Independent Detector Backend Selector */}
          <Text style={styles.fieldLabel}>Detector Execution Device</Text>
          <View style={styles.segmentedRow}>
            {(['npu', 'gpu', 'cpu', 'auto'] as ComputeBackend[]).map((b) => {
              const active = config.detectorBackend === b;
              return (
                <TouchableOpacity
                  key={b}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ detectorBackend: b })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {b.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* CPU Thread Pool Selector when CPU is used */}
          {(config.hmrBackend === 'cpu' || config.detectorBackend === 'cpu') && (
            <>
              <Text style={styles.fieldLabel}>
                XNNPACK Mobile CPU Threads ({config.cpuThreads} Threads)
              </Text>
              <View style={styles.segmentedRow}>
                {[1, 2, 4, 6, 8].map((t) => {
                  const active = config.cpuThreads === t;
                  return (
                    <TouchableOpacity
                      key={t}
                      style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                      onPress={() => onChangeConfig({ cpuThreads: t })}
                    >
                      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                        {t}T
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </>
          )}
        </View>

        {/* SECTION 3: INSTANTHMR MODEL SIZE & MESH OPTIONS */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>03</Text>
            <View>
              <Text style={styles.sectionTitle}>InstantHMR Model Size & 3D Mesh</Text>
              <Text style={styles.sectionDesc}>
                Choose between Nano, Tiny, Small, Medium, or Commercial (288×288 input crop)
              </Text>
            </View>
          </View>

          <View style={styles.verticalStack}>
            {hardware.models.map((m) => {
              const active = config.hmrModel === m.id;
              return (
                <TouchableOpacity
                  key={m.id}
                  disabled={!m.supportedOnDevice}
                  style={[
                    styles.modelOptionCard,
                    active && styles.modelOptionCardActive,
                    !m.supportedOnDevice && styles.cardDisabled,
                  ]}
                  onPress={() => onChangeConfig({ hmrModel: m.id as HmrModelVariant })}
                >
                  <View style={styles.modelOptTop}>
                    <View style={styles.accTitleGroup}>
                      <Text style={[styles.accName, active && styles.accNameActive]}>
                        {m.name}
                      </Text>
                      {m.recommended && (
                        <View style={styles.recPill}>
                          <Text style={styles.recPillText}>RECOMMENDED</Text>
                        </View>
                      )}
                    </View>
                    <Text style={styles.modelOptMetrics}>
                      {m.parametersM.toFixed(1)}M • {m.s23NpuMs.toFixed(2)}ms NPU
                    </Text>
                  </View>
                  <Text style={styles.accDescText}>{m.description}</Text>
                  <View style={styles.modelStatsPills}>
                    <Text style={styles.statTag}>3DPW: {m.threeDpwMm} mm</Text>
                    <Text style={styles.statTag}>EMDB-1: {m.emdb1Mm} mm</Text>
                    <Text style={styles.statTag}>{m.sizeMb} MB</Text>
                    <Text style={styles.statTag}>{m.license}</Text>
                  </View>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Crop Expansion Ratio */}
          <Text style={styles.fieldLabel}>
            CLIFF Bounding Box Crop Expansion ({config.cropExpand.toFixed(2)}×)
          </Text>
          <View style={styles.segmentedRow}>
            {[1.1, 1.2, 1.3, 1.4, 1.5].map((ratio) => {
              const active = Math.abs(config.cropExpand - ratio) < 0.02;
              return (
                <TouchableOpacity
                  key={ratio}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ cropExpand: ratio })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {ratio.toFixed(1)}×
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Full 18,439-Vertex MHR Mesh Toggle */}
          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.toggleTitle}>
                Reconstruct Full 18,439-Vertex 3D Body Mesh
              </Text>
              <Text style={styles.toggleSub}>
                Disable (`--no-mesh`) for fastest 70-keypoint 3D skeleton recovery (+1.3 ms when enabled)
              </Text>
            </View>
            <Switch
              value={config.withMesh}
              onValueChange={(v) => onChangeConfig({ withMesh: v })}
              trackColor={{ false: COLORS.borderDefault, true: COLORS.accentCyan }}
            />
          </View>
        </View>

        {/* SECTION 4: PERSON DETECTOR, STRIDE, & HARDWARE COMPATIBILITY */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>04</Text>
            <View>
              <Text style={styles.sectionTitle}>Person Detector & Stride Schedule</Text>
              <Text style={styles.sectionDesc}>
                Select hardware-compatible detector and frame stride (reuses stabilized boxes between passes)
              </Text>
            </View>
          </View>

          {/* Detector Stride Selector */}
          <Text style={styles.fieldLabel}>
            Detector Frame Stride: Every {config.detectorStride}{' '}
            {config.detectorStride === 1 ? 'Frame (Per-Frame)' : 'Frames'}
          </Text>
          <View style={styles.segmentedRow}>
            {[1, 2, 3, 4, 6, 8, 12].map((s) => {
              const active = config.detectorStride === s;
              return (
                <TouchableOpacity
                  key={s}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ detectorStride: s })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {s === 1 ? '1 (Every)' : `${s}×`}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Detector Variant Cards with Hardware Support Indicator */}
          <Text style={styles.fieldLabel}>Primary Person Detector</Text>
          <View style={styles.verticalStack}>
            {hardware.detectors.map((det) => {
              const active = config.detector === det.id;
              return (
                <TouchableOpacity
                  key={det.id}
                  disabled={!det.supportedOnDevice}
                  style={[
                    styles.modelOptionCard,
                    active && styles.modelOptionCardActive,
                    !det.supportedOnDevice && styles.cardDisabled,
                  ]}
                  onPress={() =>
                    onChangeConfig({
                      detector: det.id as DetectorVariant,
                      usePoseEnvelope: det.hasPoseEnvelope,
                    })
                  }
                >
                  <View style={styles.modelOptTop}>
                    <View style={styles.accTitleGroup}>
                      <Text style={[styles.accName, active && styles.accNameActive]}>
                        {det.name}
                      </Text>
                      <View
                        style={[
                          styles.hwBadge,
                          det.supportedOnNpu ? styles.hwBadgeNpu : styles.hwBadgeCpu,
                        ]}
                      >
                        <Text
                          style={[
                            styles.hwBadgeText,
                            {
                              color: det.supportedOnNpu
                                ? COLORS.accentEmerald
                                : COLORS.accentAmber,
                            },
                          ]}
                        >
                          {det.supportedOnNpu ? 'NPU + CPU' : 'CPU / GPU'}
                        </Text>
                      </View>
                    </View>
                    <Text style={styles.modelOptMetrics}>
                      ~{det.estimatedMsNpu.toFixed(1)}ms NPU • {det.sizeMb}MB
                    </Text>
                  </View>
                  <Text style={styles.accDescText}>{det.description}</Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Confidence & Max Persons */}
          <Text style={styles.fieldLabel}>
            Detector Confidence Threshold ({Math.round(config.detConfidence * 100)}%)
          </Text>
          <View style={styles.segmentedRow}>
            {[0.25, 0.35, 0.4, 0.5, 0.65].map((c) => {
              const active = Math.abs(config.detConfidence - c) < 0.02;
              return (
                <TouchableOpacity
                  key={c}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ detConfidence: c })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {Math.round(c * 100)}%
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          <Text style={styles.fieldLabel}>
            Max Tracked Persons Per Frame ({config.maxPersons})
          </Text>
          <View style={styles.segmentedRow}>
            {[1, 2, 3, 4].map((mp) => {
              const active = config.maxPersons === mp;
              return (
                <TouchableOpacity
                  key={mp}
                  style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                  onPress={() => onChangeConfig({ maxPersons: mp })}
                >
                  <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                    {mp} {mp === 1 ? 'Person' : 'Persons'}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* SECTION 5: TEMPORAL SMOOTHING & BOX STABILIZATION */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>05</Text>
            <View>
              <Text style={styles.sectionTitle}>1-Euro Temporal Filter & Box Stabilizer</Text>
              <Text style={styles.sectionDesc}>
                Zero-lag motion filtering (Casiez CHI 2012) and inter-stride bounding box damping
              </Text>
            </View>
          </View>

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.toggleTitle}>1-Euro Joint Smoother (--smooth)</Text>
              <Text style={styles.toggleSub}>
                Unit-calibrated cutoff (beta=0.02 px, beta=4.0 m) eliminating high-frequency jitter with zero lag
              </Text>
            </View>
            <Switch
              value={config.smoothJoints}
              onValueChange={(v) => onChangeConfig({ smoothJoints: v })}
              trackColor={{ false: COLORS.borderDefault, true: COLORS.accentCyan }}
            />
          </View>

          {config.smoothJoints && (
            <>
              <Text style={styles.fieldLabel}>
                1-Euro Responsiveness Beta Scale ({config.betaScale.toFixed(1)}×)
              </Text>
              <View style={styles.segmentedRow}>
                {[0.5, 0.8, 1.0, 1.5, 2.0].map((bs) => {
                  const active = Math.abs(config.betaScale - bs) < 0.05;
                  return (
                    <TouchableOpacity
                      key={bs}
                      style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                      onPress={() => onChangeConfig({ betaScale: bs })}
                    >
                      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                        {bs.toFixed(1)}×
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </>
          )}

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.toggleTitle}>Bounding Box Stabilizer (--stabilize-box)</Text>
              <Text style={styles.toggleSub}>
                Smoothly holds & filters detector bounding boxes across stride frames
              </Text>
            </View>
            <Switch
              value={config.stabilizeBox}
              onValueChange={(v) => onChangeConfig({ stabilizeBox: v })}
              trackColor={{ false: COLORS.borderDefault, true: COLORS.accentCyan }}
            />
          </View>
        </View>

        {/* SECTION 6: STUDIO PRESENTATION LAYOUT & HUD */}
        <View style={styles.sectionCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIndex}>06</Text>
            <View>
              <Text style={styles.sectionTitle}>Presentation Mode & Telemetry HUD</Text>
              <Text style={styles.sectionDesc}>
                Choose how 2D camera overlays and the 3D Skeleton Studio are arranged
              </Text>
            </View>
          </View>

          <View style={styles.twoColGrid}>
            {(
              [
                {
                  id: 'studio',
                  title: 'Studio Split (2D + 3D)',
                  sub: 'Live 2D camera overlay + interactive 3D MHR70 studio',
                },
                {
                  id: 'overlay',
                  title: 'Full 2D Overlay',
                  sub: 'Full-screen camera with 70 keypoints & depth tags',
                },
                {
                  id: 'studio3d',
                  title: 'Full 3D Studio',
                  sub: 'Full-screen interactive 3D skeleton & floor grid',
                },
                {
                  id: 'compare',
                  title: 'Dual Detector Compare',
                  sub: 'Real-time benchmark comparing two detectors side-by-side',
                },
              ] as { id: PresentationMode; title: string; sub: string }[]
            ).map((m) => {
              const active = config.mode === m.id;
              return (
                <TouchableOpacity
                  key={m.id}
                  style={[styles.choiceCard, active && styles.choiceCardActive]}
                  onPress={() => onChangeConfig({ mode: m.id })}
                >
                  <View style={styles.choiceHeaderRow}>
                    <Text style={[styles.choiceTitle, active && styles.choiceTitleActive]}>
                      {m.title}
                    </Text>
                    <View style={[styles.radioDot, active && styles.radioDotActive]} />
                  </View>
                  <Text style={styles.choiceSub}>{m.sub}</Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {config.mode === 'compare' && (
            <>
              <Text style={styles.fieldLabel}>Secondary Detector (--compare-with)</Text>
              <View style={styles.segmentedRow}>
                {(['yolo26s', 'rfdetr_nano', 'yolov8n', 'yolo26n-pose'] as DetectorVariant[]).map(
                  (cd) => {
                    const active = config.compareDetector === cd;
                    return (
                      <TouchableOpacity
                        key={cd}
                        style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                        onPress={() => onChangeConfig({ compareDetector: cd })}
                      >
                        <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
                          {DETECTORS_INFO[cd].shortName}
                        </Text>
                      </TouchableOpacity>
                    );
                  }
                )}
              </View>
            </>
          )}

          <View style={styles.toggleRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.toggleTitle}>Live Telemetry HUD (--hud)</Text>
              <Text style={styles.toggleSub}>
                Displays rolling FPS, detector ms, NPU HMR ms, and stabilizer state
              </Text>
            </View>
            <Switch
              value={config.showHud}
              onValueChange={(v) => onChangeConfig({ showHud: v })}
              trackColor={{ false: COLORS.borderDefault, true: COLORS.accentCyan }}
            />
          </View>
        </View>
      </ScrollView>

      {/* Bottom Sticky Launch Bar */}
      <View style={styles.bottomLaunchBar}>
        <View style={styles.launchSummaryCol}>
          <Text style={styles.launchSummaryTop}>
            {config.cameraPosition.toUpperCase()} CAM • {selectedModelInfo.name.toUpperCase()} •{' '}
            {selectedDetectorInfo.shortName} (S={config.detectorStride})
          </Text>
          <Text style={styles.launchSummarySub}>
            Backend: {config.hmrBackend.toUpperCase()} • Mode: {config.mode.toUpperCase()} •{' '}
            {config.smoothJoints ? '1-Euro Smooth' : 'Raw'}
          </Text>
        </View>

        <TouchableOpacity
          style={[
            styles.launchBtn,
            readyCount < requiredFiles.length && { backgroundColor: COLORS.accentAmber },
          ]}
          onPress={() => {
            if (readyCount < requiredFiles.length) {
              onOpenModelManager();
            } else {
              onStartRealtimeSession();
            }
          }}
          activeOpacity={0.85}
        >
          <Text style={styles.launchBtnText}>
            {readyCount < requiredFiles.length
              ? `Download Models (${readyCount}/${requiredFiles.length})`
              : 'Launch Live Tracking Studio'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.bgPrimary,
  },
  topHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    backgroundColor: COLORS.bgSecondary,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.borderSubtle,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  appTitle: {
    color: COLORS.textPrimary,
    fontSize: 19,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  versionPill: {
    backgroundColor: COLORS.accentCyanSoft,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.accentCyan,
  },
  versionPillText: {
    color: COLORS.accentCyan,
    fontSize: 9.5,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  deviceSub: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    marginTop: 2,
  },
  headerRightActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modelsBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  modelsBtnText: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '700',
  },
  resetBtn: {
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  resetBtnText: {
    color: COLORS.textMuted,
    fontSize: 11.5,
    fontWeight: '600',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: SPACING.lg,
    paddingBottom: 110,
    gap: SPACING.lg,
  },
  telemetryPreviewCard: {
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.lg,
    padding: SPACING.lg,
    borderWidth: 1,
    borderColor: 'rgba(0, 229, 255, 0.35)',
  },
  telemetryTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: SPACING.md,
  },
  telemetryCardLabel: {
    color: COLORS.textMuted,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.7,
  },
  telemetryCardMain: {
    color: COLORS.accentEmerald,
    fontSize: 20,
    fontWeight: '800',
    marginTop: 2,
    fontVariant: ['tabular-nums'],
  },
  npuStatusChip: {
    backgroundColor: COLORS.accentEmeraldSoft,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: RADIUS.pill,
    borderWidth: 1,
    borderColor: COLORS.accentEmerald,
  },
  npuStatusChipText: {
    color: COLORS.accentEmerald,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  telemetryBreakdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: COLORS.borderSubtle,
    gap: 8,
  },
  telemetryMetricItem: {
    flex: 1,
  },
  telemetryMetricTitle: {
    color: COLORS.textMuted,
    fontSize: 10,
    fontWeight: '600',
  },
  telemetryMetricVal: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '700',
    marginTop: 2,
    fontVariant: ['tabular-nums'],
  },
  sectionCard: {
    backgroundColor: COLORS.bgSecondary,
    borderRadius: RADIUS.lg,
    padding: SPACING.lg,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: SPACING.md,
  },
  sectionIndex: {
    color: COLORS.accentCyan,
    fontSize: 13,
    fontWeight: '900',
    backgroundColor: COLORS.accentCyanSoft,
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 6,
    overflow: 'hidden',
    fontVariant: ['tabular-nums'],
  },
  sectionTitle: {
    color: COLORS.textPrimary,
    fontSize: 15,
    fontWeight: '800',
  },
  sectionDesc: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    marginTop: 2,
  },
  fieldLabel: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    fontWeight: '700',
    marginTop: SPACING.md,
    marginBottom: SPACING.sm,
    letterSpacing: 0.2,
  },
  twoColGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
  },
  choiceCard: {
    flex: 1,
    minWidth: '46%',
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  choiceCardActive: {
    borderColor: COLORS.accentCyan,
    backgroundColor: '#131D28',
  },
  choiceHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  choiceTitle: {
    color: COLORS.textPrimary,
    fontSize: 12.5,
    fontWeight: '700',
    flex: 1,
    paddingRight: 6,
  },
  choiceTitleActive: {
    color: COLORS.accentCyan,
  },
  choiceSub: {
    color: COLORS.textMuted,
    fontSize: 10.5,
    lineHeight: 14,
  },
  radioDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: COLORS.textMuted,
  },
  radioDotActive: {
    borderColor: COLORS.accentCyan,
    backgroundColor: COLORS.accentCyan,
  },
  segmentedRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  segmentBtn: {
    flex: 1,
    minWidth: 54,
    backgroundColor: COLORS.bgCard,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
    alignItems: 'center',
  },
  segmentBtnActive: {
    backgroundColor: COLORS.accentCyanSoft,
    borderColor: COLORS.accentCyan,
  },
  segmentText: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    fontWeight: '700',
  },
  segmentTextActive: {
    color: COLORS.accentCyan,
    fontWeight: '800',
  },
  verticalStack: {
    gap: SPACING.sm,
  },
  acceleratorCard: {
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  acceleratorCardActive: {
    borderColor: COLORS.accentCyan,
    backgroundColor: '#131D28',
  },
  accTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  accTitleGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    flexWrap: 'wrap',
  },
  accName: {
    color: COLORS.textPrimary,
    fontSize: 13,
    fontWeight: '800',
  },
  accNameActive: {
    color: COLORS.accentCyan,
  },
  npuPill: {
    backgroundColor: COLORS.accentEmeraldSoft,
    paddingHorizontal: 6,
    paddingVertical: 1.5,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: COLORS.accentEmerald,
  },
  npuPillText: {
    color: COLORS.accentEmerald,
    fontSize: 9,
    fontWeight: '800',
  },
  accLatencyTag: {
    color: COLORS.accentEmerald,
    fontSize: 11.5,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  accProviderText: {
    color: COLORS.textSecondary,
    fontSize: 10.5,
    fontWeight: '600',
    marginTop: 2,
  },
  accDescText: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 4,
    lineHeight: 15,
  },
  modelOptionCard: {
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  modelOptionCardActive: {
    borderColor: COLORS.accentCyan,
    backgroundColor: '#131D28',
  },
  cardDisabled: {
    opacity: 0.45,
  },
  modelOptTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modelOptMetrics: {
    color: COLORS.textSecondary,
    fontSize: 11,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  recPill: {
    backgroundColor: COLORS.accentCyanSoft,
    paddingHorizontal: 6,
    paddingVertical: 1.5,
    borderRadius: 4,
  },
  recPillText: {
    color: COLORS.accentCyan,
    fontSize: 9,
    fontWeight: '800',
  },
  hwBadge: {
    paddingHorizontal: 6,
    paddingVertical: 1.5,
    borderRadius: 4,
    borderWidth: 1,
  },
  hwBadgeNpu: {
    backgroundColor: COLORS.accentEmeraldSoft,
    borderColor: COLORS.accentEmerald,
  },
  hwBadgeCpu: {
    backgroundColor: COLORS.accentAmberSoft,
    borderColor: COLORS.accentAmber,
  },
  hwBadgeText: {
    fontSize: 9,
    fontWeight: '800',
  },
  modelStatsPills: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 6,
  },
  statTag: {
    color: COLORS.textSecondary,
    backgroundColor: COLORS.bgPrimary,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 4,
    fontSize: 10,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
  },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: SPACING.md,
    paddingTop: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: COLORS.borderSubtle,
    gap: 12,
  },
  toggleTitle: {
    color: COLORS.textPrimary,
    fontSize: 12.5,
    fontWeight: '700',
  },
  toggleSub: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 2,
  },
  bottomLaunchBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.bgSecondary,
    borderTopWidth: 1,
    borderTopColor: COLORS.borderDefault,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  launchSummaryCol: {
    flex: 1,
  },
  launchSummaryTop: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '800',
  },
  launchSummarySub: {
    color: COLORS.accentCyan,
    fontSize: 10.5,
    fontWeight: '600',
    marginTop: 2,
  },
  launchBtn: {
    backgroundColor: COLORS.accentCyan,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: RADIUS.md,
  },
  launchBtnText: {
    color: COLORS.textInverse,
    fontSize: 13.5,
    fontWeight: '900',
    letterSpacing: 0.2,
  },
  permissionPromptCard: {
    backgroundColor: 'rgba(0, 229, 255, 0.08)',
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: 'rgba(0, 229, 255, 0.3)',
    padding: SPACING.md,
    marginBottom: SPACING.lg,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  permissionPromptTitle: {
    color: COLORS.accentCyan,
    fontSize: 13,
    fontWeight: '800',
    marginBottom: 2,
  },
  permissionPromptSub: {
    color: COLORS.textSecondary,
    fontSize: 11,
    lineHeight: 15,
  },
  permissionPromptBtn: {
    backgroundColor: COLORS.accentCyan,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: RADIUS.sm,
  },
  permissionPromptBtnText: {
    color: COLORS.bgPrimary,
    fontSize: 12,
    fontWeight: '800',
  },
});
