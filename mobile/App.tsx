/**
 * InstantHMR Mobile — Minimal Real-Time Pose & 70-Keypoint Tracking App.
 *
 * Minimal viable product:
 * 1. Checks camera permission (simple one-tap grant if needed).
 * 2. Checks/auto-downloads required models on first launch with a clean progress bar.
 * 3. Immediately opens full-screen live camera stream.
 * 4. Runs person detector + InstantHMR neural network.
 * 5. Displays the 70 InstantHMR keypoints on detected persons in real time.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  Linking,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraDevices,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import { SkeletonOverlay2D } from './src/components/SkeletonOverlay2D';
import { COLORS, RADIUS, SPACING } from './src/constants/theme';
import { InstantHmrMobileEngine } from './src/engine/inferenceEngine';
import { sharedModelManager } from './src/engine/modelManager';
import { DEFAULT_SESSION_CONFIG, FrameInferenceResult } from './src/types/config';

const REQUIRED_MODELS = ['yolo26n.onnx', 'instanthmr-nano.onnx', 'mhr-body.onnx'];

export default function App() {
  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back');
  const [viewport, setViewport] = useState({ width: 390, height: 750 });
  const [modelsReady, setModelsReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('Initializing InstantHMR...');
  const [sessionsReady, setSessionsReady] = useState(false);
  const [frameResult, setFrameResult] = useState<FrameInferenceResult | null>(null);

  // VisionCamera permissions and devices
  const { hasPermission, requestPermission } = useCameraPermission();
  const devices = useCameraDevices();
  const primaryDevice = useCameraDevice(cameraPosition);
  const device = primaryDevice || devices[0];

  // Camera format selection (720p HD at 30 FPS optimal for real-time mobile tracking)
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1280, height: 720 } },
    { fps: 30 },
  ]);
  const safeFps = format ? Math.max(format.minFps, Math.min(format.maxFps, 30)) : undefined;

  // Worklet -> JS bridge throttler
  const isProcessing = useSharedValue(false);
  const lastFrameTimestamp = useSharedValue(0);
  const { resize } = useResizePlugin();

  const engineRef = useRef<InstantHmrMobileEngine>(
    new InstantHmrMobileEngine({
      ...DEFAULT_SESSION_CONFIG,
      hmrBackend: 'cpu',
      detectorBackend: 'cpu',
      maxPersons: 1,
      mode: 'overlay',
    })
  );

  // Request camera permission on mount
  useEffect(() => {
    if (!hasPermission) {
      requestPermission().catch((err) => {
        console.warn('Camera permission request error:', err);
      });
    }
  }, [hasPermission, requestPermission]);

  // Model setup & automatic download
  useEffect(() => {
    let active = true;

    async function prepareModels() {
      setStatusMessage('Checking installed models...');
      const snapshot = await sharedModelManager.scanInstalledModels();

      const missing = REQUIRED_MODELS.filter((f) => !snapshot[f]?.installed);

      if (missing.length === 0) {
        if (!active) return;
        setDownloadProgress(1);
        setStatusMessage('Loading models into memory...');
        await initSessions(snapshot);
        return;
      }

      // Download missing models sequentially
      for (let i = 0; i < missing.length; i++) {
        const file = missing[i];
        if (!active) return;
        setStatusMessage(`Downloading ${file} (${i + 1}/${missing.length})...`);

        const unsubscribe = sharedModelManager.subscribe((states) => {
          const s = states[file];
          if (s && s.progress) {
            const overall = (i + s.progress) / missing.length;
            setDownloadProgress(overall);
          }
        });

        try {
          await sharedModelManager.downloadModel(file);
        } catch (err: any) {
          console.warn(`Error downloading ${file}:`, err);
        } finally {
          unsubscribe();
        }
      }

      const finalSnapshot = await sharedModelManager.scanInstalledModels();
      if (!active) return;
      setDownloadProgress(1);
      setStatusMessage('Initializing neural networks...');
      await initSessions(finalSnapshot);
    }

    async function initSessions(states: Record<string, any>) {
      try {
        await engineRef.current.initializeSessions({
          hmrModelUri: states['instanthmr-nano.onnx']?.localUri || null,
          bodyModelUri: states['mhr-body.onnx']?.localUri || null,
          detectorModelUri: states['yolo26n.onnx']?.localUri || null,
        });
        if (active) {
          setSessionsReady(true);
          setModelsReady(true);
        }
      } catch (err) {
        console.error('Session initialization error:', err);
        if (active) {
          setModelsReady(true);
        }
      }
    }

    prepareModels();

    return () => {
      active = false;
    };
  }, []);

  // JS-thread inference runner
  const handleFrameOnJs = useRunOnJS(
    (rgbBuffer: Uint8Array, width: number, height: number, timestampS: number) => {
      if (!sessionsReady) {
        isProcessing.value = false;
        return;
      }
      engineRef.current
        .processFrame(rgbBuffer, width, height, timestampS)
        .then((result) => {
          setFrameResult(result);
        })
        .catch((err) => {
          console.warn('Inference error:', err);
        })
        .finally(() => {
          isProcessing.value = false;
        });
    },
    [sessionsReady, isProcessing]
  );

  // Native worklet frame processor running on camera thread
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (isProcessing.value) {
        return;
      }
      const now = Date.now();
      // Throttle to max ~15-20 FPS to keep CPU cool and prevent ANR
      if (now - lastFrameTimestamp.value < 65) {
        return;
      }
      lastFrameTimestamp.value = now;
      isProcessing.value = true;

      try {
        const resized = resize(frame, {
          scale: { width: 640, height: 640 },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });
        handleFrameOnJs(resized, 640, 640, now / 1000.0);
      } catch {
        isProcessing.value = false;
      }
    },
    [resize, handleFrameOnJs, isProcessing, lastFrameTimestamp]
  );

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    if (width > 0 && height > 0) {
      setViewport({ width, height });
    }
  };

  // State 1: Camera Permission Screen
  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#0B0D11" />
        <View style={styles.centerCard}>
          <Text style={styles.icon}>📷</Text>
          <Text style={styles.title}>Camera Permission</Text>
          <Text style={styles.subtitle}>
            InstantHMR needs camera access to detect people and display real-time 3D pose keypoints.
          </Text>
          <TouchableOpacity
            style={styles.primaryBtn}
            onPress={async () => {
              const granted = await requestPermission();
              if (!granted) {
                Linking.openSettings();
              }
            }}
          >
            <Text style={styles.primaryBtnText}>Grant Camera Access</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // State 2: Model Download / Preparation Screen
  if (!modelsReady) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#0B0D11" />
        <View style={styles.centerCard}>
          <ActivityIndicator size="large" color={COLORS.accentCyan} style={{ marginBottom: 20 }} />
          <Text style={styles.title}>InstantHMR Setup</Text>
          <Text style={styles.subtitle}>{statusMessage}</Text>
          <View style={styles.progressBarTrack}>
            <View
              style={[
                styles.progressBarFill,
                { width: `${Math.round(downloadProgress * 100)}%` },
              ]}
            />
          </View>
          <Text style={styles.progressText}>{Math.round(downloadProgress * 100)}%</Text>
        </View>
      </SafeAreaView>
    );
  }

  const persons = frameResult?.persons || [];
  const fps = frameResult?.telemetry?.rollingFps || 30;

  // State 3: Live Camera + InstantHMR Keypoints Overlay
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      <View style={styles.cameraWrapper} onLayout={handleLayout}>
        {device ? (
          <Camera
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            format={format}
            fps={format != null ? safeFps : undefined}
            preview={true}
            androidPreviewViewType="texture-view"
            resizeMode="cover"
            pixelFormat="yuv"
            frameProcessor={frameProcessor}
          />
        ) : (
          <View style={styles.noCameraView}>
            <Text style={styles.noCameraText}>Connecting to Camera...</Text>
          </View>
        )}

        {/* 2D 70-Keypoint Skeleton Overlay */}
        <SkeletonOverlay2D
          persons={persons}
          viewWidth={viewport.width}
          viewHeight={viewport.height}
          sourceWidth={640}
          sourceHeight={640}
          modelLabel="InstantHMR"
          drawBox={true}
        />

        {/* Top Minimal Status HUD */}
        <View style={styles.topHudRow}>
          <View style={styles.hudPill}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: persons.length > 0 ? COLORS.accentEmerald : COLORS.accentCyan },
              ]}
            />
            <Text style={styles.hudPillText}>
              InstantHMR • {fps.toFixed(0)} FPS • {persons.length > 0 ? `${persons.length} Person` : 'Searching...'}
            </Text>
          </View>

          <TouchableOpacity
            style={styles.flipBtn}
            onPress={() => setCameraPosition((prev) => (prev === 'back' ? 'front' : 'back'))}
            activeOpacity={0.8}
          >
            <Text style={styles.flipBtnText}>🔄 {cameraPosition === 'back' ? 'Back' : 'Front'}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  cameraWrapper: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  centerCard: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.xl,
    backgroundColor: '#0B0D11',
  },
  icon: {
    fontSize: 56,
    marginBottom: SPACING.md,
  },
  title: {
    color: COLORS.textPrimary,
    fontSize: 20,
    fontWeight: '800',
    marginBottom: SPACING.xs,
    textAlign: 'center',
  },
  subtitle: {
    color: COLORS.textSecondary,
    fontSize: 13,
    textAlign: 'center',
    marginBottom: SPACING.lg,
    maxWidth: 280,
  },
  primaryBtn: {
    backgroundColor: COLORS.accentCyan,
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: RADIUS.sm,
  },
  primaryBtnText: {
    color: '#0B0D11',
    fontSize: 14,
    fontWeight: '800',
  },
  progressBarTrack: {
    width: 240,
    height: 8,
    backgroundColor: '#1C222D',
    borderRadius: 4,
    overflow: 'hidden',
    marginTop: SPACING.sm,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: COLORS.accentCyan,
  },
  progressText: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '700',
    marginTop: 8,
  },
  noCameraView: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#0B0D11',
    justifyContent: 'center',
    alignItems: 'center',
  },
  noCameraText: {
    color: COLORS.textMuted,
    fontSize: 13,
  },
  topHudRow: {
    position: 'absolute',
    top: 14,
    left: 14,
    right: 14,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 10,
  },
  hudPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(11, 13, 17, 0.85)',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: RADIUS.pill,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    gap: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  hudPillText: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '700',
  },
  flipBtn: {
    backgroundColor: 'rgba(11, 13, 17, 0.85)',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: RADIUS.pill,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  flipBtnText: {
    color: COLORS.accentCyan,
    fontSize: 11.5,
    fontWeight: '800',
  },
});
