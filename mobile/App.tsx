/**
 * InstantHMR Mobile — Minimal Real-Time Pose & 70-Keypoint Tracking App.
 *
 * Minimal viable product:
 * 1. Checks camera permission (simple one-tap grant if needed).
 * 2. Checks/auto-downloads required models on first launch with a clean progress bar.
 * 3. Immediately opens full-screen live camera stream on GPU TextureView.
 * 4. Runs person detector + InstantHMR neural network asynchronously on CPU.
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
} from 'react-native-vision-camera';
import { AlphaType, ColorType, Skia } from '@shopify/react-native-skia';
import * as FileSystem from 'expo-file-system';
import { SkeletonOverlay2D } from './src/components/SkeletonOverlay2D';
import { COLORS, RADIUS, SPACING } from './src/constants/theme';
import { InstantHmrMobileEngine } from './src/engine/inferenceEngine';
import { sharedModelManager } from './src/engine/modelManager';
import { DEFAULT_SESSION_CONFIG, FrameInferenceResult } from './src/types/config';

const REQUIRED_MODELS = ['yolo26n.onnx', 'instanthmr-nano.onnx', 'mhr-body.onnx'];

export default function App() {
  const cameraRef = useRef<Camera>(null);
  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back');
  const [viewport, setViewport] = useState({ width: 390, height: 750 });
  const [modelsReady, setModelsReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('Initializing InstantHMR...');
  const [sessionsReady, setSessionsReady] = useState(false);
  const [frameResult, setFrameResult] = useState<FrameInferenceResult | null>(null);

  const [activeBackend, setActiveBackend] = useState<'npu' | 'cpu'>('npu');
  const [activeProviderLabel, setActiveProviderLabel] = useState<string>('NPU');

  // VisionCamera permissions and devices
  const { hasPermission, requestPermission } = useCameraPermission();
  const devices = useCameraDevices();
  const primaryDevice = useCameraDevice(cameraPosition);
  const device = primaryDevice || devices[0];

  // Camera format selection (720p HD optimal for real-time mobile tracking)
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1280, height: 720 } },
    { fps: 30 },
  ]);
  const safeFps = format ? Math.max(format.minFps, Math.min(format.maxFps, 30)) : undefined;

  const engineRef = useRef<InstantHmrMobileEngine>(
    new InstantHmrMobileEngine({
      ...DEFAULT_SESSION_CONFIG,
      hmrBackend: 'npu',
      detectorBackend: 'npu',
      detectorStride: 1,
      detConfidence: 0.25,
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

  const initSessions = async (states: Record<string, any>, backend: 'npu' | 'cpu' = 'npu') => {
    try {
      engineRef.current.updateConfig({
        hmrBackend: backend,
        detectorBackend: backend,
      });
      const res = await engineRef.current.initializeSessions({
        hmrModelUri: states['instanthmr-nano.onnx']?.localUri || null,
        bodyModelUri: states['mhr-body.onnx']?.localUri || null,
        detectorModelUri: states['yolo26n.onnx']?.localUri || null,
      });
      setActiveBackend(backend);
      setActiveProviderLabel(res.hmrProvider);
      setSessionsReady(true);
      setModelsReady(true);
    } catch (err: any) {
      console.error('Session initialization error:', err);
      setStatusMessage(`Model init error: ${err?.message || err}`);
      setModelsReady(true);
    }
  };

  const toggleBackend = async () => {
    const nextBackend = activeBackend === 'npu' ? 'cpu' : 'npu';
    const snapshot = await sharedModelManager.scanInstalledModels();
    await initSessions(snapshot, nextBackend);
  };

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
        setStatusMessage('Loading models into NPU...');
        await initSessions(snapshot, 'npu');
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
      setStatusMessage('Initializing neural networks on NPU...');
      await initSessions(finalSnapshot, 'npu');
    }

    prepareModels();

    return () => {
      active = false;
    };
  }, []);

  // Real-time asynchronous inference loop (decoupled from camera thread)
  useEffect(() => {
    if (!sessionsReady || !hasPermission || !device) return;

    let isMounted = true;
    let isBusy = false;

    const runInferenceCycle = async () => {
      if (isBusy || !cameraRef.current || !isMounted) return;
      isBusy = true;

      try {
        // 1. Capture snapshot from native preview
        const photo = await cameraRef.current.takeSnapshot({ quality: 75 });
        if (!isMounted || !photo?.path) return;

        const photoUri = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`;

        // 2. Decode JPEG via native Skia stream or Base64 fallback
        let data = null;
        try {
          data = await Skia.Data.fromURI(photoUri);
        } catch {
          try {
            const b64 = await FileSystem.readAsStringAsync(photoUri, {
              encoding: FileSystem.EncodingType.Base64,
            });
            data = Skia.Data.fromBase64(b64);
          } catch {
            data = null;
          }
        }

        // Clean up temporary photo file immediately so disk space is constant
        FileSystem.deleteAsync(photoUri, { idempotent: true }).catch(() => {});

        if (!data || !isMounted) return;

        const skImage = Skia.Image.MakeImageFromEncoded(data);
        if (!skImage) return;

        // 3. Render onto 640x640 offscreen surface
        const surface = Skia.Surface.MakeOffscreen(640, 640);
        if (!surface) return;

        const canvas = surface.getCanvas();
        const src = Skia.XYWHRect(0, 0, skImage.width(), skImage.height());
        const dst = Skia.XYWHRect(0, 0, 640, 640);
        canvas.drawImageRect(skImage, src, dst, Skia.Paint());
        surface.flush();

        // 4. Read 640x640 RGBA uint8 pixels
        const snap = surface.makeImageSnapshot();
        const rgba = snap.readPixels(0, 0, {
          width: 640,
          height: 640,
          colorType: ColorType.RGBA_8888,
          alphaType: AlphaType.Unpremul,
        });

        if (rgba && isMounted) {
          const nowS = Date.now() / 1000.0;
          const result = await engineRef.current.processFrame(
            rgba as Uint8Array,
            640,
            640,
            nowS,
            4 // 4 channels RGBA
          );
          if (isMounted) {
            setFrameResult(result);
          }
        }
      } catch (err: any) {
        console.error('[InstantHMR] Pipeline inference error:', err?.message || err);
      } finally {
        isBusy = false;
      }
    };

    // Poll at ~16 FPS target, adapting naturally to CPU inference speed without queue buildup
    const intervalId = setInterval(runInferenceCycle, 60);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [sessionsReady, hasPermission, device]);

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
  const fps = frameResult?.telemetry?.rollingFps || 0;

  // State 3: Live Camera + InstantHMR Keypoints Overlay
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      <View style={styles.cameraWrapper} onLayout={handleLayout}>
        {device ? (
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            format={format}
            fps={format != null ? safeFps : undefined}
            preview={true}
            photo={true}
            androidPreviewViewType="texture-view"
            resizeMode="cover"
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
          <TouchableOpacity
            style={styles.hudPill}
            onPress={toggleBackend}
            activeOpacity={0.7}
          >
            <View
              style={[
                styles.statusDot,
                { backgroundColor: persons.length > 0 ? COLORS.accentEmerald : COLORS.accentCyan },
              ]}
            />
            <Text style={styles.hudPillText}>
              {activeBackend === 'npu' ? '⚡ ' : '⚙️ '}
              {activeProviderLabel} • {fps > 0 ? `${fps.toFixed(0)} FPS` : 'Ready'} •{' '}
              {persons.length > 0
                ? `${persons.length} Person (${Math.round(persons[0].confidence * 100)}%)`
                : 'Searching...'}
            </Text>
          </TouchableOpacity>

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
