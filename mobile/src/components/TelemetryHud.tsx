/**
 * Real-Time Telemetry HUD Bar for InstantHMR Mobile Studio.
 *
 * Ports `make_single_hud` and `make_compare_hud` from `instanthmr/visualizer.py`
 * into a native responsive mobile HUD displaying:
 * - Rolling Pipeline FPS
 * - Detector Latency (and Stride Reuse indicator)
 * - InstantHMR NPU / CPU Latency + MHR Body Decoder Latency
 * - Active Hardware Provider (CoreML ANE / Android NNAPI QNN / XNNPACK CPU)
 * - 1-Euro Smoother + BoxStabilizer badge
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { DETECTORS_INFO, HMR_MODELS_INFO } from '../constants/hardware';
import { COLORS, RADIUS, SPACING } from '../constants/theme';
import { FrameTelemetry, SessionConfig } from '../types/config';

interface TelemetryHudProps {
  telemetry: FrameTelemetry;
  config: SessionConfig;
}

export const TelemetryHud: React.FC<TelemetryHudProps> = ({ telemetry, config }) => {
  if (!config.showHud) {
    return null;
  }

  const detName = DETECTORS_INFO[config.detector].shortName;
  const hmrName = HMR_MODELS_INFO[config.hmrModel].name.replace('InstantHMR ', '');
  const isNpu = config.hmrBackend === 'npu' || config.hmrBackend === 'auto';

  return (
    <View style={styles.hudContainer}>
      {/* Top Row: Pipeline Identity + Active NPU/CPU Provider + Camera Badge */}
      <View style={styles.topRow}>
        <View style={styles.brandGroup}>
          <View style={[styles.liveDot, { backgroundColor: COLORS.accentEmerald }]} />
          <Text style={styles.brandTitle}>InstantHMR</Text>
          <View style={styles.cameraBadge}>
            <Text style={styles.cameraBadgeText}>
              {config.cameraPosition.toUpperCase()} • {config.resolution}
            </Text>
          </View>
        </View>

        <View
          style={[
            styles.providerBadge,
            {
              backgroundColor: isNpu ? COLORS.accentCyanSoft : COLORS.accentAmberSoft,
              borderColor: isNpu ? COLORS.accentCyan : COLORS.accentAmber,
            },
          ]}
        >
          <Text
            style={[
              styles.providerBadgeText,
              { color: isNpu ? COLORS.accentCyan : COLORS.accentAmber },
            ]}
          >
            {isNpu ? 'NPU ACCEL' : config.hmrBackend.toUpperCase()} • {hmrName.toUpperCase()}
          </Text>
        </View>
      </View>

      {/* Bottom Metric Pills Row */}
      <View style={styles.pillsRow}>
        {/* 1. Rolling Pipeline FPS Pill */}
        <View style={[styles.pill, styles.fpsPill]}>
          <Text style={styles.pillLabel}>PIPELINE</Text>
          <Text style={[styles.pillValue, { color: COLORS.accentEmerald }]}>
            {telemetry.rollingFps.toFixed(1)} FPS
          </Text>
        </View>

        {/* 2. Detector Latency + Stride Pill */}
        <View style={[styles.pill, styles.detPill]}>
          <Text style={styles.pillLabel}>
            {detName} (S={config.detectorStride})
          </Text>
          <Text style={[styles.pillValue, { color: COLORS.accentAmber }]}>
            {telemetry.detectorMs.toFixed(1)} ms
            {telemetry.detectorReused ? ' [Hold]' : ''}
          </Text>
        </View>

        {/* 3. InstantHMR + MHR Body Latency Pill */}
        <View style={[styles.pill, styles.hmrPill]}>
          <Text style={styles.pillLabel}>HMR + MHR70</Text>
          <Text style={[styles.pillValue, { color: COLORS.accentCyan }]}>
            {telemetry.hmrMs.toFixed(1)} + {telemetry.bodyDecoderMs.toFixed(1)} ms
          </Text>
        </View>

        {/* 4. Filter Status Pill */}
        <View style={[styles.pill, styles.filterPill]}>
          <Text style={styles.pillLabel}>STABILIZER</Text>
          <Text
            style={[
              styles.pillValue,
              {
                color: config.smoothJoints ? COLORS.accentSky : COLORS.accentCoral,
              },
            ]}
          >
            {config.smoothJoints
              ? config.stabilizeBox
                ? '1€ + Box'
                : '1-Euro'
              : 'RAW'}
          </Text>
        </View>
      </View>

      {/* Dual-Detector Comparison Subbar when in `compare` mode */}
      {config.mode === 'compare' && telemetry.secondaryDetectorMs !== undefined && (
        <View style={styles.compareRow}>
          <Text style={styles.compareText}>
            BENCHMARK: {detName} ({telemetry.detectorMs.toFixed(1)}ms) vs{' '}
            {DETECTORS_INFO[config.compareDetector].shortName} (
            {telemetry.secondaryDetectorMs.toFixed(1)}ms)
          </Text>
          <Text style={styles.compareSpeedup}>
            {telemetry.detectorMs <= telemetry.secondaryDetectorMs
              ? `${detName} is ${(
                  telemetry.secondaryDetectorMs / Math.max(0.1, telemetry.detectorMs)
                ).toFixed(1)}x Faster`
              : `${DETECTORS_INFO[config.compareDetector].shortName} is ${(
                  telemetry.detectorMs / Math.max(0.1, telemetry.secondaryDetectorMs)
                ).toFixed(1)}x Faster`}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  hudContainer: {
    backgroundColor: COLORS.bgHudGlass,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.borderSubtle,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  brandGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  brandTitle: {
    color: COLORS.textPrimary,
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  cameraBadge: {
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  cameraBadgeText: {
    color: COLORS.textSecondary,
    fontSize: 9.5,
    fontWeight: '700',
  },
  providerBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2.5,
    borderRadius: RADIUS.pill,
    borderWidth: 1,
  },
  providerBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.4,
  },
  pillsRow: {
    flexDirection: 'row',
    gap: 6,
  },
  pill: {
    flex: 1,
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.sm,
    paddingHorizontal: 7,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  fpsPill: {
    borderColor: 'rgba(16, 185, 129, 0.35)',
  },
  detPill: {
    borderColor: 'rgba(245, 158, 11, 0.35)',
  },
  hmrPill: {
    borderColor: 'rgba(0, 229, 255, 0.35)',
  },
  filterPill: {
    flex: 0.85,
    borderColor: COLORS.borderDefault,
  },
  pillLabel: {
    color: COLORS.textMuted,
    fontSize: 8.5,
    fontWeight: '700',
    letterSpacing: 0.4,
    marginBottom: 1,
  },
  pillValue: {
    fontSize: 11,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  compareRow: {
    marginTop: 6,
    paddingTop: 5,
    borderTopWidth: 1,
    borderTopColor: COLORS.borderSubtle,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  compareText: {
    color: COLORS.textSecondary,
    fontSize: 10.5,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
  },
  compareSpeedup: {
    color: COLORS.accentCyan,
    fontSize: 10.5,
    fontWeight: '800',
  },
});
