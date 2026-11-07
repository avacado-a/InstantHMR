/**
 * Model Manager & On-Demand Weight Downloader Modal for InstantHMR Mobile.
 *
 * Allows users to inspect, download, verify, or remove ONNX checkpoints (`mhr-body.onnx`,
 * `instanthmr-nano/tiny/small/medium/medium-commercial.onnx`, and `yolo26n/s/pose/rfdetr.onnx`)
 * with real-time progress bars and storage telemetry.
 */
import React from 'react';
import {
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { MODEL_ASSETS_REGISTRY, ModelAssetSpec } from '../constants/hardware';
import { COLORS, RADIUS, SPACING } from '../constants/theme';
import { ModelInstallState } from '../engine/modelManager';

interface ModelManagerModalProps {
  visible: boolean;
  onClose: () => void;
  modelStates: Record<string, ModelInstallState>;
  requiredFiles: string[];
  onDownloadModel: (filename: string) => void;
  onDownloadAllRequired: () => void;
  onDeleteModel: (filename: string) => void;
}

export const ModelManagerModal: React.FC<ModelManagerModalProps> = ({
  visible,
  onClose,
  modelStates,
  requiredFiles,
  onDownloadModel,
  onDownloadAllRequired,
  onDeleteModel,
}) => {
  const specs = Object.values(MODEL_ASSETS_REGISTRY);
  const requiredSet = new Set(requiredFiles);

  const missingRequiredCount = requiredFiles.filter(
    (f) => !modelStates[f]?.installed
  ).length;

  const renderCategorySection = (
    title: string,
    subtitle: string,
    category: ModelAssetSpec['category']
  ) => {
    const items = specs.filter((s) => s.category === category);
    return (
      <View style={styles.sectionBlock} key={category}>
        <Text style={styles.sectionTitle}>{title}</Text>
        <Text style={styles.sectionSubtitle}>{subtitle}</Text>
        {items.map((spec) => {
          const st = modelStates[spec.filename];
          const isReq = requiredSet.has(spec.filename);
          const isInstalled = st?.installed ?? false;
          const isDownloading = st?.downloading ?? false;
          const pct = Math.round((st?.progress || 0) * 100);

          return (
            <View
              key={spec.filename}
              style={[
                styles.modelCard,
                isReq && styles.modelCardRequired,
              ]}
            >
              <View style={styles.modelHeaderRow}>
                <View style={styles.modelTitleWrap}>
                  <View style={styles.titleLine}>
                    <Text style={styles.modelName}>{spec.title}</Text>
                    {isReq && (
                      <View style={styles.activeConfigBadge}>
                        <Text style={styles.activeConfigBadgeText}>SELECTED</Text>
                      </View>
                    )}
                  </View>
                  <Text style={styles.modelFilename}>
                    {spec.filename} • {spec.sizeMb.toFixed(1)} MB
                  </Text>
                </View>

                {isInstalled ? (
                  <View style={styles.actionGroup}>
                    <View style={styles.installedBadge}>
                      <Text style={styles.installedBadgeText}>READY</Text>
                    </View>
                    <TouchableOpacity
                      style={styles.deleteBtn}
                      onPress={() => onDeleteModel(spec.filename)}
                    >
                      <Text style={styles.deleteBtnText}>Remove</Text>
                    </TouchableOpacity>
                  </View>
                ) : isDownloading ? (
                  <View style={styles.downloadingBadge}>
                    <Text style={styles.downloadingText}>{pct}%</Text>
                  </View>
                ) : (
                  <TouchableOpacity
                    style={[
                      styles.downloadBtn,
                      isReq && styles.downloadBtnPrimary,
                    ]}
                    onPress={() => onDownloadModel(spec.filename)}
                  >
                    <Text
                      style={[
                        styles.downloadBtnText,
                        isReq && styles.downloadBtnTextPrimary,
                      ]}
                    >
                      Download
                    </Text>
                  </TouchableOpacity>
                )}
              </View>

              <Text style={styles.modelDesc}>{spec.description}</Text>

              {isDownloading && (
                <View style={styles.progressTrack}>
                  <View
                    style={[
                      styles.progressFill,
                      { width: `${Math.max(4, pct)}%` },
                    ]}
                  />
                </View>
              )}

              {st?.error && (
                <Text style={styles.errorText}>Error: {st.error}</Text>
              )}
            </View>
          );
        })}
      </View>
    );
  };

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          {/* Header */}
          <View style={styles.sheetHeader}>
            <View>
              <Text style={styles.sheetTitle}>ONNX Model & NPU Weights Manager</Text>
              <Text style={styles.sheetSubtitle}>
                On-demand download from Hugging Face Hub & Ultralytics into device storage
              </Text>
            </View>
            <TouchableOpacity style={styles.closeBtn} onPress={onClose}>
              <Text style={styles.closeBtnText}>Done</Text>
            </TouchableOpacity>
          </View>

          {/* Quick Action Banner for Active Selection */}
          <View style={styles.bannerBar}>
            <View style={{ flex: 1 }}>
              <Text style={styles.bannerTitle}>
                Active Pipeline Weights ({requiredFiles.length - missingRequiredCount}/{requiredFiles.length} Ready)
              </Text>
              <Text style={styles.bannerSub}>
                {requiredFiles.join(' + ')}
              </Text>
            </View>
            {missingRequiredCount > 0 && (
              <TouchableOpacity
                style={styles.downloadAllBtn}
                onPress={onDownloadAllRequired}
              >
                <Text style={styles.downloadAllBtnText}>
                  Download Required ({missingRequiredCount})
                </Text>
              </TouchableOpacity>
            )}
          </View>

          <ScrollView
            style={styles.scrollArea}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
          >
            {renderCategorySection(
              '1. Shared MHR Kinematic Body Graph',
              'Required by all InstantHMR models to decode 204 pose & 45 shape parameters into 70 3D joints and 18,439-vertex mesh.',
              'body'
            )}
            {renderCategorySection(
              '2. InstantHMR 3D Recovery Backbones (288×288)',
              'Trained on SAM 3D Body dataset. Runs in 4.6–5.5 ms per person on Mobile NPU (CoreML ANE / Snapdragon QNN).',
              'hmr'
            )}
            {renderCategorySection(
              '3. Person Detectors & Pose Envelope Models',
              'Finds person bounding boxes every N frames (Detector Stride). Pose variants expand boxes to cover outstretched limbs.',
              'detector'
            )}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(4, 6, 10, 0.82)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: COLORS.bgSecondary,
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    maxHeight: '90%',
    borderTopWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  sheetHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.borderSubtle,
  },
  sheetTitle: {
    color: COLORS.textPrimary,
    fontSize: 16,
    fontWeight: '800',
  },
  sheetSubtitle: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    marginTop: 2,
  },
  closeBtn: {
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  closeBtnText: {
    color: COLORS.accentCyan,
    fontSize: 12.5,
    fontWeight: '700',
  },
  bannerBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: COLORS.bgCard,
    marginHorizontal: SPACING.lg,
    marginTop: SPACING.md,
    padding: SPACING.md,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
    gap: 10,
  },
  bannerTitle: {
    color: COLORS.textPrimary,
    fontSize: 12.5,
    fontWeight: '700',
  },
  bannerSub: {
    color: COLORS.accentCyan,
    fontSize: 11,
    marginTop: 2,
    fontVariant: ['tabular-nums'],
  },
  downloadAllBtn: {
    backgroundColor: COLORS.accentCyan,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: RADIUS.sm,
  },
  downloadAllBtnText: {
    color: COLORS.textInverse,
    fontSize: 11.5,
    fontWeight: '800',
  },
  scrollArea: {
    paddingHorizontal: SPACING.lg,
  },
  scrollContent: {
    paddingBottom: 36,
    paddingTop: SPACING.md,
  },
  sectionBlock: {
    marginBottom: SPACING.xl,
  },
  sectionTitle: {
    color: COLORS.textPrimary,
    fontSize: 13.5,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  sectionSubtitle: {
    color: COLORS.textSecondary,
    fontSize: 11,
    marginTop: 2,
    marginBottom: SPACING.sm,
  },
  modelCard: {
    backgroundColor: COLORS.bgCard,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    marginBottom: SPACING.sm,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  modelCardRequired: {
    borderColor: 'rgba(0, 229, 255, 0.45)',
    backgroundColor: '#141B26',
  },
  modelHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modelTitleWrap: {
    flex: 1,
    paddingRight: 8,
  },
  titleLine: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    flexWrap: 'wrap',
  },
  modelName: {
    color: COLORS.textPrimary,
    fontSize: 13,
    fontWeight: '700',
  },
  activeConfigBadge: {
    backgroundColor: COLORS.accentCyanSoft,
    paddingHorizontal: 6,
    paddingVertical: 1.5,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: COLORS.accentCyan,
  },
  activeConfigBadgeText: {
    color: COLORS.accentCyan,
    fontSize: 9,
    fontWeight: '800',
  },
  modelFilename: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 2,
    fontVariant: ['tabular-nums'],
  },
  modelDesc: {
    color: COLORS.textSecondary,
    fontSize: 11,
    marginTop: 6,
    lineHeight: 15,
  },
  actionGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  installedBadge: {
    backgroundColor: COLORS.accentEmeraldSoft,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.accentEmerald,
  },
  installedBadgeText: {
    color: COLORS.accentEmerald,
    fontSize: 10,
    fontWeight: '800',
  },
  deleteBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  deleteBtnText: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '600',
  },
  downloadingBadge: {
    backgroundColor: COLORS.accentCyanSoft,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: RADIUS.sm,
  },
  downloadingText: {
    color: COLORS.accentCyan,
    fontSize: 11,
    fontWeight: '800',
    fontVariant: ['tabular-nums'],
  },
  downloadBtn: {
    backgroundColor: COLORS.bgCardElevated,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
  },
  downloadBtnPrimary: {
    backgroundColor: COLORS.accentCyan,
    borderColor: COLORS.accentCyan,
  },
  downloadBtnText: {
    color: COLORS.textPrimary,
    fontSize: 11.5,
    fontWeight: '700',
  },
  downloadBtnTextPrimary: {
    color: COLORS.textInverse,
    fontWeight: '800',
  },
  progressTrack: {
    height: 5,
    backgroundColor: COLORS.bgPrimary,
    borderRadius: 3,
    marginTop: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.accentCyan,
  },
  errorText: {
    color: COLORS.accentCoral,
    fontSize: 10.5,
    marginTop: 4,
  },
});
