/**
 * InstantHMR Mobile Studio (Android & iOS)
 *
 * Root entry point connecting:
 * 1. Hardware capability detection (Mobile NPU via CoreML ANE / Android NNAPI QNN vs XNNPACK CPU)
 * 2. Pre-Launch Studio Configurator with every option configurable before starting
 * 3. On-Demand Model Manager with progress bars
 * 4. Real-Time Live Camera & 3D MHR70 Studio
 */
import React, { useEffect, useMemo, useState } from 'react';
import { SafeAreaView, StatusBar, StyleSheet } from 'react-native';
import { LiveCameraStudio } from './src/components/LiveCameraStudio';
import { ModelManagerModal } from './src/components/ModelManagerModal';
import { PreLaunchConfigurator } from './src/components/PreLaunchConfigurator';
import { detectDeviceHardwareProfile } from './src/constants/hardware';
import { COLORS } from './src/constants/theme';
import { ModelInstallState, sharedModelManager } from './src/engine/modelManager';
import { DEFAULT_SESSION_CONFIG, SessionConfig } from './src/types/config';

export default function App() {
  const hardware = useMemo(() => detectDeviceHardwareProfile(), []);
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_SESSION_CONFIG);
  const [screen, setScreen] = useState<'setup' | 'live'>('setup');
  const [modelModalOpen, setModelModalOpen] = useState(false);
  const [modelStates, setModelStates] = useState<Record<string, ModelInstallState>>(
    () => sharedModelManager.getSnapshot()
  );

  useEffect(() => {
    const unsubscribe = sharedModelManager.subscribe(setModelStates);
    sharedModelManager.scanInstalledModels();
    return unsubscribe;
  }, []);

  const requiredFiles = useMemo(
    () =>
      sharedModelManager.getRequiredFilesForConfig(
        config.hmrModel,
        config.detector,
        config.mode === 'compare',
        config.compareDetector
      ),
    [config.hmrModel, config.detector, config.mode, config.compareDetector]
  );

  const handleUpdateConfig = (patch: Partial<SessionConfig>) => {
    setConfig((prev) => ({ ...prev, ...patch }));
  };

  const handleDownloadAllRequired = async () => {
    for (const filename of requiredFiles) {
      if (!modelStates[filename]?.installed) {
        try {
          await sharedModelManager.downloadModel(filename);
        } catch {
          // Handled inside ModelInstallState
        }
      }
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bgPrimary} />

      {screen === 'setup' ? (
        <PreLaunchConfigurator
          config={config}
          hardware={hardware}
          modelStates={modelStates}
          requiredFiles={requiredFiles}
          onChangeConfig={handleUpdateConfig}
          onResetDefaults={() => setConfig(DEFAULT_SESSION_CONFIG)}
          onOpenModelManager={() => setModelModalOpen(true)}
          onStartRealtimeSession={() => setScreen('live')}
        />
      ) : (
        <LiveCameraStudio
          config={config}
          modelStates={modelStates}
          onUpdateConfig={handleUpdateConfig}
          onExitToSetup={() => setScreen('setup')}
        />
      )}

      <ModelManagerModal
        visible={modelModalOpen}
        onClose={() => setModelModalOpen(false)}
        modelStates={modelStates}
        requiredFiles={requiredFiles}
        onDownloadModel={(f) => sharedModelManager.downloadModel(f)}
        onDownloadAllRequired={handleDownloadAllRequired}
        onDeleteModel={(f) => sharedModelManager.deleteModel(f)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.bgPrimary,
  },
});
