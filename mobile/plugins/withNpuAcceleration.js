/**
 * Expo Config Plugin: withNpuAcceleration
 *
 * Configures native Android (NNAPI / Qualcomm QNN / XNNPACK) and iOS (Apple CoreML Neural Engine)
 * build flags, permissions, and hardware acceleration dependencies for ONNX Runtime Mobile.
 */
const {
  withAndroidManifest,
  withAppBuildGradle,
  withInfoPlist,
} = require('@expo/config-plugins');

function withAndroidNpu(config) {
  config = withAndroidManifest(config, (modConfig) => {
    const mainApp = modConfig.modResults.manifest.application?.[0];
    if (mainApp) {
      mainApp.$['android:largeHeap'] = 'true';
      mainApp.$['android:hardwareAccelerated'] = 'true';
    }
    return modConfig;
  });

  config = withAppBuildGradle(config, (modConfig) => {
    if (modConfig.modResults.language === 'groovy') {
      let contents = modConfig.modResults.contents;
      if (!contents.includes('onnxruntime-android')) {
        const packagingBlock = `
    packagingOptions {
        pickFirst '**/libonnxruntime.so'
        pickFirst '**/libonnxruntime4j_jni.so'
        pickFirst '**/libc++_shared.so'
    }
`;
        contents = contents.replace(/android\s*\{/, `android {\n${packagingBlock}`);
      }
      modConfig.modResults.contents = contents;
    }
    return modConfig;
  });

  return config;
}

function withIosNeuralEngine(config) {
  return withInfoPlist(config, (modConfig) => {
    modConfig.modResults.CADisableMinimumFrameDurationOnPhone = true;
    modConfig.modResults.NSCameraUsageDescription =
      modConfig.modResults.NSCameraUsageDescription ||
      'InstantHMR needs camera access for real-time 3D human mesh and 70-keypoint pose recovery.';
    return modConfig;
  });
}

module.exports = function withNpuAcceleration(config) {
  config = withAndroidNpu(config);
  config = withIosNeuralEngine(config);
  return config;
};
