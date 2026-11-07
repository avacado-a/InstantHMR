/**
 * Expo Config Plugin: withNpuAcceleration
 *
 * Configures native Android (NNAPI / Qualcomm QNN / XNNPACK) and iOS (Apple CoreML Neural Engine)
 * build flags, permissions, and hardware acceleration dependencies for ONNX Runtime Mobile.
 *
 * Also fixes upstream onnxruntime-react-native Gradle issues:
 * 1. Replaces 'latest.integration@aar' dynamic Maven resolution with pinned '1.24.3@aar'
 *    to avoid Gradle 8.10 + Apache Ivy SAXNotRecognizedException on 'accessExternalSchema'.
 * 2. Removes broken systemProp.javax.xml.parsers.* overrides from onnxruntime-react-native/android/gradle.properties.
 * 3. Pins Kotlin 1.9.25 for Compose Compiler 1.5.15 compatibility.
 * 4. Packages libonnxruntime.so into the final APK via implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3").
 */
const {
  withAndroidManifest,
  withAppBuildGradle,
  withProjectBuildGradle,
  withGradleProperties,
  withDangerousMod,
  withInfoPlist,
} = require('@expo/config-plugins');
const fs = require('fs');
const path = require('path');

const ORT_ANDROID_VERSION = '1.24.3';
const ORT_EXTENSIONS_VERSION = '0.13.0';

function patchOnnxruntimeReactNativeNodeModule(projectRoot) {
  const ortAndroidDir = path.join(
    projectRoot,
    'node_modules',
    'onnxruntime-react-native',
    'android'
  );

  // 1. Patch onnxruntime-react-native/android/build.gradle
  const ortBuildGradlePath = path.join(ortAndroidDir, 'build.gradle');
  if (fs.existsSync(ortBuildGradlePath)) {
    let buildGradle = fs.readFileSync(ortBuildGradlePath, 'utf8');

    // Replace dynamic 'latest.integration' with concrete version to avoid Maven metadata XML SAX parser crash
    buildGradle = buildGradle.replace(
      /com\.microsoft\.onnxruntime:onnxruntime-android-qnn:latest\.integration@aar/g,
      `com.microsoft.onnxruntime:onnxruntime-android-qnn:${ORT_ANDROID_VERSION}@aar`
    );
    buildGradle = buildGradle.replace(
      /com\.microsoft\.onnxruntime:onnxruntime-android:latest\.integration@aar/g,
      `com.microsoft.onnxruntime:onnxruntime-android:${ORT_ANDROID_VERSION}@aar`
    );
    buildGradle = buildGradle.replace(
      /com\.microsoft\.onnxruntime:onnxruntime-extensions-android:latest\.integration@aar/g,
      `com.microsoft.onnxruntime:onnxruntime-extensions-android:${ORT_EXTENSIONS_VERSION}@aar`
    );

    // Ensure VersionNumber check doesn't fail on Gradle 8.10+ (RN 0.76 is >= 0.71 anyway)
    buildGradle = buildGradle.replace(
      /if\s*\(\s*VersionNumber\.parse\(REACT_NATIVE_VERSION\)\s*<\s*VersionNumber\.parse\("0\.71"\)\s*\)/g,
      'if (REACT_NATIVE_MINOR_VERSION < 71)'
    );

    fs.writeFileSync(ortBuildGradlePath, buildGradle, 'utf8');
  }

  // 2. Patch onnxruntime-react-native/android/gradle.properties to remove broken SAXParserFactory systemProps
  const ortGradlePropsPath = path.join(ortAndroidDir, 'gradle.properties');
  if (fs.existsSync(ortGradlePropsPath)) {
    let gradleProps = fs.readFileSync(ortGradlePropsPath, 'utf8');
    gradleProps = gradleProps
      .split('\n')
      .filter((line) => !line.trim().startsWith('systemProp.javax.xml.'))
      .join('\n');
    fs.writeFileSync(ortGradlePropsPath, gradleProps, 'utf8');
  }
}

function withAndroidNpu(config) {
  config = withAndroidManifest(config, (modConfig) => {
    const mainApp = modConfig.modResults.manifest.application?.[0];
    if (mainApp) {
      mainApp.$['android:largeHeap'] = 'true';
      mainApp.$['android:hardwareAccelerated'] = 'true';
    }
    return modConfig;
  });

  // Patch node_modules/onnxruntime-react-native/android during prebuild
  config = withDangerousMod(config, [
    'android',
    async (modConfig) => {
      patchOnnxruntimeReactNativeNodeModule(modConfig.modRequest.projectRoot);
      return modConfig;
    },
  ]);

  config = withAppBuildGradle(config, (modConfig) => {
    if (modConfig.modResults.language === 'groovy') {
      let contents = modConfig.modResults.contents;
      if (!contents.includes('pickFirst \'**/libonnxruntime.so\'')) {
        const packagingBlock = `
    packagingOptions {
        pickFirst '**/libonnxruntime.so'
        pickFirst '**/libonnxruntime4j_jni.so'
        pickFirst '**/libonnxruntimejsi.so'
        pickFirst '**/libc++_shared.so'
        pickFirst '**/libfbjni.so'
    }
`;
        contents = contents.replace(/android\s*\{/, `android {\n${packagingBlock}`);
      }

      // Ensure com.microsoft.onnxruntime:onnxruntime-android is included as an app dependency
      // so libonnxruntime.so is packaged into the final APK's lib/<abi>/ directory.
      if (!contents.includes('com.microsoft.onnxruntime:onnxruntime-android:')) {
        contents = contents.replace(
          /dependencies\s*\{/,
          `dependencies {\n    implementation("com.microsoft.onnxruntime:onnxruntime-android:${ORT_ANDROID_VERSION}")`
        );
      }

      modConfig.modResults.contents = contents;
    }
    return modConfig;
  });

  config = withProjectBuildGradle(config, (modConfig) => {
    if (modConfig.modResults.language === 'groovy') {
      let contents = modConfig.modResults.contents;
      contents = contents.replace(
        /kotlinVersion\s*=\s*findProperty\(['"]android\.kotlinVersion['"]\)\s*\?:\s*['"][^'"]+['"]/,
        "kotlinVersion = '1.9.25'"
      );
      if (!contents.includes('suppressKotlinVersionCompatibilityCheck')) {
        contents += `
allprojects {
    configurations.all {
        resolutionStrategy {
            force "com.microsoft.onnxruntime:onnxruntime-android:${ORT_ANDROID_VERSION}"
            force "com.microsoft.onnxruntime:onnxruntime-android-qnn:${ORT_ANDROID_VERSION}"
            force "com.microsoft.onnxruntime:onnxruntime-extensions-android:${ORT_EXTENSIONS_VERSION}"
        }
    }
    tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile).configureEach {
        kotlinOptions {
            freeCompilerArgs += [
                "-P",
                "plugin:androidx.compose.compiler.plugins.kotlin:suppressKotlinVersionCompatibilityCheck=true"
            ]
        }
    }
}
`;
      }
      modConfig.modResults.contents = contents;
    }
    return modConfig;
  });

  config = withGradleProperties(config, (modConfig) => {
    const propsToSet = {
      'android.kotlinVersion': '1.9.25',
      'kotlinVersion': '1.9.25',
      'suppressKotlinVersionCompatibilityCheck': 'true',
      'androidx.compose.compiler.plugins.kotlin.suppressKotlinVersionCompatibilityCheck': 'true',
      'reactNativeArchitectures': 'arm64-v8a,armeabi-v7a',
      'org.gradle.jvmargs': '-Xmx4096m -XX:MaxMetaspaceSize=1024m -Dfile.encoding=UTF-8',
      'org.gradle.parallel': 'true',
    };

    for (const [key, value] of Object.entries(propsToSet)) {
      const existingIdx = modConfig.modResults.findIndex(
        (item) => item.type === 'property' && item.key === key
      );
      if (existingIdx >= 0) {
        modConfig.modResults[existingIdx].value = value;
      } else {
        modConfig.modResults.push({
          type: 'property',
          key,
          value,
        });
      }
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
