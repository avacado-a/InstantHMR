const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Allow bundling .onnx and .ort model files if pre-bundled into assets
config.resolver.assetExts.push('onnx', 'ort', 'bin');

module.exports = config;
