#!/usr/bin/env bash
# ==============================================================================
# InstantHMR Mobile Studio — One-Step Installer & Builder (Android & iOS)
#
# Usage:
#   ./mobile/install_mobile.sh              # Install dependencies + verify pipeline
#   ./mobile/install_mobile.sh --android    # Install + build & launch on Android device/emulator
#   ./mobile/install_mobile.sh --ios        # Install + pod install + build & launch on iOS device/simulator
#   ./mobile/install_mobile.sh --bundle     # Also pre-download & bundle default ONNX models offline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ANDROID=0
RUN_IOS=0
BUNDLE_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --android) RUN_ANDROID=1 ;;
    --ios) RUN_IOS=1 ;;
    --bundle) BUNDLE_MODELS=1 ;;
    -h|--help)
      echo "Usage: ./mobile/install_mobile.sh [--android] [--ios] [--bundle]"
      exit 0
      ;;
  esac
done

echo "========================================================================"
echo "  InstantHMR Mobile Studio v2.0 — Easy Installer (Android & iOS)"
echo "========================================================================"

cd "${SCRIPT_DIR}"

echo "[1/4] Checking Node.js & package manager..."
if ! command -v node >/dev/null 2>&1; then
  echo "Error: Node.js is required (>= 18). Please install Node.js first."
  exit 1
fi
echo "  ✓ Node $(node -v) detected"

echo "[2/4] Installing React Native, Expo, VisionCamera, and ONNX Runtime packages..."
npm install --legacy-peer-deps

if [ "${BUNDLE_MODELS}" -eq 1 ]; then
  echo "[3/4] Downloading & bundling core ONNX models (mhr-body + instanthmr-nano + yolo26n)..."
  python3 "${SCRIPT_DIR}/scripts/prepare_mobile_models.py" --bundle
else
  echo "[3/4] Skipping offline model bundle (models download on-demand inside the app's Model Manager UI)."
  echo "      Tip: pass --bundle to pre-bundle ONNX weights into the app binary."
fi

echo "[4/4] Verifying TypeScript & InstantHMR mobile math/smoothing pipeline..."
node "${SCRIPT_DIR}/scripts/verify_pipeline.mjs"

if [ "${RUN_ANDROID}" -eq 1 ]; then
  echo "→ Prebuilding & launching Android app (NNAPI / QNN NPU + XNNPACK CPU)..."
  npx expo run:android
elif [ "${RUN_IOS}" -eq 1 ]; then
  echo "→ Prebuilding & launching iOS app (CoreML Neural Engine + Metal + CPU)..."
  npx expo run:ios
else
  echo ""
  echo "========================================================================"
  echo "  ✓ InstantHMR Mobile Studio is installed and ready!"
  echo ""
  echo "  Quick Start Commands (from ./mobile):"
  echo "    • Run on Android (NPU/CPU):   npm run android"
  echo "    • Run on iOS (CoreML ANE):    npm run ios"
  echo "    • Start Metro Dev Server:     npm run start"
  echo "    • Pre-bundle ONNX models:     npm run bundle-models"
  echo "========================================================================"
fi
