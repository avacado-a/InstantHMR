# InstantHMR Mobile Studio: Step-by-Step Installation Guide (Android & iOS)

This guide shows you the exact steps to install and run the **InstantHMR Mobile Studio** app on your physical **Android** phone or **iPhone**.

---

## 🤖 For Android Phones (Samsung Galaxy, Google Pixel, Xiaomi, OnePlus, etc.)

You have **two super easy methods**:

### Method 1: The Easiest Way — Download Standalone `.apk` File Directly (No Android Studio Needed!)

1. **Build the APK in the cloud with 1 command**:
   Inside the `mobile/` directory on your computer, run:
   ```bash
   npx eas-cli build -p android --profile preview
   ```
   *Every commit to `mobile-app` automatically compiles and publishes a fresh standalone `.apk` directly to [GitHub Releases](https://github.com/avacado-a/InstantHMR/releases).*

   *(You can also build the APK locally or via Expo EAS with `npx eas-cli build -p android --profile preview`).*


2. **Install on your phone**:
   - Open the download link on your Android phone's browser (or scan the QR code).
   - Tap **Download APK** and open it.
   - If prompted: **"Install unknown apps"**, toggle **Allow from this source** (standard on Android for direct APK installs).
   - Tap **Install**. The **InstantHMR Studio** icon will now appear on your phone's home screen!

---

### Method 2: Via USB Cable (Plug & Play)

If you have a computer with Node.js and a USB cable:

1. **Enable Developer Options on your phone**:
   - Go to **Settings** > **About phone**.
   - Tap **Build number** 7 times until you see *"You are now a developer!"*.
   - Go back to **Settings** > **System / Developer options** and turn on **USB debugging**.

2. **Plug your phone into your computer with a USB cable**:
   - When your phone asks *"Allow USB debugging?"*, tap **Always allow**.

3. **Run the installer from your terminal**:
   ```bash
   git checkout mobile-app
   cd mobile
   npm run android
   ```
   *This automatically builds the app, installs it via `adb` over the USB cable, and launches InstantHMR Studio directly on your phone's screen.*

---

## 🍏 For Apple iPhones (iPhone 12 / 13 / 14 / 15 / 16 / Pro / Max)

Because iOS uses strict Apple code signing, there are two straightforward methods:

### Method 1: Connect iPhone to Mac with USB Cable (Recommended, 5 Minutes)

1. **Enable Developer Mode on your iPhone**:
   - On iOS 16, 17, or 18: Go to **Settings** > **Privacy & Security**.
   - Scroll down to the bottom and tap **Developer Mode**.
   - Toggle it **ON** and restart your iPhone when prompted.

2. **Plug your iPhone into your Mac**:
   - Connect your iPhone to your Mac with a Lightning or USB-C cable.
   - Unlock your iPhone and tap **Trust this computer**.

3. **Install with 1 command**:
   Inside the `mobile/` directory:
   ```bash
   git checkout mobile-app
   cd mobile
   npm run ios -- --device
   ```
   *(Or generate the Xcode project with `npx expo prebuild -p ios`, open `mobile/ios/InstantHMRStudio.xcworkspace` in Xcode, select your personal free Apple ID under **Signing & Capabilities**, and press the ▶ **Run** button).*

4. **First-time Trust**:
   - Once installed on your iPhone, go to **Settings** > **General** > **VPN & Device Management**.
   - Tap your Apple ID and tap **Trust "InstantHMR Studio"**.
   - Tap the **InstantHMR Studio** icon on your home screen to launch!

---

### Method 2: Wireless Cloud Build via EAS / TestFlight

If you have an Apple Developer account:
```bash
cd mobile
npx eas-cli build -p ios --profile preview
```
Scan the resulting QR code with your iPhone's camera to install directly via Apple Internal Distribution or TestFlight.

---

## 🚀 First Launch Inside the App

1. When the app opens, tap **Allow Camera Access** when prompted.
2. On the **Pre-Launch Studio Setup** screen:
   - Select **Front Camera** (for selfies/workouts) or **Back Camera** (for room tracking).
   - Ensure **Hardware Compute** is set to **NPU** (Apple CoreML Neural Engine on iPhone, or Qualcomm Hexagon / Android NNAPI on Android).
   - Tap **Launch Live Studio** to begin real-time 3D human mesh & 70-keypoint pose estimation!
3. Tap **Models** in the top bar anytime to download larger InstantHMR model variants (`small`, `medium`) or new detectors (`yolo26s`, `rfdetr_nano`).
