/**
 * InstantHMR Mobile Model Manager (On-Demand Download, Progress Tracking, & Local Cache).
 *
 * Manages downloading and caching ONNX weights (`mhr-body.onnx`, `instanthmr-*.onnx`,
 * and `yolo26*.onnx` / `rfdetr_nano.onnx`) inside the device's persistent document directory.
 */
import * as FileSystem from 'expo-file-system';
import {
  DETECTORS_INFO,
  HMR_MODELS_INFO,
  MODEL_ASSETS_REGISTRY,
  ModelAssetSpec,
} from '../constants/hardware';
import { DetectorVariant, HmrModelVariant } from '../types/config';

export interface ModelInstallState {
  filename: string;
  installed: boolean;
  downloading: boolean;
  progress: number; // 0.0 to 1.0
  downloadedMb: number;
  totalMb: number;
  localUri: string | null;
  error: string | null;
}

const MODELS_SUBDIR = `${FileSystem.documentDirectory || 'file:///data/instanthmr/'}models/`;

export class MobileModelManager {
  private states: Map<string, ModelInstallState> = new Map();
  private listeners: Set<(states: Record<string, ModelInstallState>) => void> = new Set();

  constructor() {
    for (const [filename, spec] of Object.entries(MODEL_ASSETS_REGISTRY)) {
      this.states.set(filename, {
        filename,
        installed: false,
        downloading: false,
        progress: 0,
        downloadedMb: 0,
        totalMb: spec.sizeMb,
        localUri: null,
        error: null,
      });
    }
  }

  public subscribe(listener: (states: Record<string, ModelInstallState>) => void): () => void {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => {
      this.listeners.delete(listener);
    };
  }

  public getSnapshot(): Record<string, ModelInstallState> {
    const out: Record<string, ModelInstallState> = {};
    for (const [k, v] of this.states.entries()) {
      out[k] = { ...v };
    }
    return out;
  }

  private notify(): void {
    const snap = this.getSnapshot();
    for (const listener of this.listeners) {
      listener(snap);
    }
  }

  public async scanInstalledModels(): Promise<Record<string, ModelInstallState>> {
    try {
      const dirInfo = await FileSystem.getInfoAsync(MODELS_SUBDIR);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(MODELS_SUBDIR, { intermediates: true });
      }

      for (const [filename, spec] of Object.entries(MODEL_ASSETS_REGISTRY)) {
        const fileUri = `${MODELS_SUBDIR}${filename}`;
        const info = await FileSystem.getInfoAsync(fileUri);
        const state = this.states.get(filename)!;
        if (info.exists && (info.size || 0) > 100 * 1024) {
          state.installed = true;
          state.progress = 1.0;
          state.downloadedMb = Math.round(((info.size || 0) / (1024 * 1024)) * 10) / 10;
          state.localUri = fileUri;
        } else {
          state.installed = false;
          state.localUri = null;
        }
      }
    } catch {
      // Fallback when running in environments without native FileSystem
    }
    this.notify();
    return this.getSnapshot();
  }

  public async downloadModel(filename: string): Promise<string> {
    const spec: ModelAssetSpec | undefined = MODEL_ASSETS_REGISTRY[filename];
    if (!spec) {
      throw new Error(`Unknown model asset: ${filename}`);
    }

    const state = this.states.get(filename)!;
    if (state.installed && state.localUri) {
      return state.localUri;
    }

    state.downloading = true;
    state.progress = 0.01;
    state.error = null;
    this.notify();

    try {
      const dirInfo = await FileSystem.getInfoAsync(MODELS_SUBDIR);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(MODELS_SUBDIR, { intermediates: true });
      }

      const targetUri = `${MODELS_SUBDIR}${filename}`;
      const downloadResumable = FileSystem.createDownloadResumable(
        spec.url,
        targetUri,
        {},
        (downloadProgress) => {
          const expectedBytes =
            downloadProgress.totalBytesExpectedToWrite > 0
              ? downloadProgress.totalBytesExpectedToWrite
              : spec.sizeMb * 1024 * 1024;
          const ratio = Math.min(
            1.0,
            downloadProgress.totalBytesWritten / Math.max(expectedBytes, 1)
          );
          state.progress = ratio;
          state.downloadedMb =
            Math.round((downloadProgress.totalBytesWritten / (1024 * 1024)) * 10) / 10;
          this.notify();
        }
      );

      const res = await downloadResumable.downloadAsync();
      if (!res || res.status < 200 || res.status >= 300) {
        throw new Error(`HTTP ${res?.status || 'Download error'}`);
      }

      state.downloading = false;
      state.installed = true;
      state.progress = 1.0;
      state.downloadedMb = spec.sizeMb;
      state.localUri = res.uri;
      this.notify();
      return res.uri;
    } catch (err: any) {
      state.downloading = false;
      state.error = err?.message || 'Failed to download model';
      this.notify();
      throw err;
    }
  }

  /**
   * Ensures all models required by the user's selected `hmrModel`, `detector`, and (if in compare mode)
   * `compareDetector` are ready on device.
   */
  public getRequiredFilesForConfig(
    hmrModel: HmrModelVariant,
    detector: DetectorVariant,
    compareMode: boolean,
    compareDetector: DetectorVariant
  ): string[] {
    const files = new Set<string>(['mhr-body.onnx']);
    files.add(HMR_MODELS_INFO[hmrModel].filename);
    files.add(DETECTORS_INFO[detector].filename);
    if (compareMode) {
      files.add(DETECTORS_INFO[compareDetector].filename);
    }
    return Array.from(files);
  }

  public async deleteModel(filename: string): Promise<void> {
    const state = this.states.get(filename);
    if (!state) return;
    if (state.localUri) {
      try {
        await FileSystem.deleteAsync(state.localUri, { idempotent: true });
      } catch {
        // ignore
      }
    }
    state.installed = false;
    state.progress = 0;
    state.downloadedMb = 0;
    state.localUri = null;
    this.notify();
  }
}

export const sharedModelManager = new MobileModelManager();
