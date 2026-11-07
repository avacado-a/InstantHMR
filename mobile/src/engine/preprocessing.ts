/**
 * InstantHMR Preprocessing, CLIFF Conditioning, & Post-processing Engine.
 *
 * Matches `instanthmr/inference.py` and `instanthmr/detector.py` 1:1:
 * - Focal length estimation (`1.05 * hypot(H, W)`)
 * - CLIFF conditioning (`[atan((cx - W/2)/f), atan((cy - H/2)/f), size/f]`)
 * - Square crop extraction with `cropExpand` (1.2x - 1.4x), zero padding, and bilinear resize to 288x288
 * - ImageNet RGB normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) in CHW layout
 * - 2D keypoint unprojection from `[-1, 1]` crop coordinates back to full-frame pixel coordinates
 * - 17-keypoint Pose Envelope box padding for outstretched limbs (`yolo26n-pose` / `yolo26s-pose`)
 */
import { NUM_JOINTS } from '../constants/skeleton';
import { PersonPrediction } from '../types/config';

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;
export const FOCAL_FALLBACK_DIAG = 1.05;

export interface CropPreprocessResult {
  /** Normalized NCHW float32 tensor of shape (1, 3, inputSize, inputSize) */
  cropTensor: Float32Array;
  /** CLIFF conditioning vector of shape (3,) */
  cliffCond: Float32Array;
  /** Square crop geometry [sx, sy, side] in source image pixels */
  square: [number, number, number];
}

/**
 * Computes the camera focal length in pixels.
 * Uses user-specified focal length if provided, otherwise `1.05 * hypot(height, width)`.
 */
export function computeFocalLengthPx(
  height: number,
  width: number,
  customFocalPx: number | null = null,
  cliffFocal: boolean = true
): number {
  if (customFocalPx !== null && customFocalPx > 0) {
    return customFocalPx;
  }
  const factor = cliffFocal ? FOCAL_FALLBACK_DIAG : 1.0;
  return factor * Math.hypot(height, width);
}

/**
 * Computes the 3-element CLIFF camera conditioning vector and square crop bounds
 * for a tight person bounding box `[x1, y1, x2, y2]`.
 */
export function computeCliffAndSquareBounds(
  bbox: [number, number, number, number],
  imageWidth: number,
  imageHeight: number,
  focalPx: number,
  cropExpand: number = 1.2,
  cliffFocal: boolean = true
): {
  cliffCond: Float32Array;
  square: [number, number, number];
} {
  const [x1, y1, x2, y2] = bbox;
  const cx = (x1 + x2) * 0.5;
  const cy = (y1 + y2) * 0.5;
  const size = Math.max(x2 - x1, y2 - y1, 1.0);

  const cliffCond = new Float32Array(3);
  if (cliffFocal) {
    cliffCond[0] = Math.atan((cx - imageWidth * 0.5) / focalPx);
    cliffCond[1] = Math.atan((cy - imageHeight * 0.5) / focalPx);
    cliffCond[2] = size / focalPx;
  } else {
    cliffCond[0] = (2.0 * cx) / imageWidth - 1.0;
    cliffCond[1] = (2.0 * cy) / imageHeight - 1.0;
    cliffCond[2] = size / Math.max(imageWidth, imageHeight);
  }

  const side = size * cropExpand;
  const sx = cx - side * 0.5;
  const sy = cy - side * 0.5;

  return {
    cliffCond,
    square: [sx, sy, side],
  };
}

/**
 * Extracts a square `cropExpand` patch from an RGB/RGBA uint8 buffer, pads out-of-frame borders
 * with zeros, resizes with bilinear interpolation to `(inputSize, inputSize)`, and normalizes
 * with ImageNet mean & std into a contiguous CHW `Float32Array`.
 */
export function preprocessPersonCrop(
  imageBuffer: Uint8Array,
  imageWidth: number,
  imageHeight: number,
  channels: 3 | 4,
  bbox: [number, number, number, number],
  focalPx: number,
  inputSize: number = 288,
  cropExpand: number = 1.2,
  cliffFocal: boolean = true
): CropPreprocessResult {
  const { cliffCond, square } = computeCliffAndSquareBounds(
    bbox,
    imageWidth,
    imageHeight,
    focalPx,
    cropExpand,
    cliffFocal
  );
  const [sx, sy, side] = square;

  const planeSize = inputSize * inputSize;
  const cropTensor = new Float32Array(3 * planeSize);
  const scale = side / inputSize;

  const inv255 = 1.0 / 255.0;
  const meanR = IMAGENET_MEAN[0];
  const meanG = IMAGENET_MEAN[1];
  const meanB = IMAGENET_MEAN[2];
  const invStdR = 1.0 / IMAGENET_STD[0];
  const invStdG = 1.0 / IMAGENET_STD[1];
  const invStdB = 1.0 / IMAGENET_STD[2];

  for (let dstY = 0; dstY < inputSize; dstY++) {
    const srcY = sy + (dstY + 0.5) * scale - 0.5;
    const y0 = Math.floor(srcY);
    const y1 = y0 + 1;
    const wy = srcY - y0;
    const wy0 = 1.0 - wy;

    for (let dstX = 0; dstX < inputSize; dstX++) {
      const srcX = sx + (dstX + 0.5) * scale - 0.5;
      const x0 = Math.floor(srcX);
      const x1 = x0 + 1;
      const wx = srcX - x0;
      const wx0 = 1.0 - wx;

      let r = 0.0;
      let g = 0.0;
      let b = 0.0;

      // Sample 4 bilinear neighbors with zero-padding outside [0, W-1] x [0, H-1]
      if (y0 >= 0 && y0 < imageHeight) {
        const rowOffset0 = y0 * imageWidth * channels;
        if (x0 >= 0 && x0 < imageWidth) {
          const idx = rowOffset0 + x0 * channels;
          const w00 = wy0 * wx0;
          r += imageBuffer[idx] * w00;
          g += imageBuffer[idx + 1] * w00;
          b += imageBuffer[idx + 2] * w00;
        }
        if (x1 >= 0 && x1 < imageWidth) {
          const idx = rowOffset0 + x1 * channels;
          const w01 = wy0 * wx;
          r += imageBuffer[idx] * w01;
          g += imageBuffer[idx + 1] * w01;
          b += imageBuffer[idx + 2] * w01;
        }
      }

      if (y1 >= 0 && y1 < imageHeight) {
        const rowOffset1 = y1 * imageWidth * channels;
        if (x0 >= 0 && x0 < imageWidth) {
          const idx = rowOffset1 + x0 * channels;
          const w10 = wy * wx0;
          r += imageBuffer[idx] * w10;
          g += imageBuffer[idx + 1] * w10;
          b += imageBuffer[idx + 2] * w10;
        }
        if (x1 >= 0 && x1 < imageWidth) {
          const idx = rowOffset1 + x1 * channels;
          const w11 = wy * wx;
          r += imageBuffer[idx] * w11;
          g += imageBuffer[idx + 1] * w11;
          b += imageBuffer[idx + 2] * w11;
        }
      }

      const outIdx = dstY * inputSize + dstX;
      cropTensor[outIdx] = (r * inv255 - meanR) * invStdR;
      cropTensor[planeSize + outIdx] = (g * inv255 - meanG) * invStdG;
      cropTensor[2 * planeSize + outIdx] = (b * inv255 - meanB) * invStdB;
    }
  }

  return {
    cropTensor,
    cliffCond,
    square,
  };
}

/**
 * Unprojects normalized `[-1, 1]` crop 2D keypoints back to full-frame pixel coordinates:
 * `j2d = (joints_2d + 1.0) * 0.5 * side + [sx, sy]`
 */
export function unprojectCropJoints2d(
  normJoints2d: Float32Array,
  personOffset: number,
  square: [number, number, number]
): Float32Array {
  const [sx, sy, side] = square;
  const halfSide = 0.5 * side;
  const out = new Float32Array(NUM_JOINTS * 2);
  const base = personOffset * NUM_JOINTS * 2;

  for (let j = 0; j < NUM_JOINTS; j++) {
    const nx = normJoints2d[base + j * 2];
    const ny = normJoints2d[base + j * 2 + 1];
    out[j * 2] = (nx + 1.0) * halfSide + sx;
    out[j * 2 + 1] = (ny + 1.0) * halfSide + sy;
  }
  return out;
}

/**
 * Applies the 17-keypoint YOLO-Pose envelope expansion (`instanthmr/detector.py` lines 100-115)
 * so outstretched arms/wrists/ankles are guaranteed to stay inside the bounding box.
 */
export function applyPoseEnvelopeToBox(
  box: [number, number, number, number],
  keypoints17xy: Float32Array,
  imageWidth: number,
  imageHeight: number,
  padPx: number = 12.0
): [number, number, number, number] {
  let kMinX = Infinity;
  let kMinY = Infinity;
  let kMaxX = -Infinity;
  let kMaxY = -Infinity;
  let validCount = 0;

  for (let k = 0; k < 17; k++) {
    const kx = keypoints17xy[k * 2];
    const ky = keypoints17xy[k * 2 + 1];
    if (kx > 0 || ky > 0) {
      validCount++;
      if (kx < kMinX) kMinX = kx;
      if (ky < kMinY) kMinY = ky;
      if (kx > kMaxX) kMaxX = kx;
      if (ky > kMaxY) kMaxY = ky;
    }
  }

  if (validCount === 0) {
    return box;
  }

  return [
    Math.max(0.0, Math.min(box[0], kMinX - padPx)),
    Math.max(0.0, Math.min(box[1], kMinY - padPx)),
    Math.min(imageWidth, Math.max(box[2], kMaxX + padPx)),
    Math.min(imageHeight, Math.max(box[3], kMaxY + padPx)),
  ];
}

/**
 * Assembles a complete `PersonPrediction` from the raw ONNX network & MHR body outputs.
 */
export function buildPersonPrediction(
  id: number,
  bbox: [number, number, number, number],
  confidence: number,
  square: [number, number, number],
  normJoints2d: Float32Array,
  joints3dLocal: Float32Array,
  camTrans: [number, number, number],
  mhrParams: Float32Array,
  shapeParams: Float32Array,
  focalLength: number,
  imageShape: [number, number],
  verticesLocal?: Float32Array
): PersonPrediction {
  const joints2d = unprojectCropJoints2d(normJoints2d, 0, square);
  const joints3dCam = new Float32Array(NUM_JOINTS * 3);
  for (let j = 0; j < NUM_JOINTS; j++) {
    joints3dCam[j * 3] = joints3dLocal[j * 3] + camTrans[0];
    joints3dCam[j * 3 + 1] = joints3dLocal[j * 3 + 1] + camTrans[1];
    joints3dCam[j * 3 + 2] = joints3dLocal[j * 3 + 2] + camTrans[2];
  }

  return {
    id,
    bbox,
    confidence,
    joints2d,
    joints3dLocal,
    joints3dCam,
    camTrans,
    mhrParams,
    shapeParams,
    verticesLocal,
    focalLength,
    imageShape,
  };
}
