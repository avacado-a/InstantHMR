/**
 * Causal 1-Euro Temporal Smoother & Bounding Box Stabilizer for InstantHMR Mobile.
 *
 * Direct TypeScript port of `instanthmr/smoothing.py`:
 * - `_OneEuro` filter (Casiez, Roussel, and Vogel, CHI 2012)
 * - `BoxStabilizer` (mincutoff=0.8 Hz, beta=0.02) eliminating 4-frame detector stride jitter
 * - `TemporalSmoother` matching people across frames by IoU and smoothing:
 *     * bbox (0.8 Hz, beta=0.02)
 *     * joints2d (1.0 Hz, beta=0.02)
 *     * joints3dLocal (1.0 Hz, beta=4.0)
 *     * camTrans (0.6 Hz, beta=2.0)
 */
import { PersonPrediction } from '../types/config';

export class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  private x: Float64Array | null = null;
  private dx: Float64Array | null = null;
  private t: number | null = null;

  constructor(minCutoff: number, beta: number, dCutoff: number = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  private static alpha(cutoff: number, dt: number): number {
    const c = Math.max(cutoff, 1e-6);
    return 1.0 / (1.0 + 1.0 / (2.0 * Math.PI * c * dt));
  }

  public filterArray(input: Float32Array | number[], timestampS: number): Float32Array {
    const n = input.length;
    if (this.x === null || this.dx === null || this.t === null || this.x.length !== n) {
      this.x = new Float64Array(n);
      this.dx = new Float64Array(n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        this.x[i] = input[i];
        out[i] = input[i];
      }
      this.t = timestampS;
      return out;
    }

    const dt = timestampS > this.t ? timestampS - this.t : 1.0 / 30.0;
    this.t = timestampS;
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    const out = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const val = input[i];
      const prevX = this.x[i];
      const prevDx = this.dx[i];
      const curDx = aD * ((val - prevX) / dt) + (1.0 - aD) * prevDx;
      this.dx[i] = curDx;

      const cutoff = this.minCutoff + this.beta * Math.abs(curDx);
      const a = OneEuroFilter.alpha(cutoff, dt);
      const nextX = a * val + (1.0 - a) * prevX;
      this.x[i] = nextX;
      out[i] = nextX;
    }

    return out;
  }

  public reset(): void {
    this.x = null;
    this.dx = null;
    this.t = null;
  }
}

/**
 * Stabilizes bounding box tracking between detector runs.
 * Tuned for pixel coordinates (minCutoff=0.8 Hz, beta=0.02).
 */
export class BoxStabilizer {
  private filter: OneEuroFilter;
  private currentBox: [number, number, number, number] | null = null;
  private currentScore: number = 0.0;

  constructor(minCutoff: number = 0.8, beta: number = 0.02) {
    this.filter = new OneEuroFilter(minCutoff, beta);
  }

  public reset(): void {
    this.filter.reset();
    this.currentBox = null;
    this.currentScore = 0.0;
  }

  public updateDetection(
    box: [number, number, number, number],
    score: number,
    timestampS: number
  ): [number, number, number, number] {
    this.currentScore = score;
    const filtered = this.filter.filterArray(box, timestampS);
    this.currentBox = [filtered[0], filtered[1], filtered[2], filtered[3]];
    return this.currentBox;
  }

  public getBox(timestampS: number): [number, number, number, number] | null {
    if (!this.currentBox) {
      return null;
    }
    const filtered = this.filter.filterArray(this.currentBox, timestampS);
    this.currentBox = [filtered[0], filtered[1], filtered[2], filtered[3]];
    return this.currentBox;
  }

  public getScore(): number {
    return this.currentScore;
  }
}

export function computeIoU(
  a: [number, number, number, number],
  b: [number, number, number, number]
): number {
  const iw = Math.max(0.0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0.0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const areaA = Math.max(0.0, a[2] - a[0]) * Math.max(0.0, a[3] - a[1]);
  const areaB = Math.max(0.0, b[2] - b[0]) * Math.max(0.0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0.0 ? inter / union : 0.0;
}

interface TrackState {
  box: [number, number, number, number];
  age: number;
  bboxFilter: OneEuroFilter;
  joints2dFilter: OneEuroFilter;
  joints3dLocalFilter: OneEuroFilter;
  camTransFilter: OneEuroFilter;
}

/**
 * Multi-person temporal smoother using IoU track assignment and unit-calibrated 1-Euro filters.
 */
export class TemporalSmoother {
  private betaScale: number;
  private iouThreshold: number;
  private maxAge: number;
  private tracks: Map<number, TrackState> = new Map();
  private nextId: number = 1;

  constructor(betaScale: number = 1.0, iouThreshold: number = 0.3, maxAge: number = 5) {
    this.betaScale = betaScale;
    this.iouThreshold = iouThreshold;
    this.maxAge = maxAge;
  }

  public setBetaScale(betaScale: number): void {
    this.betaScale = betaScale;
  }

  public reset(): void {
    this.tracks.clear();
    this.nextId = 1;
  }

  public smooth(persons: PersonPrediction[], timestampS: number): PersonPrediction[] {
    const assignedIds = this.assignTracks(persons);
    const results: PersonPrediction[] = [];

    for (let i = 0; i < persons.length; i++) {
      const person = persons[i];
      const trackId = assignedIds[i];
      const track = this.tracks.get(trackId)!;

      const sb = track.bboxFilter.filterArray(person.bbox, timestampS);
      const smoothedBbox: [number, number, number, number] = [sb[0], sb[1], sb[2], sb[3]];

      const smoothedJoints2d = track.joints2dFilter.filterArray(person.joints2d, timestampS);
      const smoothedJoints3dLocal = track.joints3dLocalFilter.filterArray(
        person.joints3dLocal,
        timestampS
      );
      const sCam = track.camTransFilter.filterArray(person.camTrans, timestampS);
      const smoothedCamTrans: [number, number, number] = [sCam[0], sCam[1], sCam[2]];

      const smoothedJoints3dCam = new Float32Array(smoothedJoints3dLocal.length);
      for (let j = 0; j < smoothedJoints3dLocal.length; j += 3) {
        smoothedJoints3dCam[j] = smoothedJoints3dLocal[j] + smoothedCamTrans[0];
        smoothedJoints3dCam[j + 1] = smoothedJoints3dLocal[j + 1] + smoothedCamTrans[1];
        smoothedJoints3dCam[j + 2] = smoothedJoints3dLocal[j + 2] + smoothedCamTrans[2];
      }

      results.push({
        ...person,
        id: trackId,
        bbox: smoothedBbox,
        joints2d: smoothedJoints2d,
        joints3dLocal: smoothedJoints3dLocal,
        joints3dCam: smoothedJoints3dCam,
        camTrans: smoothedCamTrans,
      });
    }

    return results;
  }

  private assignTracks(persons: PersonPrediction[]): number[] {
    const pairs: { score: number; personIdx: number; trackId: number }[] = [];

    for (let i = 0; i < persons.length; i++) {
      for (const [trackId, tr] of this.tracks.entries()) {
        pairs.push({
          score: computeIoU(persons[i].bbox, tr.box),
          personIdx: i,
          trackId,
        });
      }
    }

    pairs.sort((a, b) => b.score - a.score);

    const ids: number[] = new Array(persons.length).fill(-1);
    const usedTracks = new Set<number>();

    for (const pair of pairs) {
      if (
        pair.score >= this.iouThreshold &&
        ids[pair.personIdx] === -1 &&
        !usedTracks.has(pair.trackId)
      ) {
        ids[pair.personIdx] = pair.trackId;
        usedTracks.add(pair.trackId);
      }
    }

    const bs = this.betaScale;
    for (let i = 0; i < persons.length; i++) {
      if (ids[i] === -1) {
        const newId = this.nextId++;
        ids[i] = newId;
        this.tracks.set(newId, {
          box: persons[i].bbox,
          age: 0,
          bboxFilter: new OneEuroFilter(0.8, 0.02 * bs),
          joints2dFilter: new OneEuroFilter(1.0, 0.02 * bs),
          joints3dLocalFilter: new OneEuroFilter(1.0, 4.0 * bs),
          camTransFilter: new OneEuroFilter(0.6, 2.0 * bs),
        });
      } else {
        const tr = this.tracks.get(ids[i])!;
        tr.box = persons[i].bbox;
        tr.age = 0;
      }
    }

    const activeIds = new Set(ids);
    for (const [trackId, tr] of this.tracks.entries()) {
      if (!activeIds.has(trackId)) {
        tr.age += 1;
        if (tr.age > this.maxAge) {
          this.tracks.delete(trackId);
        }
      }
    }

    return ids;
  }
}
