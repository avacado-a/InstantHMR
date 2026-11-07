/**
 * MHR70 Joint Ordering and Skeleton Connectivity for InstantHMR Mobile.
 *
 * Matches `instanthmr/skeleton.py` and `instanthmr/visualizer.py` 1:1.
 * The student model regresses 70 keypoints in the MHR70 ordering:
 * Body landmarks first (0-20), then right hand (21-41), then left hand (42-62),
 * then olecranon / cubital / acromion / neck (63-69).
 */
import { COLORS } from './theme';

export const NUM_JOINTS = 70;

export const JOINT_NAMES: readonly string[] = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',          // 0-4
  'left_shoulder', 'right_shoulder',                                  // 5-6
  'left_elbow', 'right_elbow',                                        // 7-8
  'left_hip', 'right_hip',                                            // 9-10
  'left_knee', 'right_knee',                                          // 11-12
  'left_ankle', 'right_ankle',                                        // 13-14
  'left_big_toe_tip', 'left_small_toe_tip', 'left_heel',              // 15-17
  'right_big_toe_tip', 'right_small_toe_tip', 'right_heel',           // 18-20
  'right_thumb_tip', 'right_thumb_first_joint',
  'right_thumb_second_joint', 'right_thumb_third_joint',              // 21-24
  'right_index_tip', 'right_index_first_joint',
  'right_index_second_joint', 'right_index_third_joint',              // 25-28
  'right_middle_tip', 'right_middle_first_joint',
  'right_middle_second_joint', 'right_middle_third_joint',            // 29-32
  'right_ring_tip', 'right_ring_first_joint',
  'right_ring_second_joint', 'right_ring_third_joint',                // 33-36
  'right_pinky_tip', 'right_pinky_first_joint',
  'right_pinky_second_joint', 'right_pinky_third_joint',              // 37-40
  'right_wrist',                                                      // 41
  'left_thumb_tip', 'left_thumb_first_joint',
  'left_thumb_second_joint', 'left_thumb_third_joint',                // 42-45
  'left_index_tip', 'left_index_first_joint',
  'left_index_second_joint', 'left_index_third_joint',                // 46-49
  'left_middle_tip', 'left_middle_first_joint',
  'left_middle_second_joint', 'left_middle_third_joint',              // 50-53
  'left_ring_tip', 'left_ring_first_joint',
  'left_ring_second_joint', 'left_ring_third_joint',                  // 54-57
  'left_pinky_tip', 'left_pinky_first_joint',
  'left_pinky_second_joint', 'left_pinky_third_joint',                // 58-61
  'left_wrist',                                                       // 62
  'left_olecranon', 'right_olecranon',                                // 63-64
  'left_cubital_fossa', 'right_cubital_fossa',                        // 65-66
  'left_acromion', 'right_acromion',                                  // 67-68
  'neck',                                                             // 69
] as const;

export const SKELETON_EDGES: readonly [number, number][] = [
  // Head <-> neck
  [0, 69],
  // Torso
  [69, 5], [69, 6],
  [5, 9], [6, 10], [9, 10],
  // Arms
  [5, 7], [7, 62],
  [6, 8], [8, 41],
  // Legs
  [9, 11], [11, 13], [13, 17],
  [10, 12], [12, 14], [14, 20],
  // Left hand fan
  [62, 42], [62, 46], [62, 50], [62, 54], [62, 58],
  // Right hand fan
  [41, 21], [41, 25], [41, 29], [41, 33], [41, 37],
] as const;

const HEAD_INDICES = new Set([0, 1, 2, 3, 4, 69]);
const ARM_INDICES = new Set([5, 6, 7, 8, 41, 62, 63, 64, 65, 66, 67, 68]);

/**
 * Returns the anatomical hex color for a bone connecting joint `idx1` and `idx2`.
 */
export function getBoneColor(idx1: number, idx2: number): string {
  if (HEAD_INDICES.has(idx1) || HEAD_INDICES.has(idx2)) {
    return COLORS.boneHead;
  }
  if ((idx1 >= 21 && idx1 <= 40) || (idx2 >= 21 && idx2 <= 40)) {
    return COLORS.boneRightHand;
  }
  if ((idx1 >= 42 && idx1 <= 61) || (idx2 >= 42 && idx2 <= 61)) {
    return COLORS.boneLeftHand;
  }
  if (ARM_INDICES.has(idx1) || ARM_INDICES.has(idx2)) {
    return COLORS.boneArms;
  }
  if ((idx1 === 5 || idx1 === 6 || idx1 === 9 || idx1 === 10) &&
      (idx2 === 5 || idx2 === 6 || idx2 === 9 || idx2 === 10)) {
    return COLORS.boneTorso;
  }
  return COLORS.boneLegs;
}

/**
 * Returns the joint radius and hex color for joint index `i`.
 */
export function getJointStyle(i: number): { radius: number; color: string } {
  const isHand = i >= 21 && i <= 61;
  if (isHand) {
    return { radius: 2.2, color: '#FFFFFF' };
  }
  if (HEAD_INDICES.has(i)) {
    return { radius: 3.5, color: COLORS.boneHead };
  }
  return { radius: 3.2, color: COLORS.boneLegs };
}
