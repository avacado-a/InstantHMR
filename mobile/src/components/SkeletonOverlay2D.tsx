/**
 * 2D Camera Skeleton, 70-Keypoint, & Stabilized Bounding Box Overlay Renderer.
 *
 * Mirrors `draw_person_overlay` from `instanthmr/visualizer.py` using `react-native-svg`
 * for crisp hardware-accelerated vector rendering at 60 FPS on iOS and Android:
 * - Corner-bracketed stabilized bounding box with confidence + metric depth tag (`Z: 2.4m`)
 * - Anatomically color-coded MHR70 bone connections
 * - 70 articulated joints (including 40 hand finger joints)
 */
import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, G, Line, Rect, Text as SvgText } from 'react-native-svg';
import { getBoneColor, getJointStyle, NUM_JOINTS, SKELETON_EDGES } from '../constants/skeleton';
import { COLORS } from '../constants/theme';
import { PersonPrediction } from '../types/config';

interface SkeletonOverlay2DProps {
  persons: PersonPrediction[];
  viewWidth: number;
  viewHeight: number;
  sourceWidth: number;
  sourceHeight: number;
  modelLabel: string;
  boxColor?: string;
  drawBox?: boolean;
  withMeshWireframe?: boolean;
  meshAlpha?: number;
}

export const SkeletonOverlay2D: React.FC<SkeletonOverlay2DProps> = ({
  persons,
  viewWidth,
  viewHeight,
  sourceWidth,
  sourceHeight,
  modelLabel,
  boxColor = COLORS.accentEmerald,
  drawBox = true,
  withMeshWireframe = false,
  meshAlpha = 0.35,
}) => {
  if (viewWidth <= 0 || viewHeight <= 0) {
    return null;
  }

  const scaleX = viewWidth / Math.max(1, sourceWidth);
  const scaleY = viewHeight / Math.max(1, sourceHeight);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg width={viewWidth} height={viewHeight}>
        {persons.map((person) => {
          const [bx1, by1, bx2, by2] = person.bbox;
          const x1 = Math.max(2, bx1 * scaleX);
          const y1 = Math.max(2, by1 * scaleY);
          const x2 = Math.min(viewWidth - 2, bx2 * scaleX);
          const y2 = Math.min(viewHeight - 2, by2 * scaleY);
          const bw = Math.max(0, x2 - x1);
          const bh = Math.max(0, y2 - y1);
          const cornerLen = Math.min(18, Math.max(6, bw * 0.2), Math.max(6, bh * 0.2));

          const zDepth = person.camTrans[2];
          const confPct = Math.round(person.confidence * 100);
          const tagText = `${modelLabel} ${confPct}% • Z:${zDepth.toFixed(2)}m`;
          const tagWidth = Math.min(bw + 40, tagText.length * 6.4 + 14);
          const tagY = Math.max(4, y1 - 22);

          const j2d = person.joints2d;

          return (
            <G key={`person-${person.id}`}>
              {/* Optional Volumetric Mesh Silhouette Ribbons when withMesh is enabled */}
              {withMeshWireframe && (
                <G opacity={meshAlpha}>
                  {[
                    [5, 6, 10, 9],
                    [5, 7, 62],
                    [6, 8, 41],
                    [9, 11, 13],
                    [10, 12, 14],
                  ].map((chain, cIdx) => {
                    const els: React.ReactNode[] = [];
                    for (let k = 0; k < chain.length - 1; k++) {
                      const a = chain[k];
                      const b = chain[k + 1];
                      els.push(
                        <Line
                          key={`mesh-rib-${cIdx}-${k}`}
                          x1={j2d[a * 2] * scaleX}
                          y1={j2d[a * 2 + 1] * scaleY}
                          x2={j2d[b * 2] * scaleX}
                          y2={j2d[b * 2 + 1] * scaleY}
                          stroke={COLORS.accentCyan}
                          strokeWidth={14}
                          strokeLinecap="round"
                        />
                      );
                    }
                    return els;
                  })}
                </G>
              )}

              {/* 1. Stabilized Bounding Box + Corner Target Brackets */}
              {drawBox && bw > 4 && bh > 4 && (
                <G>
                  <Rect
                    x={x1}
                    y={y1}
                    width={bw}
                    height={bh}
                    stroke={boxColor}
                    strokeWidth={1.2}
                    strokeOpacity={0.55}
                    fill="none"
                  />
                  {/* Top-Left Corner */}
                  <Line x1={x1} y1={y1} x2={x1 + cornerLen} y2={y1} stroke={boxColor} strokeWidth={2.5} />
                  <Line x1={x1} y1={y1} x2={x1} y2={y1 + cornerLen} stroke={boxColor} strokeWidth={2.5} />
                  {/* Top-Right Corner */}
                  <Line x1={x2} y1={y1} x2={x2 - cornerLen} y2={y1} stroke={boxColor} strokeWidth={2.5} />
                  <Line x1={x2} y1={y1} x2={x2} y2={y1 + cornerLen} stroke={boxColor} strokeWidth={2.5} />
                  {/* Bottom-Left Corner */}
                  <Line x1={x1} y1={y2} x2={x1 + cornerLen} y2={y2} stroke={boxColor} strokeWidth={2.5} />
                  <Line x1={x1} y1={y2} x2={x1} y2={y2 - cornerLen} stroke={boxColor} strokeWidth={2.5} />
                  {/* Bottom-Right Corner */}
                  <Line x1={x2} y1={y2} x2={x2 - cornerLen} y2={y2} stroke={boxColor} strokeWidth={2.5} />
                  <Line x1={x2} y1={y2} x2={x2} y2={y2 - cornerLen} stroke={boxColor} strokeWidth={2.5} />

                  {/* Depth & Confidence Telemetry Tag */}
                  <Rect
                    x={x1}
                    y={tagY}
                    width={tagWidth}
                    height={19}
                    rx={4}
                    fill={boxColor}
                  />
                  <SvgText
                    x={x1 + 6}
                    y={tagY + 13}
                    fill="#06080B"
                    fontSize={10.5}
                    fontWeight="700"
                  >
                    {tagText}
                  </SvgText>
                </G>
              )}

              {/* 2. MHR70 Skeleton Bones */}
              {SKELETON_EDGES.map(([a, b], edgeIdx) => {
                if (a >= NUM_JOINTS || b >= NUM_JOINTS) return null;
                const xA = j2d[a * 2] * scaleX;
                const yA = j2d[a * 2 + 1] * scaleY;
                const xB = j2d[b * 2] * scaleX;
                const yB = j2d[b * 2 + 1] * scaleY;
                const color = getBoneColor(a, b);
                return (
                  <Line
                    key={`edge-${edgeIdx}`}
                    x1={xA}
                    y1={yA}
                    x2={xB}
                    y2={yB}
                    stroke={color}
                    strokeWidth={2.4}
                    strokeLinecap="round"
                  />
                );
              })}

              {/* 3. 70 Articulated Keypoints */}
              {Array.from({ length: NUM_JOINTS }).map((_, jIdx) => {
                const jx = j2d[jIdx * 2] * scaleX;
                const jy = j2d[jIdx * 2 + 1] * scaleY;
                if (jx < 0 || jx > viewWidth || jy < 0 || jy > viewHeight) {
                  return null;
                }
                const { radius, color } = getJointStyle(jIdx);
                return (
                  <Circle
                    key={`joint-${jIdx}`}
                    cx={jx}
                    cy={jy}
                    r={radius}
                    fill={color}
                    stroke="#0A0D12"
                    strokeWidth={1.1}
                  />
                );
              })}
            </G>
          );
        })}
      </Svg>
    </View>
  );
};
