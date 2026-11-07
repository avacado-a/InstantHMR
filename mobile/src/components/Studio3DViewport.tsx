/**
 * Interactive 3D Skeleton & Mesh Studio Viewport for InstantHMR Mobile.
 *
 * Ports `render_3d_skeleton_studio` from `instanthmr/visualizer.py` with interactive touch orbit
 * (drag to rotate azimuth & elevation, pinch/reset, or continuous auto-rotation), perspective
 * floor grid, depth telemetry, and 3D coordinate axes widget.
 */
import React, { useMemo, useRef, useState } from 'react';
import {
  PanResponder,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import Svg, { Circle, G, Line, Polygon } from 'react-native-svg';
import { getBoneColor, getJointStyle, NUM_JOINTS, SKELETON_EDGES } from '../constants/skeleton';
import { COLORS, RADIUS, SPACING } from '../constants/theme';
import { PersonPrediction } from '../types/config';

interface Studio3DViewportProps {
  persons: PersonPrediction[];
  width: number;
  height: number;
  baseAzim: number;
  baseElev: number;
  rotateSpeed: number;
  frameIndex: number;
  modelTag: string;
  withMesh: boolean;
  meshAlpha: number;
}

export const Studio3DViewport: React.FC<Studio3DViewportProps> = ({
  persons,
  width,
  height,
  baseAzim,
  baseElev,
  rotateSpeed,
  frameIndex,
  modelTag,
  withMesh,
  meshAlpha,
}) => {
  const [userAzimDelta, setUserAzimDelta] = useState(0);
  const [userElevDelta, setUserElevDelta] = useState(0);
  const dragStartRef = useRef({ azim: 0, elev: 0 });

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: (_, g) =>
          Math.abs(g.dx) > 3 || Math.abs(g.dy) > 3,
        onPanResponderGrant: () => {
          dragStartRef.current = { azim: userAzimDelta, elev: userElevDelta };
        },
        onPanResponderMove: (_, gestureState) => {
          setUserAzimDelta(dragStartRef.current.azim + gestureState.dx * 0.45);
          setUserElevDelta(
            Math.max(
              -60,
              Math.min(60, dragStartRef.current.elev - gestureState.dy * 0.35)
            )
          );
        },
      }),
    [userAzimDelta, userElevDelta]
  );

  const effectiveAzim = baseAzim + userAzimDelta + frameIndex * rotateSpeed * 6.0;
  const effectiveElev = baseElev + userElevDelta;

  const radAz = (effectiveAzim * Math.PI) / 180.0;
  const radEl = (effectiveElev * Math.PI) / 180.0;
  const cosAz = Math.cos(radAz);
  const sinAz = Math.sin(radAz);
  const cosEl = Math.cos(radEl);
  const sinEl = Math.sin(radEl);

  const primaryPerson = persons[0] || null;
  const zDepth = primaryPerson ? primaryPerson.camTrans[2] : null;

  // Project 3D joints with R = R_el @ R_az matching `render_3d_skeleton_studio`
  const projected = useMemo(() => {
    if (!primaryPerson || width <= 10 || height <= 10) {
      return null;
    }

    const j3d = primaryPerson.joints3dLocal;
    const rotX = new Float32Array(NUM_JOINTS);
    const rotY = new Float32Array(NUM_JOINTS);
    const rotZ = new Float32Array(NUM_JOINTS);

    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;

    for (let i = 0; i < NUM_JOINTS; i++) {
      const x = j3d[i * 3];
      const y = j3d[i * 3 + 1];
      const z = j3d[i * 3 + 2];

      // R_az around Y axis
      const x1 = x * cosAz + z * sinAz;
      const y1 = y;
      const z1 = -x * sinAz + z * cosAz;

      // R_el around X axis
      const x2 = x1;
      const y2 = y1 * cosEl - z1 * sinEl;
      const z2 = y1 * sinEl + z1 * cosEl;

      rotX[i] = x2;
      rotY[i] = y2;
      rotZ[i] = z2;

      sumX += x2;
      sumY += y2;
      sumZ += z2;
    }

    const meanX = sumX / NUM_JOINTS;
    const meanY = sumY / NUM_JOINTS;

    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < NUM_JOINTS; i++) {
      const cy = rotY[i] - meanY;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
    }

    const spanY = Math.max(0.15, maxY - minY);
    const scale = (height * 0.64) / spanY;
    const centerX = width * 0.5;
    const centerY = height * 0.48;

    const u = new Float32Array(NUM_JOINTS);
    const v = new Float32Array(NUM_JOINTS);
    for (let i = 0; i < NUM_JOINTS; i++) {
      u[i] = (rotX[i] - meanX) * scale + centerX;
      v[i] = (rotY[i] - meanY) * scale + centerY;
    }

    return { u, v, rotZ };
  }, [primaryPerson, width, height, cosAz, sinAz, cosEl, sinEl]);

  const cyFloor = Math.round(height * 0.85);
  const floorStepX = Math.max(22, Math.floor(width / 11));
  const floorStepY = Math.max(12, Math.floor((height - cyFloor) / 4));

  return (
    <View
      style={[styles.container, { width, height }]}
      {...panResponder.panHandlers}
    >
      <Svg width={width} height={height}>
        {/* Perspective Studio Floor Grid */}
        <G>
          {Array.from({ length: 13 }).map((_, idx) => {
            const gx = 16 + idx * floorStepX;
            if (gx >= width - 8) return null;
            const endX = gx + (gx - width * 0.5) * 0.52;
            return (
              <Line
                key={`grid-v-${idx}`}
                x1={gx}
                y1={cyFloor}
                x2={endX}
                y2={height - 6}
                stroke={COLORS.studioGridLine}
                strokeWidth={1}
              />
            );
          })}
          {Array.from({ length: 5 }).map((_, idx) => {
            const gy = cyFloor + idx * floorStepY;
            if (gy >= height - 4) return null;
            return (
              <Line
                key={`grid-h-${idx}`}
                x1={10}
                y1={gy}
                x2={width - 10}
                y2={gy}
                stroke={COLORS.studioGridLine}
                strokeWidth={1}
              />
            );
          })}
        </G>

        {projected && (
          <G>
            {/* Optional 3D Volumetric MHR Body Hull Proxy when withMesh is enabled */}
            {withMesh && (
              <Polygon
                points={`${projected.u[5]},${projected.v[5]} ${projected.u[6]},${projected.v[6]} ${projected.u[10]},${projected.v[10]} ${projected.u[9]},${projected.v[9]}`}
                fill={COLORS.accentCyan}
                fillOpacity={meshAlpha * 0.65}
                stroke={COLORS.accentCyan}
                strokeOpacity={meshAlpha}
                strokeWidth={1.5}
              />
            )}

            {/* 3D MHR70 Bones with Outer Glow & Core Stroke */}
            {SKELETON_EDGES.map(([a, b], edgeIdx) => {
              if (a >= NUM_JOINTS || b >= NUM_JOINTS) return null;
              const color = getBoneColor(a, b);
              return (
                <G key={`s3d-edge-${edgeIdx}`}>
                  <Line
                    x1={projected.u[a]}
                    y1={projected.v[a]}
                    x2={projected.u[b]}
                    y2={projected.v[b]}
                    stroke={color}
                    strokeOpacity={0.28}
                    strokeWidth={5.5}
                    strokeLinecap="round"
                  />
                  <Line
                    x1={projected.u[a]}
                    y1={projected.v[a]}
                    x2={projected.u[b]}
                    y2={projected.v[b]}
                    stroke={color}
                    strokeWidth={2.4}
                    strokeLinecap="round"
                  />
                </G>
              );
            })}

            {/* 70 Articulated 3D Keypoints */}
            {Array.from({ length: NUM_JOINTS }).map((_, jIdx) => {
              const { radius, color } = getJointStyle(jIdx);
              return (
                <Circle
                  key={`s3d-joint-${jIdx}`}
                  cx={projected.u[jIdx]}
                  cy={projected.v[jIdx]}
                  r={radius}
                  fill={color}
                  stroke="#080A0E"
                  strokeWidth={1}
                />
              );
            })}
          </G>
        )}

        {/* Bottom-Right 3D Coordinate Axes Gizmo */}
        <G>
          <Line
            x1={width - 42}
            y1={height - 26}
            x2={width - 20}
            y2={height - 26}
            stroke="#F43F5E"
            strokeWidth={2}
          />
          <Line
            x1={width - 42}
            y1={height - 26}
            x2={width - 42}
            y2={height - 48}
            stroke="#10B981"
            strokeWidth={2}
          />
          <Line
            x1={width - 42}
            y1={height - 26}
            x2={width - 54}
            y2={height - 14}
            stroke="#38BDF8"
            strokeWidth={2}
          />
        </G>
      </Svg>

      {/* Studio Header Badge */}
      <View style={styles.headerOverlay} pointerEvents="box-none">
        <View>
          <Text style={styles.studioTitle}>3D SKELETON STUDIO (MHR 70)</Text>
          <Text style={styles.studioSubtitle}>
            {modelTag} • {zDepth !== null ? `Depth Z: ${zDepth.toFixed(2)}m` : '70 Articulated Joints'} • Azim {Math.round(effectiveAzim % 360)}°
          </Text>
        </View>
        {(userAzimDelta !== 0 || userElevDelta !== 0) && (
          <TouchableOpacity
            style={styles.resetOrbitBtn}
            onPress={() => {
              setUserAzimDelta(0);
              setUserElevDelta(0);
            }}
          >
            <Text style={styles.resetOrbitText}>Reset View</Text>
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.orbitHintBadge} pointerEvents="none">
        <Text style={styles.orbitHintText}>Drag to Orbit 3D</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.studioFloorBg,
    overflow: 'hidden',
    position: 'relative',
  },
  headerOverlay: {
    position: 'absolute',
    top: SPACING.sm,
    left: SPACING.md,
    right: SPACING.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  studioTitle: {
    color: COLORS.accentCyan,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  studioSubtitle: {
    color: COLORS.textSecondary,
    fontSize: 10.5,
    marginTop: 2,
    fontVariant: ['tabular-nums'],
  },
  resetOrbitBtn: {
    backgroundColor: COLORS.bgCardElevated,
    borderWidth: 1,
    borderColor: COLORS.borderDefault,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: RADIUS.sm,
  },
  resetOrbitText: {
    color: COLORS.textPrimary,
    fontSize: 10,
    fontWeight: '600',
  },
  orbitHintBadge: {
    position: 'absolute',
    bottom: 8,
    left: 12,
    backgroundColor: 'rgba(19, 24, 32, 0.75)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: RADIUS.sm,
    borderWidth: 1,
    borderColor: COLORS.borderSubtle,
  },
  orbitHintText: {
    color: COLORS.textMuted,
    fontSize: 9.5,
    fontWeight: '600',
  },
});
