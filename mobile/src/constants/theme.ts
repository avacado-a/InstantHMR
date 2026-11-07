/**
 * Design System Tokens & Curated Studio Palette for InstantHMR Mobile.
 *
 * Crafted with an Obsidian & Cyan/Emerald technical telemetry aesthetic:
 * - Zero purple-on-dark cliches
 * - High-contrast tabular monospace numbers for real-time latency & FPS HUD
 * - Subtle elevation, glassmorphic surfaces, and anatomical color coding
 */

export const COLORS = {
  // Base Obsidian Surfaces
  bgPrimary: '#080A0E',
  bgSecondary: '#0E1218',
  bgCard: '#131820',
  bgCardElevated: '#19202B',
  bgGlass: 'rgba(19, 24, 32, 0.84)',
  bgHudGlass: 'rgba(10, 13, 18, 0.88)',

  // Borders & Dividers
  borderSubtle: '#1E2633',
  borderDefault: '#283242',
  borderFocus: '#00E5FF',

  // Typography
  textPrimary: '#F3F6FA',
  textSecondary: '#9BA6B8',
  textMuted: '#636E80',
  textInverse: '#080A0E',

  // Primary Technical Accents (Cyan & Emerald)
  accentCyan: '#00E5FF',
  accentCyanSoft: 'rgba(0, 229, 255, 0.14)',
  accentEmerald: '#10B981',
  accentEmeraldSoft: 'rgba(16, 185, 129, 0.14)',
  accentAmber: '#F59E0B',
  accentAmberSoft: 'rgba(245, 158, 11, 0.14)',
  accentCoral: '#F43F5E',
  accentCoralSoft: 'rgba(244, 63, 94, 0.14)',
  accentSky: '#38BDF8',
  accentSkySoft: 'rgba(56, 189, 248, 0.14)',

  // Anatomical MHR70 Skeleton Color Coding (RGB Hex)
  boneHead: '#3CDFFF',      // Electric Cyan (Head & Neck)
  boneTorso: '#326EFF',     // Cobalt Blue (Spine, Shoulders, Hips)
  boneArms: '#FFA028',      // Warm Amber (Upper & Forearms)
  boneRightHand: '#FFF03C', // Solar Yellow (Right Hand 20 articulations)
  boneLeftHand: '#00F5D4',  // Mint Teal (Left Hand 20 articulations)
  boneLegs: '#50E678',      // Emerald Green (Knees, Ankles, Feet)

  // Studio 3D Floor Grid
  studioFloorBg: '#0B0E14',
  studioGridLine: '#1B2230',
  studioGridHighlight: '#253042',
} as const;

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
} as const;

export const RADIUS = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
  pill: 999,
} as const;
