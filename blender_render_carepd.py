"""
Blender Photorealistic Clinical Gait Renderer
=============================================
Executed directly via Blender CLI:
  blender -b -P blender_render_carepd.py -- data/care_pd/PD-GaM.pkl data/blender_clinical_gait_1080p.mp4 95
"""

import sys
import os
import pickle
import math
import subprocess
import shutil
import numpy as np

try:
    import bpy
    import mathutils
except ImportError:
    pass

# SMPL 24 Kinematic Tree
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-6:
        return np.eye(3, dtype=np.float32)
    k = r / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def compute_smpl_fk(pose_72, trans):
    offsets = np.array([
        [0.0, 0.0, 0.0],         # 0: Pelvis
        [-0.09, -0.09, 0.0],     # 1: L_Hip
        [0.09, -0.09, 0.0],      # 2: R_Hip
        [0.0, 0.12, 0.0],        # 3: Spine1
        [-0.04, -0.38, 0.0],     # 4: L_Knee
        [0.04, -0.38, 0.0],      # 5: R_Knee
        [0.0, 0.12, 0.0],        # 6: Spine2
        [0.0, -0.38, 0.0],       # 7: L_Ankle
        [0.0, -0.38, 0.0],       # 8: R_Ankle
        [0.0, 0.12, 0.0],        # 9: Spine3
        [0.04, -0.06, 0.12],     # 10: L_Foot
        [-0.04, -0.06, 0.12],    # 11: R_Foot
        [0.0, 0.18, 0.0],        # 12: Neck
        [-0.08, 0.12, 0.0],      # 13: L_Collar
        [0.08, 0.12, 0.0],       # 14: R_Collar
        [0.0, 0.18, 0.0],        # 15: Head
        [-0.14, 0.0, 0.0],       # 16: L_Shoulder
        [0.14, 0.0, 0.0],        # 17: R_Shoulder
        [0.0, -0.26, 0.0],       # 18: L_Elbow
        [0.0, -0.26, 0.0],       # 19: R_Elbow
        [0.0, -0.24, 0.0],       # 20: L_Wrist
        [0.0, -0.24, 0.0],       # 21: R_Wrist
        [0.0, -0.08, 0.0],       # 22: L_Hand
        [0.0, -0.08, 0.0],       # 23: R_Hand
    ], dtype=np.float32)
    
    T = len(pose_72)
    joints_3d = np.zeros((T, 24, 3), dtype=np.float32)
    for t in range(T):
        rot_mats = [rodrigues(pose_72[t, i*3:(i+1)*3]) for i in range(24)]
        world_pos = np.zeros((24, 3), dtype=np.float32)
        world_rot = [np.eye(3, dtype=np.float32) for _ in range(24)]
        world_pos[0] = trans[t]
        world_rot[0] = rot_mats[0]
        
        for i in range(1, 24):
            p = SMPL_PARENTS[i]
            world_rot[i] = world_rot[p] @ rot_mats[i]
            world_pos[i] = world_pos[p] + world_rot[p] @ offsets[i]
        joints_3d[t] = world_pos
    return joints_3d

def build_pbr_material(name, base_color, roughness=0.4, metallic=0.0, specular=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = specular
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = specular
    return mat

def setup_blender_scene(joints_3d, output_mp4, fps=25):
    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    
    # Frames temp dir
    frames_dir = "/tmp/blender_carepd_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Render Settings (1080p)
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(frames_dir, "frame_")
    
    T = len(joints_3d)
    scene.frame_start = 1
    scene.frame_end = T
    
    # 1. PBR Materials
    skin_mat = build_pbr_material("Skin_Mat", (0.82, 0.65, 0.54), roughness=0.55, specular=0.3)
    shirt_mat = build_pbr_material("Shirt_Mat", (0.15, 0.22, 0.35), roughness=0.7)
    pants_mat = build_pbr_material("Pants_Mat", (0.08, 0.10, 0.14), roughness=0.8)
    shoe_mat = build_pbr_material("Shoe_Mat", (0.04, 0.04, 0.04), roughness=0.3)
    floor_mat = build_pbr_material("Floor_Mat", (0.75, 0.78, 0.82), roughness=0.25, specular=0.6)
    
    # 2. Clinic Floor Plane
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.95))
    floor_obj = bpy.context.active_object
    floor_obj.data.materials.append(floor_mat)
    
    # 3. Studio 3-Point Lighting
    bpy.ops.object.light_add(type='AREA', location=(3.5, -3.0, 4.5))
    key_light = bpy.context.active_object
    key_light.data.energy = 450
    key_light.data.size = 2.5
    
    bpy.ops.object.light_add(type='AREA', location=(-3.5, -2.5, 3.5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 220
    fill_light.data.size = 3.0
    
    bpy.ops.object.light_add(type='SUN', location=(0, 4.0, 5.0))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 3.5
    
    # 4. Cinematic Physical Camera
    bpy.ops.object.camera_add(location=(0, -3.8, 0.25), rotation=(math.radians(88), 0, 0))
    cam = bpy.context.active_object
    scene.camera = cam
    cam.data.lens = 45
    
    # 5. Build Rigged Character Hierarchy
    limbs = [
        # (j1, j2, radius, mat, name)
        (0, 3, 0.13, shirt_mat, "pelvis"),
        (3, 6, 0.12, shirt_mat, "abdomen"),
        (6, 9, 0.13, shirt_mat, "chest"),
        (9, 12, 0.11, shirt_mat, "upper_chest"),
        (12, 15, 0.07, skin_mat, "neck"),
        # Legs
        (0, 1, 0.09, pants_mat, "l_thigh"),
        (1, 4, 0.08, pants_mat, "l_leg_upper"),
        (4, 7, 0.065, pants_mat, "l_shin"),
        (7, 10, 0.05, shoe_mat, "l_foot"),
        (0, 2, 0.09, pants_mat, "r_thigh"),
        (2, 5, 0.08, pants_mat, "r_leg_upper"),
        (5, 8, 0.065, pants_mat, "r_shin"),
        (8, 11, 0.05, shoe_mat, "r_foot"),
        # Arms
        (9, 13, 0.07, shirt_mat, "l_collar"),
        (13, 16, 0.065, skin_mat, "l_bicep"),
        (16, 18, 0.055, skin_mat, "l_elbow"),
        (18, 20, 0.045, skin_mat, "l_forearm"),
        (20, 22, 0.035, skin_mat, "l_hand"),
        (9, 14, 0.07, shirt_mat, "r_collar"),
        (14, 17, 0.065, skin_mat, "r_bicep"),
        (17, 19, 0.055, skin_mat, "r_elbow"),
        (19, 21, 0.045, skin_mat, "r_forearm"),
        (21, 23, 0.035, skin_mat, "r_hand"),
    ]
    
    limb_objects = []
    for j1, j2, radius, mat, name in limbs:
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1.0)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(mat)
        limb_objects.append((obj, j1, j2, radius))
        
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(0, 0, 0))
    head_obj = bpy.context.active_object
    head_obj.name = "Head_Sphere"
    head_obj.data.materials.append(skin_mat)
    
    # 6. Keyframe Timeline
    print(f"🎬 Keyframing {T} frames of SMPL gait into Blender...")
    for frame_idx in range(1, T + 1):
        scene.frame_set(frame_idx)
        j = joints_3d[frame_idx - 1]
        
        head_pos = mathutils.Vector((j[15, 0], j[15, 2], j[15, 1]))
        head_obj.location = head_pos
        head_obj.keyframe_insert(data_path="location", frame=frame_idx)
        
        for obj, j1, j2, radius in limb_objects:
            p1 = mathutils.Vector((j[j1, 0], j[j1, 2], j[j1, 1]))
            p2 = mathutils.Vector((j[j2, 0], j[j2, 2], j[j2, 1]))
            
            mid = (p1 + p2) * 0.5
            vec = p2 - p1
            dist = vec.length
            
            obj.location = mid
            obj.scale = mathutils.Vector((1, 1, max(1e-4, dist)))
            
            if dist > 1e-4:
                rot_quat = mathutils.Vector((0, 0, 1)).rotation_difference(vec)
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = rot_quat
                
            obj.keyframe_insert(data_path="location", frame=frame_idx)
            obj.keyframe_insert(data_path="scale", frame=frame_idx)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
            
        root = mathutils.Vector((j[0, 0], j[0, 2], j[0, 1]))
        cam.location = root + mathutils.Vector((0, -3.6, 0.25))
        cam.keyframe_insert(data_path="location", frame=frame_idx)
        
    print(f"🚀 Rendering {T} frames with Blender EEVEE...")
    bpy.ops.render.render(animation=True)
    
    # 7. Encode frames into high-bitrate MP4 with FFmpeg
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-r", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        os.path.abspath(output_mp4)
    ]
    print(f"📦 Encoding frames to {output_mp4}...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ Full Blender 1080p Render Finished: {output_mp4}")

if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        pkl_path = args[idx + 1]
        out_path = args[idx + 2]
        max_f = int(args[idx + 3]) if len(args) > idx + 3 else 95
    else:
        pkl_path = "data/care_pd/PD-GaM.pkl"
        out_path = "data/blender_clinical_gait_1080p.mp4"
        max_f = 95
        
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sub = list(data.keys())[0]
    walk = list(data[sub].keys())[0]
    poses = data[sub][walk]["pose"][:max_f]
    trans = data[sub][walk]["trans"][:max_f]
    fps = data[sub][walk].get("fps", 25) or 25
    
    joints_3d = compute_smpl_fk(poses, trans)
    setup_blender_scene(joints_3d, out_path, fps=fps)
