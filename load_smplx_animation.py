import bpy
import numpy as np
from pathlib import Path
from mathutils import Vector, Quaternion


MODULE_DIR = Path(__file__).resolve().parent

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


def animation_data_clear(obj):

    obj.animation_data_clear()
    obj.data.animation_data_clear()


def call_smplx_operator(name):
    op = getattr(bpy.ops.object, name, None)
    if op is None:
        print(f"WARNING: bpy.ops.object.{name} not found. Install/enable the SMPL-X Blender add-on for full visualization.")
        return False
    try:
        op('EXEC_DEFAULT')
    except Exception as exc:
        print(f"WARNING: bpy.ops.object.{name} failed: {exc}")
        return False
    return True


def ensure_object_visible(obj):
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj.hide_set(False)
    except Exception:
        pass

    parent = obj.parent
    while parent is not None:
        parent.hide_viewport = False
        parent.hide_render = False
        try:
            parent.hide_set(False)
        except Exception:
            pass
        parent = parent.parent

    for collection in obj.users_collection:
        collection.hide_viewport = False
        collection.hide_render = False


def _pose_frames(file, keys, num_frames, num_joints):
    value = next((file[key] for key in keys if key in file), None)
    if value is None:
        return np.zeros((num_frames, num_joints, 3), dtype=np.float64)

    frames = np.asarray(value, dtype=np.float64).reshape(-1, num_joints, 3)
    if len(frames) == 1 and num_frames > 1:
        frames = np.repeat(frames, num_frames, axis=0)
    if len(frames) != num_frames:
        raise ValueError(
            f"{keys[0]} contains {len(frames)} frames, expected {num_frames}"
        )
    return frames


def _set_linear_keyframes(fcurve):
    for point in fcurve.keyframe_points:
        point.interpolation = "LINEAR"


def load_smplx_animation(
    file,
    obj,
    action_name=None,
    start_frame=0,
    fps=30,
):
    ensure_object_visible(obj)
    animation_data_clear(obj)

    armature = obj.parent
    if armature is None or armature.type != "ARMATURE":
        raise ValueError(f"{obj.name} must be parented to an SMPL-X armature")
    ensure_object_visible(armature)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj  # mesh needs to be active object for recalculating joint locations

    with np.load(MODULE_DIR / "smplx_handposes.npz", allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate((left_hand_pose, right_hand_pose)).reshape(NUM_SMPLX_HANDJOINTS * 2, 3)

    translation = np.asarray(file["transl"], dtype=np.float64).reshape(-1, 3)
    global_orient = np.asarray(file["global_orient"], dtype=np.float64).reshape(-1, 3)
    body_pose = np.asarray(file["body_pose"], dtype=np.float64).reshape(
        -1, NUM_SMPLX_BODYJOINTS, 3
    )
    num_keyframes = len(body_pose)
    if len(translation) != num_keyframes or len(global_orient) != num_keyframes:
        raise ValueError(
            "transl, global_orient, and body_pose must contain the same frame count"
        )

    left_hand_pose = _pose_frames(
        file,
        ("left_hand", "left_hand_pose"),
        num_keyframes,
        NUM_SMPLX_HANDJOINTS,
    )
    right_hand_pose = _pose_frames(
        file,
        ("right_hand", "right_hand_pose"),
        num_keyframes,
        NUM_SMPLX_HANDJOINTS,
    )
    betas_value = file.get("betas", np.zeros(10, dtype=np.float64))
    betas = np.asarray(betas_value, dtype=np.float64).reshape(-1)[:10]
    if len(betas) < 10:
        betas = np.pad(betas, (0, 10 - len(betas)))

    bpy.ops.object.mode_set(mode='OBJECT')
    if obj.data.shape_keys is None:
        print(f"WARNING: {obj.name} has no shape keys; SMPL-X betas were not applied")
    else:
        for index, beta in enumerate(betas):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = float(beta)
            else:
                print(f"ERROR: No key block for: {key_block_name}")

    call_smplx_operator('smplx_update_joint_locations')

    global_orients = np.array([get_quat_from_rodrigues(rodrigues) for rodrigues in global_orient])
    body_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues) for rodrigues in body_pose[:, index, :]])
                  for index, bone_name in enumerate(SMPLX_JOINT_NAMES[1: NUM_SMPLX_BODYJOINTS + 1])}

    start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
    left_hand_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues, hand_pose_relaxed[i]) for rodrigues in left_hand_pose[:, i, :]])
                       for i, bone_name in enumerate(SMPLX_JOINT_NAMES[start_name_index: start_name_index + NUM_SMPLX_HANDJOINTS])}

    start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
    right_hand_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues, hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]) for rodrigues in right_hand_pose[:, i, :]])
                        for i, bone_name in enumerate(SMPLX_JOINT_NAMES[start_name_index: start_name_index + NUM_SMPLX_HANDJOINTS])}

    body_poses = {**body_poses, **left_hand_poses, **right_hand_poses}
    body_poses['pelvis'] = global_orients

    for bone_name in body_poses:
        bone = armature.pose.bones.get(bone_name)
        if bone is not None:
            bone.rotation_mode = "QUATERNION"

    animation_data = armature.animation_data_create()
    if action_name is None:
        action_name = f"{armature.name}Action"
    existing_action = bpy.data.actions.get(action_name)
    if existing_action is not None:
        bpy.data.actions.remove(existing_action)
    action = animation_data.action = bpy.data.actions.new(action_name)
    keyframe_numbers = range(start_frame, start_frame + num_keyframes)
    for i in range(3):
        fcurve = action.fcurves.new('pose.bones["root"].location', index=i)
        fcurve.keyframe_points.add(count=num_keyframes)
        fcurve.keyframe_points.foreach_set(
            "co",
            [x for co in zip(keyframe_numbers, translation[:, i]) for x in co],
        )
        _set_linear_keyframes(fcurve)
        fcurve.update()
    for bone_name, quaternions in body_poses.items():
        for i in range(4):
            fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_quaternion', index=i)
            fcurve.keyframe_points.add(count=num_keyframes)
            fcurve.keyframe_points.foreach_set("co",
                                               [x for co in zip(keyframe_numbers, quaternions[:, i]) for x in co])
            _set_linear_keyframes(fcurve)
            fcurve.update()

    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = start_frame + num_keyframes - 1
    scene.frame_preview_start = scene.frame_start
    scene.frame_preview_end = scene.frame_end
    scene.render.fps = int(fps)
    scene.frame_set(start_frame)

    # Activate corrective poseshapes
    call_smplx_operator('smplx_set_poseshapes')

    return {'FINISHED'}


def get_quat_from_rodrigues(rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    quat = Quaternion((1.0, 0.0, 0.0, 0.0)) if angle_rad < 1e-12 else Quaternion(rod.normalized(), angle_rad)

    if rodrigues_reference is None:
       return quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        quat_result = (
            Quaternion((1.0, 0.0, 0.0, 0.0))
            if angle_rad_result < 1e-12
            else Quaternion(rod_result.normalized(), angle_rad_result)
        )
        return quat_result


def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return
