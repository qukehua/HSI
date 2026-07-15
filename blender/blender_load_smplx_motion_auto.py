import importlib
import os
import pickle as pkl
import sys
from pathlib import Path

import bpy


PROJECT_ROOT = Path(r"D:\code\HSI")
SOURCE_BLEND = PROJECT_ROOT / "blender" / "vis.blend"
MOTION_FILE = PROJECT_ROOT / "results" / "outputs" / "output__demo-21__0.pkl"
SMPLX_MESH_NAME = "SMPLX-mesh-male"
SMPLX_ARMATURE_NAME = "SMPLX-male"


def find_smplx_mesh():
    exact = bpy.data.objects.get(SMPLX_MESH_NAME)
    if exact is not None:
        return exact

    for obj in bpy.data.objects:
        name = obj.name.lower()
        if obj.type == "MESH" and "smplx" in name and "mesh" in name:
            return obj
    return None


def append_object_from_blend(source_blend, object_name):
    source_blend = Path(source_blend)
    if not source_blend.exists():
        raise FileNotFoundError(f"Cannot find source blend: {source_blend}")

    if bpy.data.objects.get(object_name) is not None:
        return bpy.data.objects[object_name]

    directory = str(source_blend) + "\\Object\\"
    bpy.ops.wm.append(directory=directory, filename=object_name)
    return bpy.data.objects.get(object_name)


def ensure_smplx_mesh():
    obj = find_smplx_mesh()
    if obj is not None:
        return obj

    print(f"{SMPLX_MESH_NAME} not found in current scene. Appending from {SOURCE_BLEND}...")
    armature = append_object_from_blend(SOURCE_BLEND, SMPLX_ARMATURE_NAME)
    mesh = append_object_from_blend(SOURCE_BLEND, SMPLX_MESH_NAME)

    if mesh is None:
        raise KeyError(f"Could not append {SMPLX_MESH_NAME} from {SOURCE_BLEND}")
    if mesh.parent is None and armature is not None:
        mesh.parent = armature

    return mesh


def prepare_project_imports():
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def load_motion():
    prepare_project_imports()

    import load_smplx_animation
    importlib.reload(load_smplx_animation)
    from load_smplx_animation import load_smplx_animation

    obj = ensure_smplx_mesh()

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    with open(MOTION_FILE, "rb") as f:
        data = pkl.load(f)

    load_smplx_animation(data, obj)
    print(f"Loaded SMPL-X motion from {MOTION_FILE} onto {obj.name}")


if __name__ == "__main__":
    load_motion()
