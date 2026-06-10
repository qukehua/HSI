import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np


SCENE_VIS_GRID = np.array([-4.0, 0.0, -6.0, 4.0, 2.0, 6.0, 400.0, 100.0, 600.0])


def dataset_to_blender(points):
    # Dataset coordinates are y-up: (x, y, z). Blender is z-up.
    # Match the conversion used by utils.yup_to_zup: (x, -z, y).
    out = points[:, [0, 2, 1]].copy()
    out[:, 1] *= -1.0
    return out


def clear_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is not None:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def create_voxel_mesh(name, centers, voxel_size, color):
    collection_name = f"{name}_voxels"
    clear_collection(collection_name)
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    mat = make_material(f"{name}_mat", color)
    half = np.asarray(voxel_size, dtype=np.float64) / 2.0
    offsets = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ]
    )
    cube_faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    verts = []
    faces = []
    for center in centers:
        base = len(verts)
        verts.extend((center + offsets).tolist())
        faces.extend(tuple(base + idx for idx in face) for face in cube_faces)

    mesh = bpy.data.meshes.new(f"{name}_voxel_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_occupied_voxels", mesh)
    obj.data.materials.append(mat)
    collection.objects.link(obj)

    return collection


def create_point_cloud(name, centers, color, point_radius):
    collection_name = f"{name}_points"
    clear_collection(collection_name)
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    mat = make_material(f"{name}_point_mat", color)
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(centers.tolist(), [], [])
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_occupied_points", mesh)
    obj.show_name = True
    obj.data.materials.append(mat)
    collection.objects.link(obj)

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    empty = bpy.context.object
    empty.name = f"{name}_origin"
    collection.objects.link(empty)
    bpy.context.scene.collection.objects.unlink(empty)

    return collection


def add_marker(name, dataset_xyz, color=(1.0, 0.1, 0.1, 1.0), radius=0.08):
    loc = dataset_to_blender(np.asarray(dataset_xyz, dtype=np.float64).reshape(1, 3))[0]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    mat = make_material(f"{name}_mat", color)
    obj.data.materials.append(mat)
    return obj


def load_scene(scene_path, stride, y_min, y_max, max_points, as_cubes):
    scene_path = Path(scene_path)
    occ = np.load(scene_path)
    while occ.ndim > 3:
        occ = occ[0]
    occ = occ.astype(bool)

    grid = SCENE_VIS_GRID
    lower = grid[:3]
    dims = grid[6:].astype(int)
    voxel = (grid[3:6] - lower) / dims

    y_low = max(0, int((y_min - lower[1]) / voxel[1]))
    y_high = min(dims[1], int((y_max - lower[1]) / voxel[1]) + 1)
    indices = np.argwhere(occ[::stride, y_low:y_high:stride, ::stride])
    indices[:, 0] *= stride
    indices[:, 1] = indices[:, 1] * stride + y_low
    indices[:, 2] *= stride

    if len(indices) > max_points:
        rng = np.random.default_rng(1234)
        indices = indices[rng.choice(len(indices), size=max_points, replace=False)]

    centers = lower + (indices + 0.5) * voxel
    centers_blender = dataset_to_blender(centers)

    name = scene_path.stem
    if as_cubes:
        create_voxel_mesh(name, centers_blender, voxel[[0, 2, 1]] * stride, (0.2, 0.65, 1.0, 0.35))
    else:
        create_point_cloud(name, centers_blender, (0.2, 0.65, 1.0, 1.0), voxel.max() * stride)

    add_marker("dataset_origin", [0.0, 0.0, 0.0], (1.0, 0.1, 0.1, 1.0))
    print(f"Loaded {len(centers)} occupied voxels from {scene_path}")
    print("Coordinate conversion: dataset (x, y, z) -> Blender (x, -z, y)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--y-min", type=float, default=0.05)
    parser.add_argument("--y-max", type=float, default=1.80)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--cubes", action="store_true", default=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    args = parser.parse_args(argv)
    load_scene(args.scene, args.stride, args.y_min, args.y_max, args.max_points, args.cubes)


if __name__ == "__main__":
    main()
