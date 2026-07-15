import bpy


SMPLX_MESH_NAME = "SMPLX-mesh-male"
SNAPSHOT_COLLECTION = "HSI_motion_sequence_snapshots"

# Set to [] to sample frames automatically from the current timeline.
FRAMES = [1, 30, 60, 90,180, 300]
NUM_SNAPSHOTS = 6

# Choose one color style. The body color stays fixed; only transparency changes over time.
# Options: "purple", "gold", "pink", "gradient", or "light blue".
COLOR_STYLE = "light blue"

# Fade the motion trail from early transparent poses to later solid poses.
START_ALPHA = 0.16
END_ALPHA = 0.92

# Keep the animated source body visible after creating snapshots.
KEEP_SOURCE_VISIBLE = False


def find_smplx_mesh():
    obj = bpy.data.objects.get(SMPLX_MESH_NAME)
    if obj is not None:
        return obj
    for candidate in bpy.data.objects:
        name = candidate.name.lower()
        if candidate.type == "MESH" and "smplx" in name and "mesh" in name:
            return candidate
    raise KeyError(f"Cannot find SMPL-X mesh object: {SMPLX_MESH_NAME}")


def clear_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is not None:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def set_principled_input(bsdf, socket_names, value):
    wanted = {name.lower() for name in socket_names}
    for socket in bsdf.inputs:
        name = str(getattr(socket, "name", "")).lower()
        ident = str(getattr(socket, "identifier", "")).lower()
        if name in wanted or ident in wanted:
            socket.default_value = value
            return


def make_material(name, color, alpha=1.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    rgba = (color[0], color[1], color[2], alpha)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    mat.show_transparent_back = True
    mat.use_screen_refraction = alpha < 1.0

    bsdf = next((node for node in mat.node_tree.nodes if getattr(node, "type", None) == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        set_principled_input(bsdf, ("Base Color", "基础色", "base_color"), rgba)
        set_principled_input(bsdf, ("Alpha", "透明度", "alpha"), alpha)
        set_principled_input(bsdf, ("Roughness", "粗糙度", "roughness"), 0.62)
        set_principled_input(bsdf, ("Metallic", "金属度", "metallic"), 0.0)
    return mat


def color_style_key():
    return COLOR_STYLE.strip().lower().replace("_", " ")


def color_for_index(index, total):
    style = color_style_key()
    if style == "purple":
        return (0.47, 0.34, 1.0)
    if style == "pink":
        return (1.0, 0.38, 0.58)
    if style == "gradient":
        return (0.66, 0.50, 1.0)
    if style in {"light blue", "blue"}:
        return (0.38, 0.72, 1.0)
    if style == "gold":
        return (1.0, 0.66, 0.18)
    raise ValueError(f"Unknown COLOR_STYLE={COLOR_STYLE!r}")


def alpha_for_index(index, total):
    t = 1.0 if total <= 1 else index / float(total - 1)
    return START_ALPHA * (1.0 - t) + END_ALPHA * t


def assign_single_material(mesh, material):
    mesh.materials.clear()
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.material_index = 0


def auto_frames():
    scene = bpy.context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if end <= start:
        end = start + max(NUM_SNAPSHOTS - 1, 1)
    if NUM_SNAPSHOTS <= 1:
        return [start]
    return [round(start + (end - start) * i / (NUM_SNAPSHOTS - 1)) for i in range(NUM_SNAPSHOTS)]


def create_snapshot(source_obj, frame, material, collection):
    scene = bpy.context.scene
    scene.frame_set(int(frame))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source_obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    mesh.name = f"HSI_body_snapshot_mesh_{frame:04d}"
    assign_single_material(mesh, material)

    snapshot = bpy.data.objects.new(f"HSI_body_snapshot_{frame:04d}", mesh)
    snapshot.matrix_world = evaluated.matrix_world.copy()
    collection.objects.link(snapshot)
    return snapshot


def create_motion_sequence_snapshots():
    source_obj = find_smplx_mesh()
    frames = [int(frame) for frame in (FRAMES if FRAMES else auto_frames())]
    collection = clear_collection(SNAPSHOT_COLLECTION)

    for index, frame in enumerate(frames):
        mat = make_material(
            f"HSI_motion_snapshot_{color_style_key().replace(' ', '_')}_{index:02d}",
            color_for_index(index, len(frames)),
            alpha_for_index(index, len(frames)),
        )
        create_snapshot(source_obj, frame, mat, collection)

    source_obj.hide_viewport = not KEEP_SOURCE_VISIBLE
    source_obj.hide_render = not KEEP_SOURCE_VISIBLE
    print(f"Created {len(frames)} motion snapshots from {source_obj.name}: {frames}")


if __name__ == "__main__":
    create_motion_sequence_snapshots()
