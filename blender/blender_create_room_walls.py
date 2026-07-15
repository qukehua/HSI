import bpy


# Edit these numbers after running once if the room does not align with your sofa.
ROOM_WIDTH = 5.0
ROOM_DEPTH = 4.2
WALL_HEIGHT = 2.8
WALL_THICKNESS = 0.08
FLOOR_Z = 0.0

# The shell is centered at the origin by default.
CENTER_X = 0.0
CENTER_Y = 0.0

# Back-wall window, like the paper figure. Set ENABLE_WINDOW = False for a plain wall.
ENABLE_WINDOW = True
WINDOW_WIDTH = 1.2
WINDOW_HEIGHT = 1.3
WINDOW_CENTER_X = 1.15
WINDOW_CENTER_Z = 1.65
WINDOW_FRAME_THICKNESS = 0.045


def make_mat(name, color, roughness=0.7):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = next((node for node in mat.node_tree.nodes if getattr(node, "type", None) == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        for socket in bsdf.inputs:
            name_l = str(getattr(socket, "name", "")).lower()
            ident = str(getattr(socket, "identifier", "")).lower()
            if name_l in {"base color", "基础色"} or ident == "base_color":
                socket.default_value = color
            elif name_l in {"roughness", "粗糙度"} or ident == "roughness":
                socket.default_value = roughness
    return mat


def get_collection(name):
    old = bpy.data.collections.get(name)
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def cube_obj(name, location, scale, mat, collection):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)

    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def add_back_wall_with_window(collection, wall_mat, frame_mat):
    back_y = CENTER_Y + ROOM_DEPTH / 2.0
    z_mid = FLOOR_Z + WALL_HEIGHT / 2.0

    if not ENABLE_WINDOW:
        cube_obj(
            "HSI_back_wall",
            (CENTER_X, back_y, z_mid),
            (ROOM_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
            wall_mat,
            collection,
        )
        return

    wx0 = CENTER_X + WINDOW_CENTER_X - WINDOW_WIDTH / 2.0
    wx1 = CENTER_X + WINDOW_CENTER_X + WINDOW_WIDTH / 2.0
    wz0 = FLOOR_Z + WINDOW_CENTER_Z - WINDOW_HEIGHT / 2.0
    wz1 = FLOOR_Z + WINDOW_CENTER_Z + WINDOW_HEIGHT / 2.0

    left_width = max(wx0 - (CENTER_X - ROOM_WIDTH / 2.0), 0.01)
    right_width = max((CENTER_X + ROOM_WIDTH / 2.0) - wx1, 0.01)
    bottom_height = max(wz0 - FLOOR_Z, 0.01)
    top_height = max((FLOOR_Z + WALL_HEIGHT) - wz1, 0.01)

    cube_obj(
        "HSI_back_wall_left",
        (CENTER_X - ROOM_WIDTH / 2.0 + left_width / 2.0, back_y, z_mid),
        (left_width, WALL_THICKNESS, WALL_HEIGHT),
        wall_mat,
        collection,
    )
    cube_obj(
        "HSI_back_wall_right",
        (wx1 + right_width / 2.0, back_y, z_mid),
        (right_width, WALL_THICKNESS, WALL_HEIGHT),
        wall_mat,
        collection,
    )
    cube_obj(
        "HSI_back_wall_below_window",
        (CENTER_X + WINDOW_CENTER_X, back_y, FLOOR_Z + bottom_height / 2.0),
        (WINDOW_WIDTH, WALL_THICKNESS, bottom_height),
        wall_mat,
        collection,
    )
    cube_obj(
        "HSI_back_wall_above_window",
        (CENTER_X + WINDOW_CENTER_X, back_y, wz1 + top_height / 2.0),
        (WINDOW_WIDTH, WALL_THICKNESS, top_height),
        wall_mat,
        collection,
    )

    frame_depth = WALL_THICKNESS * 1.6
    cube_obj(
        "HSI_window_frame_top",
        (CENTER_X + WINDOW_CENTER_X, back_y - 0.01, wz1),
        (WINDOW_WIDTH + WINDOW_FRAME_THICKNESS * 2, frame_depth, WINDOW_FRAME_THICKNESS),
        frame_mat,
        collection,
    )
    cube_obj(
        "HSI_window_frame_bottom",
        (CENTER_X + WINDOW_CENTER_X, back_y - 0.01, wz0),
        (WINDOW_WIDTH + WINDOW_FRAME_THICKNESS * 2, frame_depth, WINDOW_FRAME_THICKNESS),
        frame_mat,
        collection,
    )
    cube_obj(
        "HSI_window_frame_left",
        (wx0, back_y - 0.01, WINDOW_CENTER_Z),
        (WINDOW_FRAME_THICKNESS, frame_depth, WINDOW_HEIGHT),
        frame_mat,
        collection,
    )
    cube_obj(
        "HSI_window_frame_right",
        (wx1, back_y - 0.01, WINDOW_CENTER_Z),
        (WINDOW_FRAME_THICKNESS, frame_depth, WINDOW_HEIGHT),
        frame_mat,
        collection,
    )


def create_room_shell():
    collection = get_collection("HSI_room_shell")
    wall_mat = make_mat("HSI_room_wall_warm_white", (0.88, 0.86, 0.82, 1.0), 0.82)
    floor_mat = make_mat("HSI_room_wood_floor", (0.58, 0.40, 0.22, 1.0), 0.52)
    trim_mat = make_mat("HSI_room_light_trim", (0.78, 0.70, 0.58, 1.0), 0.6)

    floor_thickness = 0.035
    cube_obj(
        "HSI_wood_floor",
        (CENTER_X, CENTER_Y, FLOOR_Z - floor_thickness / 2.0),
        (ROOM_WIDTH, ROOM_DEPTH, floor_thickness),
        floor_mat,
        collection,
    )

    add_back_wall_with_window(collection, wall_mat, trim_mat)

    left_x = CENTER_X - ROOM_WIDTH / 2.0
    right_x = CENTER_X + ROOM_WIDTH / 2.0
    z_mid = FLOOR_Z + WALL_HEIGHT / 2.0
    cube_obj(
        "HSI_left_wall",
        (left_x, CENTER_Y, z_mid),
        (WALL_THICKNESS, ROOM_DEPTH, WALL_HEIGHT),
        wall_mat,
        collection,
    )
    cube_obj(
        "HSI_right_wall",
        (right_x, CENTER_Y, z_mid),
        (WALL_THICKNESS, ROOM_DEPTH, WALL_HEIGHT),
        wall_mat,
        collection,
    )

    baseboard_height = 0.08
    baseboard_depth = 0.035
    cube_obj(
        "HSI_back_baseboard",
        (CENTER_X, CENTER_Y + ROOM_DEPTH / 2.0 - WALL_THICKNESS, FLOOR_Z + baseboard_height / 2.0),
        (ROOM_WIDTH, baseboard_depth, baseboard_height),
        trim_mat,
        collection,
    )
    cube_obj(
        "HSI_left_baseboard",
        (left_x + WALL_THICKNESS, CENTER_Y, FLOOR_Z + baseboard_height / 2.0),
        (baseboard_depth, ROOM_DEPTH, baseboard_height),
        trim_mat,
        collection,
    )

    print("Created HSI_room_shell. Move/scale this collection to align it with your sofa and SMPL-X motion.")


if __name__ == "__main__":
    create_room_shell()
