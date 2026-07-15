import bpy
from mathutils import Vector


COLLECTION_NAME = "HSI_room_shell"
CONTROLLER_NAME = "HSI_room_shell_controller"


def collection_center(objects):
    points = []
    for obj in objects:
        if not hasattr(obj, "bound_box"):
            points.append(obj.matrix_world.translation)
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return Vector((0.0, 0.0, 0.0))
    center = Vector((0.0, 0.0, 0.0))
    for point in points:
        center += point
    return center / len(points)


def parent_room_shell_to_empty():
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        raise KeyError(f"Collection not found: {COLLECTION_NAME}")

    room_objects = [obj for obj in collection.objects if obj.name != CONTROLLER_NAME]
    if not room_objects:
        raise RuntimeError(f"No objects found in collection: {COLLECTION_NAME}")

    empty = bpy.data.objects.get(CONTROLLER_NAME)
    if empty is None:
        empty = bpy.data.objects.new(CONTROLLER_NAME, None)
        empty.empty_display_type = "CUBE"
        empty.empty_display_size = 0.6
        collection.objects.link(empty)

    empty.location = collection_center(room_objects)

    for obj in room_objects:
        world = obj.matrix_world.copy()
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()
        obj.matrix_world = world

    bpy.ops.object.select_all(action="DESELECT")
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty
    print(f"Parented {len(room_objects)} room objects to {CONTROLLER_NAME}. Move this Empty to move the whole room.")


if __name__ == "__main__":
    parent_room_shell_to_empty()
