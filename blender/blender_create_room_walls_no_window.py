import importlib.util
from pathlib import Path


BASE_SCRIPT = Path(r"D:\code\HSI\blender\blender_create_room_walls.py")


def load_room_wall_script():
    spec = importlib.util.spec_from_file_location("blender_create_room_walls_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load room wall script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_room_wall_script()
    module.ENABLE_WINDOW = False
    module.create_room_shell()
    print("Created HSI_room_shell without a window.")


if __name__ == "__main__":
    main()
