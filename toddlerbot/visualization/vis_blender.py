import os
import pickle
import xml.etree.ElementTree as ET

import bpy


def clear_collections(include_list=["Visual", "Collision"]):
    for col in bpy.data.collections:
        if col.name in include_list:
            bpy.data.collections.remove(col)


def create_collection(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    return bpy.data.collections[collection_name]


def import_mesh_to_blender(mesh_name, mesh_path, collection):
    # Check file extension and use the appropriate importer
    if mesh_path.lower().endswith(".stl"):
        bpy.ops.import_mesh.stl(filepath=mesh_path)
    elif mesh_path.lower().endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported file format for {mesh_path}")

    obj = bpy.context.selected_objects[-1]
    obj.name = mesh_name

    for col in obj.users_collection:
        col.objects.unlink(obj)

    collection.objects.link(obj)


def import_meshes(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # First clear existing collections
    visual_collection = create_collection("Visual")
    collision_collection = create_collection("Collision")
    # Make collision collection invisible by default
    collision_collection.hide_viewport = True

    for mesh in root.findall(".//mesh"):
        mesh_name = mesh.get("name")
        mesh_file = mesh.get("file")
        if mesh_name and mesh_file:
            mesh_path = os.path.join(os.path.dirname(xml_path), mesh_file)
            if "visual" in mesh_name:
                import_mesh_to_blender(mesh_name, mesh_path, visual_collection)
            elif "collision" in mesh_name:
                import_mesh_to_blender(mesh_name, mesh_path, collision_collection)
            else:
                raise ValueError(f"Unsupported mesh name {mesh_name}")


def get_body2mesh_mapping(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    body2mesh_visual = {}
    body2mesh_collision = {}
    for body in root.findall(".//body"):
        body_name = body.get("name")
        for geom in body.findall("./geom"):
            mesh_name = geom.get("mesh")
            if mesh_name:
                if "visual" in mesh_name:
                    body2mesh_visual[body_name] = mesh_name
                elif "collision" in mesh_name:
                    body2mesh_collision[body_name] = mesh_name
                else:
                    raise ValueError(f"Unsupported mesh name {mesh_name}")

    return body2mesh_visual, body2mesh_collision


def set_frame_range(anim_data):
    scene = bpy.context.scene
    max_frame = -float("inf")

    # Iterate through all animation data to determine the min and max frame
    for keyframes in anim_data.values():
        for frame_idx, _ in enumerate(keyframes):
            if frame_idx > max_frame:
                max_frame = frame_idx

    # Set the scene frame start and end
    scene.frame_start = 1
    scene.frame_end = max_frame + 1

    print(f"Frame start: {scene.frame_start}, Frame end: {scene.frame_end}")


def clear_keyframes(obj, data_paths):
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        for data_path in data_paths:
            for fcurve in action.fcurves:
                if fcurve.data_path == data_path:
                    # Remove the fcurve
                    action.fcurves.remove(fcurve)


def import_animation(root_path, body2mesh_visual, body2mesh_collision):
    anim_data_path = os.path.join(root_path, "anim_data.pkl")
    with open(anim_data_path, "rb") as f:
        anim_data = pickle.load(f)

    set_frame_range(anim_data)  # Set the frame range based on animation data

    for obj_key, timestamped_obj_poses in anim_data.items():
        if obj_key in body2mesh_visual:
            blender_obj_name = body2mesh_visual[obj_key]
        elif obj_key in body2mesh_collision:
            blender_obj_name = body2mesh_collision[obj_key]
        else:
            print(f"Unsupported body {obj_key}")
            continue

        blender_obj = bpy.data.objects.get(blender_obj_name)

        if not blender_obj:
            print(f"Object named {blender_obj_name} not found in the scene.")
            continue

        # Clear existing keyframes for location and rotation
        clear_keyframes(blender_obj, ["location", "rotation_quaternion"])

        for frame_idx, (timestamp, pos, quat) in enumerate(timestamped_obj_poses):
            blender_obj.location = pos
            blender_obj.rotation_mode = "QUATERNION"
            blender_obj.rotation_quaternion = quat
            blender_obj.keyframe_insert(data_path="location", frame=frame_idx + 1)
            blender_obj.keyframe_insert(
                data_path="rotation_quaternion", frame=frame_idx + 1
            )


if __name__ == "__main__":
    robot_name = "toddlerbot_legs"
    reimport = False
    exp_folder_path = "results/20240420_140344_walk_toddlerbot_legs_mujoco"

    blend_file_path = bpy.path.abspath("//")
    repo_path = os.path.join(blend_file_path, "../../")

    robot_dir = os.path.join(repo_path, "toddlerbot", "descriptions", robot_name)
    if os.path.exists(robot_dir):
        xml_path = os.path.join(robot_dir, robot_name + ".xml")
    else:
        raise FileNotFoundError(f"Robot description not found at {robot_dir}")

    if reimport:
        clear_collections()
        import_meshes(xml_path)

    body2mesh_visual, body2mesh_collision = get_body2mesh_mapping(xml_path)

    root_path = os.path.join(repo_path, exp_folder_path)
    import_animation(root_path, body2mesh_visual, body2mesh_collision)
