import mujoco  # type: ignore
import numpy as np


def compare_models(model, model_copy, atol=1e-6, rtol=1e-6):
    # Iterate over all attributes of the model object
    for attr in dir(model):
        # Skip internal/private attributes and methods
        if not attr.startswith("_") and not callable(getattr(model, attr)):
            # Get the attribute values from both models
            model_attr = getattr(model, attr)
            model_copy_attr = getattr(model_copy, attr)

            try:
                # Check if the attributes are numerical arrays or scalars and compare
                if isinstance(model_attr, (np.ndarray, list)) and isinstance(
                    model_copy_attr, (np.ndarray, list)
                ):
                    if not np.allclose(
                        model_attr, model_copy_attr, atol=atol, rtol=rtol
                    ):
                        print(f"Difference in attribute {attr}:")
                        # print(f"model: {model_attr}")
                        # print(f"model_copy: {model_copy_attr}")
                elif isinstance(model_attr, (int, float)) and isinstance(
                    model_copy_attr, (int, float)
                ):
                    if not np.isclose(
                        model_attr, model_copy_attr, atol=atol, rtol=rtol
                    ):
                        print(f"Difference in attribute {attr}:")
                        # print(f"model: {model_attr}")
                        # print(f"model_copy: {model_copy_attr}")
                else:
                    # Compare non-numerical attributes (strings, etc.)
                    if model_attr != model_copy_attr:
                        pass
                        # print(f"Difference in attribute {attr}:")
                        # print(f"model: {model_attr}")
                        # print(f"model_copy: {model_copy_attr}")
            except Exception as e:
                print(f"Error comparing attribute {attr}: {e}")


xml_path = "toddlerbot/robot_descriptions/toddlerbot/toddlerbot_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
data = mujoco.MjData(model)


def get_body_mass_attr_range(model, data, num_env):
    body_mass_attr_range = []
    body_mass_delta_range = np.linspace(-0.5, 0.5, num_env)
    body_mass = np.array(model.body("torso").mass).copy()
    body_inertia = np.array(model.body("torso").inertia).copy()
    for body_mass_delta in body_mass_delta_range:
        model.body("torso").mass = body_mass + body_mass_delta
        model.body("torso").inertia = (
            (body_mass + body_mass_delta) / body_mass * body_inertia
        )
        mujoco.mj_setConst(model, data)
        body_mass_attr_range.append(
            {
                "body_mass": np.array(model.body_mass).copy(),
                "body_inertia": np.array(model.body_inertia).copy(),
                "actuator_acc0": np.array(model.actuator_acc0).copy(),
                "body_invweight0": np.array(model.body_invweight0).copy(),
                "body_subtreemass": np.array(model.body_subtreemass).copy(),
                "dof_M0": np.array(model.dof_M0).copy(),
                "dof_invweight0": np.array(model.dof_invweight0).copy(),
                "tendon_invweight0": np.array(model.tendon_invweight0).copy(),
            }
        )

    return body_mass_attr_range


body_mass_attr_range = get_body_mass_attr_range(model, data, 11)

model_new = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
for attr_name, attr in body_mass_attr_range[6].items():
    setattr(model_new, attr_name, attr)

xml_path_copy = "toddlerbot/robot_descriptions/toddlerbot/toddlerbot_scene_copy.xml"
model_copy = mujoco.MjModel.from_xml_path(xml_path_copy)  # type: ignore

compare_models(model_new, model_copy)
