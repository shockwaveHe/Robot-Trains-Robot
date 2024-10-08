from pathlib import Path

import mink
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

# <body name="com_target" pos="0.0 0 .3442" mocap="true">
#     <geom type="box" size=".05 .05 .05" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
#     <site type="sphere" size="0.01" rgba="1 1 1 1" group="1"/>
# </body>
# <body name="ank_roll_link_target" pos="0.0 0 .3442" mocap="true">
#     <geom type="box" size=".01 .01 .01" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
#     <site type="sphere" size="0.01" rgba="1 1 1 1" group="1"/>
# </body>
# <body name="ank_roll_link_2_target" pos="0.0 0 .3442" mocap="true">
#     <geom type="box" size=".01 .01 .01" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
#     <site type="sphere" size="0.01" rgba="1 1 1 1" group="1"/>
# </body>
# <body name="hand_target" pos="0.1 0.1 0.3442" mocap="true">
#     <geom type="box" size=".03 .03 .03" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
#     <site type="sphere" size="0.05" rgba="1 0 1 0" group="1"/>
# </body>
# <body name="hand_2_target" pos="0.1 -0.1 0.3442" mocap="true">
#     <geom type="box" size=".03 .03 .03" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
#     <site type="sphere" size="0.05" rgba="1 0 1 0" group="1"/>
# </body>

_HERE = Path(__file__).parent
_XML = (
    _HERE
    / ".."
    / "toddlerbot"
    / "robot_descriptions"
    / "toddlerbot_OP3"
    / "toddlerbot_OP3_scene.xml"
)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["ank_roll_link", "ank_roll_link_2"]
    hands = ["hand", "hand_2"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="torso",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="body",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="body",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective bodies.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "body")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "body")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
