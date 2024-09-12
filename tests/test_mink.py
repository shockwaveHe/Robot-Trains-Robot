import time

import mink
import mujoco
import mujoco.viewer
import pygame

from toddlerbot.tools.teleop.joystick import get_controller_input, initialize_joystick

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(
        "toddlerbot/robot_descriptions/toddlerbot/toddlerbot_fixed_scene.xml"
    )

    configuration = mink.Configuration(model)

    hands = ["hand", "hand_2"]

    tasks = [
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    tasks = []
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

    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    joystick = initialize_joystick()

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand in hands:
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        while viewer.is_running():
            # Process joystick events
            pygame.event.pump()

            # Control mocap bodies using joystick
            controller_input = get_controller_input(
                joystick, [[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]]
            )

            data.mocap_pos[hands_mid[0], 0] += (
                controller_input[0] * 0.01
            )  # Move in x direction
            data.mocap_pos[hands_mid[0], 1] += (
                controller_input[1] * 0.01
            )  # Move in y direction

            # Update task targets.
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            vel = mink.solve_ik(configuration, tasks, 0.005, solver, 1e-1)
            configuration.integrate_inplace(vel, 0.005)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS
            time.sleep(0.005)
            viewer.sync()
