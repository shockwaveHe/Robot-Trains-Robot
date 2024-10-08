import os
from pathlib import Path

os.environ["MUJOCO_LOG"] = "error"
import mink
import mujoco
import mujoco.viewer
import mujoco_py
import numpy as np
from loop_rate_limiters import RateLimiter

_HERE = Path(__file__).parent
_XML = _HERE / "Archive" / "toddlerbot_scene.xml"

mujoco_py.ignore_mujoco_warnings()
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # mink configuration
    configuration = mink.Configuration(model)

    ##############################
    #  The block of code below is used when calling
    #  data.ctrl[actuator_ids] = configuration.q[dof_ids] during the simulation
    #  Not used in this example: reference to arm_aloha.py
    dof_ids = []
    joint_names = {}
    actuator_names = {}
    actuator_ids = []
    for actuator_id in range(model.nu):  # model.nu is the number of actuators
        actuator_ids.append(actuator_id)
        joint_id = model.actuator_trnid[actuator_id][
            0
        ]  # Get the joint ID controlled by this actuator
        dof_ids.append(joint_id)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        joint_names[joint_id] = joint_name
        actuator_names[actuator_id] = actuator_name

    dof_ids = np.array(dof_ids)
    actuator_ids = np.array(actuator_ids)
    ##############################

    # Define tasks.

    ##############################
    # Tasks for balancing the body.
    body_task = mink.FrameTask(
        frame_name="torso", frame_type="body", position_cost=10.0, orientation_cost=1.0
    )

    left_knee_task = mink.FrameTask(
        frame_name="xm430_plate",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=1.0,
    )

    right_knee_task = mink.FrameTask(
        frame_name="xm430_plate_2",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=1.0,
    )

    left_foot_task = mink.FrameTask(
        frame_name="ank_roll_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=10.0,
    )

    right_foot_task = mink.FrameTask(
        frame_name="ank_roll_link_2",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=10.0,
    )

    waist_task = mink.FrameTask(
        frame_name="waist_gears",
        frame_type="body",
        position_cost=10.0,
        orientation_cost=1.0,
    )
    ##############################

    # task for moving the gripper
    end_effector_task = mink.FrameTask(
        frame_name="rail_2",
        frame_type="body",
        position_cost=20.0,
        orientation_cost=20.0,
    )

    # right_elbow_task = mink.FrameTask(
    #     frame_name="2xl430_gears_5",
    #     frame_type="body",
    #     position_cost=0.1,
    #     orientation_cost=0.1)

    tasks = [body_task, left_foot_task, right_foot_task, waist_task, end_effector_task]
    ############################################

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 1

    # Enable collision avoidance between (wrist3, floor) and (wrist3, wall).

    # waist_link_geoms = mink.get_body_geom_ids(model, model.body("waist_link").id)
    # collision_pairs = [
    #     (waist_link_geoms, ['ank_roll_link_2_collision']),
    # ]

    # limits = [
    #     mink.ConfigurationLimit(model=configuration.model),
    #     mink.CollisionAvoidanceLimit(
    #         model=configuration.model,
    #         geom_pairs=collision_pairs,
    #     ),
    # ]

    # Obtaining defined target mocap ids for each task
    # Targets are defined in the torddlerbot_scene.xml
    # TODO: Find a way to define the target in the code
    body_target_mid = model.body("torso_target").mocapid[0]
    left_knee_target_mid = model.body("left_knee_target").mocapid[0]
    right_knee_target_mid = model.body("right_knee_target").mocapid[0]
    waist_target_mid = model.body("waist_target").mocapid[0]
    gripper_target_mid = model.body("gripper_target").mocapid[0]
    left_foot_target_mid = model.body("left_foot_target").mocapid[0]
    right_foot_target_mid = model.body("right_foot_target").mocapid[0]
    # right_elbow_target_mid = model.body("right_elbow_target").mocapid[0]

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.flags[mujoco.mjtFrame.mjFRAME_WORLD] = True
        rate = RateLimiter(frequency=100.0)
        dt = rate.dt
        t = 0

        configuration.update_from_keyframe("home")
        body_initial_T_w = configuration.get_transform_frame_to_world("torso", "body")
        center_translation = body_initial_T_w.translation()[:2]
        circle_radius = 0.5
        body_task.set_target_from_configuration(configuration)

        # initial target positions: for balancing the body
        left_knee_initial_T = configuration.get_transform_frame_to_world(
            "xm430_plate", "body"
        )
        right_knee_initial_T = configuration.get_transform_frame_to_world(
            "xm430_plate_2", "body"
        )
        waist_initial_T = configuration.get_transform_frame_to_world(
            "waist_gears", "body"
        )
        gripper_target_initial_T = configuration.get_transform_frame_to_world(
            "gripper_target", "body"
        )
        left_foot_initial_T = configuration.get_transform_frame_to_world(
            "ank_roll_link", "body"
        )
        right_foot_initial_T = configuration.get_transform_frame_to_world(
            "ank_roll_link_2", "body"
        )
        # right_elbow_initial_T = configuration.get_transform_frame_to_world("2xl430_gears_5", "body")

        while viewer.is_running():
            # # Update task target positions
            T = body_task.transform_target_to_world.copy()

            zero_rot = mink.SO3.from_rpy_radians(0.0, 0.0, 0).wxyz

            translation = T.translation()
            data.mocap_pos[body_target_mid] = translation
            data.mocap_quat[body_target_mid] = zero_rot
            data.mocap_pos[left_knee_target_mid] = left_knee_initial_T.translation()
            data.mocap_quat[left_knee_target_mid] = left_knee_initial_T.rotation().wxyz
            data.mocap_pos[right_knee_target_mid] = right_knee_initial_T.translation()
            data.mocap_quat[right_knee_target_mid] = (
                right_knee_initial_T.rotation().wxyz
            )
            data.mocap_pos[waist_target_mid] = waist_initial_T.translation()
            data.mocap_quat[waist_target_mid] = waist_initial_T.rotation().wxyz
            data.mocap_pos[gripper_target_mid] = gripper_target_initial_T.translation()
            data.mocap_pos[left_foot_target_mid] = left_foot_initial_T.translation()
            data.mocap_quat[left_foot_target_mid] = left_foot_initial_T.rotation().wxyz
            data.mocap_pos[right_foot_target_mid] = right_foot_initial_T.translation()
            data.mocap_quat[right_foot_target_mid] = (
                right_foot_initial_T.rotation().wxyz
            )
            # data.mocap_pos[right_elbow_target_mid] = right_elbow_initial_T.translation()
            # data.mocap_quat[right_elbow_target_mid] = right_elbow_initial_T.rotation().wxyz

            # end effector target position: a sinusoidal trajectory
            data.mocap_pos[gripper_target_mid][1] += 0.05 * np.cos(0.5 * t + np.pi)
            data.mocap_pos[gripper_target_mid][2] += 0.02 * np.sin(0.5 * t)

            # Update task targets
            body_task.set_target(mink.SE3.from_mocap_id(data, body_target_mid))
            left_knee_task.set_target(
                mink.SE3.from_mocap_id(data, left_knee_target_mid)
            )
            right_knee_task.set_target(
                mink.SE3.from_mocap_id(data, right_knee_target_mid)
            )
            waist_task.set_target(mink.SE3.from_mocap_id(data, waist_target_mid))
            left_foot_task.set_target(
                mink.SE3.from_mocap_id(data, left_foot_target_mid)
            )
            right_foot_task.set_target(
                mink.SE3.from_mocap_id(data, right_foot_target_mid)
            )
            # right_elbow_task.set_target(mink.SE3.from_mocap_id(data, right_elbow_target_mid))
            end_effector_task.set_target(
                mink.SE3.from_mocap_id(data, gripper_target_mid)
            )

            for i in range(max_iters):
                # solve IK
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-2)
                configuration.integrate_inplace(vel, rate.dt)

                # Termination condition
                end_effector_error = end_effector_task.compute_error(configuration)
                end_effector_achieved = (
                    np.linalg.norm(end_effector_error) <= pos_threshold
                )

                if end_effector_achieved:
                    print(f"Exiting after {i} iterations.")
                    break

            # data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_camlight(model, data)
            mujoco.mj_step(model, data)

            # Manually reset configuration.data to mujoco data
            # Capture any changes in simulation that are not done by mink
            configuration.data = data
            viewer.sync()
            rate.sleep()
            t += dt
