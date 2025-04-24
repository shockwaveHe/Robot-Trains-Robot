import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R


class MinkRetarget:
    def __init__(
        self,
        model_path,
        ik_match_table,
        scale,
        ground=0.0,
        solver="quadprog",
        damping=1e-1,
    ) -> None:
        self.model = mj.MjModel.from_xml_path(model_path)
        self.ik_match_table = ik_match_table
        self.scale = scale
        self.ground = ground * np.array([0, 0, 1])

        self.solver = solver
        self.damping = damping

        self.vicon2task = {}
        self.pos_offsets = {}
        self.rot_offsets = {}

        self.task_errors = {}

        self.setup_retarget_configuration()

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        for frame_name, entry in self.ik_match_table.items():
            # add frame_type, modified by Yao He
            vicon_name, frame_type, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type=frame_type,
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.vicon2task[vicon_name] = task
                self.pos_offsets[vicon_name] = np.array(pos_offset) - self.ground
                self.rot_offsets[vicon_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks.append(task)
                self.task_errors[task] = []

    def update_targets(self, vicon_data):
        # Update frame task targets
        for vicon_name in self.vicon2task.keys():
            task = self.vicon2task[vicon_name]
            pos_offset = self.pos_offsets[vicon_name]
            rot_offset = self.rot_offsets[vicon_name]

            pos = (
                self.scale * np.array(vicon_data[vicon_name][0])
            ) + pos_offset  # Only scale the Vicon positions, not the offset
            rot = (
                R.from_quat(vicon_data[vicon_name][1], scalar_first=True) #* rot_offset
            ).as_quat(scalar_first=True)

            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget(self, vicon_data):
        # Update the task targets
        self.update_targets(vicon_data)

        # Solve the IK problem
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration, self.tasks, dt, self.solver, self.damping
        )
        self.configuration.integrate_inplace(vel, dt)

        # Return the solution
        return self.configuration.data.qpos

    def retarget_v2(self, vicon_data, dt, max_iters=100, pos_threshold = 1e-4):
        # Update the task targets
        self.update_targets(vicon_data)

        for i in range(max_iters):
            # solve IK
            vel = mink.solve_ik(self.configuration, self.tasks, dt, self.solver, self.damping)
            self.configuration.integrate_inplace(vel, dt)
        
        return self.configuration.data.qpos
    def update_task_errors(self):
        for task in self.tasks:
            self.task_errors[task].append(task.compute_error(self.configuration))

    def error(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks]
            )
        )
