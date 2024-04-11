import time

import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from toddlerbot.sim import *
from toddlerbot.utils.constants import GRAVITY
from toddlerbot.utils.file_utils import find_description_path


class MuJoCoSim(BaseSim):
    def __init__(self, robot, xml_path=None, fixed: bool = False):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()
        self.name = "mujoco"

        self.model = None
        self.data = None

        self.last_mom = None
        self.last_angmom = None
        self.last_t = None

        self.foot_size = robot.foot_size

        if xml_path is None:
            xml_path = find_description_path(robot.name, suffix="_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        if not fixed:
            self.put_robot_on_ground(robot)

    def put_robot_on_ground(self, robot: HumanoidRobot, z_offset: float = 0.02):
        """
        Adjust the robot's position to place its lowest point at a specified offset above the ground.

        Args:
            robot (HumanoidRobot): The humanoid robot.
            z_offset (float): The offset from the ground to place the robot. Default is 0.01.
        """
        lowest_z = float("inf")

        mujoco.mj_kinematics(self.model, self.data)
        # Iterate through all body parts to find the lowest point
        for i in range(self.model.nbody):
            if self.data.body(i).name == "world":
                continue
            # To correpond to the PyBullet code, we use xipos instead of xpos
            body_pos = self.data.body(i).xipos
            lowest_z = min(lowest_z, body_pos[2])

        body_link_name = robot.config.canonical_name2link_name["body_link"]
        base_pos = self.data.body(body_link_name).xpos
        desired_z = base_pos[2] - lowest_z + z_offset
        if lowest_z < 0:
            raise ValueError(
                f"Robot is below the ground. Change the z value of {body_link_name} to be {desired_z}"
            )
        elif lowest_z > z_offset:
            raise ValueError(
                f"Robot is too high above the ground. Change the z value of {body_link_name} as {desired_z}"
            )

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_pos = self.data.body(link_name).xpos
        return np.array(link_pos)

    def get_link_quat(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_quat = self.data.body(link_name).xquat
        return np.array(link_quat)

    def get_torso_pose(self, robot: HumanoidRobot):
        mujoco.mj_kinematics(self.model, self.data)
        torso_pos = self.data.site("torso").xpos.copy()
        torso_mat = self.data.site("torso").xmat.copy().reshape(3, 3)
        return torso_pos, torso_mat

    def get_joint_state(self, robot: HumanoidRobot):
        joint_state_dict = {}
        time_curr = time.time()
        for name, info in robot.joints_info.items():
            if info["active"]:
                joint_state_dict[name] = JointState(
                    time=time_curr, pos=self.data.joint(name).qpos.item()
                )

        return joint_state_dict

    def _compute_dmom(self):
        if self.last_t is None:
            self.last_mom = np.zeros(3)
            self.last_angmom = np.zeros(3)
            self.last_t = time.time()
        else:
            # Calculate current momentum and angular momentum
            mom = np.zeros(3)
            angmom = np.zeros(3)
            for i in range(self.model.nbody):
                body_name = self.model.body(i).name
                if body_name == "world":
                    continue

                body_mass = self.model.body(i).mass
                body_inertia = self.model.body(i).inertia
                body_pos = self.data.body(i).xipos  # Position
                body_xmat = self.data.body(i).xmat.reshape(3, 3)  # Rotation matrix
                # the translation component comes after the rotation component
                body_vel = self.data.body(i).cvel[3:]  # Linear Velocity
                body_ang_vel = self.data.body(i).cvel[:3]  # Angular velocity

                # Eq. 3.63-3.66 on p.94 in "Introduction to Humanoid Robotics" by Shuuji Kajita
                P = body_mass * body_vel
                L = (
                    np.cross(body_pos, P)
                    + body_xmat @ np.diag(body_inertia) @ body_xmat.T @ body_ang_vel
                )
                mom += P
                angmom += L

            time_curr = time.time()
            dt = time_curr - self.last_t

            self.dP = (mom - self.last_mom) / dt
            self.dL = (angmom - self.last_angmom) / dt

            # Update last states
            self.last_mom = mom
            self.last_angmom = angmom
            self.last_t = time_curr

    def get_zmp(self, com_pos, pz=0.0):
        M = self.model.body(0).subtreemass
        cx, cy = com_pos
        # print(f"dP: {self.dP}, dL: {self.dL}")
        # Eq. 3.73-3.74 on p.96 in "Introduction to Humanoid Robotics" by Shuuji Kajita
        px_zmp = (M * GRAVITY * cx + pz * self.dP[0] - self.dL[1]) / (
            M * GRAVITY + self.dP[2]
        )
        py_zmp = (M * GRAVITY * cy + pz * self.dP[1] + self.dL[0]) / (
            M * GRAVITY + self.dP[2]
        )
        self.zmp = [px_zmp.item(), py_zmp.item()]

        return self.zmp

    def set_joint_angles(
        self,
        robot,
        joint_angles,
        interp=True,
        interp_method="cubic",
        control_dt=0.01,
    ):
        joint_order = list(joint_angles.keys())
        pos = np.array(list(joint_angles.values()))

        def set_pos_helper(pos):
            force_dict = {}
            for name, p in zip(joint_order, pos):
                kp = robot.config.motor_params[name].kp
                kv = robot.config.motor_params[name].kv
                force = (
                    kp * (p - self.data.joint(name).qpos)
                    - kv * self.data.joint(name).qvel
                )
                force_dict[name] = force.item()
                self.data.actuator(f"{name}_act").ctrl = force

            # log(f"Force: {force_dict}", header=self.name, level="debug")

        if interp:
            pos_start = np.array(
                [state.pos for state in self.get_joint_state(robot).values()]
            )
            max_step_size = 0.5
            if np.max(np.abs(pos - pos_start)) > max_step_size:
                interpolate_pos(
                    set_pos_helper,
                    pos_start,
                    pos,
                    control_dt,
                    interp_method,
                    self.name,
                    sleep_time=control_dt,
                )
            else:
                time_start = time.time()
                set_pos_helper(pos)
                time_elapsed = time.time() - time_start
                time_until_next_step = control_dt - time_elapsed
                if time_until_next_step > 0:
                    sleep(time_until_next_step)
        else:
            set_pos_helper(pos)

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        vis_flags: Optional[List] = [],
        sleep_time: float = 0.0,
    ):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if step_params is not None:
            sim_step_idx, path, foot_steps, com_traj = step_params

        def vis_foot_steps():
            i = viewer.user_scn.ngeom
            for foot_step in foot_steps:
                if foot_step.support_leg == "both":
                    continue

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                    size=[
                        self.foot_size[0] / 2,
                        self.foot_size[1] / 2,
                        self.foot_size[2] / 2,
                    ],
                    pos=np.array(
                        [
                            foot_step.position[0],
                            foot_step.position[1],
                            self.foot_size[2] / 2,
                        ]
                    ),
                    mat=euler2mat(0, 0, foot_step.position[2]).flatten(),
                    rgba=(
                        [0, 0, 1, 1]
                        if foot_step.support_leg == "left"
                        else [0, 1, 0, 1]
                    ),
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_com_traj():
            i = viewer.user_scn.ngeom
            for com_pos in com_traj:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.001, 0.001, 0.001]),
                    pos=np.array([com_pos[0], com_pos[1], 0.005]),
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1],
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_path():
            i = viewer.user_scn.ngeom
            for j in range(len(path) - 1):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([1, 1, 1]),
                    pos=np.array([0, 0, 0]),
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 0, 1],
                )
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    100,
                    np.array([*path[j], 0.0]),
                    np.array([*path[j + 1], 0.0]),
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_torso():
            i = viewer.user_scn.ngeom
            torso_pos = self.data.site("torso").xpos
            torso_mat = self.data.site("torso").xmat
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.005, 0.005, 0.15]),
                pos=torso_pos,
                mat=torso_mat,
                rgba=[1, 0, 0, 1],
            )
            viewer.user_scn.ngeom = i + 1

        try:
            while viewer.is_running():
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                self._compute_dmom()

                with viewer.lock():
                    if step_func is not None:
                        if step_params is None:
                            step_func()
                        else:
                            step_params = step_func(*step_params)

                            viewer.user_scn.ngeom = 0
                            if "foot_steps" in vis_flags:
                                vis_foot_steps()
                            if "com_traj" in vis_flags:
                                vis_com_traj()
                            if "path" in vis_flags:
                                vis_path()
                            if "torso" in vis_flags:
                                vis_torso()

                viewer.sync()

                time_until_next_step = sleep_time - (time.time() - step_start)
                # time_until_next_step = self.model.opt.timestep - (
                #     time.time() - step_start
                # )
                if time_until_next_step > 0:
                    sleep(time_until_next_step)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()

    def close(self):
        pass


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = MuJoCoSim()
    sim.simulate()
