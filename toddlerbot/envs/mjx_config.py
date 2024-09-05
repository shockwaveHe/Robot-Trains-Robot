from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MJXConfig:
    @dataclass
    class SimConfig:
        timestep: float = 0.002
        solver: int = 2  # Newton
        iterations: int = 1
        ls_iterations: int = 4

    @dataclass
    class ObsConfig:
        @dataclass
        class ObsScales:
            lin_vel: float = 2.0
            ang_vel: float = 1.0
            dof_pos: float = 1.0
            dof_vel: float = 0.05
            euler: float = 1.0
            # height_measurements: float = 5.0

        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 101
        num_single_privileged_obs: int = 140
        scales: ObsScales = ObsScales()

    @dataclass
    class ActionConfig:
        contact_force_threshold: float = 5.0
        n_frames: int = 6

    @dataclass
    class RewardsConfig:
        @dataclass
        class RewardScales:
            torso_pos: float = 0.0  # 1.0
            torso_quat: float = 1.5
            lin_vel_xy: float = 1.5
            lin_vel_z: float = 0.5
            ang_vel_xy: float = 0.5
            ang_vel_z: float = 0.5
            leg_joint_pos: float = 5.0
            leg_joint_vel: float = 0.0  # 1e-4
            arm_joint_pos: float = 0.0  # 0.1
            arm_joint_vel: float = 0.0  # 1e-3
            neck_joint_pos: float = 0.0  # 0.1
            neck_joint_vel: float = 0.0  # 1e-3
            waist_joint_pos: float = 0.0  # 50.0
            waist_joint_vel: float = 0.0  # 1e-3
            motor_torque: float = 5e-3
            joint_acc: float = 5e-7
            leg_action_rate: float = 1e-2
            leg_action_acc: float = 1e-2
            arm_action_rate: float = 0.0  # 1e-2
            arm_action_acc: float = 0.0  # 1e-2
            neck_action_rate: float = 0.0  # 1e-2
            neck_action_acc: float = 0.0  # 1e-2
            waist_action_rate: float = 1e-2
            waist_action_acc: float = 1e-2
            collision: float = 0.0  # 1.0
            survival: float = 10.0

            def reset(self):
                for key in vars(self):
                    setattr(self, key, 0.0)

        healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 0.4])
        tracking_sigma: float = 5.0
        min_feet_distance: float = 0.06
        max_feet_distance: float = 0.15
        target_feet_z_delta: float = 0.03
        scales: RewardScales = RewardScales()

    @dataclass
    class CommandsConfig:
        resample_time: float = 5.0

    @dataclass
    class DomainRandConfig:
        friction_range: Optional[List[float]] = field(
            default_factory=lambda: [0.6, 1.4]
        )
        gain_range: Optional[List[float]] = field(default_factory=lambda: [-5, 5])
        damping_range: Optional[List[float]] = field(default_factory=lambda: [0.8, 1.2])
        armature_range: Optional[List[float]] = field(
            default_factory=lambda: [0.8, 1.2]
        )
        # TODO: add mass_range
        added_mass_range: Optional[List[float]] = None
        push_interval_s: int = 2  # seconds
        push_vel: float = 0.05

    @dataclass
    class NoiseConfig:
        reset_noise_pos: float = 0.1
        obs_noise_scale: float = 0.1
        dof_pos: float = 1.0
        dof_vel: float = 2.0
        ang_vel: float = 2.0
        euler: float = 1.0

    def __init__(self):
        self.sim = self.SimConfig()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.rewards = self.RewardsConfig()
        self.commands = self.CommandsConfig()
        self.domain_rand = self.DomainRandConfig()
        self.noise = self.NoiseConfig()
