from dataclasses import dataclass, field
from typing import List


@dataclass
class MJXConfig:
    @dataclass
    class SimConfig:
        timestep: float = 0.004
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
        action_scale: float = 0.25
        filter_type: str = "none"
        filter_order: int = 4
        filter_cutoff: float = 10.0
        contact_force_threshold: float = 1.0
        n_steps_delay: int = 1
        n_frames: int = 5

    @dataclass
    class RewardsConfig:
        @dataclass
        class RewardScales:
            torso_pos: float = 0.0  # 1.0
            torso_quat: float = 1.0
            lin_vel_xy: float = 1.0
            lin_vel_z: float = 0.5
            ang_vel_xy: float = 0.5
            ang_vel_z: float = 1.0
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
            feet_contact: float = 0.5
            feet_contact_number: float = 0.0
            collision: float = 0.0  # 1.0
            survival: float = 10.0

            def reset(self):
                for key in vars(self):
                    setattr(self, key, 0.0)

        healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 0.4])
        tracking_sigma: float = 5.0
        min_feet_y_dist: float = 0.05
        max_feet_y_dist: float = 0.13
        target_feet_z_delta: float = 0.02
        torso_pitch_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])
        scales: RewardScales = RewardScales()

    @dataclass
    class CommandsConfig:
        num_commands: int = 3
        resample_time: float = 5.0
        command_list: List[List[float]] = field(default_factory=lambda: [[0.0]])

    @dataclass
    class DomainRandConfig:
        friction_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
        damping_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        armature_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        frictionloss_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        added_mass_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])
        kp_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        kd_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        tau_max_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        q_dot_tau_max_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        q_dot_max_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        push_interval_s: int = 4  # seconds
        push_vel: float = 0.1

    @dataclass
    class NoiseConfig:
        reset_noise_pos: float = 0.05
        obs_noise_scale: float = 0.05
        dof_pos: float = 1.0
        dof_vel: float = 2.0
        ang_vel: float = 2.0
        euler: float = 1.0
        backlash_scale: float = 0.02
        backlash_activation: float = 0.1

    def __init__(self):
        self.sim = self.SimConfig()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.rewards = self.RewardsConfig()
        self.commands = self.CommandsConfig()
        self.domain_rand = self.DomainRandConfig()
        self.noise = self.NoiseConfig()
