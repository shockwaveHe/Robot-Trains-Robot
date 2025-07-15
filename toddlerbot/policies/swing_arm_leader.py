import itertools
import struct
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import serial
from scipy.signal import hilbert

from toddlerbot.finetuning.finetune_config import FinetuneConfig, get_finetune_config
from toddlerbot.finetuning.utils import Timer
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.keyboard import Keyboard
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode

matplotlib.use("TkAgg")


class SwingArmLeaderPolicy(BasePolicy, policy_name="swing_arm_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "192.168.0.70",
        exp_folder: Optional[str] = "",
        finetune_cfg: Optional[Dict] = None,
    ):
        super().__init__(name, robot, init_motor_pos)
        if finetune_cfg is None:
            finetune_cfg = get_finetune_config("swing", exp_folder)

        self.finetune_cfg: FinetuneConfig = finetune_cfg
        self.zmq_sender = ZMQNode(type="sender", ip=ip)
        self.zmq_receiver = ZMQNode(type="receiver")

        print(f"ZMQ Connected to {ip}")

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None
        self.keyboard = Keyboard()
        self.control_dt = 0.02  # 0.02
        self.total_steps = 0
        self.follower_steps = 0

        self.system_start = False

        self.timer = Timer()
        self.timer.start()

        g = 9.81
        L = 0.55
        self.desired_fx_freq = np.sqrt(g / L) / (2 * np.pi)
        # self.modes = ["free", "free", "free"]
        self.modes = ["enhance", "free", "surpress"]
        self.mode = "free"
        self.expected_total_steps = 25000

        # different mode partitioning
        
        # self.mode_partition = (
        #     np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * self.expected_total_steps
        # )
        # self.mode_sequence = ["free_0", "enhance", "free_1", "surpress", "free_2"]

        # self.mode_partition = np.array([1.0]) * self.expected_total_steps
        # self.mode_sequence = ["free_0"]

        self.mode_partition = np.array([0.4, 0.8, 1.0]) * self.expected_total_steps
        self.mode_sequence = ["free_0", "enhance", "free_1"]

        # self.mode_partition = np.array([0.4, 0.8, 1.0]) * self.expected_total_steps
        # self.mode_sequence = ["free_0", "surpress", "free_1"]

        assert len(self.mode_sequence) == len(self.mode_partition)
        self.mode_sequence_steps = {
            self.mode_sequence[i]: self.mode_partition[i]
            for i in range(len(self.mode_sequence))
        }

        print("mode sequence:", self.mode_sequence_steps)

        # self.exernal_guidance_mode = "systematic"
        self.external_force_period = 8
        self.external_force_cycle = 0

        self.external_guidance_mode = "random"
        # self.external_guidance_mode = "systematic"
        self.guidance_list = range(1, 5)
        self.guidance_num = 3
        self.guidance_steps = np.random.choice(
            self.guidance_list, size=self.guidance_num, replace=False
        )
        print("guidance steps:", self.guidance_steps)

        self.supress_list = range(0, 4)
        self.supress_num = 1
        self.supress_steps = np.random.choice(
            self.supress_list, size=self.supress_num, replace=False
        )
        print("supress steps:", self.supress_steps)
        self.init_speed = 0.05
        self.warmup_time = 0.0
        self.z_pos_period = 1.0 / self.desired_fx_freq  # 1.5
        self.signal_len = int(
            self.z_pos_period // self.control_dt * 4
        )  # scale it to prevent noise, roughtly 6 seconds
        self.z_pos_range = [-0.025, 0.025]
        self.paused = False
        self.healthy_force_x = [-10.0, 10.0]
        self.healthy_force_y = [-3.0, 3.0]

        self.close_loop = True
        self.swing_buffer_size = self.finetune_cfg.swing_buffer_size

        self.ee_pos1_visualization_scale = 1.0
        self.ee_pos1_visualization_offset = 0.4
        self.fx_visualization_scale = 0.02
        self.visualize_downscale = 1
        self.fx_buffer = deque(maxlen=self.swing_buffer_size)

        self.fx_visualization_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        self.fy_buffer = deque(maxlen=self.swing_buffer_size)
        self.fz_buffer = deque(maxlen=self.swing_buffer_size)
        self.ee_pos1_visualization_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        # for phase analysis
        self.ee_pos1_buffer = deque(maxlen=self.swing_buffer_size)

        self.timestamp_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )

        self.rp_visualization_scale = 1.0
        self.rp_visualization_offset = 0.0
        self.reference_pos_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )

        self.fx_phase_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        self.ee_pos1_phase_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        self.ref_pos_phase_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )

        self.fx_freq_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        self.ee_pos1_freq_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )
        self.reference_pos_freq_buffer = deque(
            maxlen=self.swing_buffer_size // self.visualize_downscale
        )

        self.force = 20.0
        self.z_pos = 0.0  # z_pos is the target z position (not delta for now)
        self.last_z_pos = 0.0

        self.visualize = False
        self.visualization_update_interval = 0.2
        self.last_visualization_time = 0.0

        self.real_control_start_time = -1
        self.real_current_control_time = 0
        self.n_real_steps_total = 0
        if self.visualize:
            # plt.ion()
            self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(
                3, 1, figsize=(6, 10)
            )
            self.ee_pos1_curve = self.ax.plot(
                [],
                [],
                lw=2,
                label=f"ee_pos[1] x {self.ee_pos1_visualization_scale} + {self.ee_pos1_visualization_offset}",
            )[0]
            self.fx_curve = self.ax.plot(
                [], [], lw=2, label=f"fx x {self.fx_visualization_scale}"
            )[0]

            self.reference_pos_curve = self.ax.plot(
                [],
                [],
                lw=2,
                label=f"reference_pos x {self.rp_visualization_scale} + {self.rp_visualization_offset}",
            )[0]
            self.ax.legend()

            self.fx_phase_curve = self.ax2.plot([], [], lw=2, label="fx_phase")[0]
            self.ee_pos1_phase_curve = self.ax2.plot(
                [], [], lw=2, label="ee_pos1_phase"
            )[0]
            self.ref_pos_phase_curve = self.ax2.plot(
                [], [], lw=2, label="reference_pos_phase"
            )[0]
            self.ax2.legend()

            self.fx_freq_curve = self.ax3.plot([], [], lw=2, label="fx_freq")[0]
            self.ee_pos1_freq_curve = self.ax3.plot([], [], lw=2, label="ee_pos1_freq")[
                0
            ]
            self.reference_pos_freq_curve = self.ax3.plot(
                [], [], lw=2, label="reference_pos_freq"
            )[0]
            self.ax3.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            print("visualization started")
            plt.show(block=False)
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        shm_name = "force_shm"
        try:
            print("Creating shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=120
            )
        except FileExistsError:
            print("Using existing shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=False, size=120
            )
        self.fx_phase = 0
        self.fx_amplitude = 0.0
        self.ee_pos1_phase = 0
        self.reference_pos_phase = 0
        self.fx_freq = 0
        self.ee_pos1_freq = 0
        self.reference_pos_freq = 0

        self.arm_shm.buf[:8] = struct.pack("d", self.force)
        self.arm_shm.buf[8:16] = struct.pack("d", self.z_pos)
        self.ee_force = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[16:40]
        )
        self.ee_torque = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[40:64]
        )
        self.ee_pos = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[64:88]
        )
        self.ee_vel = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[88:112]
        )

    def close(self):
        self.zmq_sender.close()
        self.zmq_receiver.close()
        try:
            self.arm_shm.close()
            self.arm_shm.unlink()
        except FileNotFoundError:
            print("Shared memory not found")

    def get_enforcing_amplitude_scale(self, fx_amplitude):
        return 1

    def is_done(self, obs: Obs):
        # TODO: handle done situation
        # return obs.ee_force[0] < self.healthy_force_x[0] or obs.ee_force[0] > self.healthy_force_x[1] \
        #     or obs.ee_force[1] < self.healthy_force_y[0] or obs.ee_force[1] > self.healthy_force_y[1]
        return False

    def get_current_fx_angle(self):
        """
        Having the current fx curve, assuming it is sin curve
        Unknown time (since in training we don't know when the swing starts successfully)
        Directly get the current fx angle
        return: current fx angle
        """
        # Use fft to get the amplitude
        if len(self.fx_buffer) < self.swing_buffer_size // 5:
            # print("Not enough data to get the current fx angle")
            return 0.0
        signal = np.array(self.fx_buffer)[-self.signal_len :]
        analytical_signal = hilbert(signal)
        instantaneous_phase = np.angle(analytical_signal)
        current_phase = instantaneous_phase[-1]
        return current_phase

    def arm_angle(self, mode):
        if len(self.fx_buffer) < self.swing_buffer_size // 5:
            # print("Not enough data to get the current fx angle")
            return 0.0
        if "enhance" in mode:
            if self.external_guidance_mode == "systematic":
                # if self.external_force_cycle % self.external_force_period == 0:
                return self.get_current_fx_angle()
                # else:
                #     return 0.0
            elif self.external_guidance_mode == "random":
                if self.external_force_cycle in self.guidance_steps:
                    return self.get_current_fx_angle()
                else:
                    return 0.0
        elif "surpress" in mode:
            if self.external_guidance_mode == "systematic":
                # if self.external_force_cycle % self.external_force_period == 0:
                return self.get_current_fx_angle() + np.pi
                # else:
                #     return 0.0
            elif self.external_guidance_mode == "random":
                if self.external_force_cycle in self.supress_steps:
                    return self.get_current_fx_angle() + np.pi
                else:
                    return 0.0
        elif "free" in mode:
            return 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def z_pos_schedule(self, phase_shift):
        current_time = self.timer.elapsed()
        # print(f"current_time: {current_time}")
        min_z_pos, max_z_pos = self.z_pos_range

        # Sinusoidal oscillation between min_speed and max_speed
        phase = (
            (current_time - self.warmup_time) / self.z_pos_period * 2 * np.pi
        )  # + (phase_shift)
        # z_pos = min_z_pos + (max_z_pos - min_z_pos) * (0.5 * (1 - np.cos(phase)))

        z_pos = 0.5 * (max_z_pos - min_z_pos) * np.cos(phase) + min_z_pos
        # z_pos = z_pos - self.last_z_pos
        # self.last_z_pos = z_pos
        # return z_pos
        return z_pos, current_time - self.warmup_time

    def mode_schedule(self):
        # determine mode based on some metric
        for key, value in self.mode_sequence_steps.items():
            if self.follower_steps <= value:
                self.mode = key
                return key
        self.mode = self.mode_sequence[-1]
        return self.mode_sequence[-1]
        # for i in range(len(self.mode_partition)):
        #     if self.follower_steps <= self.mode_partition[i]:
        #         self.mode = self.modes[i]
        #         return self.modes[i]
        # self.mode = self.modes[-1]
        # return self.modes[-1]

    def z_pos_schedule_dynamic(self, mode):
        current_time = self.timer.elapsed()
        min_z_pos, max_z_pos = self.z_pos_range

        desired_angle = self.arm_angle(mode)
        z_pos = 0.5 * (max_z_pos - min_z_pos) * np.cos(desired_angle) + min_z_pos
        return z_pos, current_time - self.warmup_time

    def animate_update(self):
        self.current_time = time.time()
        if (
            self.current_time - self.last_visualization_time
            < self.visualization_update_interval
        ):
            return
        self.last_visualization_time = self.current_time
        self.fig.canvas.restore_region(self.bg)
        self.ee_pos1_curve.set_data(
            self.timestamp_buffer, self.ee_pos1_visualization_buffer
        )
        self.fx_curve.set_data(self.timestamp_buffer, self.fx_visualization_buffer)
        self.reference_pos_curve.set_data(
            self.timestamp_buffer, self.reference_pos_buffer
        )

        self.ax.draw_artist(self.ee_pos1_curve)
        self.ax.draw_artist(self.fx_curve)
        self.ax.draw_artist(self.reference_pos_curve)

        self.ax.relim()
        self.ax.autoscale_view(
            scalex=True,
        )

        self.fx_phase_curve.set_data(self.timestamp_buffer, self.fx_phase_buffer)
        self.ee_pos1_phase_curve.set_data(
            self.timestamp_buffer, self.ee_pos1_phase_buffer
        )
        self.ref_pos_phase_curve.set_data(
            self.timestamp_buffer, self.ref_pos_phase_buffer
        )
        self.ax2.draw_artist(self.fx_phase_curve)
        self.ax2.draw_artist(self.ee_pos1_phase_curve)
        self.ax2.draw_artist(self.ref_pos_phase_curve)
        self.ax2.relim()
        self.ax2.autoscale_view(
            scalex=True,
        )
        self.fx_freq_curve.set_data(self.timestamp_buffer, self.fx_freq_buffer)
        self.ee_pos1_freq_curve.set_data(
            self.timestamp_buffer, self.ee_pos1_freq_buffer
        )
        self.reference_pos_freq_curve.set_data(
            self.timestamp_buffer, self.reference_pos_freq_buffer
        )
        self.ax3.draw_artist(self.fx_freq_curve)
        self.ax3.draw_artist(self.ee_pos1_freq_curve)
        self.ax3.draw_artist(self.reference_pos_freq_curve)
        self.ax3.relim()
        self.ax3.autoscale_view(
            scalex=True,
        )
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        keyboard_inputs = self.keyboard.get_keyboard_input()
        self.stopped = keyboard_inputs["stop"]
        if self.paused and keyboard_inputs["resume"]:
            print("Resuming the system")
            self.paused = False
        if not self.paused and keyboard_inputs["pause"]:
            print("Pausing the system")
            self.paused = True
        self.keyboard.reset()

        action = self.default_motor_pos.copy()
        control_inputs = {}
        # print('lin vel', lin_vel, 'speed', self.speed, 'arm_vel', obs.arm_ee_vel)
        obs.ee_force = self.ee_force
        obs.ee_torque = self.ee_torque
        obs.arm_ee_pos = self.ee_pos
        obs.arm_ee_vel = self.ee_vel

        # print(f"fx_phase: {self.fx_phase}, n_buffer: {n_buffer}")
        is_done = self.is_done(obs)
        # self.z_pos, delta_t = self.z_pos_schedule(
        #     self.fx_phase, self.fx_amplitude
        # )  # + np.pi / 2)
        self.z_pos, delta_t = self.z_pos_schedule_dynamic(self.mode_schedule())
        # print(
        #     f"z_pos: {self.z_pos}, ee_force: {self.ee_force}, ee_torque: {self.ee_torque}, ee_pos: {self.ee_pos}, ee_vel: {self.ee_vel}"
        # )
        if self.follower_steps > 0 and self.close_loop:
            self.fx_buffer.append(self.ee_force[0])
            self.fy_buffer.append(self.ee_force[1])
            self.fz_buffer.append(self.ee_force[2])
            self.ee_pos1_buffer.append(self.ee_pos[1])
            # if self.visualize:
            self.timestamp_buffer.append(delta_t)
            self.ee_pos1_visualization_buffer.append(
                self.ee_pos[1] * self.ee_pos1_visualization_scale
                + self.ee_pos1_visualization_offset
            )
            self.fx_visualization_buffer.append(
                self.ee_force[0] * self.fx_visualization_scale
            )
            self.reference_pos_buffer.append(
                self.z_pos * self.rp_visualization_scale + self.rp_visualization_offset
            )

            n_buffer = len(self.fx_buffer)
            if n_buffer < self.swing_buffer_size / 4:
                self.fx_phase = 0
                self.ee_pos1_phase = 0
                self.reference_pos_phase = 0
                self.fx_freq = 0
                self.ee_pos1_freq = 0
                self.reference_pos_freq = 0
            else:
                fx_fft_vals = np.fft.rfft(self.fx_buffer)
                fx_fft_freqs = np.fft.rfftfreq(n_buffer, d=self.control_dt)
                fx_fft_phase = np.angle(fx_fft_vals)
                fx_freq_idx = (
                    np.argmax(np.abs(fx_fft_vals[1:])) + 1
                )  # Skip DC com ponent
                self.fx_phase = fx_fft_phase[fx_freq_idx]
                self.fx_freq = fx_fft_freqs[fx_freq_idx]

                ee_pos1_fft_vals = np.fft.rfft(self.ee_pos1_buffer)
                ee_pos1_fft_freqs = np.fft.rfftfreq(
                    len(self.ee_pos1_buffer), d=self.control_dt
                )
                ee_pos1_fft_phase = np.angle(ee_pos1_fft_vals)
                ee_pos1_freq_idx = np.argmax(np.abs(ee_pos1_fft_vals[1:])) + 1
                self.ee_pos1_phase = ee_pos1_fft_phase[ee_pos1_freq_idx]
                self.ee_pos1_freq = ee_pos1_fft_freqs[ee_pos1_freq_idx]

                reference_pos_fft_vals = np.fft.rfft(self.reference_pos_buffer)
                reference_pos_fft_freqs = np.fft.rfftfreq(
                    len(self.reference_pos_buffer), d=self.control_dt
                )
                reference_pos_fft_phase = np.angle(reference_pos_fft_vals)
                reference_pos_freq_idx = (
                    np.argmax(np.abs(reference_pos_fft_vals[1:])) + 1
                )
                self.reference_pos_phase = reference_pos_fft_phase[
                    reference_pos_freq_idx
                ]
                self.reference_pos_freq = reference_pos_fft_freqs[
                    reference_pos_freq_idx
                ]

            self.fx_phase_buffer.append(self.fx_phase)
            self.ee_pos1_phase_buffer.append(self.ee_pos1_phase)
            self.ref_pos_phase_buffer.append(self.reference_pos_phase)

            self.fx_freq_buffer.append(self.fx_freq)
            self.ee_pos1_freq_buffer.append(self.ee_pos1_freq)
            self.reference_pos_freq_buffer.append(self.reference_pos_freq)

            # print(f"reference_pos_freq: {self.reference_pos_freq}")

        # print(f"len(self.fx_buffer): {len(self.fx_buffer)}")
        # print(f"len(self.reference_pos_buffer): {len(self.reference_pos_buffer)}")
        if self.visualize:
            self.animate_update()
        if self.stopped:
            self.z_pos = 0.0
        # self.z_pos = 0.0
        self.arm_shm.buf[8:16] = struct.pack("d", self.z_pos)
        msg = ZMQMessage(
            time=time.time(),
            arm_force=self.ee_force,
            arm_torque=self.ee_torque,
            arm_ee_pos=obs.arm_ee_pos,
            arm_ee_vel=obs.arm_ee_vel,
            ee_pos_target=self.z_pos,
            is_stopped=is_done or self.stopped,
            is_paused=self.paused,
            external_guidance_stage=self.mode,
        )

        self.zmq_sender.send_msg(msg)
        msg_recv = self.zmq_receiver.get_msg()
        if msg_recv is not None:
            if msg_recv.total_steps > 0:
                self.follower_steps = msg_recv.total_steps
                self.external_force_cycle += 1
                print(
                    f"follower steps: {self.follower_steps}, mode: {self.mode}, self.external_force_cycle: {self.external_force_cycle}"
                )
            if msg_recv.is_stopped:
                self.timer.reset()
                self.external_force_cycle = 0
                if self.external_guidance_mode == "random":
                    if "enhance" in self.mode:
                        self.guidance_steps = np.random.choice(
                            self.guidance_list, size=self.guidance_num, replace=False
                        )
                        print("guidance steps:", self.guidance_steps)
                    elif "surpress" in self.mode:
                        self.supress_steps = np.random.choice(
                            self.supress_list, size=self.supress_num, replace=False
                        )
                        print("surpress steps:", self.supress_steps)

                print("Waiting for the follower to start...")
                while True:
                    msg_recv = self.zmq_receiver.get_msg()
                    if msg_recv is not None and not msg_recv.is_stopped:
                        break
                    time.sleep(0.1)
                print("Follower started")
                self.total_steps = 0
                self.timer.start()
            # self.x_pos_delta = msg.x_pos_delta

        time_elapsed = self.timer.elapsed()
        if time_elapsed < self.total_steps * self.control_dt:
            time.sleep(self.total_steps * self.control_dt - time_elapsed)
        self.total_steps += 1

        return control_inputs, action, obs

    def reset(self, obs: Obs = None) -> Obs:
        self.timer.reset()
        # control_inputs = {}
        # msg = ZMQMessage(
        #     time=time.time(),
        #     control_inputs=control_inputs,
        #     is_done=True
        # )
        # self.zmq_sender.send_msg(msg)
        print("Reset done")

        self.timer.start()
        return obs
