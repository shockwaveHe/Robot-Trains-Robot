# type: ignore
import collections

import cv2
import numpy as np
import numpy.typing as npt
from pynput import keyboard

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

default_pose = np.array(
    [
        -0.60745645,
        -0.9265244,
        0.02147579,
        1.2348546,
        0.52922344,
        0.49394178,
        -1.125942,
        0.5123496,
        -0.96180606,
        -0.25003886,
        1.2195148,
        -0.35128164,
        -0.6504078,
        -1.1535536,
    ]
)


class DPPolicy(BasePolicy, policy_name="dp"):
    def __init__(self, robot: Robot, model_path: str, dest: str):
        super().__init__(name="replay_fixed", robot=robot, init_motor_pos=default_pose)

        # self.default_action = np.array(
        #     list(robot.default_motor_angles.values()), dtype=np.float32
        # )

        self.model_path = model_path
        self.toggle_motor = False

        self.log = False
        self.stop_inference = True
        self.blend_percentage = 0.0
        self.default_pose = default_pose

        self.camera = Camera(camera_id=0)

        self.load_model()
        # deque for observation
        self.obs_deque = collections.deque([], maxlen=self.model.obs_horizon)
        self.model_action_seq = []

        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.stop_inference = not self.stop_inference
                    self.blend_percentage = 0.0
                    if self.stop_inference:
                        print("\nInference stopped, resetting to default pose\n")
                    else:
                        print("\nInference started, leader controlled by model now.\n")
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def load_model(self):
        from diffusion_policy_minimal.datasets.teleop_dataset import TeleopImageDataset
        from diffusion_policy_minimal.inference_class import DPModel

        model_path = "/home/weizhuo2/Documents/gits/diffusion_policy_minimal/checkpoints/teleop_model.pth"
        pred_horizon, obs_horizon, action_horizon = 16, 2, 8
        lowdim_obs_dim, action_dim = 14, 14

        # create dataset from file
        dataset_path = "/home/weizhuo2/Documents/gits/diffusion_policy_minimal/teleop_data/teleop_dataset.lz4"
        dataset = TeleopImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
        )
        # save training data statistics (min, max) for each dim
        stats = dataset.stats

        self.model = DPModel(
            model_path,
            stats,
            pred_horizon,
            obs_horizon,
            action_horizon,
            lowdim_obs_dim,
            action_dim,
        )

    def inference_step(self) -> npt.NDArray[np.float32]:
        model_action = self.model.get_action_from_obs(self.obs_deque)
        return list(model_action)

    def reset_slowly(self, obs_real):
        leader_action = (
            self.default_pose * self.blend_percentage
            + obs_real.motor_pos * (1 - self.blend_percentage)
        )
        self.blend_percentage += 0.002
        self.blend_percentage = min(1, self.blend_percentage)
        return leader_action

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        # manage obs_deque
        camera_frame = self.camera.get_state()
        camera_frame = cv2.resize(camera_frame, (171, 96))[:96, 38:134] / 255.0
        camera_frame = camera_frame.transpose(2, 0, 1)
        obs_entry = {"image": camera_frame, "agent_pos": obs_real.motor_pos}
        self.obs_deque.append(obs_entry)

        leader_action = obs_real.motor_pos
        if len(self.obs_deque) == self.model.obs_horizon:
            if len(self.model_action_seq) == 0:
                self.model_action_sequence = self.inference_step()
            sim_action = self.model_action_sequence.pop(0)
        else:
            sim_action = obs_real.motor_pos

        if self.stop_inference:
            leader_action = self.reset_slowly(obs_real)
            # if self.blend_percentage >= 0.99:
            # self.log = True
            # self.toggle_motor = True
        # else:
        #     leader_action = sim_action
        return sim_action, leader_action
