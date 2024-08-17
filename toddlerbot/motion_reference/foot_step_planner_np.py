from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.interpolate import CubicHermiteSpline  # type: ignore

from toddlerbot.visualization.vis_plot import plot_footsteps


@dataclass
class FootStepPlanParameters:
    max_stride: np.ndarray  # x, y, theta
    t_step: float
    y_offset_com_to_foot: float


@dataclass
class FootStep:
    time: float
    position: np.ndarray  # x, y, theta
    support_leg: str = ""


class FootStepPlanner:
    def __init__(self, params: FootStepPlanParameters):
        self.params = params

    def compute_steps(
        self,
        curr_pose: np.ndarray,
        target_pose: np.ndarray,
    ) -> List[FootStep]:
        y_offset = self.params.y_offset_com_to_foot
        foot_steps = []
        time = 0.0

        foot_steps.append(FootStep(time, curr_pose, "both"))
        time += self.params.t_step

        sampled_spline_x, sampled_spline_y = self.generate_and_sample_hermite_path(
            curr_pose, target_pose
        )
        # Combine x and y components into a single path
        path = np.vstack((sampled_spline_x, sampled_spline_y)).T

        # Generate footsteps along the path
        # Ignore the first point and the last point
        dx = np.gradient(sampled_spline_x)
        dy = np.gradient(sampled_spline_y)

        # Normalize the tangent vectors to get the direction of the tangent at each point
        norms = np.sqrt(dx**2 + dy**2)
        nx = -dy / norms
        ny = dx / norms

        nx[0] = -np.sin(curr_pose[2])
        ny[0] = np.cos(curr_pose[2])
        nx[-1] = -np.sin(target_pose[2])
        ny[-1] = np.cos(target_pose[2])

        left_foot_x = sampled_spline_x + nx * y_offset
        left_foot_y = sampled_spline_y + ny * y_offset
        right_foot_x = sampled_spline_x - nx * y_offset
        right_foot_y = sampled_spline_y - ny * y_offset

        left_first_step_length = np.linalg.norm(
            [left_foot_x[1] - left_foot_x[0], left_foot_y[1] - left_foot_y[0]]
        )
        right_first_step_length = np.linalg.norm(
            [right_foot_x[1] - right_foot_x[0], right_foot_y[1] - right_foot_y[0]]
        )
        support_leg = (
            "left" if left_first_step_length > right_first_step_length else "right"
        )
        for i in range(len(left_foot_x)):
            if support_leg == "left":
                position = np.array(
                    [left_foot_x[i], left_foot_y[i], np.arctan2(-nx[i], ny[i])]
                )
            else:
                position = np.array(
                    [right_foot_x[i], right_foot_y[i], np.arctan2(-nx[i], ny[i])]
                )
            foot_steps.append(FootStep(time, position, support_leg))

            time += self.params.t_step
            support_leg = "left" if support_leg == "right" else "right"

        # Add the final step(s) with the foot together or stopped position
        foot_steps.append(FootStep(time, target_pose, "both"))
        # Add another step for the robot to stabilize
        time += self.params.t_step
        foot_steps.append(FootStep(time, target_pose, "both"))

        return path, foot_steps

    def generate_and_sample_hermite_path(
        self, curr_pose, target_pose, high_res_stride=1e-3
    ):
        x0, y0, theta0 = curr_pose
        xn, yn, thetan = target_pose

        # High-resolution Hermite spline interpolation
        path_distance = np.linalg.norm([xn - x0, yn - y0])
        num_high_res_points = int(path_distance / high_res_stride)
        t_high_res = np.linspace(0, 1, num_high_res_points)
        t0_vec = np.array([np.cos(theta0), np.sin(theta0)]) * path_distance
        tn_vec = np.array([np.cos(thetan), np.sin(thetan)]) * path_distance
        spline_x = CubicHermiteSpline([0, 1], [x0, xn], [t0_vec[0], tn_vec[0]])(
            t_high_res
        )
        spline_y = CubicHermiteSpline([0, 1], [y0, yn], [t0_vec[1], tn_vec[1]])(
            t_high_res
        )

        dx = np.gradient(spline_x)
        dy = np.gradient(spline_y)
        spline_theta = np.arctan2(dy, dx)

        sampled_path_x = [x0]
        sampled_path_y = [y0]
        last_x, last_y, last_theta = x0, y0, theta0
        for i in range(1, len(spline_x)):
            # Calculate distance from the last sampled point to the current point
            l1_distance = np.abs(
                [
                    spline_x[i] - last_x,
                    spline_y[i] - last_y,
                    spline_theta[i] - last_theta,
                ]
            )

            # If adding the next point would exceed the max_stride, or it's the last point, sample it
            if np.any(l1_distance >= self.params.max_stride) or i == len(spline_x) - 1:
                sampled_path_x.append(spline_x[i])
                sampled_path_y.append(spline_y[i])
                last_x, last_y, last_theta = spline_x[i], spline_y[i], spline_theta[i]

        return np.array(sampled_path_x), np.array(sampled_path_y)


if __name__ == "__main__":
    import random

    planner_params = FootStepPlanParameters(
        max_stride=np.array(([0.05, 0.05, np.pi / 8])),
        t_step=0.75,
        y_offset_com_to_foot=0.04,
    )
    planner = FootStepPlanner(planner_params)

    target_pose = np.array([0.5, -0.5, np.pi / 2])  # type: ignore
    path, foot_steps = planner.compute_steps(
        curr_pose=np.array([0, 0, 0]),
        target_pose=target_pose,
    )

    plot_footsteps(
        path,
        foot_steps,
        [0.1, 0.05],
        planner_params.y_offset_com_to_foot,
        fig_size=(8, 8),
        title=f"Footsteps Planning: {target_pose[0]:.2f} {target_pose[1]:.2f} {target_pose[2]:.2f}",
        x_label="Position X",
        y_label="Position Y",
        save_config=True,
        save_path="results/plots",
        file_suffix=f"{i}",
    )()
