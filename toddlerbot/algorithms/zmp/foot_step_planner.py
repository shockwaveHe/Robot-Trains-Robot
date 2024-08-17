from dataclasses import dataclass
from typing import List, Tuple

from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


@dataclass
class FootStepPlanParameters:
    max_stride: ArrayType  # x, y, theta
    t_step: float
    foot_to_com_y: float


@dataclass
class FootStep:
    time: float
    position: ArrayType  # x, y, theta
    support_leg: str = ""


def cubic_hermite_spline(
    t: ArrayType, p0: ArrayType, p1: ArrayType, m0: ArrayType, m1: ArrayType
) -> ArrayType:
    """
    Computes the cubic Hermite spline at parameter t.

    Args:
        t (ArrayType): The parameter, should be between 0 and 1.
        p0 (ArrayType): The starting point (position).
        p1 (ArrayType): The ending point (position).
        m0 (ArrayType): The tangent at the starting point.
        m1 (ArrayType): The tangent at the ending point.

    Returns:
        ArrayType: The interpolated value at parameter t.
    """
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1  # type: ignore


def generate_hermite_spline(
    curr_pose: ArrayType, target_pose: ArrayType, path_resolution: float = 0.01
) -> Tuple[ArrayType, ArrayType]:
    x0, y0, theta0 = curr_pose
    xn, yn, thetan = target_pose

    # Calculate tangents
    path_distance = np.linalg.norm(np.array([xn - x0, yn - y0]))  # type: ignore
    t0_vec = np.array([np.cos(theta0), np.sin(theta0)]) * path_distance  # type: ignore
    tn_vec = np.array([np.cos(thetan), np.sin(thetan)]) * path_distance  # type: ignore

    # Generate parameter values t for interpolation
    t_values = np.linspace(0, 1, int(path_distance / path_resolution))  # type: ignore

    # Calculate spline for x and y components separately
    spline_x = cubic_hermite_spline(t_values, x0, xn, t0_vec[0], tn_vec[0])  # type: ignore
    spline_y = cubic_hermite_spline(t_values, y0, yn, t0_vec[1], tn_vec[1])  # type: ignore

    return spline_x, spline_y


class FootStepPlanner:
    def __init__(self, params: FootStepPlanParameters):
        self.params = params

    def compute_steps(
        self,
        curr_pose: ArrayType,
        target_pose: ArrayType,
    ) -> Tuple[ArrayType, List[FootStep]]:
        y_offset = self.params.foot_to_com_y
        foot_steps: List[FootStep] = []
        time = 0.0

        foot_steps.append(FootStep(time, curr_pose, "both"))
        time += self.params.t_step

        sampled_spline_x, sampled_spline_y = self.generate_and_sample_path(
            curr_pose, target_pose
        )

        # Combine x and y components into a single path
        path = np.vstack((sampled_spline_x, sampled_spline_y)).T  # type: ignore

        # Generate footsteps along the path
        dx = np.gradient(sampled_spline_x)  # type: ignore
        dy = np.gradient(sampled_spline_y)  # type: ignore

        # Normalize the tangent vectors to get the direction of the tangent at each point
        norms = np.sqrt(dx**2 + dy**2)  # type: ignore
        nx = -dy / norms  # type: ignore
        ny = dx / norms

        nx = inplace_update(nx, 0, -np.sin(curr_pose[2]))  # type: ignore
        ny = inplace_update(ny, 0, np.cos(curr_pose[2]))  # type: ignore
        nx = inplace_update(nx, -1, -np.sin(target_pose[2]))  # type: ignore
        ny = inplace_update(ny, -1, np.cos(target_pose[2]))  # type: ignore

        left_foot_x = sampled_spline_x + nx * y_offset
        left_foot_y = sampled_spline_y + ny * y_offset
        right_foot_x = sampled_spline_x - nx * y_offset
        right_foot_y = sampled_spline_y - ny * y_offset

        left_first_step_length = np.linalg.norm(  # type: ignore
            np.array([left_foot_x[1] - left_foot_x[0], left_foot_y[1] - left_foot_y[0]])  # type: ignore
        )
        right_first_step_length = np.linalg.norm(  # type: ignore
            np.array(  # type: ignore
                [right_foot_x[1] - right_foot_x[0], right_foot_y[1] - right_foot_y[0]]
            )
        )
        support_leg = (
            "left" if left_first_step_length > right_first_step_length else "right"
        )

        for i in range(len(left_foot_x)):
            if support_leg == "left":
                position = np.array(  # type: ignore
                    [left_foot_x[i], left_foot_y[i], np.arctan2(-nx[i], ny[i])]  # type: ignore
                )
            else:
                position = np.array(  # type: ignore
                    [right_foot_x[i], right_foot_y[i], np.arctan2(-nx[i], ny[i])]  # type: ignore
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

    def generate_and_sample_path(
        self,
        curr_pose: ArrayType,
        target_pose: ArrayType,
        high_res_stride: float = 1e-3,
    ) -> Tuple[ArrayType, ArrayType]:
        # Use the custom Hermite spline function
        sampled_spline_x, sampled_spline_y = generate_hermite_spline(
            curr_pose, target_pose, path_resolution=high_res_stride
        )

        dx = np.gradient(sampled_spline_x)  # type: ignore
        dy = np.gradient(sampled_spline_y)  # type: ignore
        spline_theta = np.arctan2(dy, dx)  # type: ignore

        # Rest of your sampling logic remains unchanged
        sampled_path_x = [curr_pose[0]]
        sampled_path_y = [curr_pose[1]]
        last_x, last_y, last_theta = curr_pose

        for i in range(1, len(sampled_spline_x)):
            l1_distance = np.abs(  # type: ignore
                np.array(  # type: ignore
                    [
                        sampled_spline_x[i] - last_x,
                        sampled_spline_y[i] - last_y,
                        spline_theta[i] - last_theta,
                    ]
                )
            )

            if (
                np.any(l1_distance >= self.params.max_stride)  # type: ignore
                or i == len(sampled_spline_x) - 1
            ):
                sampled_path_x.append(sampled_spline_x[i])
                sampled_path_y.append(sampled_spline_y[i])
                last_x, last_y, last_theta = (
                    sampled_spline_x[i],
                    sampled_spline_y[i],
                    spline_theta[i],
                )

        return np.array(sampled_path_x), np.array(sampled_path_y)  # type: ignore


if __name__ == "__main__":
    from toddlerbot.visualization.vis_plot import plot_footsteps

    planner_params = FootStepPlanParameters(
        max_stride=np.array(([0.05, 0.05, np.pi / 8])),  # type: ignore
        t_step=0.75,
        foot_to_com_y=0.04,
    )
    planner = FootStepPlanner(planner_params)

    target_pose = np.array([0.5, -0.5, np.pi / 2])  # type: ignore
    path, foot_steps = planner.compute_steps(
        curr_pose=np.array([0, 0, 0]),  # type: ignore
        target_pose=target_pose,
    )

    import numpy

    # You can plot the footsteps with your existing plotting utility here
    plot_footsteps(
        numpy.asarray(path, dtype=numpy.float32),
        numpy.array(
            [numpy.asarray(fs.position) for fs in foot_steps], dtype=numpy.float32
        ),
        [fs.support_leg for fs in foot_steps],
        (0.1, 0.05),
        planner_params.foot_to_com_y,
        fig_size=(8, 8),
        title=f"Footsteps Planning: {target_pose[0]:.2f} {target_pose[1]:.2f} {target_pose[2]:.2f}",
        x_label="Position X",
        y_label="Position Y",
        save_config=False,
        save_path=".",
        file_name="footsteps.png",
    )()
