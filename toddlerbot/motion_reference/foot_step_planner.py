from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp

from toddlerbot.visualization.vis_plot import plot_footsteps


@dataclass
class FootStepPlanParameters:
    max_stride: jax.Array  # x, y, theta
    t_step: float
    y_offset_com_to_foot: float


@dataclass
class FootStep:
    time: float
    position: jax.Array  # x, y, theta
    support_leg: str = ""


def cubic_hermite_spline(
    t: jax.Array, p0: jax.Array, p1: jax.Array, m0: jax.Array, m1: jax.Array
) -> jax.Array:
    """
    Computes the cubic Hermite spline at parameter t.

    Args:
        t (jax.Array): The parameter, should be between 0 and 1.
        p0 (jax.Array): The starting point (position).
        p1 (jax.Array): The ending point (position).
        m0 (jax.Array): The tangent at the starting point.
        m1 (jax.Array): The tangent at the ending point.

    Returns:
        jax.Array: The interpolated value at parameter t.
    """
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def generate_hermite_spline(
    curr_pose: jax.Array, target_pose: jax.Array, path_resolution: float = 0.01
) -> Tuple[jax.Array, jax.Array]:
    x0, y0, theta0 = curr_pose
    xn, yn, thetan = target_pose

    # Calculate tangents
    path_distance = jnp.linalg.norm(jnp.array([xn - x0, yn - y0]))  # type: ignore
    t0_vec = jnp.array([jnp.cos(theta0), jnp.sin(theta0)]) * path_distance  # type: ignore
    tn_vec = jnp.array([jnp.cos(thetan), jnp.sin(thetan)]) * path_distance  # type: ignore

    # Generate parameter values t for interpolation
    t_values = jnp.linspace(0, 1, int(path_distance / path_resolution))  # type: ignore

    # Calculate spline for x and y components separately
    spline_x = cubic_hermite_spline(t_values, x0, xn, t0_vec[0], tn_vec[0])  # type: ignore
    spline_y = cubic_hermite_spline(t_values, y0, yn, t0_vec[1], tn_vec[1])  # type: ignore

    return spline_x, spline_y


class FootStepPlanner:
    def __init__(self, params: FootStepPlanParameters):
        self.params = params

    def compute_steps(
        self,
        curr_pose: jax.Array,
        target_pose: jax.Array,
    ) -> Tuple[jax.Array, List[FootStep]]:
        y_offset = self.params.y_offset_com_to_foot
        foot_steps: List[FootStep] = []
        time = 0.0

        foot_steps.append(FootStep(time, curr_pose, "both"))
        time += self.params.t_step

        sampled_spline_x, sampled_spline_y = self.generate_and_sample_path(
            curr_pose, target_pose
        )
        # Combine x and y components into a single path
        path = jnp.vstack((sampled_spline_x, sampled_spline_y)).T  # type: ignore

        # Generate footsteps along the path
        # Ignore the first point and the last point
        dx = jnp.gradient(sampled_spline_x)  # type: ignore
        dy = jnp.gradient(sampled_spline_y)  # type: ignore

        # Normalize the tangent vectors to get the direction of the tangent at each point
        norms = jnp.sqrt(dx**2 + dy**2)  # type: ignore
        nx = -dy / norms  # type: ignore
        ny = dx / norms  # type: ignore

        nx = nx.at[0].set(-jnp.sin(curr_pose[2]))  # type: ignore
        ny = ny.at[0].set(jnp.cos(curr_pose[2]))  # type: ignore
        nx = nx.at[-1].set(-jnp.sin(target_pose[2]))  # type: ignore
        ny = ny.at[-1].set(jnp.cos(target_pose[2]))  # type: ignore

        left_foot_x = sampled_spline_x + nx * y_offset  # type: ignore
        left_foot_y = sampled_spline_y + ny * y_offset
        right_foot_x = sampled_spline_x - nx * y_offset  # type: ignore
        right_foot_y = sampled_spline_y - ny * y_offset

        left_first_step_length = jnp.linalg.norm(  # type: ignore
            jnp.array(  # type: ignore
                [left_foot_x[1] - left_foot_x[0], left_foot_y[1] - left_foot_y[0]]
            )
        )
        right_first_step_length = jnp.linalg.norm(  # type: ignore
            jnp.array(  # type: ignore
                [right_foot_x[1] - right_foot_x[0], right_foot_y[1] - right_foot_y[0]]
            )
        )
        support_leg = (
            "left" if left_first_step_length > right_first_step_length else "right"
        )
        for i in range(len(left_foot_x)):  # type: ignore
            if support_leg == "left":
                position = jnp.array(  # type: ignore
                    [left_foot_x[i], left_foot_y[i], jnp.arctan2(-nx[i], ny[i])]  # type: ignore
                )
            else:
                position = jnp.array(  # type: ignore
                    [right_foot_x[i], right_foot_y[i], jnp.arctan2(-nx[i], ny[i])]  # type: ignore
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
        curr_pose: jax.Array,
        target_pose: jax.Array,
        high_res_stride: float = 1e-3,
    ) -> Tuple[jax.Array, jax.Array]:
        # Use the custom Hermite spline function
        sampled_spline_x, sampled_spline_y = generate_hermite_spline(
            curr_pose, target_pose, path_resolution=high_res_stride
        )

        dx = jnp.gradient(sampled_spline_x)  # type: ignore
        dy = jnp.gradient(sampled_spline_y)  # type: ignore
        spline_theta = jnp.arctan2(dy, dx)  # type: ignore

        # Rest of your sampling logic remains unchanged
        sampled_path_x = [curr_pose[0]]
        sampled_path_y = [curr_pose[1]]
        last_x, last_y, last_theta = curr_pose

        for i in range(1, len(sampled_spline_x)):
            l1_distance = jnp.abs(  # type: ignore
                jnp.array(  # type: ignore
                    [
                        sampled_spline_x[i] - last_x,
                        sampled_spline_y[i] - last_y,
                        spline_theta[i] - last_theta,
                    ]
                )
            )

            if (
                jnp.any(l1_distance >= self.params.max_stride)  # type: ignore
                or i == len(sampled_spline_x) - 1
            ):
                sampled_path_x.append(sampled_spline_x[i])
                sampled_path_y.append(sampled_spline_y[i])
                last_x, last_y, last_theta = (
                    sampled_spline_x[i],
                    sampled_spline_y[i],
                    spline_theta[i],
                )

        return jnp.array(sampled_path_x), jnp.array(sampled_path_y)  # type: ignore


if __name__ == "__main__":
    planner_params = FootStepPlanParameters(
        max_stride=jnp.array(([0.05, 0.05, jnp.pi / 8])),  # type: ignore
        t_step=0.75,
        y_offset_com_to_foot=0.04,
    )
    planner = FootStepPlanner(planner_params)

    target_pose = jnp.array([0.5, -0.5, jnp.pi / 2])  # type: ignore
    path, foot_steps = planner.compute_steps(
        curr_pose=jnp.array([0, 0, 0]),  # type: ignore
        target_pose=target_pose,
    )

    # You can plot the footsteps with your existing plotting utility here
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
    )()
