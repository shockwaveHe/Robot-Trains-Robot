import csv
import math
import os
import time
from collections import defaultdict
from queue import Queue
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from scipy.signal import lfilter, lfilter_zi
from sklearn.manifold import TSNE

from toddlerbot.sim import Obs

# for writing figures without an active X server


class FinetuneLogger:
    def __init__(
        self,
        exp_folder: str,
        log_interval_steps: int = 5,
        plot_interval_steps: int = 1024,
        update_csv: str = "training_updates.csv",
        reward_csv: str = "training_rewards.csv",
        enable_logging: bool = True,
        enable_profiling: bool = False,
        smooth_factor: float = 0.0,
        exp_type: str = "walk",  # "walk" or "swing"
    ):
        """
        :param exp_folder: where to store CSV logs and plots
        :param log_interval_steps: how often (in env steps) to write reward CSV
        :param plot_interval_steps: how often (in env steps) to update reward plots
        :param update_csv: CSV file for update logs
        :param reward_csv: CSV file for reward logs
        :param enable_logging: if False, all logging calls are no-ops
        :param enable_profiling: if True, we measure time spent in each logging function
        :param smooth_factor: EMA smoothing factor for plots (None or 0 disables smoothing)
        """
        self.exp_folder = exp_folder
        assert os.path.exists(exp_folder), (
            f"Experiment folder '{exp_folder}' does not exist."
        )

        self.enable_logging = enable_logging
        self.enable_profiling = enable_profiling
        self.smooth_factor = smooth_factor
        if exp_type == "walk":
            self.exp_type = "walk"
            self.latent_trajectory = {}
            with open("toddlerbot/finetuning/initial_latents.pt", "rb") as f:
                train_initial_latents = torch.load(f)

            with open("toddlerbot/finetuning/latent_z_release.pt", "rb") as f:
                initial_latents = torch.load(f)
                if type(initial_latents) == dict:
                    initial_latents = initial_latents["latent_z"]
            self.latent_trajectory["initial_latents"] = train_initial_latents.clone()
            self.latent_trajectory["optimized_latents"] = [initial_latents.clone()]
        elif exp_type == "swing":
            self.exp_type = "swing"
        else:
            raise ValueError(f"Unknown exp_type: {exp_type}. Use 'walk' or 'swing'.")
        # Timers and counts for profiling
        self.profiling_data = defaultdict(float)  # e.g. {"log_step": 0.034, ...}
        self.profiling_counts = defaultdict(int)  # e.g. {"log_step": 100, ...}

        # ====== Per-STEP (reward) logging ======
        self.env_step_counter = 0
        self.reward_term_histories = {}  # reward_term -> list of floats
        self.log_interval_steps = log_interval_steps
        self.plot_interval_steps = plot_interval_steps

        # CSV for environment-step reward logs
        self.reward_csv_path = os.path.join(self.exp_folder, reward_csv)
        self.reward_header_written = False

        # ====== Per-UPDATE logging (BPPO, Q, etc) ======
        self.update_step_counter = 0
        self.update_metrics_list = []  # list of dicts, each dict = { 'time':..., 'update_step':..., 'some_metric':... }
        self.update_csv_path = os.path.join(self.exp_folder, update_csv)

        # Plotting queue and thread
        self.plot_queue = Queue()
        self.plot_thread = Thread(target=self._plot_worker, daemon=True)
        self.plot_thread.start()

        self.start_time = time.time()

    def save_state(self, exp_folder: str):
        """Saves the current state of the logger to a pickle file."""
        state = {
            "env_step_counter": self.env_step_counter,
            "reward_term_histories": self.reward_term_histories,
            "update_step_counter": self.update_step_counter,
            "update_metrics_list": self.update_metrics_list,
            "profiling_data": self.profiling_data,
            "profiling_counts": self.profiling_counts,
        }
        np.savez_compressed(os.path.join(exp_folder, "logger.npz"), **state)
        print(f"Logger state saved to {exp_folder}")

    def load_state(self, exp_folder: str):
        """Loads the logger state from a pickle file."""
        logger_path = os.path.join(exp_folder, "logger.npz")
        state = np.load(logger_path, allow_pickle=True)

        self.env_step_counter = state["env_step_counter"]
        self.reward_term_histories = state["reward_term_histories"]
        self.update_step_counter = state["update_step_counter"]
        self.update_metrics_list = state["update_metrics_list"]
        self.profiling_data = state["profiling_data"]
        self.profiling_counts = state["profiling_counts"]

        print(f"Logger state loaded from {logger_path}")

    def _plot_worker(self):
        """Worker thread that handles plotting tasks to avoid blocking the main thread."""
        while True:
            task = self.plot_queue.get()
            if task is None:  # Sentinel to shut down the thread
                break
            func, args = task
            func(*args)
            self.plot_queue.task_done()

    def _ema(self, data, alpha):
        """
        Compute EMA using an IIR filter.
        y[0] = x[0]
        y[i] = (1 - smoothing_factor)*x[i] + smoothing_factor*y[i-1]
        """
        b = [1 - alpha]
        a = [1, -alpha]
        # Compute steady-state initial condition for a constant input equal to data[0]
        zi = lfilter_zi(b, a) * data[0]
        y, _ = lfilter(b, a, data, zi=zi)
        return y

    # ------------------------------------------------------------------
    # 1) PER-STEP REWARD LOGGING
    # ------------------------------------------------------------------
    def log_step(self, reward_dict: dict, obs: Obs, **kwargs):
        """
        Called every environment step to record each reward term.
        reward_dict: { 'torso_pos': float, 'torso_quat': float, ... }
        """
        if not self.enable_logging:
            return

        self.env_step_counter += 1
        log_dict = {f"rew_{key}": value for key, value in reward_dict.items()}
        log_dict["time"] = obs.time
        log_dict["lin_vel_x"] = obs.lin_vel[0]
        log_dict["lin_vel_y"] = obs.lin_vel[1]
        # log_dict["lin_vel_z"] = obs.lin_vel[2]
        log_dict["ang_vel_x"] = obs.ang_vel[0]
        log_dict["ang_vel_y"] = obs.ang_vel[1]
        log_dict["ang_vel_z"] = obs.ang_vel[2]
        log_dict["ee_force_x"] = obs.ee_force[0]
        log_dict["ee_force_y"] = obs.ee_force[1]
        log_dict["ee_force_z"] = obs.ee_force[2]
        log_dict["ee_pos_x"] = obs.arm_ee_pos[0]
        log_dict["ee_pos_y"] = obs.arm_ee_pos[1]
        log_dict["ee_pos_z"] = obs.arm_ee_pos[2]
        log_dict["torso_roll"] = obs.euler[0]
        log_dict["torso_pitch"] = obs.euler[1]
        log_dict["torso_yaw"] = obs.euler[2]
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                log_dict[key] = value
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    log_dict[key] = value.item()
                else:
                    assert value.size == 3, (
                        f"Expected 3-element array for {key}, got {value}"
                    )
                    log_dict[f"{key}_x"] = value[0]
                    log_dict[f"{key}_y"] = value[1]
                    log_dict[f"{key}_z"] = value[2]
        # store each reward term in reward_term_histories
        for rname, rval in log_dict.items():
            if rname not in self.reward_term_histories:
                self.reward_term_histories[rname] = []
            self.reward_term_histories[rname].append(rval)

        # if a reward term was previously recorded but not present now, we can append 0 or None
        for existing_rname in self.reward_term_histories.keys():
            if existing_rname not in log_dict:
                self.reward_term_histories[existing_rname].append(0.0)

        # optionally write CSV line
        if self.env_step_counter % self.log_interval_steps == 0:
            self._write_reward_csv_line()

        # optionally update reward plots
        # if self.env_step_counter % self.plot_interval_steps == 0:
        #     self.plot_queue.put((self.plot_updates, []))

    def set_exp_folder(self, exp_folder: str):
        """Sets the experiment folder for saving logs and plots."""
        self.exp_folder = exp_folder
        self.reward_csv_path = os.path.join(exp_folder, "training_rewards.csv")
        self.update_csv_path = os.path.join(exp_folder, "training_updates.csv")
        self.reward_header_written = False

    def reset(self):
        """Clears all reward histories and resets counters."""
        self.env_step_counter = 0
        self.reward_term_histories = {}
        self.reward_header_written = False
        self.update_step_counter = 0
        self.update_metrics_list = []
        self.profiling_data = defaultdict(float)
        self.profiling_counts = defaultdict(int)
        self.start_time = time.time()

    def _write_reward_csv_line(self):
        """
        Appends one row to the 'reward_csv_path' with the current step's reward data.
        """
        if not self.enable_logging:
            return

        # Build the columns
        column_names = ["env_step"]
        column_values = [self.env_step_counter]

        # for consistent ordering
        sorted_rnames = sorted(self.reward_term_histories.keys())
        for rname in sorted_rnames:
            column_names.append(rname)
            column_values.append(self.reward_term_histories[rname][-1])

        write_header_now = False
        if not self.reward_header_written:
            write_header_now = True
            self.reward_header_written = True

        with open(self.reward_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header_now:
                writer.writerow(column_names)
            writer.writerow(column_values)

    def plot_rewards(self):
        """
        Creates a grid of subplots, one for each reward term.
        """
        if not self.enable_logging:
            return

        reward_term_names = sorted(list(self.reward_term_histories.keys()))
        if len(reward_term_names) == 0:
            return

        plt.switch_backend("Agg")
        ncols = 3
        nrows = math.ceil(len(reward_term_names) / ncols)
        # import ipdb; ipdb.set_trace()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True
        )
        axes = axes.flatten()

        for idx, rname in enumerate(reward_term_names):
            ax = axes[idx]
            data = self.reward_term_histories[rname]
            if self.smooth_factor:
                data = self._ema(data, self.smooth_factor)
            ax.plot(data, label=rname)
            ax.set_title(rname)
            ax.set_xlabel("Env Steps")
            ax.set_ylabel("Reward Value")
            ax.legend()

        for i in range(len(reward_term_names), len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        path = os.path.join(self.exp_folder, "rewards_grid.png")
        plt.savefig(path)
        plt.close(fig)

        print(f"Saved reward plot to {path}")

        # Create a stacked area plot of reward proportions
        # First, gather the reward-term data (and apply smoothing if needed)
        term_data = {}
        for term in self.reward_term_histories:
            if term.startswith("rew_"):
                data = np.array(self.reward_term_histories[term])
                if self.smooth_factor:
                    data = self._ema(data, 0.995)
                term_data[term] = data

        # Assume that all reward-term arrays have the same length.
        T = len(next(iter(term_data.values())))
        steps = np.arange(T)

        # Compute the total reward at each time step.
        total = 1e-8
        try:
            for key, data in term_data.items():
                total += data.mean()
        except ValueError as _:
            import traceback

            traceback.print_exc()
            print(key, data.shape)
        # Avoid division by zero (if total is 0 at any step)

        # Compute the proportion of each reward term at each time step.
        term_props = {}
        for term, data in term_data.items():
            term_props[term] = data.mean() / total

        # Sort the reward terms by their overall (e.g. average) proportion,
        # so that the term with the highest average is at the bottom of the stack.
        avg_props = {term: np.mean(props) for term, props in term_props.items()}
        sorted_terms = sorted(avg_props, key=avg_props.get, reverse=True)

        # Prepare the data in sorted order.
        props_len = min([len(term_data[term]) for term in sorted_terms])
        props_sorted = [term_data[term][:props_len] for term in sorted_terms]
        # Create the stacked area plot.
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Generate distinct colors for each term (using, e.g., tab10)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_terms)))
        ax2.stackplot(
            steps,
            *props_sorted,
            labels=[term[4:] for term in sorted_terms],
            colors=colors,
        )
        ax2.set_title("Stacked Reward Proportions Over Time")
        ax2.set_xlabel("Env Steps")
        ax2.set_ylabel("Proportion of Total Reward")
        ax2.legend(loc="upper left")

        path2 = os.path.join(self.exp_folder, "rewards_stack.png")
        plt.savefig(path2)
        plt.close(fig2)
        print(f"Saved stacked reward proportion plot to {path2}")

    # ------------------------------------------------------------------
    # 2) PER-UPDATE LOGGING (for Q, V, Policy, OPE, etc.)
    # ------------------------------------------------------------------
    def log_update(self, latent_z=None, **kwargs):
        """
        Called each time you do an offline update (BPPO, Q, V, OPE, etc.).
        We allow arbitrary keyword arguments.
        """
        if not self.enable_logging:
            return

        self.update_step_counter += 1
        data_point = {"time": time.time(), "update_step": self.update_step_counter}
        # store all user-provided metrics
        for key, val in kwargs.items():
            if val is None:
                continue

            data_point[key] = val

        # add to our list of updates
        self.update_metrics_list.append(data_point)

        if latent_z is not None:
            self.latent_trajectory["optimized_latents"].append(latent_z.clone())

        # if you want immediate CSV writing, you can do it here:
        # if self.update_step_counter % self.log_interval_steps == 0:
        #     self._flush_update_csv()

        # if self.update_step_counter % self.plot_interval_steps == 0:
        #     self.plot_queue.put((self.plot_updates, []))

    def _flush_update_csv(self):
        """
        Writes all update metrics so far into a CSV.
        If new metric keys have appeared, we incorporate them automatically by scanning all data points.
        """
        if not self.enable_logging or len(self.update_metrics_list) == 0:
            return

        # 1) discover all keys
        all_keys = set()
        for dp in self.update_metrics_list:
            all_keys.update(dp.keys())
        # we'd like a sorted, consistent ordering
        all_keys = sorted(list(all_keys))

        # Ensure all keys are present in each row, filling missing values with None
        rows = []
        for dp in self.update_metrics_list:
            row = {key: None for key in all_keys}
            row.update(dp)
            rows.append(row)

        # 2) write CSV
        with open(self.update_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

    def plot_updates(self):
        """
        Creates a grid of subplots for any metrics that have been logged via log_update().
        Each metric becomes its own subplot.
        """
        if self.exp_type == "walk":
            plt.switch_backend("Agg")
        if not self.enable_logging:
            return

        if len(self.update_metrics_list) == 0:
            return

        # gather all keys except 'time' and 'update_step'
        all_keys = set()
        for dp in self.update_metrics_list:
            all_keys.update(dp.keys())
        all_keys.discard("time")
        all_keys.discard("update_step")
        metric_keys = sorted(list(all_keys))
        if len(metric_keys) == 0:
            return

        ncols = 3
        nrows = math.ceil(len(metric_keys) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=False
        )
        axes = axes.flatten()

        for idx, mkey in enumerate(metric_keys):
            ax = axes[idx]
            # gather this metric's values for each data point
            yvals = []
            for dp in self.update_metrics_list:
                val = dp.get(mkey, None)
                if val is not None:
                    yvals.append(val)
            if self.smooth_factor and len(yvals):
                yvals = self._ema(yvals, self.smooth_factor)
            ax.plot(yvals, label=mkey)
            ax.set_title(mkey)
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Value")
            ax.legend()

        for i in range(len(metric_keys), len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        path = os.path.join(self.exp_folder, "updates_grid.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"Saved update plot to {path}")
        if self.exp_type == "walk":
            if len(self.latent_trajectory["optimized_latents"]) > 1:
                self.visualize_latent_dynamics(self.exp_folder)
                self.visualize_latent_dynamics(self.exp_folder, dim=3)

    def close(self):
        """Shut down the plotting thread."""
        self.plot_rewards()
        self.plot_updates()
        self._flush_update_csv()
        self._write_reward_csv_line()
        self.plot_queue.put(None)  # Sentinel to shut down the thread
        self.plot_thread.join()

    def visualize_latent_dynamics(self, log_dir, dim=2):
        """
        Visualize latent dynamics using t-SNE

        Args:
            initial_latents: torch.Tensor of shape (n_samples, latent_dim) - initial latent points
            latent_trajectory: torch.Tensor of shape (n_steps, latent_dim) - optimization trajectory
            output_path: str - path to save the visualization
            dim: int - 2 or 3 for 2D or 3D visualization
        """
        initial_latents = self.latent_trajectory["initial_latents"]
        latent_trajectory = self.latent_trajectory["optimized_latents"]
        if len(latent_trajectory) == 0:
            print("No latent trajectory to visualize.")
            return
        # Convert to numpy if they're torch tensors
        initial_latents = initial_latents.cpu().numpy()
        latent_trajectory = torch.concatenate(latent_trajectory).cpu().numpy()

        # Combine all points for t-SNE
        all_points = np.vstack([initial_latents, latent_trajectory])

        # Apply t-SNE
        tsne = TSNE(n_components=dim, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(all_points)

        # Split back into initial points and trajectory
        n_initial = initial_latents.shape[0]
        initial_embedded = embedded[:n_initial]
        trajectory_embedded = embedded[n_initial:]

        # Create plot
        fig = plt.figure(figsize=(8, 6))

        if dim == 2:
            # Plot initial points
            plt.scatter(
                initial_embedded[:, 0],
                initial_embedded[:, 1],
                color="blue",
                alpha=0.5,
                label="Initial Latents",
            )

            # Plot trajectory with color progression
            points = trajectory_embedded.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from time step to colors
            norm = plt.Normalize(0, len(trajectory_embedded))
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(np.arange(len(trajectory_embedded)))
            lc.set_linewidth(2)
            line = plt.gca().add_collection(lc)

            # Add colorbar
            plt.colorbar(line, label="Optimization Step")

            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")

        elif dim == 3:
            ax = plt.axes(projection="3d")

            # Plot initial points
            ax.scatter3D(
                initial_embedded[:, 0],
                initial_embedded[:, 1],
                initial_embedded[:, 2],
                color="blue",
                alpha=0.5,
                label="Initial Latents",
            )

            # Plot trajectory with color progression
            sc = ax.scatter3D(
                trajectory_embedded[:, 0],
                trajectory_embedded[:, 1],
                trajectory_embedded[:, 2],
                c=np.arange(len(trajectory_embedded)),
                cmap="viridis",
                label="Optimization Trajectory",
            )

            # Add colorbar
            plt.colorbar(sc, label="Optimization Step")

            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")

        plt.title("Latent Space Dynamics Visualization")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(log_dir, f"latent_dynamics_{dim}d.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Visualization saved to {output_path}")
