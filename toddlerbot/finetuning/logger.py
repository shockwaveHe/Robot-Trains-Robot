import os
import time
import csv
import math
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for writing figures without an active X server
import matplotlib.pyplot as plt
from collections import defaultdict
from threading import Thread
from queue import Queue
from toddlerbot.sim import Obs

class FinetuneLogger:
    def __init__(
        self,
        exp_folder: str,
        log_interval_steps: int = 20,
        plot_interval_steps: int = 5000,
        update_csv: str = "training_updates.csv",
        reward_csv: str = "training_rewards.csv",
        enable_logging: bool = True,
        enable_profiling: bool = False,
        smooth_factor: float = 0.0
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
        assert os.path.exists(exp_folder), f"Experiment folder '{exp_folder}' does not exist."

        self.enable_logging = enable_logging
        self.enable_profiling = enable_profiling
        self.smooth_factor = smooth_factor

        # Timers and counts for profiling
        self.profiling_data = defaultdict(float)   # e.g. {"log_step": 0.034, ...}
        self.profiling_counts = defaultdict(int)   # e.g. {"log_step": 100, ...}

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


    def save_state(self, filepath: str):
        """Saves the current state of the logger to a pickle file."""
        state = {
            "env_step_counter": self.env_step_counter,
            "reward_term_histories": self.reward_term_histories,
            "update_step_counter": self.update_step_counter,
            "update_metrics_list": self.update_metrics_list,
            "profiling_data": self.profiling_data,
            "profiling_counts": self.profiling_counts,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Logger state saved to {filepath}")

    def load_state(self, filepath: str):
        """Loads the logger state from a pickle file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        self.env_step_counter = state["env_step_counter"]
        self.reward_term_histories = state["reward_term_histories"]
        self.update_step_counter = state["update_step_counter"]
        self.update_metrics_list = state["update_metrics_list"]
        self.profiling_data = state["profiling_data"]
        self.profiling_counts = state["profiling_counts"]
        
        print(f"Logger state loaded from {filepath}")


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
        """Apply exponential moving average smoothing to a list of data using a convolution-based approach."""
        if alpha is None or alpha <= 0 or len(data) < 2:
            return data
        kernel = np.array([(1 - alpha) ** i for i in range(len(data))])
        kernel = kernel / kernel.sum()  # Normalize the kernel
        smoothed = np.convolve(data, kernel, mode='full')[:len(data)]
        return smoothed

    # -------------
    # Profiling helpers
    # -------------
    def _start_profile(self, name: str):
        """Call at the start of a function to track time, if profiling is on."""
        if self.enable_profiling:
            self.profiling_counts[name] += 1
            return time.perf_counter()
        return None

    def _end_profile(self, name: str, start_time: float):
        """Call at the end of a function to track time, if profiling is on."""
        if self.enable_profiling and start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.profiling_data[name] += elapsed

    def print_profiling_data(self):
        """Print out total, call count, and average time for each profiled method."""
        if not self.enable_profiling:
            print("Profiling is disabled.")
            return
        print("===== Profiling Results =====")
        for func_name, total_time in self.profiling_data.items():
            count = self.profiling_counts[func_name]
            avg_time = total_time / count if count != 0 else 0
            print(f"{func_name}: total={total_time:.4f}s, calls={count}, avg={avg_time:.6f}s")
        print("==============================")

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

        start_t = self._start_profile("log_step")

        self.env_step_counter += 1
        log_dict = {f"rew_{key}": value for key, value in reward_dict.items()}
        log_dict["time"] = obs.time
        log_dict["lin_vel_x"] = obs.lin_vel[0]
        log_dict["lin_vel_y"] = obs.lin_vel[1]
        log_dict["lin_vel_z"] = obs.lin_vel[2]
        log_dict["ang_vel_x"] = obs.ang_vel[0]
        log_dict["ang_vel_y"] = obs.ang_vel[1]
        log_dict["ang_vel_z"] = obs.ang_vel[2]
        log_dict["ee_force_x"] = obs.ee_force[0]
        log_dict["ee_force_y"] = obs.ee_force[1]
        log_dict["ee_force_z"] = obs.ee_force[2]
        log_dict["ee_pos_z"] = obs.arm_ee_pos[2]
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                log_dict[key] = value
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    log_dict[key] = value.item()
                else:
                    assert value.size == 3, f"Expected 3-element array for {key}, got {value}"
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
        #     self.plot_queue.put((self.plot_rewards, []))

        self._end_profile("log_step", start_t)

    def _write_reward_csv_line(self):
        """
        Appends one row to the 'reward_csv_path' with the current step's reward data.
        """
        if not self.enable_logging:
            return

        start_t = self._start_profile("_write_reward_csv_line")

        # Build the columns
        column_names = ["time", "env_step"]
        column_values = [time.time(), self.env_step_counter]

        # for consistent ordering
        sorted_rnames = sorted(self.reward_term_histories.keys())
        for rname in sorted_rnames:
            column_names.append(rname)
            column_values.append(self.reward_term_histories[rname][-1])

        write_header_now = False
        if not self.reward_header_written:
            write_header_now = True
            self.reward_header_written = True

        with open(self.reward_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header_now:
                writer.writerow(column_names)
            writer.writerow(column_values)

        self._end_profile("_write_reward_csv_line", start_t)

    def plot_rewards(self):
        """
        Creates a grid of subplots, one for each reward term.
        """
        if not self.enable_logging:
            return

        start_t = self._start_profile("plot_rewards")

        reward_term_names = sorted(list(self.reward_term_histories.keys()))
        if len(reward_term_names) == 0:
            self._end_profile("plot_rewards", start_t)
            return

        ncols = 3
        nrows = math.ceil(len(reward_term_names) / ncols)
        # import ipdb; ipdb.set_trace()
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True)
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
        self._end_profile("plot_rewards", start_t)

    # ------------------------------------------------------------------
    # 2) PER-UPDATE LOGGING (for Q, V, Policy, OPE, etc.)
    # ------------------------------------------------------------------
    def log_update(self, **kwargs):
        """
        Called each time you do an offline update (BPPO, Q, V, OPE, etc.).
        We allow arbitrary keyword arguments.
        """
        if not self.enable_logging:
            return

        start_t = self._start_profile("log_update")

        self.update_step_counter += 1
        data_point = {
            "time": time.time(),
            "update_step": self.update_step_counter
        }
        # store all user-provided metrics
        for key, val in kwargs.items():
            data_point[key] = val

        # add to our list of updates
        self.update_metrics_list.append(data_point)

        # if you want immediate CSV writing, you can do it here:
        if self.update_step_counter % self.log_interval_steps == 0:
            self._flush_update_csv()
        
        # if self.update_step_counter % self.plot_interval_steps == 0:
        #     self.plot_queue.put((self.plot_updates, []))

        self._end_profile("log_update", start_t)

    def _flush_update_csv(self):
        """
        Writes all update metrics so far into a CSV.
        If new metric keys have appeared, we incorporate them automatically by scanning all data points.
        """
        if not self.enable_logging:
            return

        start_t = self._start_profile("_flush_update_csv")

        if len(self.update_metrics_list) == 0:
            self._end_profile("_flush_update_csv", start_t)
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
        with open(self.update_csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

        self._end_profile("_flush_update_csv", start_t)


    def plot_updates(self):
        """
        Creates a grid of subplots for any metrics that have been logged via log_update().
        Each metric becomes its own subplot.
        """
        if not self.enable_logging:
            return

        start_t = self._start_profile("plot_updates")

        if len(self.update_metrics_list) == 0:
            self._end_profile("plot_updates", start_t)
            return

        # gather all keys except 'time' and 'update_step'
        all_keys = set()
        for dp in self.update_metrics_list:
            all_keys.update(dp.keys())
        all_keys.discard("time")
        all_keys.discard("update_step")
        metric_keys = sorted(list(all_keys))
        if len(metric_keys) == 0:
            self._end_profile("plot_updates", start_t)
            return

        ncols = 3
        nrows = math.ceil(len(metric_keys) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=False)
        axes = axes.flatten()

        for idx, mkey in enumerate(metric_keys):
            ax = axes[idx]
            # gather this metric's values for each data point
            yvals = []
            for dp in self.update_metrics_list:
                val = dp.get(mkey, None)
                if val is not None:
                    yvals.append(val)
            if self.smooth_factor:
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
        self._end_profile("plot_updates", start_t)

    def close(self):
        """Shut down the plotting thread."""
        self.plot_rewards()
        self.plot_updates()
        self._flush_update_csv()
        self._write_reward_csv_line()
        self.plot_queue.put(None)  # Sentinel to shut down the thread
        self.plot_thread.join()
