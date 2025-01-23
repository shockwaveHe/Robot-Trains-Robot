import os
import time
import csv
import math
import matplotlib
matplotlib.use('Agg')  # for writing figures without an active X server
import matplotlib.pyplot as plt

class FinetuneLogger:
    def __init__(
        self,
        exp_folder: str,
        log_interval_steps: int = 5,
        plot_interval_steps: int = 20,
        update_csv: str = "training_updates.csv",
        reward_csv: str = "training_rewards.csv"
    ):
        """
        :param exp_folder: where to store CSV logs and plots
        :param log_interval_steps: how often (in env steps) to write reward CSV
        :param plot_interval_steps: how often (in env steps) to update reward plots
        :param update_csv_name: CSV file for update logs
        :param reward_csv_name: CSV file for reward logs
        """
        self.exp_folder = exp_folder
        assert os.path.exists(exp_folder), f"Experiment folder '{exp_folder}' does not exist."

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

    # ------------------------------------------------------------------
    # 1) PER-STEP REWARD LOGGING
    # ------------------------------------------------------------------
    def log_step(self, reward_dict: dict):
        """
        Called every environment step to record each reward term (e.g. torso_pos, torso_quat, etc.).
        reward_dict: { 'torso_pos': float, 'torso_quat': float, ... }
        """
        self.env_step_counter += 1

        # store each reward term in reward_term_histories
        for rname, rval in reward_dict.items():
            if rname not in self.reward_term_histories:
                self.reward_term_histories[rname] = []
            self.reward_term_histories[rname].append(rval)

        # if a reward term was previously recorded but not present now, we can append 0 or None
        for existing_rname in self.reward_term_histories.keys():
            if existing_rname not in reward_dict:
                self.reward_term_histories[existing_rname].append(0.0)

        # optionally write CSV line
        if self.env_step_counter % self.log_interval_steps == 0:
            self._write_reward_csv_line()

        # optionally update reward plots
        if self.env_step_counter % self.plot_interval_steps == 0:
            self.plot_rewards()

    def _write_reward_csv_line(self):
        """
        Appends one row to the 'reward_csv_path' with the current step's reward data.
        """
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

    def plot_rewards(self):
        """
        Creates a grid of subplots, one for each reward term.
        """
        import math
        reward_term_names = sorted(list(self.reward_term_histories.keys()))
        if len(reward_term_names) == 0:
            return

        ncols = 3
        nrows = math.ceil(len(reward_term_names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True)
        axes = axes.flatten()

        for idx, rname in enumerate(reward_term_names):
            ax = axes[idx]
            ax.plot(self.reward_term_histories[rname], label=rname)
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

    # ------------------------------------------------------------------
    # 2) PER-UPDATE LOGGING (for Q, V, Policy, OPE, etc.)
    # ------------------------------------------------------------------
    def log_update(self, **kwargs):
        """
        Called each time you do an offline update (BPPO, Q, V, OPE, etc.).
        We allow arbitrary keyword arguments, e.g. `policy_loss=..., q_loss=..., dynamics_loss=...`.
        A new subplot and CSV column is created automatically for each key.
        """
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
        
        if self.update_step_counter % self.plot_interval_steps == 0:
            self.plot_updates()

    def _flush_update_csv(self):
        """
        Writes all update metrics so far into a CSV.
        If new metric keys have appeared, we incorporate them automatically by scanning all data points.
        """
        if len(self.update_metrics_list) == 0:
            return

        # 1) discover all keys
        all_keys = set()
        for dp in self.update_metrics_list:
            all_keys.update(dp.keys())
        # we'd like a sorted, consistent ordering
        all_keys = sorted(list(all_keys))

        # 2) write CSV
        with open(self.update_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # header
            writer.writerow(all_keys)
            # each row
            for dp in self.update_metrics_list:
                row = []
                for k in all_keys:
                    row.append(dp.get(k, None))  # None if that key wasn't logged in this dp
                writer.writerow(row)

    def plot_updates(self):
        """
        Creates a grid of subplots for any metrics that have been logged via log_update().
        Each metric becomes its own subplot.
        """
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

        # convert each metric into a list for plotting

        ncols = 3
        nrows = math.ceil(len(metric_keys) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=False)
        axes = axes.flatten()

        for idx, mkey in enumerate(metric_keys):
            ax = axes[idx]
            # gather this metric's values for each data point, defaulting to None if missing
            yvals = []
            for dp in self.update_metrics_list:
                val = dp.get(mkey, None)
                if val is not None:
                    yvals.append(val)
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
