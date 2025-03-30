#!/usr/bin/env python3
import argparse
import collections
import csv
import io
import math
import os
import subprocess
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class CSVMonitor:
    def __init__(
        self,
        file_name,
        task="walk_finetune",
        user="arya",
        host="10.5.6.248",
        window=100,
        ema=0.2,
        ncols=4,
        interval=100,
    ):
        """
        Initialize the CSV monitor.
        """
        # Use TkAgg to allow key press events.
        import matplotlib

        matplotlib.use("TkAgg")

        self.file_name = file_name
        self.user = user
        self.host = host
        self.task = task
        self.remote_file = None
        self.window = window
        self.ema_alpha = ema
        self.ncols = ncols
        self.interval = interval

        self.header = None
        self.num_columns = None
        self.ema_data = None
        self.data_lock = threading.Lock()

        self.tail_proc = None
        self.reader_thread = None

        self.fig = None
        self.axes = None
        self.lines = None
        self.anim = None

    def find_latest_results_folder(self):
        """
        Find the latest results folder on the remote machine.
        (This command lists directories matching a pattern and returns the newest one.)
        """
        find_cmd = (
            f"ssh {self.user}@{self.host} "
            f'"ls -td ~/projects/toddlerbot_internal/results/toddlerbot_2xm_{self.task}_real_world_* 2>/dev/null | head -n 1"'
        )
        try:
            latest_folder = subprocess.check_output(
                find_cmd, shell=True, universal_newlines=True
            ).strip()
            if latest_folder:
                print("Latest results folder:", latest_folder)
                return latest_folder
        except subprocess.CalledProcessError:
            pass
        return None

    def fetch_header(self):
        """
        Fetch the header line from the remote CSV file.
        This method blocks until a valid file and header are found.
        """
        print("Searching for the latest results folder...")
        while True:
            remote_folder = self.find_latest_results_folder()
            if remote_folder:
                self.remote_file = os.path.join(remote_folder, f"{self.file_name}.csv")
                break
            print("No results folder found yet, waiting...")
            time.sleep(2)

        head_cmd = [
            "ssh",
            f"{self.user}@{self.host}",
            "head",
            "-n",
            "1",
            self.remote_file,
        ]
        print("Waiting for CSV file and headers to be available...")
        while True:
            try:
                header_output = subprocess.check_output(
                    head_cmd, universal_newlines=True
                )
                header_line = header_output.strip()
                if header_line:
                    header_reader = csv.reader(io.StringIO(header_line))
                    self.header = next(header_reader)
                    self.num_columns = len(self.header)
                    self.ema_data = [
                        collections.deque(maxlen=self.window)
                        for _ in range(self.num_columns)
                    ]
                    print("Detected columns:", self.header)
                    return
            except subprocess.CalledProcessError:
                # The file may not yet be available.
                pass
            except Exception as e:
                print(f"Error fetching header, retrying: {e}")
            time.sleep(2)

    def _remote_data_reader(self):
        """
        Continuously read lines from the tail process and update EMA deques.
        This function runs in a background thread.
        """
        for line in self.tail_proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                row = next(csv.reader(io.StringIO(line)))
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue

            if len(row) != self.num_columns:
                print("Warning: unexpected number of columns in line:", row)
                continue

            with self.data_lock:
                for i, field in enumerate(row):
                    try:
                        val = float(field)
                    except ValueError:
                        continue
                    # If no previous value exists, initialize the EMA with the current value.
                    if len(self.ema_data[i]) == 0:
                        ema_val = val
                    else:
                        ema_val = (
                            self.ema_alpha * val
                            + (1 - self.ema_alpha) * self.ema_data[i][-1]
                        )
                    self.ema_data[i].append(ema_val)

    def start_tail(self):
        """
        Start the SSH tail process to follow the remote CSV file.
        The '-n {window}' option grabs the last few lines initially.
        """
        tail_cmd = [
            "ssh",
            f"{self.user}@{self.host}",
            "tail",
            "-n",
            str(self.window),
            "-F",
            self.remote_file,
        ]
        try:
            self.tail_proc = subprocess.Popen(
                tail_cmd, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1
            )
        except Exception as e:
            raise RuntimeError(f"Error starting tail process: {e}")
        self.reader_thread = threading.Thread(target=self._remote_data_reader)
        self.reader_thread.daemon = True
        self.reader_thread.start()

    def init_plot(self):
        """
        Initialize the Matplotlib figure, subplots, and the animation.
        Also, set up the key press handler.
        """
        ncols = self.ncols
        nrows = math.ceil(self.num_columns / ncols)
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        self.fig.canvas.manager.set_window_title(
            f"CSV Monitor - {self.file_name.replace('_', ' ').title()}"
        )

        # Flatten the axes array.
        if isinstance(self.axes, plt.Axes):
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()

        # Hide any extra subplots.
        for ax in self.axes[self.num_columns :]:
            ax.set_visible(False)

        self.lines = []
        for i in range(self.num_columns):
            ax = self.axes[i]
            (line,) = ax.plot([], [], lw=1)
            ax.set_title(self.header[i], fontsize=10, pad=2)
            ax.grid(True)
            # Remove x-axis ticks except for the subplots in the last row.
            if (i // ncols) < (nrows - 1):
                ax.set_xticklabels([])
                ax.set_xticks([])
            self.lines.append(line)

        def init_func():
            for line in self.lines:
                line.set_data([], [])
            return self.lines

        def animate(frame):
            with self.data_lock:
                for i in range(self.num_columns):
                    y_data = list(self.ema_data[i])
                    x_data = list(range(len(y_data)))
                    self.lines[i].set_data(x_data, y_data)
                    if x_data:
                        self.axes[i].set_xlim(x_data[0], x_data[-1])
                        ymin, ymax = min(y_data), max(y_data)
                        if ymin == ymax:
                            ymin -= 1
                            ymax += 1
                        self.axes[i].set_ylim(ymin, ymax)
            return self.lines

        self.anim = animation.FuncAnimation(
            self.fig,
            animate,
            init_func=init_func,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )
        self.fig.tight_layout()

        # Connect the key press events.
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_key(self, event):
        """
        Handle key press events.
         - 'q' closes the plot window (and quits the program).
         - 'r' reloads the file by calling find_latest_results_folder and restarting the tail.
        """
        if event.key == "q":
            print("Exiting...")
            plt.close(self.fig)
        elif event.key == "r":
            print("Reloading file...")
            self.reload_file()

    def reload_file(self):
        """
        Reload the remote CSV file:
          1. Check for a (possibly) new results folder.
          2. If a new file is found, terminate the current tail process,
             clear the stored data, update the header (if possible),
             and start tailing the new file.
        """
        new_folder = self.find_latest_results_folder()
        if new_folder:
            new_file = os.path.join(new_folder, f"{self.file_name}.csv")
            if new_file == self.remote_file:
                print("Remote file is the same. No update.")
                return
            print("Switching to new file:", new_file)
            # Terminate the current tail process.
            if self.tail_proc:
                self.tail_proc.terminate()
                self.tail_proc = None
            # Update the remote file and clear the stored EMA data.
            self.remote_file = new_file
            with self.data_lock:
                self.ema_data = [
                    collections.deque(maxlen=self.window)
                    for _ in range(self.num_columns)
                ]
            # Optionally, update the header from the new file.
            head_cmd = [
                "ssh",
                f"{self.user}@{self.host}",
                "head",
                "-n",
                "1",
                self.remote_file,
            ]
            try:
                header_output = subprocess.check_output(
                    head_cmd, universal_newlines=True
                )
                header_line = header_output.strip()
                header_reader = csv.reader(io.StringIO(header_line))
                new_header = next(header_reader)
                if len(new_header) != self.num_columns:
                    print(
                        "Warning: new file has a different number of columns. Continuing with old configuration."
                    )
                else:
                    self.header = new_header
                    # Update the subplot titles.
                    for i, title in enumerate(self.header):
                        self.axes[i].set_title(title, fontsize=10, pad=2)
            except Exception as e:
                print("Error reading header from new file:", e)
            # Start tailing the new file.
            self.start_tail()
        else:
            print("No new results folder found.")

    def stop(self):
        """
        Clean up the tail process and its reader thread.
        """
        if self.tail_proc:
            self.tail_proc.terminate()
        if self.reader_thread:
            self.reader_thread.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor a remote CSV file over SSH and plot each column (with EMA smoothing)."
    )
    parser.add_argument("--user", help="SSH username", default="arya")
    parser.add_argument("--host", help="SSH host (or IP address)", default="10.5.6.248")
    parser.add_argument(
        "--window", type=int, default=500, help="Number of data points to display"
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.5,
        help="EMA smoothing factor (alpha) between 0 and 1",
    )
    parser.add_argument(
        "--ncols", type=int, default=5, help="Number of subplots per row"
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Plot update interval in milliseconds"
    )
    parser.add_argument("--task", default="walk_finetune")
    args = parser.parse_args()

    file_name = "training_rewards"
    monitor = CSVMonitor(
        file_name=file_name,
        user=args.user,
        host=args.host,
        task=args.task,
        window=args.window,
        ema=args.ema,
        ncols=args.ncols,
        interval=args.interval,
    )

    # Block until a valid CSV file and its header are found.
    monitor.fetch_header()
    # Start tailing the remote file.
    monitor.start_tail()
    # Initialize the plot and key event handling.
    monitor.init_plot()
    # Start the GUI event loop.
    plt.show()
    # When the window is closed, stop background processes.
    monitor.stop()
