#!/usr/bin/env python3
import argparse
import collections
import csv
import io
import math
import multiprocessing
import os
import subprocess
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

class CSVMonitor:
    def __init__(self, file_name, user='toddie', host='10.5.6.248', window=100, ema=0.2, ncols=4, interval=100):
        """
        Initializes the CSVMonitor.
        
        :param user: SSH username.
        :param host: SSH host or IP address.
        :param remote_file: Full path to the CSV file on the remote host.
        :param window: Number of data points to display.
        :param ema: EMA smoothing factor (alpha, between 0 and 1).
        :param ncols: Number of subplots per row.
        :param interval: Plot update interval in milliseconds.
        """
        self.file_name = file_name
        self.user = user
        self.host = host
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

        # Use a separate process to run the plot so that its GUI runs in its main thread.
        self._plot_process = None

    
    def find_latest_results_folder(self):
        """Find the latest results.csv file for the given experiment_name on the remote machine."""
        find_cmd = (
            f'ssh {self.user}@{self.host} '
            f'"ls -td ~/projects/toddlerbot/results/toddlerbot_walk_finetune_real_world_* 2>/dev/null | head -n 1"'
        )
        try:
            latest_file = subprocess.check_output(find_cmd, shell=True, universal_newlines=True).strip()
            if latest_file:
                # print(f"Found latest results file: {latest_file}")
                return latest_file
        except subprocess.CalledProcessError:
            pass  # No matching files found

        return None

    def fetch_header(self):
        """Fetch the header line from the remote CSV file via SSH, waiting if necessary."""
        print("Searching for the latest results file...")
        while True:
            remote_folder = self.find_latest_results_folder()
            self.remote_file = os.path.join(remote_folder, f'{self.file_name}.csv') if remote_folder else None
            if self.remote_file:
                break
            print("No results file found yet, waiting...")
            time.sleep(2)

        head_cmd = ['ssh', f'{self.user}@{self.host}', 'head', '-n', '1', self.remote_file]
        print("Waiting for CSV file and headers to be available...")
        while True:
            try:
                header_output = subprocess.check_output(head_cmd, universal_newlines=True)
                header_line = header_output.strip()
                if header_line:  # Header is found, proceed
                    header_reader = csv.reader(io.StringIO(header_line))
                    self.header = next(header_reader)
                    self.num_columns = len(self.header)
                    self.ema_data = [collections.deque(maxlen=self.window) for _ in range(self.num_columns)]
                    # print("Detected columns:", self.header)
                    return
            except subprocess.CalledProcessError:
                # Likely the file does not exist yet.
                pass
            except Exception as e:
                print(f"Error fetching header, retrying: {e}")
            time.sleep(2)  # Wait a couple of seconds before trying again

    def _remote_data_reader(self):
        """
        Continuously read lines from the tail process and update EMA deques.
        This method is intended to run in a background thread.
        """
        for line in self.tail_proc.stdout:
            line = line.strip()
            if not line:
                continue
            f = io.StringIO(line)
            try:
                row = next(csv.reader(f))
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
                    # Compute EMA: if no previous value, use the current value.
                    if len(self.ema_data[i]) == 0:
                        ema_val = val
                    else:
                        ema_val = self.ema_alpha * val + (1 - self.ema_alpha) * self.ema_data[i][-1]
                    self.ema_data[i].append(ema_val)

    def start_tail(self):
        """
        Start the tail process via SSH to follow the remote CSV file.
        Using `-n {window}` grabs the last `window` lines initially.
        """
        tail_cmd = ['ssh', f'{self.user}@{self.host}', 'tail', '-n', str(self.window), '-F', self.remote_file]
        try:
            self.tail_proc = subprocess.Popen(tail_cmd, stdout=subprocess.PIPE,
                                              universal_newlines=True, bufsize=1)
        except Exception as e:
            raise RuntimeError(f"Error starting tail process: {e}")

        self.reader_thread = threading.Thread(target=self._remote_data_reader)
        self.reader_thread.daemon = True
        self.reader_thread.start()

    def init_plot(self):
        """Initialize the Matplotlib figure and subplots."""
        ncols = self.ncols
        nrows = math.ceil(self.num_columns / ncols)
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        self.fig.canvas.manager.set_window_title(f"CSV Monitor - {self.file_name.replace('_', ' ').title()}")

        # Ensure axes is a flat list.
        if isinstance(self.axes, plt.Axes):
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()

        # Hide any extra subplots if the grid has more axes than needed.
        for ax in self.axes[self.num_columns:]:
            ax.set_visible(False)

        self.lines = []
        for i in range(self.num_columns):
            ax = self.axes[i]
            line, = ax.plot([], [], lw=1)
            # Set smaller font size and bring the title closer to the plot.
            ax.set_title(self.header[i], fontsize=10, pad=2)
            ax.grid(True)
            # Disable x-axis ticks for all subplots except those in the last row.
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

        # Create the animation; disable caching to avoid warnings.
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init_func,
                                            interval=self.interval, blit=False,
                                            cache_frame_data=True)
        plt.tight_layout()

    def _run(self):
        """
        The main function that runs in the monitor process.
        It sets up the header, tail, plot, and then calls plt.show() to start the GUI event loop.
        """
        self.fetch_header()
        self.start_tail()
        self.init_plot()
        # Now that everything is set up, the GUI event loop runs in this process’s main thread.
        plt.show()  # This will block until the window is closed.
        # When plt.show() returns, we clean up.
        self.stop()

    def start(self):
        """
        Starts the monitor in a separate process so that this method does not block.
        After calling start(), you can perform other tasks in your main thread.
        """
        self._plot_process = multiprocessing.Process(target=self._run)
        self._plot_process.start()

    def close(self):
        """
        Stops the monitor and closes the plot window.
        Call this method when your other tasks are done and you want to shut down the monitor.
        """
        if self._plot_process is not None:
            # Terminate the monitor process.
            self._plot_process.terminate()
            self._plot_process.join()
            print("Monitor closed.")

    def stop(self):
        """Clean up the tail process and background thread."""
        if self.tail_proc:
            self.tail_proc.terminate()
        if self.reader_thread:
            self.reader_thread.join(timeout=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Monitor a remote CSV file over SSH and plot each column (with EMA smoothing)."
    )
    parser.add_argument('--user', help="SSH username", default="toddie")
    parser.add_argument('--host', help="SSH host (or IP address)", default="10.5.6.248")
    parser.add_argument('--window', type=int, default=500,
                        help="Number of data points to display (default: 100)")
    parser.add_argument('--ema', type=float, default=0.2,
                        help=("EMA smoothing factor (alpha) between 0 and 1. "
                              "1.0 means no smoothing (default: 1.0)"))
    parser.add_argument('--ncols', type=int, default=4,
                        help="Number of subplots per row (default: 4)")
    parser.add_argument('--interval', type=int, default=100,
                        help="Plot update interval in milliseconds (default: 100)")
    args = parser.parse_args()

    monitor = CSVMonitor(user=args.user, host=args.host,
                         window=args.window, ema=args.ema, ncols=args.ncols, interval=args.interval)
    monitor.start()

    try:
        print("Monitor is running in the background. Do your other tasks here.")
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing monitor...")
        monitor.close()