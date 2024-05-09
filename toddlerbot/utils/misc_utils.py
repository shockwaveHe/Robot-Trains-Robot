import asyncio
import functools
import inspect
import logging
import subprocess
import time

from colorama import Fore, init
from line_profiler import LineProfiler

# Initialize colorama to auto-reset color codes after each print statement
init(autoreset=True)

# Set up basic configuration for logging
logging.basicConfig(level=logging.WARNING)

# Configure your specific logger
my_logger = logging.getLogger("my_logger")
my_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
my_logger.addHandler(handler)
my_logger.propagate = False


def log(message, header=None, level="info"):
    """
    Log a message to the console with color coding.

    Parameters:
    - message: The message to log.
    - level: The log level (ERROR, WARNING, INFO).
    """
    header_msg = f"[{header}] " if header is not None else ""
    if level == "debug":
        my_logger.debug(Fore.CYAN + "[Debug] " + header_msg + message)
    elif level == "error":
        my_logger.error(Fore.RED + "[Error] " + header_msg + message)
    elif level == "warning":
        my_logger.warning(Fore.YELLOW + "[Warning] " + header_msg + message)
    else:
        my_logger.info(Fore.WHITE + "[Info] " + header_msg + message)


def precise_sleep(duration):
    """
    Sleep for a specified amount of time.

    Parameters:
    - time: The amount of time to sleep (in seconds).
    """
    try:
        # Convert to seconds and subtract a little
        target = time.perf_counter_ns() + duration * 1e9

        # Leave 0.05s for active wait
        while time.perf_counter_ns() < target - 1e6:
            time.sleep(0)

        # Active waiting for the last 1ms
        while time.perf_counter_ns() < target:
            pass

    except KeyboardInterrupt:
        raise KeyboardInterrupt("Sleep interrupted by user.")


# Create a global profiler instance
global_profiler = LineProfiler()


def profile():
    def decorator(func):
        # Register function to the global profiler
        global_profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global_profiler.enable_by_count()  # Enable profiling
            try:
                if inspect.iscoroutinefunction(func):
                    # Handle coroutine functions
                    result = asyncio.run(func(*args, **kwargs))
                else:
                    # Handle regular functions
                    result = func(*args, **kwargs)
            finally:
                global_profiler.disable_by_count()  # Disable profiling

            return result

        return wrapper

    return decorator


def dump_profiling_data(prof_path="profile_output.lprof"):
    # Dump all profiling data into a single file
    global_profiler.dump_stats(prof_path)
    txt_path = prof_path.replace(".lprof", ".txt")
    subprocess.run(f"python -m line_profiler {prof_path} > {txt_path}", shell=True)

    log(f"Profile results saved to {txt_path}.", header="Profiler")


def snake2camel(snake_str):
    """
    Convert a snake_case string to CamelCase.

    Parameters:
    - snake_str: The snake_case string to convert.

    Returns:
    - The CamelCase string.
    """
    return "".join(word.title() for word in snake_str.split("_"))


def camel2snake(camel_str):
    """
    Convert a CamelCase string to snake_case.

    Parameters:
    - camel_str: The CamelCase string to convert.

    Returns:
    - The snake_case string.
    """
    return "".join(["_" + c.lower() if c.isupper() else c for c in camel_str]).lstrip(
        "_"
    )


def set_seed(seed):
    import os
    import random

    import numpy as np
    import torch

    if seed == -1:
        seed = np.random.randint(0, 10000)

    log(f"Setting seed: {seed}", header="Seed")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
