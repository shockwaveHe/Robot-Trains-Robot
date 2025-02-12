import ast
import asyncio
import functools
import inspect
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Callable, TypeVar

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


def log(message: str, header: str = "", level: str = "info"):
    """Logs a message with an optional header and severity level.

    Args:
        message (str): The message to log.
        header (str, optional): An optional header for the log message. Defaults to an empty string.
        level (str, optional): The severity level of the log message (e.g., 'info', 'warning', 'error'). Defaults to 'info'.
    """
    header_msg = f"[{header}] "
    if level == "debug":
        my_logger.debug(Fore.CYAN + "[Debug] " + header_msg + message)
    elif level == "error":
        my_logger.error(Fore.RED + "[Error] " + header_msg + message)
    elif level == "warning":
        my_logger.warning(Fore.YELLOW + "[Warning] " + header_msg + message)
    else:
        my_logger.info(Fore.WHITE + "[Info] " + header_msg + message)


def precise_sleep(duration: float):
    """Sleeps for a specified duration with high precision.

    Args:
        duration (float): The time to sleep in seconds.
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
F = TypeVar("F", bound=Callable[..., Any])


def profile() -> Callable[[F], F]:
    """Convert a snake_case string to CamelCase.

    Args:
        snake_str: The snake_case string to convert.

    Returns:
        The CamelCase string.
    """

    def decorator(func: F) -> F:
        # Register function to the global profiler
        global_profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Enable profiling
            global_profiler.enable_by_count()
            try:
                if inspect.iscoroutinefunction(func):
                    # Handle coroutine functions
                    result = asyncio.run(func(*args, **kwargs))
                else:
                    # Handle regular functions
                    result = func(*args, **kwargs)
            finally:
                # Disable profiling
                global_profiler.disable_by_count()

            return result

        return wrapper  # type: ignore

    return decorator


def dump_profiling_data(prof_path: str = "profile_output.lprof"):
    """Save profiling data to a specified file path.

    Args:
        prof_path (str): The file path where the profiling data will be saved. Defaults to "profile_output.lprof".
    """
    # Dump all profiling data into a single file
    global_profiler.dump_stats(prof_path)
    txt_path = prof_path.replace(".lprof", ".txt")
    subprocess.run(f"python -m line_profiler {prof_path} > {txt_path}", shell=True)

    log(f"Profile results saved to {txt_path}.", header="Profiler")


def snake2camel(snake_str: str) -> str:
    """Converts a snake_case string to CamelCase.

    Args:
        snake_str (str): The snake_case string to convert.

    Returns:
        str: The CamelCase string.
    """
    return "".join(word.title() for word in snake_str.split("_"))


def camel2snake(camel_str: str) -> str:
    """Converts a CamelCase string to snake_case.

    Args:
        camel_str (str): The CamelCase string to be converted.

    Returns:
        str: The converted snake_case string.
    """
    return "".join(["_" + c.lower() if c.isupper() else c for c in camel_str]).lstrip(
        "_"
    )


def set_seed(seed: int):
    """Sets the random seed for various libraries to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and the
    environment variable `PYTHONHASHSEED`. If the provided seed is -1, a random
    seed is generated and used.

    Args:
        seed (int): The seed value to set. If -1, a random seed is generated.
    """
    import os
    import random

    import numpy as np
    # import torch

    if seed == -1:
        seed = np.random.randint(0, 10000)

    log(f"Setting seed: {seed}", header="Seed")

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def parse_value(value: str):
    """Recursively converts a dataclass to a dictionary, handling nested dataclasses by directly accessing their fields.

    Args:
        value (str): The string representation of the value to be parsed.

    Returns:
        dict: A dictionary representation of the dataclass.
    """

    # Trim any extra whitespace
    value = value.strip()

    # Check for boolean values
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False

    # Check for list formatted as "[value1, value2, ...]"
    if value.startswith("[") and value.endswith("]"):
        try:
            # Use ast.literal_eval to safely parse the string into a list
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, list):
                return parsed_value
        except (SyntaxError, ValueError):
            raise ValueError(f"Invalid list format: {value}")

    # Try to convert to int or float
    try:
        # If there’s a decimal point, treat as float
        if "." in value:
            return float(value)
        # Otherwise, treat as int
        return int(value)
    except ValueError:
        # If neither int nor float, return the original string
        return value


def dataclass2dict(obj):
    """Converts a dataclass instance to a dictionary, including nested dataclasses.

    Args:
        obj: A dataclass instance to be converted.

    Returns:
        dict: A dictionary representation of the dataclass, with nested dataclasses
        also converted to dictionaries.
    """
    assert is_dataclass(obj)
    if len(asdict(obj)) == 0:
        return {
            attr: asdict(getattr(obj, attr))
            for attr in dir(obj)
            if not attr.startswith("_") and not callable(getattr(obj, attr))
        }
    else:
        return {key: value for key, value in asdict(obj).items()}


def find_latest_file_with_time_str(directory: str, file_prefix: str = "") -> str | None:
    """
    Finds the file with the latest timestamp (YYYYMMDD_HHMMSS) in the given directory,
    for files ending with the specified suffix.

    Args:
        directory (str): Directory to search for files.
        file_suffix (str): The suffix to match (e.g., '.pkl', '_updated.pkl').

    Returns:
        str | None: Full path of the latest file or None if no matching file is found.
    """
    pattern = re.compile(r".*" + re.escape(file_prefix) + r".*(\d{8}_\d{6}).*")

    latest_file = None
    latest_time = None

    # Iterate through files in the directory
    for file in os.listdir(directory):
        match = pattern.search(file)  # Check if the file matches the pattern
        if match:
            # Extract the timestamp and parse it into a datetime object
            timestamp_str = match.group(1)
            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            # Update the latest file if this timestamp is more recent
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file

    return os.path.join(directory, latest_file) if latest_file else None
