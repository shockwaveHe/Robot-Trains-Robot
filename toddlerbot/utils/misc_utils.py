import logging

from colorama import Fore, Style, init

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
    if level == "error":
        my_logger.error(Fore.RED + "[Error] " + header_msg + message)
    elif level == "warning":
        my_logger.warning(Fore.YELLOW + "[Warning] " + header_msg + message)
    else:
        my_logger.info(Fore.WHITE + "[Info] " + header_msg + message)
