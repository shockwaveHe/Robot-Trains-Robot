import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from colorama import Fore, Style, init

# Initialize colorama to auto-reset color codes after each print statement
init(autoreset=True)

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def log(message, header=None, level="info"):
    """
    Log a message to the console with color coding.

    Parameters:
    - message: The message to log.
    - level: The log level (ERROR, WARNING, INFO).
    """
    header_msg = f"[{header}] " if header is not None else ""
    if level == "error":
        logging.error(Fore.RED + "[Error] " + header_msg + message + Style.RESET_ALL)
    elif level == "warning":
        logging.warning(
            Fore.YELLOW + "[Warning] " + header_msg + message + Style.RESET_ALL
        )
    else:
        logging.info(Fore.WHITE + "[Info] " + header_msg + message + Style.RESET_ALL)
