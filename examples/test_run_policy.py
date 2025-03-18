# test_run_policy.py

import os
import threading
import time

import gdown

from toddlerbot.policies.run_policy import main


def run_policy(policy: str, args: list = [], sleep_time: int = 1):
    """
    Test that the policy runs for 5 seconds and can be interrupted.
    """
    # Start the policy in a separate thread
    policy_thread = threading.Thread(
        target=main, args=(["--policy", policy, "--vis", "none", "--no-plot", *args],)
    )
    policy_thread.start()

    # Let it run for 1 seconds
    time.sleep(sleep_time)

    # Send a keyboard interrupt to the thread
    policy_thread.join(timeout=0.1)  # Wait a bit for the thread to respond
    if policy_thread.is_alive():
        # Simulate a keyboard interrupt
        import ctypes

        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(policy_thread.ident), ctypes.py_object(KeyboardInterrupt)
        )

    # Wait for the thread to finish
    policy_thread.join()

    # Assert that the thread has stopped
    assert not policy_thread.is_alive()


def test_stand_policy():
    run_policy("stand")


def test_push_up_policy():
    run_policy("replay", ["--run-name", "push_up"])


def test_walk_policy():
    run_policy("walk")


def test_hug_policy():
    result_path = os.path.join("results", "toddlerbot_hug_dp_20250109_235450")
    ckpt_path = os.path.join(result_path, "best_ckpt.pth")
    if not os.path.exists(ckpt_path):
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1KZ6vMi3UzX8IxAdUEY6xGaYItKecuCny",
            output=result_path,
            quiet=True,
        )

    run_policy("dp", ["--task", "hug", "--ckpt", "20250109_235450"], sleep_time=5)
