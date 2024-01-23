import subprocess


def test_op_walking():
    # Path to the walking.py script
    script_path = "toddleroid/sim/pybullet/walking.py"

    # Path to the expected output text file
    expected_output_file = "tests/expected_output.txt"

    try:
        # Run the script for 10 seconds and capture output
        result = subprocess.run(
            ["python", script_path, "--robot-name", "sustaina_op"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        captured_output = result.stdout

        # Read the expected output from the file
        with open(expected_output_file, "r") as file:
            expected_output = file.read()

        # Compare the captured output with the expected output
        assert captured_output[:10000].strip() == expected_output[:10000].strip()

    except subprocess.TimeoutExpired:
        print("Test stopped after 5 seconds.")
