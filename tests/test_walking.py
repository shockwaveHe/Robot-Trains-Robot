import subprocess


def run_test(task_name: str, robot_name: str):
    # Path to the walking.py script
    script_path = f"toddlerbot/tasks/{task_name}.py"

    # Path to the expected output text file
    expected_output_file = f"tests/{robot_name}_{task_name}_out.txt"

    try:
        result = subprocess.run(
            ["python", script_path, "--robot", robot_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        captured_output = result.stdout

    except subprocess.TimeoutExpired as e:
        print("Script timed out. Partial output captured.")
        captured_output = e.stdout.decode("utf-8") if e.stdout else ""

    # Find the first occurrence of "joint_angles"
    start_index = captured_output.find("joint_angles")
    if start_index != -1:
        captured_output = captured_output[start_index:]

    # Read the expected output from the file
    with open(expected_output_file, "r") as file:
        expected_output = file.read()

    # Perform assertion
    assert captured_output[: int(1e4)].strip() == expected_output[: int(1e4)].strip()


def test_op_walking():
    run_test("walking", "sustaina_op")


def test_op3_walking():
    run_test("walking", "robotis_op3")


if __name__ == "__main__":
    # test_op_walking()
    pass
