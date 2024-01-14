import time

import pybullet as p
import pybullet_data

from toddleroid.sim.pybullet.utils import find_urdf_path


def setup_simulation() -> None:
    """
    Set up the PyBullet simulation environment.

    Initializes the PyBullet environment in GUI mode and sets the gravity for the simulation.
    """

    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")


def put_robot_on_ground(robot_id: int, z_offset: float) -> None:
    """
    Adjust the robot's position to place its lowest point at a specified offset above the ground.

    This function finds the lowest point along the z-axis of the robot and adjusts the robot's
    base position so that this point is at `z_offset` above the ground level (z=0).

    Args:
        robot_id: The unique ID of the robot in PyBullet.
        z_offset: The offset above the ground at which the lowest point of the robot should be placed.

    Example:
        robot_id = p.loadURDF("path_to_urdf_file")
        put_robot_on_ground(robot_id, z_offset=0.01)
    """
    num_joints = p.getNumJoints(robot_id)
    lowest_z = float("inf")

    for i in range(-1, num_joints):  # -1 for the base link
        if i == -1:
            link_pos, _ = p.getBasePositionAndOrientation(robot_id)
        else:
            link_state = p.getLinkState(robot_id, i, computeForwardKinematics=True)
            link_pos = link_state[0]

        lowest_z = min(lowest_z, link_pos[2])

    # Calculate new base position
    base_pos, base_ori = p.getBasePositionAndOrientation(robot_id)
    new_base_pos = [base_pos[0], base_pos[1], base_pos[2] - lowest_z + z_offset]

    p.resetBasePositionAndOrientation(robot_id, new_base_pos, base_ori)


def run_simulation() -> None:
    """
    Run the main simulation loop.

    Sets up the simulation, loads the robot, and runs a basic simulation loop that continuously
    updates the physics simulation and prints the robot's position and orientation.
    """
    setup_simulation()
    TIME_STEP = 1.0 / 240.0

    urdf_path = find_urdf_path("Robotis_OP3")
    robot_id = p.loadURDF(urdf_path)
    put_robot_on_ground(robot_id, z_offset=0.01)

    try:
        while True:
            p.stepSimulation()
            time.sleep(TIME_STEP)
            pos, ori = p.getBasePositionAndOrientation(robot_id)
            print(f"Robot Position: {pos}, Orientation: {ori}")
    finally:
        p.disconnect()


if __name__ == "__main__":
    run_simulation()
