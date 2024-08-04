import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.planning.zmp_feedback_planner import ZMPFeedbackPlanner


def plot_results(data_dict, suffix=""):
    # Figure 1
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(data_dict["time"], data_dict["desired_zmp"][:, 0], "r")
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 0], "b")
    plt.plot(data_dict["time"], data_dict["cop"][:, 0], "g")
    plt.plot(data_dict["time"], data_dict["x"][:, 0], "c")
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.legend(["desired zmp", "planned com", "planned cop", "actual com"])

    plt.subplot(2, 1, 2)
    plt.plot(data_dict["time"], data_dict["desired_zmp"][:, 1], "r")
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 1], "b")
    plt.plot(data_dict["time"], data_dict["cop"][:, 1], "g")
    plt.plot(data_dict["time"], data_dict["x"][:, 1], "c")
    plt.xlabel("time [s]")
    plt.ylabel("y [m]")
    plt.legend(["desired zmp", "planned com", "planned cop", "actual com"])
    plt.savefig(f"plot{suffix}_1.png")

    # Figure 2
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 2], "b")
    plt.plot(data_dict["time"], data_dict["x"][:, 2], "c")
    plt.xlabel("time [s]")
    plt.ylabel("xd [m/s]")
    plt.legend(["planned comd", "actual comd"])

    plt.subplot(2, 1, 2)
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 3], "b")
    plt.plot(data_dict["time"], data_dict["x"][:, 3], "c")
    plt.xlabel("time [s]")
    plt.ylabel("yd [m/s]")
    plt.legend(["planned comd", "actual comd"])
    plt.savefig(f"plot{suffix}_2.png")

    # Figure 3
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(data_dict["time"], data_dict["u"][:, 0], "r")
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 4], "b.")
    plt.xlabel("time [s]")
    plt.ylabel("xdd [m/s2]")
    plt.legend(["comdd from policy", "nominal comdd"])

    plt.subplot(2, 1, 2)
    plt.plot(data_dict["time"], data_dict["u"][:, 1], "r")
    plt.plot(data_dict["time"], data_dict["nominal_com"][:, 5], "b.")
    plt.xlabel("time [s]")
    plt.ylabel("ydd [m/s2]")
    plt.legend(["comdd from policy", "nominal comdd"])
    plt.savefig(f"plot{suffix}_3.png")


def print_results(data_dict):
    print("Time: \n", data_dict["time"])
    print("Desired ZMP: \n", data_dict["desired_zmp"])
    print("Nominal COM: \n", data_dict["nominal_com"])
    print("Center of Pressure (COP): \n", data_dict["cop"])
    print("State Vector (x): \n", data_dict["x"])
    print("Control Inputs (u): \n", data_dict["u"])


def print_results_to_files(data_dict, suffix=""):
    with open(f"cpp_results{suffix}.txt", "w") as file:
        file.write("Time:\n" + str(data_dict["time"]) + "\n\n")
        file.write("Desired ZMP:\n" + str(data_dict["desired_zmp"]) + "\n\n")
        file.write("Nominal COM:\n" + str(data_dict["nominal_com"]) + "\n\n")
        file.write("Center of Pressure (COP):\n" + str(data_dict["cop"]) + "\n\n")
        file.write("State Vector (x):\n" + str(data_dict["x"]) + "\n\n")
        file.write("Control Inputs (u):\n" + str(data_dict["u"]) + "\n\n")


def simulate_zmp_policy(zmp_planner, x0, dt, extra_time_at_the_end):
    time_steps, _ = zmp_planner.get_desired_zmp_traj()
    N = int((time_steps[-1] + extra_time_at_the_end - time_steps[0]) / dt)

    traj = {
        "time": np.zeros(N),
        "x": np.zeros((N, 4)),
        "u": np.zeros((N, 2)),
        "cop": np.zeros((N, 2)),
        "desired_zmp": np.zeros((N, 2)),
        "nominal_com": np.zeros((N, 6)),
    }

    for i in range(N):
        traj["time"][i] = time_steps[0] + i * dt
        if i == 0:
            traj["x"][i, :] = x0
        else:
            xd = np.hstack((traj["x"][i - 1, 2:], traj["u"][i - 1, :]))
            traj["x"][i, :] = traj["x"][i - 1, :] + xd * dt

        traj["u"][i, :] = zmp_planner.compute_optimal_com_acc(
            traj["time"][i], traj["x"][i, :]
        )
        traj["cop"][i, :] = zmp_planner.com_acc_to_cop(traj["x"][i, :], traj["u"][i, :])

        traj["desired_zmp"][i, :] = zmp_planner.get_desired_zmp(traj["time"][i])
        traj["nominal_com"][i, :2] = zmp_planner.get_nominal_com(traj["time"][i])
        traj["nominal_com"][i, 2:4] = zmp_planner.get_nominal_com_vel(traj["time"][i])
        traj["nominal_com"][i, 4:] = zmp_planner.get_nominal_com_acc(traj["time"][i])

        # Debug output can be enabled here to trace computations
        # print(f"Time: {traj['time'][i]}")
        # print(f"State: {traj['x'][i, :]}")
        # print(f"Control: {traj['u'][i, :]}")
        # print(f"COP: {traj['cop'][i, :]}")
        # print(f"Desired ZMP: {traj['desired_zmp'][i, :]}")
        # print(f"Nominal COM: {traj['nominal_com'][i, :]}")

    return traj


def generate_desired_zmp_trajs(
    footsteps, double_support_duration, single_support_duration
):
    time_steps = []
    zmp_d = []

    time = 0

    time_steps.append(time)
    zmp_d.append(footsteps[0])
    time += single_support_duration
    time_steps.append(time)
    zmp_d.append(footsteps[0])

    for i in range(1, len(footsteps)):
        time += double_support_duration
        time_steps.append(time)
        zmp_d.append(footsteps[i])

        time += single_support_duration
        time_steps.append(time)
        zmp_d.append(footsteps[i])

    return time_steps, zmp_d


def main():
    footsteps = [
        np.array([0, 0]),
        np.array([0.5, 0.1]),
        np.array([1, -0.1]),
        np.array([1.5, 0.1]),
        np.array([2, -0.1]),
        np.array([2.5, 0]),
    ]

    time_steps, zmp_d = generate_desired_zmp_trajs(footsteps, 0.5, 1)

    x0 = np.array([0, 0, 0, 0], dtype=float)
    com_z = 1
    zmp_planner = ZMPFeedbackPlanner()
    zmp_planner.plan(time_steps, zmp_d, x0, com_z, Qy=np.eye(2), R=np.eye(2) * 0.1)

    sample_dt = 0.01
    # Perturb the initial state a bit.
    x0 = np.array([0, 0, 0.2, -0.1])
    data_dict = simulate_zmp_policy(zmp_planner, x0, sample_dt, 2)

    plot_results(data_dict)
    print_results(data_dict)
    print_results_to_files(data_dict)


if __name__ == "__main__":
    main()
