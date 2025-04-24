import argparse
import json
import os
import pathlib

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from retarget_lib.mink_retarget import MinkRetarget
from retarget_lib.mocap.vicon_datastream import (
    get_subject_names,
    get_vicon_data,
    setup_vicon,
    setup_vicon_cache,
)
from retarget_lib.utils.draw import draw_frame


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.environ.get("MOCAP_SERVER_IP", "localhost:801"),
        help="Address of the Vicon stream.",
    )
    parser.add_argument(
        "--ik_config",
        default=HERE / ".." / "ik_configs" / "vicon2g1_29dof_rev_1_0.json",
        help="Path to the IK config JSON file",
    )
    parser.add_argument(
        "--xml_file",
        default=os.environ.get("XML_FILE", None),
        help="Path to the Mujoco XML file with the skeleton to control",
    )
    args = parser.parse_args()

    # Vicon initialization
    client = setup_vicon(args.host)
    subject_names = get_subject_names(client)

    assert len(subject_names) == 1, "Only one subject should be present in the scene."
    subject = subject_names[0]

    setup_vicon_cache(client, subject=subject)

    # Load the IK config
    with open(args.ik_config) as f:
        ik_config = json.load(f)

    ik_match_table = ik_config.pop("ik_match_table")

    # Initialize the retargeting system
    retarget = MinkRetarget(
        args.xml_file,
        ik_match_table,
        scale=ik_config["scale"],
        ground=ik_config["ground_height"],
    )
    model = mj.MjModel.from_xml_path(args.xml_file)
    data = mj.MjData(model)

    with mjv.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Clean custom geometry
            viewer.user_scn.ngeom = 0

            # Update task targets.
            vicon_data = get_vicon_data(client, subject)

            # Draw the task targets for reference
            for robot_link, ik_data in ik_match_table.items():
                if ik_data[0] not in vicon_data:
                    continue
                draw_frame(
                    ik_config["scale"] * np.array(vicon_data[ik_data[0]][0])
                    - retarget.ground,
                    R.from_quat(
                        vicon_data[ik_data[0]][1], scalar_first=True
                    ).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1], scalar_first=True),
                )

            # Retarget and pose the model
            retarget.update_targets(vicon_data)
            data.qpos[:] = retarget.retarget(vicon_data).copy()
            mj.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
    client.disconnect()
