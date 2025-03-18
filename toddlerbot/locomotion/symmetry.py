import torch


@torch.no_grad()
def data_augmentation_func(obs, actions, env, is_critic):
    device = obs.device if obs is not None else actions.device

    num_motors = 30
    num_action = 12
    frame_stack = 15
    left_indices = torch.tensor(
        [*range(4, 10), *range(16, 23)], dtype=torch.long, device=device
    )
    right_indices = torch.tensor(
        [*range(10, 16), *range(23, 30)], dtype=torch.long, device=device
    )
    flip_motor_mask = torch.tensor(
        [-1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1],
        dtype=torch.float32,
        device=device,
    )
    args = (
        num_motors,
        num_action,
        frame_stack,
        left_indices,
        right_indices,
        flip_motor_mask,
    )

    if is_critic:
        obs_batch = flip_critic_obs(obs, *args)
    else:
        obs_batch = flip_actor_obs(obs, *args)

    mean_actions_batch = flip_actions(actions, *args)

    return obs_batch, mean_actions_batch


def flip_actor_obs(obs, *args):
    if obs is None:
        return obs

    (
        num_motors,
        num_action,
        frame_stack,
        left_indices,
        right_indices,
        flip_motor_mask,
    ) = args

    flip_obs = torch.zeros_like(obs)
    feature_dim = obs.shape[-1] // frame_stack

    for i in range(frame_stack):
        start = i * feature_dim
        end = (i + 1) * feature_dim

        # phase signal
        flip_obs[..., start : start + 1] = obs[..., start + 1 : start + 2]
        flip_obs[..., start + 1 : start + 2] = obs[..., start : start + 1]

        # command
        flip_obs[..., start + 2 : start + 5] = obs[..., start + 2 : start + 5]

        # motor_pos_delta
        flip_obs[..., start + 5 : start + 9] = obs[..., start + 5 : start + 9]
        flip_obs[..., start + 5 + left_indices] = (
            obs[..., start + 5 + right_indices] * flip_motor_mask
        )
        flip_obs[..., start + 5 + right_indices] = (
            obs[..., start + 5 + left_indices] * flip_motor_mask
        )

        # motor_vel
        flip_obs[..., start + 5 + num_motors : start + 5 + num_motors + 4] = obs[
            ..., start + 5 + num_motors : start + 5 + num_motors + 4
        ]
        flip_obs[..., start + 5 + num_motors + left_indices] = (
            obs[..., start + 5 + num_motors + right_indices] * flip_motor_mask
        )
        flip_obs[..., start + 5 + num_motors + right_indices] = (
            obs[..., start + 5 + num_motors + left_indices] * flip_motor_mask
        )

        # last_action
        flip_obs[
            ...,
            start + 5 + 2 * num_motors : start + 5 + 2 * num_motors + num_action // 2,
        ] = (
            obs[
                ...,
                start + 5 + 2 * num_motors + num_action // 2 : start
                + 5
                + 2 * num_motors
                + num_action,
            ]
            * flip_motor_mask[: num_action // 2]
        )
        flip_obs[
            ...,
            start + 5 + 2 * num_motors + num_action // 2 : start
            + 5
            + 2 * num_motors
            + num_action,
        ] = (
            obs[
                ...,
                start + 5 + 2 * num_motors : start
                + 5
                + 2 * num_motors
                + num_action // 2,
            ]
            * flip_motor_mask[: num_action // 2]
        )

        # angular_vel and euler
        flip_obs[..., start + 5 + 2 * num_motors + num_action : end] = obs[
            ..., start + 5 + 2 * num_motors + num_action : end
        ]

    # print("flip_obs[0]", flip_obs[0])
    # print("obs[0]", obs[0])

    obs_batch = torch.cat([obs, flip_obs], dim=0)

    return obs_batch


def flip_critic_obs(obs, *args):
    if obs is None:
        return obs

    (
        num_motors,
        num_action,
        frame_stack,
        left_indices,
        right_indices,
        flip_motor_mask,
    ) = args

    flip_obs = torch.zeros_like(obs)
    feature_dim = obs.shape[-1] // frame_stack

    for i in range(frame_stack):
        start = i * feature_dim
        end = (i + 1) * feature_dim

        # phase signal
        flip_obs[..., start : start + 1] = obs[..., start + 1 : start + 2]
        flip_obs[..., start + 1 : start + 2] = obs[..., start : start + 1]

        # command
        flip_obs[..., start + 2 : start + 5] = obs[..., start + 2 : start + 5]

        # motor_pos_delta
        flip_obs[..., start + 5 : start + 9] = obs[..., start + 5 : start + 9]
        flip_obs[..., start + 5 + left_indices] = (
            obs[..., start + 5 + right_indices] * flip_motor_mask
        )
        flip_obs[..., start + 5 + right_indices] = (
            obs[..., start + 5 + left_indices] * flip_motor_mask
        )

        # motor_vel
        flip_obs[..., start + 5 + num_motors : start + 5 + num_motors + 4] = obs[
            ..., start + 5 + num_motors : start + 5 + num_motors + 4
        ]
        flip_obs[..., start + 5 + num_motors + left_indices] = (
            obs[..., start + 5 + num_motors + right_indices] * flip_motor_mask
        )
        flip_obs[..., start + 5 + num_motors + right_indices] = (
            obs[..., start + 5 + num_motors + left_indices] * flip_motor_mask
        )

        # last_action
        flip_obs[
            ...,
            start + 5 + 2 * num_motors : start + 5 + 2 * num_motors + num_action // 2,
        ] = (
            obs[
                ...,
                start + 5 + 2 * num_motors + num_action // 2 : start
                + 5
                + 2 * num_motors
                + num_action,
            ]
            * flip_motor_mask[: num_action // 2]
        )
        flip_obs[
            ...,
            start + 5 + 2 * num_motors + num_action // 2 : start
            + 5
            + 2 * num_motors
            + num_action,
        ] = (
            obs[
                ...,
                start + 5 + 2 * num_motors : start
                + 5
                + 2 * num_motors
                + num_action // 2,
            ]
            * flip_motor_mask[: num_action // 2]
        )

        # motor_pos_error
        flip_obs[
            ...,
            start + 5 + 2 * num_motors + num_action : start
            + 5
            + 2 * num_motors
            + num_action
            + 4,
        ] = obs[
            ...,
            start + 5 + 2 * num_motors + num_action : start
            + 5
            + 2 * num_motors
            + num_action
            + 4,
        ]
        flip_obs[..., start + 5 + 2 * num_motors + num_action + left_indices] = (
            obs[..., start + 5 + 2 * num_motors + num_action + right_indices]
            * flip_motor_mask
        )
        flip_obs[..., start + 5 + 2 * num_motors + num_action + right_indices] = (
            obs[..., start + 5 + 2 * num_motors + num_action + left_indices]
            * flip_motor_mask
        )

        # linear_vel, angular_vel, euler
        flip_obs[
            ...,
            start + 5 + 3 * num_motors + num_action : start
            + 5
            + 3 * num_motors
            + num_action
            + 9,
        ] = obs[
            ...,
            start + 5 + 3 * num_motors + num_action : start
            + 5
            + 3 * num_motors
            + num_action
            + 9,
        ]

        # stance mask and reference stance mask
        flip_obs[..., start + 5 + 3 * num_motors + num_action + 9] = obs[
            ..., start + 5 + 3 * num_motors + num_action + 10
        ]
        flip_obs[..., start + 5 + 3 * num_motors + num_action + 10] = obs[
            ..., start + 5 + 3 * num_motors + num_action + 9
        ]
        flip_obs[..., start + 5 + 3 * num_motors + num_action + 11] = obs[
            ..., start + 5 + 3 * num_motors + num_action + 12
        ]
        flip_obs[..., start + 5 + 3 * num_motors + num_action + 12] = obs[
            ..., start + 5 + 3 * num_motors + num_action + 11
        ]

        # push linear_vel and push angular_vel
        flip_obs[..., start + 5 + 3 * num_motors + num_action + 13 : end] = obs[
            ..., start + 5 + 3 * num_motors + num_action + 13 : end
        ]

    # print("flipped_critic_obs.shape", flip_obs.shape)
    # print("flipped_critic_obs", flip_obs[0])
    # print("critic_obs", obs[0])

    obs_batch = torch.cat([obs, flip_obs], dim=0)

    return obs_batch


def flip_actions(actions, *args):
    if actions is None:
        return None

    (_, num_action, _, _, _, flip_motor_mask) = args

    flip_actions = torch.zeros_like(actions)
    flip_actions[..., : num_action // 2] = (
        actions[..., num_action // 2 :] * flip_motor_mask[: num_action // 2]
    )
    flip_actions[..., num_action // 2 :] = (
        actions[..., : num_action // 2] * flip_motor_mask[: num_action // 2]
    )

    # print("flipped_actions", flip_actions[0])
    # print("actions", actions[0])

    action_batch = torch.cat([actions, flip_actions], dim=0)

    return action_batch
