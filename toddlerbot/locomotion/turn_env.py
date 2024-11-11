from typing import Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.walk_env import WalkEnv


class TurnEnv(WalkEnv, env_name="turn"):
    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        rng, rng_1, rng_2 = jax.random.split(rng, 3)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            # TODO: Bring the random pose sampling back
            pose_command = pose_command.at[:5].set(0.0)

        # Parametric equation of ellipse
        x = jnp.zeros(1)
        y = jnp.zeros(1)
        z = jax.random.uniform(
            rng_2,
            (1,),
            minval=self.command_range[7][0],
            maxval=self.command_range[7][1],
        )
        command = jnp.concatenate([pose_command, x, y, z])

        # Set small commands to zero based on norm condition
        mask = (jnp.linalg.norm(command[5:]) > self.deadzone).astype(jnp.float32)
        command = command.at[5:].set(command[5:] * mask)

        return command
