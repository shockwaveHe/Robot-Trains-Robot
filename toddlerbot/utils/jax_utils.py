import jax
import jax.numpy as jnp
from jax import jit  # type: ignore


@jit
def quat_apply(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    """Apply a quaternion rotation to a vector."""
    q = quaternion
    # Extend vector to a quaternion
    v = jnp.array([0.0, *vector])  # type: ignore
    # Quaternion conjugate
    q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])  # type: ignore

    # Quaternion multiplication: q * v * q_conjugate
    v_rotated = quat_mult(quat_mult(q, v), q_conj)  # type: ignore
    # Return rotated vector (ignore the scalar part)
    return v_rotated[1:]  # type: ignore


@jit
def quat_mult(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(  # type: ignore
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


@jit
def wrap_to_pi(angle: jax.Array) -> jax.Array:
    """Wrap angles to the range [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
