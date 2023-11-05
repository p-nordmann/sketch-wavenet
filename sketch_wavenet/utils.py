import jax.numpy as jnp
from jaxtyping import Array, Float32


def build_mask(
    N_max: int, N_s: Float32[Array, "batch_size"]
) -> Float32[Array, "batch_size N_max"]:
    return jnp.arange(N_max) < N_s[:, jnp.newaxis]
