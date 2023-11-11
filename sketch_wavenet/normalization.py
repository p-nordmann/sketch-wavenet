import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float32


class RMSLayerNorm(eqx.Module):
    """Similar to layer normalization, without the mean estimate.

    Known to give similar results to layer norm, with reduced compute.
    """

    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.weight = jnp.ones(shape=(dim,))
        self.eps = eps

    def __call__(self, x: Float32[Array, "dim"]) -> Float32[Array, "dim"]:
        moment_2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(moment_2 + self.eps)
        return self.weight * x_normed