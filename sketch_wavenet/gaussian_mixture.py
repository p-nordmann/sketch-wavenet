import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32


def build_mask(
    N_max: int, N_s: Float32[Array, "batch_size"]
) -> Float32[Array, "batch_size N_max"]:
    return jnp.arange(N_max) < N_s[:, jnp.newaxis]


def inner_mixture_loss(
    target_x,
    target_y,
    out_pis,
    out_mu_xs,
    out_mu_ys,
    out_s_xs,
    out_s_ys,
    out_r_xys,
):
    """Computes the mixture loss for one time step."""
    pis_term = out_pis - jax.nn.logsumexp(
        out_pis
    )  # shape = [M] (broadcasting [M] - [])
    standalone_term = jnp.log(jnp.cosh(out_r_xys)) - out_s_xs - out_s_ys  # shape = [M]

    x_normalized = (target_x - out_mu_xs) / jnp.exp(
        out_s_xs
    )  # shape = [M] (broadcasting ([] - [M])/[M])
    y_normalized = (target_y - out_mu_ys) / jnp.exp(
        out_s_ys
    )  # shape = [M] (broadcasting ([] - [M])/[M])
    real_part = (
        jnp.cosh(out_r_xys) * x_normalized - jnp.sinh(out_r_xys) * y_normalized
    )  # shape = [M]
    imag_part = y_normalized  # shape = [M]
    quadratic_term = -0.5 * (real_part**2 + imag_part**2)  # shape = [M]

    return jax.nn.logsumexp(pis_term + standalone_term + quadratic_term)  # shape = []


def mixture_loss(
    target_x,
    target_y,
    out_pis,
    out_mu_xs,
    out_mu_ys,
    out_s_xs,
    out_s_ys,
    out_r_xys,
    mask,
):
    """Computes the mixture loss for all time steps.

    All arguments have one more dimension for time. There is a mask to apply, too.
    """
    return -jnp.sum(
        mask
        * (
            jax.vmap(inner_mixture_loss)(
                target_x,
                target_y,
                out_pis,
                out_mu_xs,
                out_mu_ys,
                out_s_xs,
                out_s_ys,
                out_r_xys,
            )
            - jnp.log(2 * jnp.pi)
        )
    )
