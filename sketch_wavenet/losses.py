import jax
import jax.numpy as jnp
import optax
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
):
    """Computes the mixture loss for one time step."""
    # We retrieve a constant term to make pis_term more stable.
    # We also add a 0 term to the denominator to make the model able to turn off ALL of the gaussians.
    retrieve_cst = jnp.max(jnp.append(out_pis, 0))
    pis_term = (
        out_pis - retrieve_cst - jax.nn.logsumexp(jnp.append(out_pis, 0) - retrieve_cst)
    )  # shape = [M] (broadcasting [M] - [])

    standalone_term = -out_s_xs - out_s_ys  # shape = [M]

    x_normalized = (target_x - out_mu_xs) / jnp.exp(
        out_s_xs
    )  # shape = [M] (broadcasting ([] - [M])/[M])
    y_normalized = (target_y - out_mu_ys) / jnp.exp(
        out_s_ys
    )  # shape = [M] (broadcasting ([] - [M])/[M])
    quadratic_term = -0.5 * (x_normalized**2 + y_normalized**2)  # shape = [M]

    # We retrieve a constant term to make the final logsumexp more stable.
    inside_term = pis_term + standalone_term + quadratic_term
    max_value = jnp.max(inside_term)
    return max_value + jax.nn.logsumexp(inside_term - max_value)  # shape = []


def mixture_loss(
    target_x,
    target_y,
    out_pis,
    out_mu_xs,
    out_mu_ys,
    out_s_xs,
    out_s_ys,
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
            )
            - jnp.log(2 * jnp.pi)
        )
    )


def reconstruction_loss(model, inputs, M, key=None):
    N_s = jnp.argmax(inputs[:, :, -1], axis=-1) - 1
    N_max = inputs.shape[1]
    out = jax.vmap(model, in_axes=[0, None, None])(
        inputs[:, :-1],
        key is not None,
        key,
    )
    mask = build_mask(N_max, N_s)

    out_pis = out[:, :, 0:M]
    out_mu_xs = out[:, :, M : 2 * M]
    out_mu_ys = out[:, :, 2 * M : 3 * M]
    out_s_xs = out[:, :, 3 * M : 4 * M]
    out_s_ys = out[:, :, 4 * M : 5 * M]

    target_x = inputs[:, 1:, 0]
    target_y = inputs[:, 1:, 1]

    logits = out[:, :, 5 * M :]
    target_logits = inputs[:, 1:, 2:]

    return (
        jnp.mean(
            jax.vmap(mixture_loss)(
                target_x,
                target_y,
                out_pis,
                out_mu_xs,
                out_mu_ys,
                out_s_xs,
                out_s_ys,
                mask[:, 1:],
            )
            / N_max
        )
        + jnp.mean(optax.softmax_cross_entropy(logits, target_logits)),
        {
            "out_pis": jnp.linalg.norm(out_pis),
            "out_mu_xs": jnp.linalg.norm(out_mu_xs),
            "out_mu_ys": jnp.linalg.norm(out_mu_ys),
            "out_s_xs": jnp.linalg.norm(out_s_xs),
            "out_s_ys": jnp.linalg.norm(out_s_ys),
            "logits": jnp.linalg.norm(logits),
        },
    )
