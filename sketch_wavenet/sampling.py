import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def sample_from_mixture(M, out, key, deterministic):
    out_pis = out[0:M]
    out_mu_xs = out[M : 2 * M]
    out_mu_ys = out[2 * M : 3 * M]
    out_s_xs = out[3 * M : 4 * M]
    out_s_ys = out[4 * M : 5 * M]
    out_r_xys = out[5 * M : 6 * M]
    logits = out[6 * M :]

    key_1, key_2, key_3 = jax.random.split(key, 3)
    choice = jax.random.categorical(key_1, logits)
    if deterministic:
        choice = jnp.argmax(logits)
    gaussian_choice = jax.random.categorical(key_2, out_pis)
    if deterministic:
        gaussian_choice = jnp.argmax(out_pis)

    mu_x, mu_y, s_x, s_y, r_xy = (
        out_mu_xs[gaussian_choice],
        out_mu_ys[gaussian_choice],
        out_s_xs[gaussian_choice],
        out_s_ys[gaussian_choice],
        out_r_xys[gaussian_choice],
    )
    var_x, var_y, cov_xy = (
        jnp.exp(2 * s_x),
        jnp.exp(2 * s_y),
        jnp.tanh(r_xy) * jnp.exp(s_x) * jnp.exp(s_y),
    )
    coords = jax.random.multivariate_normal(
        key_3, jnp.array([mu_x, mu_y]), jnp.array([[var_x, cov_xy], [cov_xy, var_y]])
    )
    if deterministic:
        coords = jnp.array([mu_x, mu_y])

    return coords, choice


def sample(config, model, key, n=200, deterministic=False):
    x = jnp.zeros(shape=(n, 5), dtype=jnp.float32)
    x = x.at[0, 2].set(1)  # Origin token.
    for k in range(n):
        out = eqx.filter_jit(model)(x)[k]
        coords, choice = eqx.filter_jit(sample_from_mixture)(
            config.M, out, key, deterministic
        )
        x = x.at[k + 1, 2 + choice].set(1)
        x = x.at[k + 1, :2].set(coords)
        if choice == 2:
            return np.array(x[: k + 2])
    return np.array(x)
