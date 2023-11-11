import jax
import jax.numpy as jnp
import pytest

from sketch_wavenet.gaussian_mixture import inner_mixture_loss


@pytest.fixture(scope="session")
def jax_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_inner_mixture_loss_one_gaussian():
    eps = 1e-6
    target_x = jnp.array([0, 0, 0], dtype=jnp.float32)
    target_y = jnp.array([0, 0, 0], dtype=jnp.float32)
    out_pis = jnp.array([[0], [0], [0]], dtype=jnp.float32)
    out_mu_xs = jnp.array([[0], [1], [0]], dtype=jnp.float32)
    out_mu_ys = jnp.array([[0], [0], [2]], dtype=jnp.float32)
    out_s_xs = jnp.array([[0], [jnp.log(2)], [0]], dtype=jnp.float32)
    out_s_ys = jnp.array([[0], [jnp.log(2)], [0]], dtype=jnp.float32)
    out_r_xys = jnp.array([[0], [0], [1]], dtype=jnp.float32)
    got = jax.vmap(inner_mixture_loss)(
        target_x,
        target_y,
        out_pis,
        out_mu_xs,
        out_mu_ys,
        out_s_xs,
        out_s_ys,
        out_r_xys,
    )
    want = jnp.log(
        jnp.array(
            [
                1,
                jnp.exp(-1 / 8) / 4,
                jnp.exp(-2 * jnp.cosh(1) ** 2) * jnp.cosh(1),
            ],
            dtype=jnp.float32,
        )
    )
    assert jnp.allclose(got, want, atol=eps).item()