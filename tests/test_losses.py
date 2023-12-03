import jax
import jax.numpy as jnp

from sketch_wavenet.losses import build_mask, inner_mixture_loss


def test_inner_mixture_loss_one_gaussian():
    eps = 1e-6
    target_x = jnp.array([0, 0, 0], dtype=jnp.float32)
    target_y = jnp.array([0, 0, 0], dtype=jnp.float32)
    out_pis = jnp.array([[0], [0], [0]], dtype=jnp.float32)
    out_mu_xs = jnp.array([[0], [1], [0]], dtype=jnp.float32)
    out_mu_ys = jnp.array([[0], [0], [2]], dtype=jnp.float32)
    out_s_xs = jnp.array([[0], [jnp.log(2)], [0]], dtype=jnp.float32)
    out_s_ys = jnp.array([[0], [jnp.log(2)], [0]], dtype=jnp.float32)
    got = jax.vmap(inner_mixture_loss)(
        target_x,
        target_y,
        out_pis,
        out_mu_xs,
        out_mu_ys,
        out_s_xs,
        out_s_ys,
    )
    want = jnp.log(
        jnp.array(
            [
                1,
                jnp.exp(-1 / 8) / 4,
                jnp.exp(-2),
            ],
            dtype=jnp.float32,
        )
    ) - jnp.log(
        2
    )  # retrieve log(2) because we added a 1 to the pis denominator
    assert jnp.allclose(got, want, atol=eps).item()


def test_build_mask():
    N_max = 6
    N_s = jnp.array([3, 4, 1, 1, 6])
    batch_size = N_s.shape[0]

    mask = build_mask(N_max, N_s)

    h, w = mask.shape
    assert h == batch_size
    assert w == N_max

    comparison = mask == jnp.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    assert jnp.sum(comparison).item() == batch_size * N_max
