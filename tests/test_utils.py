import jax
import jax.numpy as jnp
import pytest
from sketch_wavenet.normalization import RMSLayerNorm
from sketch_wavenet.utils import build_mask


@pytest.fixture(scope="session")
def jax_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


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
