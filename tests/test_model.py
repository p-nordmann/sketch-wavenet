import jax
import jax.numpy as jnp
import pytest
from sketch_wavenet.normalization import RMSLayerNorm


@pytest.fixture(scope="session")
def jax_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_rms_layernorm():
    eps = 1e-2
    norm = RMSLayerNorm(10)
    x = jnp.reshape(jnp.arange(0, 20, dtype=jnp.float16), newshape=(2, 10))
    got = jax.vmap(norm)(x)
    want = x / jnp.array([[5.339] * 10, [14.782] * 10], dtype=jnp.float16)
    assert jnp.allclose(got, want, atol=eps).item()
