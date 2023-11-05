from typing import TypeVar
from jaxtyping import Array, Float32, PRNGKeyArray

import jax
import jax.numpy as jnp
import equinox as eqx

from .normalization import RMSLayerNorm

# Base field for all tensor spaces.
K = Float32

T = TypeVar("T")
Pair = tuple[T, T]


def dilated_conv(dim: int, dilation: int, *, key: PRNGKeyArray) -> eqx.nn.Conv1d:
    return eqx.nn.Conv1d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=2,
        dilation=dilation,
        padding=[(dilation, 0)],  # We try to preserve the same length as input.
        key=key,
    )


def pointwise_conv(dim: int, *, key: PRNGKeyArray) -> eqx.nn.Conv1d:
    return eqx.nn.Conv1d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=1,
        key=key,
    )


class WavenetLayer(eqx.Module):
    filter_conv: eqx.nn.Conv1d
    gate_conv: eqx.nn.Conv1d
    residual_conv: eqx.nn.Conv1d
    skip_conv: eqx.nn.Conv1d
    norm: RMSLayerNorm

    def __init__(self, dim: int, dilation: int, *, key: PRNGKeyArray):
        key_1, key_2, key_3, key_4 = jax.random.split(key, 4)

        self.filter_conv = dilated_conv(dim, dilation, key=key_1)
        self.gate_conv = dilated_conv(dim, dilation, key=key_2)

        self.residual_conv = pointwise_conv(dim, key=key_3)
        self.skip_conv = pointwise_conv(dim, key=key_4)

        self.norm = RMSLayerNorm(dim)

    def __call__(self, x: K[Array, "dim time"]) -> Pair[K[Array, "dim time"]]:
        x_normalized = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        z = jax.nn.tanh(self.filter_conv(x_normalized)) * jax.nn.sigmoid(
            self.gate_conv(x_normalized)
        )
        return self.residual_conv(z) + x, self.skip_conv(z)


class WavenetHead(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    norm: RMSLayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self, size_in: int, size_hidden: int, size_out: int, *, key: PRNGKeyArray
    ):
        key_1, key_2 = jax.random.split(key)
        self.linear_in = eqx.nn.Linear(size_in, size_hidden, key=key_1)
        self.linear_out = eqx.nn.Linear(size_hidden, size_out, key=key_2)
        self.norm = RMSLayerNorm(size_in)
        self.dropout = eqx.nn.Dropout()

    def __call__(
        self,
        x: K[Array, "size_in"],
        enable_dropout: bool = False,
        key: None | PRNGKeyArray = None,
    ) -> K[Array, "size_out"]:
        x_normalized = self.norm(x)
        z = jax.nn.relu(self.linear_in(x_normalized))
        z = self.dropout(z, inference=not enable_dropout, key=key)
        return self.linear_out(z)


class WavenetInput(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(self, size_in: int, dim: int, kernel_size: int, *, key: PRNGKeyArray):
        self.conv = eqx.nn.Conv1d(
            in_channels=size_in,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=[(kernel_size - 1, 0)],
            key=key,
        )

    def __call__(self, x: K[Array, "time size_in"]) -> K[Array, "dim time"]:
        return self.conv(jnp.transpose(x))


class Wavenet(eqx.Module):
    input: WavenetInput
    layers: list[WavenetLayer]
    head: WavenetHead

    def __init__(
        self,
        size_in: int,
        dim: int,
        size_out: int,
        dilations: list[int],
        *,
        key: PRNGKeyArray,
    ):
        key_input, key_layers, key_head = jax.random.split(key, 3)
        self.input = WavenetInput(size_in, dim, 32, key=key_input)
        self.layers = [
            WavenetLayer(dim, dilation, key=key)
            for dilation, key in zip(
                dilations, jax.random.split(key_layers, len(dilations))
            )
        ]
        self.head = WavenetHead(dim, 4 * dim, size_out, key=key_head)

    def __call__(
        self,
        x: K[Array, "time"],
        enable_dropout: bool = False,
        key: None | PRNGKeyArray = None,
    ) -> K[Array, "time size_out"]:
        if len(x.shape) == 1:
            x = x[:, jnp.newaxis]
        z = self.input(x)
        sum_out = jnp.zeros_like(z)
        for layer in self.layers:
            z, out = layer(z)
            sum_out += out
        out = jax.nn.relu(sum_out)
        return jax.vmap(self.head, in_axes=[1, None, None])(out, enable_dropout, key)
