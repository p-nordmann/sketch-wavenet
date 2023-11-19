from typing import Iterable

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray

from .losses import reconstruction_loss


def make_epoch(
    data: jax.Array,
    batch_size: int,
    augment: bool = False,
    *,
    key: PRNGKeyArray,
) -> Iterable[jax.Array]:
    n = data.shape[0]

    # Make permutation of the data.
    key, key_permutation = jax.random.split(key)
    idx = jax.random.permutation(x=data.shape[0], key=key_permutation)

    # Build batches.
    for k in range(0, n, batch_size):
        if k + batch_size > n:  # Skip the end
            break
        batch = data[idx[k : k + batch_size]]
        if augment:
            key, key_augmentation = jax.random.split(key)
            augmentation_factor = jax.random.uniform(
                key=key_augmentation,
                shape=(batch.shape[0], 1, 1),
                minval=0.9,
                maxval=1.1,
            )
            batch = batch.at[:, :, :2].multiply(augmentation_factor)
        yield batch


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    inputs: jax.Array,
    M: int,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    key: PRNGKeyArray,
):
    loss, grads = eqx.filter_value_and_grad(reconstruction_loss)(model, inputs, M, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def make_eval_step(
    model: eqx.Module,
    inputs: jax.Array,
    M: int,
):
    return reconstruction_loss(model, inputs, M)
