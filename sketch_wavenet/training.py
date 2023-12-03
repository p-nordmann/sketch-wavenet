from typing import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from .losses import reconstruction_loss


def tree_norm(grads):
    leaves, _ = jax.tree_util.tree_flatten(grads)
    squared_norms = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
    total_norm = jnp.sqrt(sum(squared_norms))
    return total_norm


def grad_norm(grads):
    grad_norm = {}
    grad_norm["wavenet_input"] = tree_norm(grads.wavenet_input)
    for k in range(len(grads.wavenet_layers)):
        grad_norm[f"wavenet_layer_{k}"] = tree_norm(grads.wavenet_layers[k])
    grad_norm["wavenet_head"] = tree_norm(grads.wavenet_head)
    return grad_norm


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
    (loss, aux), grads = eqx.filter_value_and_grad(reconstruction_loss, has_aux=True)(
        model, inputs, M, key
    )
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return (
        loss,
        model,
        opt_state,
        {**aux, **grad_norm(grads), "model_weights": tree_norm(model)},
    )


@eqx.filter_jit
def make_eval_step(
    model: eqx.Module,
    inputs: jax.Array,
    M: int,
):
    loss, aux = reconstruction_loss(model, inputs, M)
    return (loss, {**aux, "model_weights": tree_norm(model)})
