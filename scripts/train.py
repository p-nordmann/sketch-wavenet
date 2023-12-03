import random

adjectives = ["fluffy", "sparkly", "tiny", "happy", "sunny", "cozy"]
nouns = ["kitten", "cupcake", "rainbow", "butterfly", "flower", "puppy"]
cute_name = random.choice(adjectives) + "_" + random.choice(nouns)


import argparse
import json
import os
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from eqx_wavenet import Wavenet
from PIL import Image
from tqdm import tqdm

from sketch_wavenet.config import read_tomls, write_toml
from sketch_wavenet.data_processing import build_dataset, normalize_data, prepare_splits
from sketch_wavenet.drawing import Drawing
from sketch_wavenet.logging import TensorboardLogger
from sketch_wavenet.sampling import sample
from sketch_wavenet.training import make_epoch, make_eval_step, make_step


def load_quickdraw_file(path):
    data = []
    with open(path, "r") as f:
        while line := f.readline():
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains sketch-wavenet.")
    parser.add_argument(
        "config_files", nargs="+", help="A list of config files (.toml)"
    )
    args = parser.parse_args()
    config = read_tomls(args.config_files)
    print(config)

    # PRNG keys
    key_data = jax.random.PRNGKey(config.random.seed_data)
    key_model = jax.random.PRNGKey(config.random.seed_model)
    key_training = jax.random.PRNGKey(config.random.seed_training)

    # Read data and build dataset.
    data = []
    for path in config.data.files:
        data.extend(load_quickdraw_file(path))
    stroke_5_data = build_dataset(data)
    dataset = normalize_data(
        stroke_5_data, config.data.max_stroke_len, config.data.rescale_data
    )
    key_data, key_splits = jax.random.split(key_data)
    X_train, X_dev, X_test = prepare_splits(
        dataset,
        config.data.training_prop,
        config.data.dev_prop,
        config.data.test_prop,
        key=key_splits,
    )

    # Make model.
    model = Wavenet(config=config.model.wavenet, key=key_model)

    # Util: number of training steps.
    epoch_steps = X_train.shape[0] // config.training.batch_size
    total_steps = config.training.epochs * epoch_steps

    # Make learning rate.
    learning_rate = config.training.base_learning_rate
    match config.training.schedule:
        case "constant":
            learning_rate = optax.constant_schedule(learning_rate)
        case "1cycle":
            learning_rate = optax.cosine_onecycle_schedule(
                transition_steps=total_steps,
                peak_value=config.training.peak_learning_rate,
                pct_start=config.training.pct_start,
                div_factor=config.training.div_factor,
                final_div_factor=config.training.final_div_factor,
            )
        case _:
            raise NotImplementedError(
                f"schedule {config.training.schedule} is not implemented"
            )

    # Make optimizer.
    opt_cls = getattr(optax, config.training.optimizer)
    opt = opt_cls(learning_rate)
    if config.training.weight_decay:
        opt = opt_cls(learning_rate, weight_decay=config.training.weight_decay)
    if config.training.use_gradient_clipping:
        opt = optax.chain(optax.clip(1.0), opt)
    opt_state = opt.init(model)

    # Loggers.
    logger = None
    if config.files.log_dir is not None:
        logger = TensorboardLogger(config.files.log_dir, cute_name)
        write_toml(os.path.join(config.files.log_dir, cute_name, "config.toml"), config)

    # Train.
    progress_bar = tqdm(
        desc="training",
        total=total_steps,
        unit="steps",
        dynamic_ncols=True,
    )

    for epoch in range(config.training.epochs):
        key_training, key_dev = jax.random.split(key_training)
        epoch_dev = make_epoch(X_dev, config.training.batch_size, key=key_dev)

        key_training, key_train = jax.random.split(key_training)
        for steps, inputs in enumerate(
            make_epoch(
                jnp.array(X_train),
                config.training.batch_size,
                config.data.use_data_augmentation,
                key=key_train,
            )
        ):
            key_training, key_dropout = jax.random.split(key_training)
            loss_train, model, opt_state, aux = make_step(
                model,
                inputs,
                config.model.num_gaussians,
                opt,
                opt_state,
                key=key_dropout if config.training.use_dropout else None,
            )

            if logger is not None:
                if (steps + 1) % config.training.validate_each == 0:
                    try:
                        inputs = next(epoch_dev)  # type: ignore
                    except StopIteration:
                        key_training, key_dev = jax.random.split(key_training)
                        epoch_dev = make_epoch(
                            X_dev, config.training.batch_size, key=key_dev
                        )
                        inputs = next(epoch_dev)
                    loss_dev, aux_dev = make_eval_step(
                        model, inputs, config.model.num_gaussians
                    )
                    logger.log(
                        {
                            "loss_train": cast(float, loss_train),
                            "loss_dev": cast(float, loss_dev),
                            "learning_rate": cast(
                                float, learning_rate(epoch * epoch_steps + steps)
                            ),
                            **aux,
                            **aux_dev,
                        }
                    )
                else:
                    logger.log(
                        {
                            "loss_train": cast(float, loss_train),
                            "learning_rate": cast(
                                float, learning_rate(epoch * epoch_steps + steps)
                            ),
                            **aux,
                        }
                    )  # type: ignore

            progress_bar.update(1)

        # Draw an example each epoch.
        if config.files.examples_dir is not None:
            if not os.path.isdir(config.files.examples_dir):
                os.makedirs(config.files.examples_dir)
            try:
                file_path = os.path.join(
                    config.files.examples_dir, f"epoch_{epoch+1}.png"
                )

                # Generate image and save it to png.
                key_model, key_example = jax.random.split(key_model)
                strokes = sample(config.model.num_gaussians, model, key_example, T=0.1)
                Drawing.from_stroke5(strokes).render().save_png(file_path)

                if logger:
                    # Read the image back and log it to tensorboard.
                    logger.log_image(
                        "example",
                        np.array(Image.open(file_path)).transpose((2, 0, 1)),
                    )
            except Exception as e:
                print("Error drawing:" + "\n" + str(e))

    # Dump model weights.
    if not os.path.isdir(config.files.out_dir):
        os.makedirs(config.files.out_dir)
    write_toml(os.path.join(config.files.out_dir, "config.toml"), config)
    eqx.tree_serialise_leaves(os.path.join(config.files.out_dir, "model.eqx"), model)

    # Test.
    # TODO test
