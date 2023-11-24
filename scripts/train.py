import random

adjectives = ["fluffy", "sparkly", "tiny", "happy", "sunny", "cozy"]
nouns = ["kitten", "cupcake", "rainbow", "butterfly", "flower", "puppy"]
cute_name = random.choice(adjectives) + "_" + random.choice(nouns)


import argparse
import json
import os
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import toml
from eqx_wavenet import Wavenet, WavenetConfig
from PIL import Image
from tqdm import tqdm

from sketch_wavenet.data_processing import build_dataset, normalize_data, prepare_splits
from sketch_wavenet.drawing import Drawing
from sketch_wavenet.logging import TensorboardLogger
from sketch_wavenet.sampling import sample
from sketch_wavenet.training import make_epoch, make_eval_step, make_step

parser = argparse.ArgumentParser(description="Trains sketch-wavenet.")
parser.add_argument("files", nargs="+", help="A list of path to data files (.ndjson)")
parser.add_argument("--optimizer", choices=["adam", "adamw", "lion"], required=True)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--use_gradient_clipping", action="store_true")
parser.add_argument("--base_learning_rate", type=float, required=True)
parser.add_argument("--schedule", choices=["constant", "1cycle"], required=True)
parser.add_argument("--peak_learning_rate", type=float, default=1e-3)
parser.add_argument("--use_dropout", action="store_true")
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument(
    "--validate_each",
    type=int,
    required=True,
    help="Specifies how often a step of validation will be run.",
)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--examples_dir", type=str)
parser.add_argument("--model_dir", type=str, required=True)

parser.add_argument("--max_stroke_len", type=int, default=200)
parser.add_argument("--rescale_data", action="store_true")
parser.add_argument("--num_gaussians", type=int, default=20)
parser.add_argument("--use_data_augmentation", action="store_true")


def to_toml(path: str, config: NamedTuple) -> None:
    with open(path, "w") as f:
        toml.dump(config._asdict(), f)


def load_quickdraw_file(path):
    data = []
    with open(path, "r") as f:
        while line := f.readline():
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    args = parser.parse_args()
    print("Args:", args)

    key = jax.random.PRNGKey(0)

    data = []
    for path in args.files:
        data.extend(load_quickdraw_file(path))

    stroke_5_data = build_dataset(data)
    dataset = normalize_data(stroke_5_data, args.max_stroke_len, args.rescale_data)

    key, key_splits = jax.random.split(key)
    X_train, X_dev, X_test = prepare_splits(dataset, key=key_splits)

    # TODO configure wavenet
    wavenet_config = WavenetConfig(
        num_layers=9,
        layer_dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        size_in=5,
        input_kernel_size=1,
        size_layers=64,
        size_hidden=256,
        size_out=6 * args.num_gaussians + 3,
    )

    # Make model.
    key, key_model = jax.random.split(key)
    model = Wavenet(
        config=wavenet_config,
        key=key_model,
    )

    # Util: number of training steps.
    epoch_steps = X_train.shape[0] // args.batch_size
    total_steps = args.epochs * epoch_steps

    # Make learning rate.
    learning_rate = args.base_learning_rate
    match args.schedule:
        case "constant":
            optax.constant_schedule(learning_rate)
        case "1cycle":
            learning_rate = optax.cosine_onecycle_schedule(
                transition_steps=total_steps,
                peak_value=args.peak_learning_rate,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e3,
            )

    # Make optimizer.
    opt_cls = getattr(optax, args.optimizer)
    opt = opt_cls(learning_rate)
    if args.weight_decay:
        opt = opt_cls(learning_rate, weight_decay=args.weight_decay)
    if args.use_gradient_clipping:
        opt = optax.chain(optax.clip(1.0), opt)
    opt_state = opt.init(model)

    # Loggers.
    logger = None
    if args.log_dir is not None:
        logger = TensorboardLogger(args.log_dir, cute_name)

    # Train.
    progress_bar = tqdm(
        desc="training",
        total=total_steps,
        unit=" steps",
        dynamic_ncols=True,
    )

    for epoch in range(args.epochs):
        key, key_dev = jax.random.split(key)
        epoch_dev = make_epoch(X_dev, args.batch_size, key=key_dev)

        key, key_train = jax.random.split(key)
        for steps, inputs in enumerate(
            make_epoch(
                jnp.array(X_train),
                args.batch_size,
                args.use_data_augmentation,
                key=key_train,
            )
        ):
            key, key_step = jax.random.split(key)
            loss_train, model, opt_state = make_step(
                model,
                inputs,
                args.num_gaussians,
                opt,
                opt_state,
                key=key_step if args.use_dropout else None,
            )

            if logger is not None:
                if (steps + 1) % args.validate_each == 0:
                    try:
                        inputs = next(epoch_dev)  # type: ignore
                    except StopIteration:
                        key, key_dev = jax.random.split(key)
                        epoch_dev = make_epoch(X_dev, args.batch_size, key=key_dev)
                        inputs = next(epoch_dev)
                    loss_dev = make_eval_step(model, inputs, args.num_gaussians)
                    logger.log(
                        {
                            "loss_train": cast(float, loss_train),
                            "loss_dev": cast(float, loss_dev),
                            "learning_rate": cast(
                                float, learning_rate(epoch * epoch_steps + steps)
                            ),
                        }
                    )
                else:
                    logger.log(
                        {
                            "loss_train": cast(float, loss_train),
                            "learning_rate": cast(
                                float, learning_rate(epoch * epoch_steps + steps)
                            ),
                        }
                    )  # type: ignore

            progress_bar.update(1)

        # Draw an example each epoch.
        if args.examples_dir is not None:
            if not os.path.isdir(args.examples_dir):
                os.makedirs(args.examples_dir)
            try:
                file_path = os.path.join(args.examples_dir, f"epoch_{epoch+1}.png")

                # Generate image and save it to png.
                key, key_example = jax.random.split(key)
                strokes = sample(args.num_gaussians, model, key_example)
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
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    to_toml(os.path.join(args.model_dir, "model_config.toml"), wavenet_config)
    eqx.tree_serialise_leaves(os.path.join(args.model_dir, "model.eqx"), model)

    # Test.
    # TODO test
