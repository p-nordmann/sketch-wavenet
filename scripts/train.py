import argparse
import json

import jax
import optax
from eqx_wavenet import Wavenet, WavenetConfig
from tqdm import tqdm

from sketch_wavenet.data_processing import build_dataset, normalize_data, prepare_splits
from sketch_wavenet.logging import TensorboardLogger
from sketch_wavenet.training import make_epoch, make_eval_step, make_step

parser = argparse.ArgumentParser(description="Trains sketch-wavenet.")
parser.add_argument("files", nargs="+", help="A list of path to data files (.ndjson)")
parser.add_argument("--optimizer", choices=["adam", "adamw", "lion"], required=True)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--use_gradient_clipping", action="store_true")
parser.add_argument("--base_learning_rate", type=float, required=True)
parser.add_argument("--schedule", choices=["constant", "1cycle"], required=True)
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

    # Make optimizer.
    opt_cls = getattr(optax, args.optimizer)
    opt = opt_cls(args.base_learning_rate)
    if args.weight_decay:
        opt = opt_cls(args.base_learning_rate, weight_decay=args.weight_decay)
    if args.use_gradient_clipping:
        opt = optax.chain(optax.clip(1.0), opt)
    opt_state = opt.init(model)

    # Loggers.
    if args.log_dir is not None:
        logger_train = TensorboardLogger(args.log_dir, "train")
        logger_dev = TensorboardLogger(args.log_dir, "dev")

    # Train.
    epoch_steps = X_train.shape[0] // args.batch_size
    total_steps = args.epochs * epoch_steps
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
            make_epoch(X_train, args.batch_size, key=key_train)
        ):
            key, key_step = jax.random.split(key)
            loss, model, opt_state = make_step(
                model,
                inputs,
                args.num_gaussians,
                opt,
                opt_state,
                key=key_step if args.use_dropout else None,
            )

            if args.log_dir is not None:
                logger_train.log({"loss": loss})  # type: ignore

            if (steps + 1) % args.validate_each == 0:
                try:
                    inputs = next(epoch_dev)  # type: ignore
                except StopIteration:
                    key, key_dev = jax.random.split(key)
                    epoch_dev = make_epoch(X_dev, args.batch_size, key=key_dev)
                    inputs = next(epoch_dev)

                loss = make_eval_step(model, inputs, args.num_gaussians)

                if args.log_dir is not None:
                    logger_dev.step = logger_train.step  # type: ignore
                    logger_dev.log({"loss": loss})  # type: ignore

            progress_bar.update(1)

    # Test.
    # TODO test
