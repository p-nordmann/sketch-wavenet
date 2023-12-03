import argparse
import json
import os
from typing import NamedTuple

import equinox as eqx
import jax
from eqx_wavenet import Wavenet
from tqdm import tqdm

from sketch_wavenet.config import read_tomls
from sketch_wavenet.drawing import Drawing
from sketch_wavenet.sampling import sample

parser = argparse.ArgumentParser(description="Generate examples with trained model.")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--examples_dir", type=str, required=True)
parser.add_argument("--num_examples", type=int, default=5)


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_tomls([os.path.join(args.model_dir, "config.toml")])
    print(args)
    print(config.model.wavenet)

    key = jax.random.PRNGKey(0)

    key, key_model = jax.random.split(key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.model_dir, "model.eqx"),
        Wavenet(config=config.model.wavenet, key=key_model),
    )

    if not os.path.isdir(args.examples_dir):
        os.makedirs(args.examples_dir)

    for k in tqdm(range(args.num_examples)):
        try:
            key, key_example = jax.random.split(key)
            Drawing.from_stroke5(
                sample(config.model.num_gaussians, model, key_example, config.data.max_stroke_len)  # type: ignore
            ).render().save_png(os.path.join(args.examples_dir, f"example_{k+1}.png"))
        except Exception as e:
            print(f"Error drawing example {k+1}:" + "\n" + str(e))
