import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from .drawing import Drawing, RawDrawing, Stroke5


def build_dataset(data: list[RawDrawing]):
    stroke_5_data = []
    for raw_drawing in data:
        drawing = Drawing(raw_drawing)
        stroke_5_data.append(drawing.to_stroke5())
    return stroke_5_data


def normalize_data(
    stroke_5_data: list[list[Stroke5]],
    max_len: int | None = None,
    rescale: bool = False,
) -> list[list[Stroke5]]:
    max_len = max_len or max(map(len, stroke_5_data))
    std_norm = None
    if rescale:
        std_norm = np.std(
            [
                np.linalg.norm([dx, dy])
                for stroke_5 in stroke_5_data
                for dx, dy, *_ in stroke_5
            ]
        ).item()
    return [
        normalize_drawing(drawing, max_len, std_norm)
        for drawing in filter(lambda x: len(x) <= max_len, stroke_5_data)
    ]


def normalize_drawing(
    drawing: list[Stroke5], max_len: int, std_norm: float | None
) -> list[Stroke5]:
    return [
        (dx / std_norm if std_norm else dx, dy / std_norm if std_norm else dy, a, b, c)
        for dx, dy, a, b, c in drawing
    ] + [(0.0, 0.0, 0, 0, 1)] * (max_len - len(drawing))


def prepare_splits(
    dataset: list[list[Stroke5]],
    training_prop: float,
    dev_prop: float,
    test_prop: float,
    *,
    key: PRNGKeyArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.array(dataset, dtype=np.float32)

    # Splits.
    n = data.shape[0]
    a = int(training_prop * n / (training_prop + dev_prop + test_prop))
    b = int((training_prop + dev_prop) * n / (training_prop + dev_prop + test_prop))

    # Shuffle data using jax for reproduceability.
    idx = np.array(jax.random.permutation(x=n, key=key))
    return (data[idx[:a]], data[idx[a:b]], data[idx[b:]])
